"""Train a randomly initialized GPT on one prepared text dataset."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from .data import ROOT, Tokenizer, get_batch
from .model import GPT, GPTConfig


def choose_device(requested):
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested, but no CUDA device is available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested, but no MPS device is available")
    return requested


def precision(device):
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def evaluate(model, tokens, batch_size, batches, seed, device):
    was_training = model.training
    model.eval()
    # Reuse the same validation windows without consuming the training RNG.
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(batches):
        x, y = get_batch(tokens, batch_size, model.config.block_size, generator, device)
        with precision(device):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train(was_training)
    result = sum(losses) / len(losses)
    if not math.isfinite(result):
        raise FloatingPointError("Validation loss is not finite")
    return result


def save_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/shakespeare-char.pt")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=200, help="Number of additional optimizer updates")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--embed", type=int)
    parser.add_argument("--lr", type=float, help="Constant learning rate; defaults to 3e-4")
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4, help="CPU threads; avoids using every shared-machine core")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.grad_accum, args.eval_every, args.eval_batches, args.threads) <= 0:
        parser.error("Steps, batch sizes, evaluation intervals, and threads must be positive")
    if args.lr is not None and (not math.isfinite(args.lr) or args.lr <= 0):
        parser.error("Learning rate must be finite and positive")
    device = choose_device(args.device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    dataset = torch.load(args.data, map_location="cpu", weights_only=True)
    with args.data.open("rb") as stream:
        data_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    tokenizer = Tokenizer(dataset["tokenizer"])
    requested = {"block_size": args.block_size, "n_layer": args.layers, "n_head": args.heads, "n_embd": args.embed}
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint["data_sha256"] != data_sha256:
            parser.error("Resume dataset differs from the checkpoint's dataset")
        config = GPTConfig(**checkpoint["model_config"])
        for key, value in requested.items():
            if value is not None and value != getattr(config, key):
                parser.error(f"Cannot change {key} when resuming a checkpoint")
    else:
        config = GPTConfig(vocab_size=tokenizer.vocab_size, **{key: value for key, value in requested.items() if value is not None})
    for split in ("train", "val"):
        if len(dataset[split]) <= config.block_size:
            parser.error(f"{split} split is too short for context length {config.block_size}")
    output = args.out_dir or (args.resume.parent if args.resume else ROOT / "runs/tiny")
    if not args.resume and output.exists() and any(output.iterdir()):
        parser.error(f"Run directory is not empty: {output}. Use --resume or a new --out-dir")
    model = GPT(config).to(device)
    optimizer = model.configure_optimizer(args.lr if args.lr is not None else 3e-4)
    generator = torch.Generator().manual_seed(args.seed + 1)
    step, best_val, seed = 0, float("inf"), args.seed
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        generator.set_state(checkpoint["batch_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        step, best_val, seed = checkpoint["step"], checkpoint["best_val"], checkpoint["seed"]
        if args.lr is not None:
            for group in optimizer.param_groups:
                group["lr"] = args.lr
    output.mkdir(parents=True, exist_ok=True)
    details = {
        "model": asdict(config), "tokenizer": dataset["tokenizer"], "device": device,
        "data_sha256": data_sha256, "torch_version": str(torch.__version__),
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "learning_rate": optimizer.param_groups[0]["lr"], "seed": seed,
    }
    (output / "config.json").write_text(json.dumps(details, indent=2) + "\n")
    count = sum(parameter.numel() for parameter in model.parameters())
    tokens_per_step = args.batch_size * config.block_size * args.grad_accum
    print(f"Device: {device}; parameters: {count:,}; tokens/update: {tokens_per_step:,}", flush=True)
    print(f"Run: {output.resolve()}", flush=True)

    def validate_and_save():
        nonlocal best_val
        val_loss = evaluate(model, dataset["val"], args.batch_size, args.eval_batches, seed + 2, device)
        improved = val_loss < best_val
        best_val = min(best_val, val_loss)
        payload = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "model_config": asdict(config), "tokenizer": dataset["tokenizer"],
            "data_sha256": data_sha256, "step": step, "best_val": best_val, "seed": seed,
            "batch_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else None,
        }
        save_checkpoint(output / "latest.pt", payload)
        if improved:
            save_checkpoint(output / "best.pt", payload)
        metrics.write(json.dumps({"step": step, "val_loss": val_loss}) + "\n")
        metrics.flush()
        print(f"step {step:5d} | validation loss {val_loss:.4f}", flush=True)

    with (output / "metrics.jsonl").open("a") as metrics:
        validate_and_save()
        final_step = step + args.steps
        while step < final_step:
            started = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_loss = 0.0
            for _ in range(args.grad_accum):
                x, y = get_batch(dataset["train"], args.batch_size, config.block_size, generator, device)
                with precision(device):
                    _, loss = model(x, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Training loss is not finite")
                (loss / args.grad_accum).backward()
                train_loss += loss.detach().item() / args.grad_accum
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()
            step += 1
            elapsed = time.perf_counter() - started
            metrics.write(json.dumps({"step": step, "train_loss": train_loss, "grad_norm": norm.item(), "tokens_per_second": tokens_per_step / elapsed}) + "\n")
            if step % 10 == 0 or step == final_step:
                print(f"step {step:5d} | train loss {train_loss:.4f} | {tokens_per_step / elapsed:,.0f} tokens/s", flush=True)
            if step % args.eval_every == 0 or step == final_step:
                validate_and_save()


if __name__ == "__main__":
    main()

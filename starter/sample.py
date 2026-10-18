"""Generate text from a checkpoint trained by starter.train."""

import argparse
import math
from pathlib import Path

import torch

from .data import Tokenizer
from .model import GPT, GPTConfig
from .train import choose_device, precision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="First Citizen:\n")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--output", type=Path, help="Optionally save the generated text")
    args = parser.parse_args()
    if not args.prompt or args.tokens < 0 or args.top_k <= 0 or args.threads <= 0:
        parser.error("Supply a nonempty prompt, nonnegative token count, and positive top-k/threads")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("Temperature must be finite and positive")
    device = choose_device(args.device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tokenizer = Tokenizer(checkpoint["tokenizer"])
    model = GPT(GPTConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokens = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    with torch.inference_mode():
        for _ in range(args.tokens):
            with precision(device):
                logits, _ = model(tokens[:, -model.config.block_size:])
            logits = logits[:, -1].float() / args.temperature
            values, indices = torch.topk(logits, min(args.top_k, logits.size(-1)))
            choice = torch.multinomial(torch.softmax(values, dim=-1), num_samples=1)
            tokens = torch.cat((tokens, indices.gather(-1, choice)), dim=1)
    text = tokenizer.decode(tokens[0].tolist())
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

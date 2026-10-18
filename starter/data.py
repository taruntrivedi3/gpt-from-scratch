"""Prepare a small text corpus and sample correctly shifted next-token batches."""

import argparse
import hashlib
import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


class Tokenizer:
    def __init__(self, metadata):
        self.metadata = metadata
        self.kind = metadata["kind"]
        if self.kind == "char":
            self.characters = metadata["characters"]
            self.to_id = {char: index for index, char in enumerate(self.characters)}
            self.unknown_id = len(self.characters)
            self.vocab_size = self.unknown_id + 1
        elif self.kind == "gpt2":
            import tiktoken

            os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(ROOT / ".cache/tiktoken"))
            self.encoding = tiktoken.get_encoding("gpt2")
            self.vocab_size = self.encoding.n_vocab
        else:
            raise ValueError(f"Unknown tokenizer: {self.kind}")

    def encode(self, text):
        if self.kind == "char":
            return [self.to_id.get(char, self.unknown_id) for char in text]
        return self.encoding.encode_ordinary(text)

    def decode(self, tokens):
        if self.kind == "char":
            return "".join(self.characters[index] if index < len(self.characters) else "�" for index in tokens)
        return self.encoding.decode(tokens)


def prepare(source, output, kind="char", val_fraction=0.1):
    source, output = Path(source), Path(output)
    if not 0 < val_fraction < 1:
        raise ValueError("Validation fraction must be between 0 and 1")
    if output.exists():
        raise FileExistsError(f"Dataset already exists: {output}; choose a new output path")
    raw = source.read_bytes()
    text = raw.decode("utf-8")
    boundary = int(len(text) * (1 - val_fraction))
    train_text, val_text = text[:boundary], text[boundary:]
    if min(len(train_text), len(val_text)) < 2:
        raise ValueError("Both text splits must contain at least two characters")
    metadata = {"kind": kind}
    if kind == "char":
        # Fit only on training text. Unseen validation characters use an UNK id.
        metadata["characters"] = "".join(sorted(set(train_text)))
    tokenizer = Tokenizer(metadata)
    payload = {
        "train": torch.tensor(tokenizer.encode(train_text), dtype=torch.int32),
        "val": torch.tensor(tokenizer.encode(val_text), dtype=torch.int32),
        "tokenizer": metadata,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source": str(source.resolve()),
        "val_fraction": val_fraction,
    }
    if min(len(payload["train"]), len(payload["val"])) < 2:
        raise ValueError("Both token splits must contain at least two tokens")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(f"Saved {output}: {len(payload['train']):,} train / {len(payload['val']):,} validation tokens")
    print(f"Tokenizer: {kind}; vocabulary: {tokenizer.vocab_size:,}")
    if kind == "char":
        print(f"Unknown validation characters: {(payload['val'] == tokenizer.unknown_id).sum().item()}")


def get_batch(tokens, batch_size, block_size, generator, device="cpu"):
    if len(tokens) <= block_size:
        raise ValueError(f"Split has {len(tokens)} tokens; need at least {block_size + 1}")
    starts = torch.randint(len(tokens) - block_size, (batch_size,), generator=generator)
    positions = starts[:, None] + torch.arange(block_size)
    # y[t] is the token immediately after x[t], even at the window boundary.
    return tokens[positions].long().to(device), tokens[positions + 1].long().to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "input.txt")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tokenizer", choices=("char", "gpt2"), default="char")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()
    output = args.output or ROOT / f"data/shakespeare-{args.tokenizer}.pt"
    prepare(args.input, output, args.tokenizer, args.val_fraction)


if __name__ == "__main__":
    main()

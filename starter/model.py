"""GPT-2 architecture, adapted from Andrej Karpathy's MIT-licensed build-nanogpt.

Reference: train_gpt2.py at commit 6104ab1b53920f6e2159749676073ff7d815c1fa.
Unlike that teaching script, importing this module never starts training.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128

    def __post_init__(self):
        if min(self.vocab_size, self.block_size, self.n_layer, self.n_head, self.n_embd) <= 0:
            raise ValueError("Model dimensions must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("Embedding width must be divisible by the number of heads")


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = True

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        # Split channels into heads: (B, T, C) -> (B, H, T, C/H).
        q, k, v = [
            tensor.view(batch, length, self.n_head, width // self.n_head).transpose(1, 2)
            for tensor in (q, k, v)
        ]
        # Position t must never see tokens after t during next-token prediction.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch, length, width)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = True

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "wpe": nn.Embedding(config.block_size, config.n_embd),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": nn.LayerNorm(config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Input embeddings and output classifier share one parameter tensor.
        self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            std = 0.02
            if getattr(module, "NANOGPT_SCALE_INIT", False):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens, targets=None):
        _, length = tokens.shape
        if not 0 < length <= self.config.block_size:
            raise ValueError(f"Sequence length must be between 1 and {self.config.block_size}")
        positions = torch.arange(length, device=tokens.device)
        x = self.transformer.wte(tokens) + self.transformer.wpe(positions)
        for block in self.transformer.h:
            x = block(x)
        logits = self.lm_head(self.transformer.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def configure_optimizer(self, learning_rate):
        parameters = list(self.parameters())
        # Biases and layer-norm gains are not weight-decayed.
        groups = [
            {"params": [p for p in parameters if p.ndim >= 2], "weight_decay": 0.1},
            {"params": [p for p in parameters if p.ndim < 2], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)

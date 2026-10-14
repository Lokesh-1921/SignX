"""Variable-length temporal Transformer encoder."""

from __future__ import annotations

import torch
from torch import nn


class TemporalTransformer(nn.Module):
    def __init__(self, hidden_dim: int, layers: int, heads: int, dropout: float = 0.1, max_length: int = 512) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by temporal heads")
        self.position = nn.Parameter(torch.zeros(1, max_length, hidden_dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        block = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads, dim_feedforward=hidden_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.max_length = max_length

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"expected tokens [B,T,H], got {tuple(tokens.shape)}")
        if tokens.shape[1] > self.max_length:
            raise ValueError(f"sequence length {tokens.shape[1]} exceeds max_length {self.max_length}")
        values = tokens + self.position[:, : tokens.shape[1]]
        padding_mask = None if attention_mask is None else ~attention_mask.bool()
        return self.norm(self.encoder(values, src_key_padding_mask=padding_mask))

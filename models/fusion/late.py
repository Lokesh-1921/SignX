"""Configurable late fusion of aligned modality tokens."""

from __future__ import annotations

import torch
from torch import nn


class LateFusion(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, rgb: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        if rgb.shape != pose.shape:
            raise ValueError(f"RGB and pose shapes must match, got {rgb.shape} and {pose.shape}")
        gate = self.gate(torch.cat([rgb, pose], dim=-1))
        return self.norm(rgb + self.dropout(gate * pose))

"""Pose/hand/face landmark feature encoder."""

from __future__ import annotations

import torch
from torch import nn


class PoseEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("pose input_dim must be positive")
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        if pose.ndim != 3:
            raise ValueError(f"expected pose [B,T,F], got {tuple(pose.shape)}")
        return self.network(pose)

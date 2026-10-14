"""Small trainable RGB encoder for baseline and smoke-test experiments."""

from __future__ import annotations

import torch
from torch import nn

from models.common import VisualEncoder


class TinyFrameEncoder(VisualEncoder):
    """A deliberately modest ConvNet; replace with VideoMAE/ViT via the interface."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, embedding_dim, kernel_size=3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.embedding_dim = embedding_dim

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError(f"expected frames [B,T,3,H,W], got {tuple(frames.shape)}")
        batch, time = frames.shape[:2]
        flat = frames.reshape(batch * time, *frames.shape[2:]).contiguous()
        features = self.features(flat).reshape(batch * time, -1)
        return features.reshape(batch, time, self.embedding_dim)

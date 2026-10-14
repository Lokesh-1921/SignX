"""Interfaces shared by visual and multimodal encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class VisualEncoder(nn.Module, ABC):
    """Common contract for swappable image/video backbones."""

    @abstractmethod
    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Map ``[batch, time, channel, height, width]`` to frame embeddings."""

    def encode_video(self, frames: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        embeddings = self.encode_frames(frames)
        if attention_mask is None:
            return embeddings.mean(dim=1)
        weights = attention_mask.unsqueeze(-1).to(embeddings.dtype)
        return (embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)

    def get_embedding(self, frames: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.encode_video(frames, attention_mask)

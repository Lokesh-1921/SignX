"""The first working SignBridge isolated-sign recognition model."""

from __future__ import annotations

import torch
from torch import nn

from .fusion import LateFusion
from .pose import PoseEncoder
from .temporal import TemporalTransformer
from .vision import TinyFrameEncoder


class SignBridgeBaseline(nn.Module):
    """RGB or RGB+pose temporal classifier with reusable sign embeddings."""

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        temporal_layers: int = 4,
        temporal_heads: int = 8,
        dropout: float = 0.1,
        use_pose: bool = False,
        pose_dim: int = 0,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("isolated-sign baseline requires at least two classes")
        self.visual_encoder = TinyFrameEncoder(hidden_dim)
        self.use_pose = use_pose
        if use_pose:
            self.pose_encoder = PoseEncoder(pose_dim, hidden_dim, dropout)
            self.fusion = LateFusion(hidden_dim, dropout)
        self.temporal_encoder = TemporalTransformer(hidden_dim, temporal_layers, temporal_heads, dropout)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def get_sign_embedding(
        self, frames: torch.Tensor, attention_mask: torch.Tensor | None = None, pose: torch.Tensor | None = None
    ) -> torch.Tensor:
        tokens = self.visual_encoder.encode_frames(frames)
        if self.use_pose:
            if pose is None:
                raise ValueError("model was configured for pose, but no pose tensor was supplied")
            tokens = self.fusion(tokens, self.pose_encoder(pose))
        encoded = self.temporal_encoder(tokens, attention_mask)
        if attention_mask is None:
            return encoded.mean(dim=1)
        weights = attention_mask.unsqueeze(-1).to(encoded.dtype)
        return (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)

    def forward(
        self, frames: torch.Tensor, attention_mask: torch.Tensor | None = None, pose: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        embedding = self.get_sign_embedding(frames, attention_mask, pose)
        return {"logits": self.classifier(embedding), "embedding": embedding}

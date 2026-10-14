"""Configurable spatial and temporal augmentations for sign-language video.

Sign-language-aware design choices:
- Horizontal flip swaps left/right semantically, so it is off by default
  and must be explicitly enabled when the downstream task is flip-invariant.
- Temporal speed perturbation preserves the *meaning* of motion speed, so
  the range is intentionally narrow (0.8×–1.2×).
- All transforms operate on [T, C, H, W] float tensors (after normalisation).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AugmentationConfig:
    """Declarative augmentation specification loaded from YAML."""

    # Spatial
    random_crop: bool = True
    crop_scale: tuple[float, float] = (0.8, 1.0)
    random_horizontal_flip: bool = False  # off by default: sign-language semantics
    flip_probability: float = 0.5
    color_jitter: bool = True
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.05
    random_rotation: bool = False
    rotation_degrees: float = 15.0
    random_erasing: bool = False
    erasing_probability: float = 0.25
    erasing_scale: tuple[float, float] = (0.02, 0.15)

    # Temporal
    temporal_crop: bool = False
    temporal_crop_fraction: tuple[float, float] = (0.8, 1.0)
    speed_perturbation: bool = False
    speed_range: tuple[float, float] = (0.8, 1.2)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "AugmentationConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {}
        for key, value in values.items():
            if key in known:
                spec = cls.__dataclass_fields__[key]
                if spec.type.startswith("tuple"):
                    filtered[key] = tuple(value) if isinstance(value, list) else value
                else:
                    filtered[key] = value
        return cls(**filtered)

    @classmethod
    def disabled(cls) -> "AugmentationConfig":
        """Return a config with all augmentation disabled (for validation)."""
        return cls(
            random_crop=False,
            random_horizontal_flip=False,
            color_jitter=False,
            random_rotation=False,
            random_erasing=False,
            temporal_crop=False,
            speed_perturbation=False,
        )


# ---------------------------------------------------------------------------
# Spatial transforms (operate on individual frames: [C, H, W])
# ---------------------------------------------------------------------------

def _random_crop_resize(frames: torch.Tensor, scale: tuple[float, float]) -> torch.Tensor:
    """Random crop and resize back to original size.  frames: [T, C, H, W]."""
    _, _, height, width = frames.shape
    area = height * width
    ratio = random.uniform(scale[0], scale[1])
    crop_area = int(area * ratio)
    crop_side = int(math.sqrt(crop_area))
    crop_h = min(crop_side, height)
    crop_w = min(crop_side, width)
    top = random.randint(0, height - crop_h)
    left = random.randint(0, width - crop_w)
    cropped = frames[:, :, top : top + crop_h, left : left + crop_w]
    return F.interpolate(cropped, size=(height, width), mode="bilinear", align_corners=False)


def _random_horizontal_flip(frames: torch.Tensor) -> torch.Tensor:
    """Flip all frames in a clip consistently."""
    return frames.flip(dims=[-1])


def _color_jitter(
    frames: torch.Tensor,
    brightness: float,
    contrast: float,
    saturation: float,
    hue: float,
) -> torch.Tensor:
    """Apply consistent colour jitter across all frames in a clip."""
    # Brightness
    if brightness > 0:
        factor = 1.0 + random.uniform(-brightness, brightness)
        frames = frames * factor

    # Contrast
    if contrast > 0:
        factor = 1.0 + random.uniform(-contrast, contrast)
        mean = frames.mean(dim=(-2, -1), keepdim=True)
        frames = (frames - mean) * factor + mean

    # Saturation
    if saturation > 0:
        factor = 1.0 + random.uniform(-saturation, saturation)
        gray = frames.mean(dim=-3, keepdim=True)
        frames = frames * factor + gray * (1.0 - factor)

    # Hue (simplified: rotate in channel space)
    if hue > 0:
        shift = random.uniform(-hue, hue)
        frames = frames + shift

    return frames.clamp(0.0, 1.0)


def _random_rotation(frames: torch.Tensor, max_degrees: float) -> torch.Tensor:
    """Rotate all frames by the same random angle."""
    angle = random.uniform(-max_degrees, max_degrees)
    rad = math.radians(angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    # Affine matrix for rotation around centre
    theta = torch.tensor([[cos_a, -sin_a, 0], [sin_a, cos_a, 0]], dtype=frames.dtype, device=frames.device)
    theta = theta.unsqueeze(0).expand(frames.shape[0], -1, -1)
    grid = F.affine_grid(theta, frames.size(), align_corners=False)
    return F.grid_sample(frames, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def _random_erasing(
    frames: torch.Tensor, probability: float, scale: tuple[float, float]
) -> torch.Tensor:
    """Apply the same random erasing patch to all frames."""
    if random.random() > probability:
        return frames
    _, _, height, width = frames.shape
    area = height * width
    erase_area = int(area * random.uniform(scale[0], scale[1]))
    erase_h = min(int(math.sqrt(erase_area)), height)
    erase_w = min(int(erase_area / max(erase_h, 1)), width)
    top = random.randint(0, height - erase_h)
    left = random.randint(0, width - erase_w)
    frames = frames.clone()
    frames[:, :, top : top + erase_h, left : left + erase_w] = torch.rand(
        frames.shape[0], frames.shape[1], erase_h, erase_w,
        dtype=frames.dtype, device=frames.device,
    )
    return frames


# ---------------------------------------------------------------------------
# Temporal transforms (operate on clip: [T, C, H, W])
# ---------------------------------------------------------------------------

def _temporal_crop(frames: torch.Tensor, fraction_range: tuple[float, float]) -> torch.Tensor:
    """Random contiguous temporal crop, then resample to original length."""
    time = frames.shape[0]
    if time <= 1:
        return frames
    fraction = random.uniform(fraction_range[0], fraction_range[1])
    crop_len = max(1, int(time * fraction))
    start = random.randint(0, time - crop_len)
    cropped = frames[start : start + crop_len]
    if cropped.shape[0] == time:
        return cropped
    indices = torch.linspace(0, cropped.shape[0] - 1, time).round().long()
    return cropped[indices]


def _speed_perturbation(frames: torch.Tensor, speed_range: tuple[float, float]) -> torch.Tensor:
    """Change clip speed by resampling frames."""
    time = frames.shape[0]
    if time <= 1:
        return frames
    speed = random.uniform(speed_range[0], speed_range[1])
    new_len = max(1, int(time / speed))
    indices = torch.linspace(0, time - 1, new_len).round().long()
    resampled = frames[indices]
    if resampled.shape[0] == time:
        return resampled
    final_indices = torch.linspace(0, resampled.shape[0] - 1, time).round().long()
    return resampled[final_indices]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class AugmentationPipeline:
    """Applies a configured sequence of augmentations to a video clip tensor.

    Input: [T, C, H, W] float tensor in [0, 1].
    Output: [T, C, H, W] float tensor in [0, 1].
    """

    def __init__(self, config: AugmentationConfig) -> None:
        self.config = config

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4:
            raise ValueError(f"augmentation expects [T,C,H,W], got {tuple(frames.shape)}")

        cfg = self.config

        # --- Temporal transforms (before spatial, so crop/speed affect what gets augmented) ---
        if cfg.temporal_crop:
            frames = _temporal_crop(frames, cfg.temporal_crop_fraction)
        if cfg.speed_perturbation:
            frames = _speed_perturbation(frames, cfg.speed_range)

        # --- Spatial transforms (applied consistently across all frames) ---
        if cfg.random_crop:
            frames = _random_crop_resize(frames, cfg.crop_scale)
        if cfg.random_horizontal_flip and random.random() < cfg.flip_probability:
            frames = _random_horizontal_flip(frames)
        if cfg.color_jitter:
            frames = _color_jitter(frames, cfg.brightness, cfg.contrast, cfg.saturation, cfg.hue)
        if cfg.random_rotation:
            frames = _random_rotation(frames, cfg.rotation_degrees)
        if cfg.random_erasing:
            frames = _random_erasing(frames, cfg.erasing_probability, cfg.erasing_scale)

        return frames.clamp(0.0, 1.0)

    @classmethod
    def from_config(cls, values: Mapping[str, Any] | None) -> "AugmentationPipeline | None":
        """Build from a config dict. Returns None if values is None or empty."""
        if not values:
            return None
        return cls(AugmentationConfig.from_dict(values))

    @classmethod
    def validation(cls) -> "AugmentationPipeline":
        """A no-op pipeline for validation/test splits."""
        return cls(AugmentationConfig.disabled())

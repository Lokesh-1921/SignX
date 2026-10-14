"""Tests for the data augmentation pipeline."""

import torch

from preprocessing.augmentation import AugmentationConfig, AugmentationPipeline


def _make_clip(time: int = 8, channels: int = 3, height: int = 32, width: int = 32) -> torch.Tensor:
    """Create a synthetic [T, C, H, W] clip in [0, 1]."""
    return torch.rand(time, channels, height, width)


def test_augmentation_preserves_shape():
    """All augmentations should preserve [T, C, H, W] shape."""
    config = AugmentationConfig(
        random_crop=True,
        color_jitter=True,
        random_rotation=True,
        random_erasing=True,
        speed_perturbation=True,
        temporal_crop=True,
    )
    pipeline = AugmentationPipeline(config)
    clip = _make_clip()
    result = pipeline(clip)
    assert result.shape == clip.shape


def test_augmentation_output_range():
    """Output must be clamped to [0, 1]."""
    config = AugmentationConfig(
        random_crop=True,
        color_jitter=True,
        brightness=0.5,
        contrast=0.5,
        saturation=0.5,
        hue=0.1,
    )
    pipeline = AugmentationPipeline(config)
    clip = _make_clip()
    result = pipeline(clip)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_disabled_augmentation_is_identity():
    """Disabled config should be a no-op (modulo floating-point clamp)."""
    pipeline = AugmentationPipeline(AugmentationConfig.disabled())
    clip = _make_clip()
    result = pipeline(clip)
    assert torch.allclose(result, clip.clamp(0, 1))


def test_horizontal_flip_changes_pixels():
    """Explicit horizontal flip should change the pixel values."""
    clip = torch.arange(32).float().reshape(1, 1, 4, 8).expand(2, 3, 4, 8).contiguous() / 31
    flipped = clip.flip(dims=[-1])
    assert not torch.equal(clip, flipped)


def test_config_from_dict():
    """Config should load from a dict (as from YAML)."""
    config = AugmentationConfig.from_dict({
        "random_crop": True,
        "crop_scale": [0.7, 0.9],
        "brightness": 0.3,
        "speed_perturbation": True,
    })
    assert config.random_crop is True
    assert config.crop_scale == (0.7, 0.9)
    assert config.brightness == 0.3
    assert config.speed_perturbation is True


def test_from_config_returns_none_for_empty():
    """from_config should return None for empty/None config."""
    assert AugmentationPipeline.from_config(None) is None
    assert AugmentationPipeline.from_config({}) is None


def test_validation_pipeline_is_noop():
    """validation() should return a pipeline that doesn't transform."""
    pipeline = AugmentationPipeline.validation()
    clip = _make_clip()
    result = pipeline(clip)
    assert torch.allclose(result, clip.clamp(0, 1))

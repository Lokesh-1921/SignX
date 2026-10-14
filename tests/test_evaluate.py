"""Tests for the standalone evaluation script."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.manifest import ManifestRecord, write_manifest
from models import SignBridgeBaseline
from scripts.evaluate import build_model_from_checkpoint, evaluate
from training.checkpoints import CheckpointManager
from training.device import select_device


def _create_test_checkpoint(root: Path, num_classes: int = 3, hidden_dim: int = 16) -> Path:
    """Create a minimal checkpoint for testing."""
    model = SignBridgeBaseline(num_classes=num_classes, hidden_dim=hidden_dim, temporal_layers=1, temporal_heads=4)
    config = {
        "model": {
            "name": "test",
            "num_classes": num_classes,
            "hidden_dim": hidden_dim,
            "temporal_layers": 1,
            "temporal_heads": 4,
            "dropout": 0.1,
            "use_pose": False,
            "pose_dim": 0,
        },
        "data": {"labels": []},
    }
    checkpoint_dir = root / "checkpoints"
    manager = CheckpointManager(checkpoint_dir)
    path = manager.save("test.pt", {
        "epoch": 1,
        "model": model.state_dict(),
        "optimizer": {},
        "config": config,
    })
    return path


def _create_test_manifest(root: Path, num_samples: int = 6, num_classes: int = 3) -> Path:
    """Create a manifest with synthetic .npy clips."""
    labels = [f"class_{i}" for i in range(num_classes)]
    records = []
    generator = np.random.default_rng(42)
    for i in range(num_samples):
        clip_path = root / f"clip_{i}.npy"
        frames = generator.integers(0, 255, size=(4, 16, 16, 3), dtype=np.uint8)
        np.save(clip_path, frames)
        records.append(ManifestRecord(
            sample_id=f"eval_{i}",
            dataset="test",
            video_path=str(clip_path),
            label=labels[i % num_classes],
            signer_id=f"signer_{i}",
        ))
    manifest_path = root / "test.jsonl"
    write_manifest(records, manifest_path)
    return manifest_path


def test_build_model_from_checkpoint(tmp_path):
    """Should load a model from a checkpoint."""
    checkpoint = _create_test_checkpoint(tmp_path)
    device = torch.device("cpu")
    model, config = build_model_from_checkpoint(checkpoint, device)
    assert isinstance(model, SignBridgeBaseline)
    assert "model" in config


def test_evaluate_produces_structured_results(tmp_path):
    """Evaluation should produce overall metrics, per-class, and confusions."""
    from torch.utils.data import DataLoader
    from datasets.video_dataset import SignVideoDataset, collate_sign_batch

    checkpoint = _create_test_checkpoint(tmp_path, num_classes=3, hidden_dim=16)
    manifest = _create_test_manifest(tmp_path, num_samples=6, num_classes=3)
    device_info = select_device()

    model, config = build_model_from_checkpoint(checkpoint, torch.device("cpu"))
    model = model.to(torch.device("cpu"))

    labels = [f"class_{i}" for i in range(3)]
    dataset = SignVideoDataset(manifest, labels, num_frames=4, frame_size=(16, 16))
    loader = DataLoader(dataset, batch_size=3, collate_fn=collate_sign_batch)

    results = evaluate(model, loader, torch.device("cpu"), "cpu", labels)

    assert "overall" in results
    assert "per_class" in results
    assert "top_confusions" in results
    assert results["num_samples"] == 6
    assert results["num_classes"] == 3
    assert "accuracy" in results["overall"]
    assert "f1_macro" in results["overall"]
    for label in labels:
        assert label in results["per_class"]
        assert "accuracy" in results["per_class"][label]
        assert "precision" in results["per_class"][label]

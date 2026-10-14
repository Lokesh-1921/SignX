"""Evaluate a trained SignBridge checkpoint on any manifest split.

Usage:
    python scripts/evaluate.py \\
        --checkpoint checkpoints/.../best.pt \\
        --manifest data/manifests/test.jsonl \\
        --num-frames 16 --frame-size 112 112

Outputs structured JSON with overall and per-class metrics.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.video_dataset import SignVideoDataset, collate_sign_batch
from evaluation.metrics import classification_metrics
from models import SignBridgeBaseline
from training.checkpoints import CheckpointManager
from training.device import select_device


def build_model_from_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[SignBridgeBaseline, dict[str, Any]]:
    """Load a trained model from a checkpoint file."""
    payload = CheckpointManager.load(checkpoint_path, device)
    config = payload.get("config", {})
    model_config = dict(config.get("model", {}))
    num_classes = model_config.pop("num_classes", 2)
    model_config.pop("name", None)
    model = SignBridgeBaseline(num_classes=num_classes, **model_config)
    model.load_state_dict(payload["model"])
    return model, config


def evaluate(
    model: SignBridgeBaseline,
    loader: DataLoader[Any],
    device: torch.device,
    device_kind: str,
    labels: list[str],
) -> dict[str, Any]:
    """Run inference and compute metrics."""
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            pose = batch.get("pose")
            if isinstance(pose, torch.Tensor):
                pose = pose.to(device, non_blocking=True)
            output = model(frames, attention_mask, pose)
            all_logits.append(output["logits"].cpu())
            all_labels.append(batch["labels"])

    logits = torch.cat(all_logits)
    targets = torch.cat(all_labels)

    # Overall metrics
    overall = classification_metrics(logits, targets)

    # Per-class metrics
    predictions = logits.argmax(dim=1)
    per_class: dict[str, dict[str, float]] = {}
    for class_idx, label in enumerate(labels):
        predicted = predictions == class_idx
        actual = targets == class_idx
        tp = (predicted & actual).sum().float()
        fp = (predicted & ~actual).sum().float()
        fn = (~predicted & actual).sum().float()
        total = actual.sum().item()
        precision = (tp / (tp + fp).clamp_min(1)).item()
        recall = (tp / (tp + fn).clamp_min(1)).item()
        f1 = (2 * precision * recall / max(precision + recall, 1e-12))
        accuracy = (tp / max(total, 1)).item()
        per_class[label] = {
            "samples": int(total),
            "correct": int(tp.item()),
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    # Confusion summary (top confusions)
    confusions: Counter[tuple[str, str]] = Counter()
    for pred, true in zip(predictions.tolist(), targets.tolist()):
        if pred != true:
            confusions[(labels[true], labels[pred])] += 1

    top_confusions = [
        {"true": pair[0], "predicted": pair[1], "count": count}
        for pair, count in confusions.most_common(20)
    ]

    return {
        "overall": {k: round(v, 4) for k, v in overall.items()},
        "per_class": per_class,
        "top_confusions": top_confusions,
        "num_samples": int(targets.shape[0]),
        "num_classes": len(labels),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--manifest", required=True, help="JSONL manifest to evaluate")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--frame-size", type=int, nargs=2, default=[112, 112])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", help="Write results to JSON file (default: stdout)")
    arguments = parser.parse_args()

    device_info = select_device()
    model, config = build_model_from_checkpoint(arguments.checkpoint, device_info.device)
    model = model.to(device_info.device)

    # Resolve labels from config or manifest
    data_config = config.get("data", {})
    labels = data_config.get("labels") or None

    dataset = SignVideoDataset(
        arguments.manifest,
        labels,
        arguments.num_frames,
        tuple(arguments.frame_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=arguments.num_workers,
        collate_fn=collate_sign_batch,
    )

    results = evaluate(model, loader, device_info.device, device_info.kind, dataset.labels)
    results["checkpoint"] = str(arguments.checkpoint)
    results["manifest"] = str(arguments.manifest)
    results["device"] = device_info.kind

    output_json = json.dumps(results, indent=2)
    if arguments.output:
        Path(arguments.output).write_text(output_json, encoding="utf-8")
        print(f"Results written to {arguments.output}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()

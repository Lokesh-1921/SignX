"""Execute the complete first SignBridge pipeline on labelled synthetic clips.

This is a systems test only. It does not measure sign-language performance.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import subprocess

import numpy as np
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.manifest import ManifestRecord, write_manifest
from scripts.validate_dataset import validate_manifest


def make_smoke_data(root: Path) -> tuple[Path, Path]:
    records: list[ManifestRecord] = []
    generator = np.random.default_rng(7)
    for split, count in (("train", 8), ("val", 4)):
        split_records: list[ManifestRecord] = []
        for index in range(count):
            label = "bright" if index % 2 else "dark"
            base = 220 if label == "bright" else 25
            frames = np.clip(generator.normal(base, 8, size=(6, 32, 32, 3)), 0, 255).astype(np.uint8)
            clip_path = root / f"{split}_{index}.npy"
            np.save(clip_path, frames)
            split_records.append(ManifestRecord(
                sample_id=f"smoke_{split}_{index}", dataset="synthetic_smoke", video_path=str(clip_path),
                signer_id=f"{split}_signer_{index}", label=label, language="synthetic", split=split,
            ))
        manifest = root / f"{split}.jsonl"
        write_manifest(split_records, manifest)
        records.extend(split_records)
    return root / "train.jsonl", root / "val.jsonl"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="signbridge_smoke_") as temporary:
        root = Path(temporary)
        train_manifest, val_manifest = make_smoke_data(root)
        validation = validate_manifest(train_manifest, decode=True)
        config = {
            "experiment": {"name": "smoke", "seed": 7, "output_dir": str(root / "runs"), "checkpoint_dir": str(root / "checkpoints")},
            "data": {"train_manifest": str(train_manifest), "val_manifest": str(val_manifest), "num_frames": 4, "frame_size": [32, 32], "num_workers": 0, "labels": ["bright", "dark"]},
            "model": {"name": "SignBridge-Smoke", "num_classes": 2, "hidden_dim": 16, "temporal_layers": 1, "temporal_heads": 4, "dropout": 0.0, "use_pose": False, "pose_dim": 0},
            "training": {"batch_size": 2, "epochs": 1, "learning_rate": 0.003, "weight_decay": 0, "gradient_clip_norm": 1.0, "amp": True, "checkpoint_every": 1, "resume": None},
            "evaluation": {"topk": [1, 5]},
        }
        config_path = root / "smoke.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, "train.py", "--config", str(config_path)],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
        )
        if completed.returncode:
            raise RuntimeError(f"training command failed:\n{completed.stderr}\n{completed.stdout}")
        summary = json.loads(completed.stdout)
        result = {"status": "passed", "manifest_validation": validation, **summary}
        print(json.dumps(result, indent=2))
        if not (Path(summary["checkpoint_dir"]) / "best.pt").exists():
            raise RuntimeError("smoke run completed without a best checkpoint")


if __name__ == "__main__":
    main()

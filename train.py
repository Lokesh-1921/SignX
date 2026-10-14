"""Train the SignBridge isolated-sign baseline from a YAML experiment config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from datasets.video_dataset import SignVideoDataset, collate_sign_batch
from models import SignBridgeBaseline
from training.experiment_tracker import ExperimentTracker
from training.trainer import Trainer


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {"experiment", "data", "model", "training", "evaluation"}
    missing = required.difference(config or {})
    if missing:
        raise ValueError(f"configuration missing sections: {sorted(missing)}")
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(config: dict[str, Any]) -> tuple[DataLoader[Any], DataLoader[Any], list[str]]:
    data_config = config["data"]
    frame_size = tuple(data_config["frame_size"])
    labels = data_config.get("labels") or None
    augmentation_config = data_config.get("augmentation")
    train_dataset = SignVideoDataset(
        data_config["train_manifest"], labels, data_config["num_frames"], frame_size,
        augmentation_config=augmentation_config,
    )
    val_dataset = SignVideoDataset(
        data_config["val_manifest"], train_dataset.labels, data_config["num_frames"], frame_size,
    )
    options = {
        "batch_size": data_config["batch_size"] if "batch_size" in data_config else config["training"]["batch_size"],
        "num_workers": data_config.get("num_workers", 0),
        "pin_memory": data_config.get("pin_memory", False),
        "collate_fn": collate_sign_batch,
    }
    return DataLoader(train_dataset, shuffle=True, **options), DataLoader(val_dataset, shuffle=False, **options), train_dataset.labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml", help="YAML experiment configuration")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    seed_everything(int(config["experiment"].get("seed", 42)))
    train_loader, val_loader, labels = build_loaders(config)
    model_config = dict(config["model"])
    configured_count = model_config.pop("num_classes", len(labels))
    if configured_count != len(labels):
        raise ValueError(f"model num_classes={configured_count} does not match the {len(labels)} manifest labels")
    model = SignBridgeBaseline(num_classes=len(labels), **{key: value for key, value in model_config.items() if key != "name"})
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"].get("weight_decay", 0)))
    tracker = ExperimentTracker(str(config["experiment"]["name"]), config["experiment"].get("output_dir", "experiments/runs"))
    tracker.save_config(config)
    tracker.log("started", labels=labels, train_samples=len(train_loader.dataset), val_samples=len(val_loader.dataset))
    trainer = Trainer(model, optimizer, config, tracker)
    summary = trainer.fit(train_loader, val_loader)
    print(json.dumps({"experiment_id": tracker.experiment_id, "device": trainer.device_info.kind, **summary}, indent=2))


if __name__ == "__main__":
    main()

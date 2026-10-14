"""Lazy manifest-backed PyTorch dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .manifest import ManifestRecord, iter_manifest
from preprocessing.augmentation import AugmentationPipeline
from preprocessing.video import VideoLoader


class SignVideoDataset(Dataset[dict[str, object]]):
    """Loads only the selected sample at access time, never a dataset at once."""

    def __init__(
        self,
        manifest_path: str | Path,
        labels: Sequence[str] | None,
        num_frames: int,
        frame_size: tuple[int, int],
        require_labels: bool = True,
        augmentation: AugmentationPipeline | None = None,
        augmentation_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.records = list(iter_manifest(self.manifest_path))
        if not self.records:
            raise ValueError(f"manifest is empty: {self.manifest_path}")
        discovered = sorted({record.label for record in self.records if record.label is not None})
        if require_labels and any(record.label is None for record in self.records):
            raise ValueError("all training records must have a label")
        self.labels = list(labels) if labels else discovered
        if require_labels and not self.labels:
            raise ValueError("classification requires explicit labels in the manifest or config")
        self.label_to_index = {label: index for index, label in enumerate(self.labels)}
        unknown = {record.label for record in self.records if record.label and record.label not in self.label_to_index}
        if unknown:
            raise ValueError(f"labels missing from configured vocabulary: {sorted(unknown)}")
        self.loader = VideoLoader(num_frames=num_frames, frame_size=frame_size)

        # Augmentation: explicit pipeline takes priority, then config dict
        if augmentation is not None:
            self.augmentation = augmentation
        elif augmentation_config:
            self.augmentation = AugmentationPipeline.from_config(augmentation_config)
        else:
            self.augmentation = None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        frames = self.loader.load(record.video_path)
        if self.augmentation is not None:
            frames = self.augmentation(frames)
        pose = self._load_pose(record, frames.shape[0])
        label = self.label_to_index.get(record.label, -1)
        return {
            "frames": frames,
            "pose": pose,
            "attention_mask": torch.ones(frames.shape[0], dtype=torch.bool),
            "label": torch.tensor(label, dtype=torch.long),
            "sample_id": record.sample_id,
        }

    @staticmethod
    def _load_pose(record: ManifestRecord, frame_count: int) -> torch.Tensor | None:
        if not record.landmarks_path:
            return None
        pose_path = Path(record.landmarks_path)
        if not pose_path.exists():
            raise FileNotFoundError(f"landmarks missing for {record.sample_id}: {pose_path}")
        values = np.load(pose_path, allow_pickle=False)
        if isinstance(values, np.lib.npyio.NpzFile):
            values = values["pose"]
        pose = torch.as_tensor(np.asarray(values), dtype=torch.float32)
        if pose.ndim != 2:
            raise ValueError(f"pose data must be [time, features], got {tuple(pose.shape)}")
        if pose.shape[0] != frame_count:
            indices = torch.linspace(0, pose.shape[0] - 1, frame_count).round().long()
            pose = pose[indices]
        return pose


def collate_sign_batch(items: list[dict[str, object]]) -> dict[str, object]:
    """Collate fixed-window samples and enforce uniform optional pose presence."""

    has_pose = [item["pose"] is not None for item in items]
    if any(has_pose) and not all(has_pose):
        raise ValueError("a batch cannot mix samples with and without pose features")
    batch: dict[str, object] = {
        "frames": torch.stack([item["frames"] for item in items]),
        "attention_mask": torch.stack([item["attention_mask"] for item in items]),
        "labels": torch.stack([item["label"] for item in items]),
        "sample_ids": [str(item["sample_id"]) for item in items],
    }
    if all(has_pose):
        batch["pose"] = torch.stack([item["pose"] for item in items])
    return batch

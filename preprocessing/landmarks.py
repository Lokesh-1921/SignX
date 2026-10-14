"""Standardized cached landmark representation.

Extraction (MediaPipe/OpenPose/etc.) remains pluggable; this module guarantees
the stable on-disk contract consumed by models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class LandmarkSequence:
    hand_left: np.ndarray
    hand_right: np.ndarray
    pose: np.ndarray
    face: np.ndarray | None = None
    confidence: np.ndarray | None = None

    def __post_init__(self) -> None:
        frame_counts = {self.hand_left.shape[0], self.hand_right.shape[0], self.pose.shape[0]}
        if self.face is not None:
            frame_counts.add(self.face.shape[0])
        if len(frame_counts) != 1:
            raise ValueError("all landmark modalities must have the same frame count")

    def flattened_pose(self, include_face: bool = True) -> np.ndarray:
        components = [self.hand_left.reshape(len(self.hand_left), -1), self.hand_right.reshape(len(self.hand_right), -1), self.pose.reshape(len(self.pose), -1)]
        if include_face and self.face is not None:
            components.append(self.face.reshape(len(self.face), -1))
        return np.concatenate(components, axis=1).astype(np.float32, copy=False)


def save_landmarks(sequence: LandmarkSequence, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"hand_left": sequence.hand_left, "hand_right": sequence.hand_right, "pose": sequence.pose}
    if sequence.face is not None:
        payload["face"] = sequence.face
    if sequence.confidence is not None:
        payload["confidence"] = sequence.confidence
    np.savez_compressed(target, **payload)


def load_landmarks(path: str | Path) -> LandmarkSequence:
    with np.load(path, allow_pickle=False) as payload:
        return LandmarkSequence(
            hand_left=payload["hand_left"], hand_right=payload["hand_right"], pose=payload["pose"],
            face=payload["face"] if "face" in payload else None,
            confidence=payload["confidence"] if "confidence" in payload else None,
        )

"""Lazy video and array loading with deterministic temporal sampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2
except ImportError:  # Allows manifest-only work before optional local installation.
    cv2 = None


class VideoLoader:
    def __init__(self, num_frames: int, frame_size: tuple[int, int]) -> None:
        if num_frames < 1:
            raise ValueError("num_frames must be positive")
        self.num_frames = num_frames
        self.frame_size = frame_size

    def load(self, source: str | Path) -> torch.Tensor:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"video source does not exist: {path}")
        if path.suffix.lower() in {".npy", ".npz"}:
            frames = self._read_array(path)
        else:
            frames = self._read_video(path)
        return self._normalise_and_sample(frames)

    @staticmethod
    def _read_array(path: Path) -> np.ndarray:
        loaded = np.load(path, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            if "frames" not in loaded:
                raise ValueError(f"{path} must contain a 'frames' array")
            loaded = loaded["frames"]
        array = np.asarray(loaded)
        if array.ndim != 4:
            raise ValueError(f"{path} must be [T,H,W,C] or [T,C,H,W], got {array.shape}")
        return array

    @staticmethod
    def _read_video(path: Path) -> np.ndarray:
        if cv2 is None:
            raise RuntimeError("OpenCV is required for video decoding; install requirements.txt")
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"could not decode video: {path}")
        frames: list[np.ndarray] = []
        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
        if not frames:
            raise ValueError(f"video contains no decodable frames: {path}")
        return np.stack(frames)

    def _normalise_and_sample(self, frames: np.ndarray) -> torch.Tensor:
        values = torch.as_tensor(frames)
        if values.shape[-1] in {1, 3, 4}:
            values = values[..., :3].permute(0, 3, 1, 2)
        elif values.shape[1] in {1, 3, 4}:
            values = values[:, :3]
        else:
            raise ValueError("frames must contain 1, 3, or 4 channels")
        if values.shape[0] == 0:
            raise ValueError("source contains no frames")
        indices = torch.linspace(0, values.shape[0] - 1, self.num_frames).round().long()
        values = values[indices].float()
        if values.max() > 1.0:
            values = values / 255.0
        values = F.interpolate(values, size=self.frame_size, mode="bilinear", align_corners=False)
        return values.clamp(0.0, 1.0)

"""Atomic model checkpoint persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class CheckpointManager:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)

    def save(self, name: str, payload: dict[str, Any]) -> Path:
        target = self.directory / name
        temporary = target.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(target)
        return target

    @staticmethod
    def load(path: str | Path, device: torch.device) -> dict[str, Any]:
        return torch.load(path, map_location=device, weights_only=False)

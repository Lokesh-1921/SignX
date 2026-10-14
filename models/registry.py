"""Versioned local model registry; never overwrites an existing registration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelRecord:
    name: str
    version: str
    checkpoint: str
    dataset: str
    metrics: dict[str, float]
    created_at: str


class ModelRegistry:
    def __init__(self, path: str | Path = "models/registry.json") -> None:
        self.path = Path(path)

    def list(self) -> list[ModelRecord]:
        if not self.path.exists():
            return []
        values: list[dict[str, Any]] = json.loads(self.path.read_text(encoding="utf-8"))
        return [ModelRecord(**value) for value in values]

    def register(self, record: ModelRecord) -> None:
        records = self.list()
        if any(item.name == record.name and item.version == record.version for item in records):
            raise ValueError(f"model already registered: {record.name}@{record.version}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([*(asdict(item) for item in records), asdict(record)], indent=2), encoding="utf-8")

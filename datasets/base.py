"""Dataset adapter contract and provenance metadata."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .manifest import ManifestRecord


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    name: str
    source: str
    license_note: str
    sign_language: str
    annotation_format: str
    citation: str = ""


class DatasetAdapter(ABC):
    """Converts a legally obtained dataset's native annotations into JSONL.

    Adapters intentionally do not download data. Each implementation requires
    caller-supplied paths and preserves raw annotation provenance in ``extra``.
    """

    metadata: DatasetMetadata

    @abstractmethod
    def build_records(self, raw_root: Path) -> Iterable[ManifestRecord]:
        """Yield records from ``raw_root``; raise clear errors for bad input."""


adapter_registry: dict[str, type[DatasetAdapter]] = {}


def register_adapter(adapter_type: type[DatasetAdapter]) -> type[DatasetAdapter]:
    name = adapter_type.metadata.name
    if name in adapter_registry:
        raise ValueError(f"adapter already registered: {name}")
    adapter_registry[name] = adapter_type
    return adapter_type

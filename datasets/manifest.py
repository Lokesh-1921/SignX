"""Universal, streaming JSONL manifest used across sign-language datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterator, Mapping


class ManifestError(ValueError):
    """Raised when a manifest entry violates the project data contract."""


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    """A dataset-neutral description of one sign-language video sample.

    Unknown fields are retained in ``extra`` so dataset-specific provenance is
    preserved instead of discarded during adapter conversion.
    """

    sample_id: str
    dataset: str
    video_path: str
    signer_id: str | None = None
    label: str | None = None
    gloss: str | None = None
    translation: str | None = None
    language: str | None = None
    region: str | None = None
    duration: float | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    split: str | None = None
    landmarks_path: str | None = None
    context_before: str | None = None
    context_after: str | None = None
    annotation_version: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    REQUIRED = frozenset({"sample_id", "dataset", "video_path"})
    _KNOWN = frozenset(
        {
            "sample_id", "dataset", "video_path", "signer_id", "label", "gloss",
            "translation", "language", "region", "duration", "fps", "width",
            "height", "split", "landmarks_path", "context_before", "context_after",
            "annotation_version", "extra",
        }
    )

    def __post_init__(self) -> None:
        for field_name in self.REQUIRED:
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ManifestError(f"{field_name} must be a non-empty string")
        if self.duration is not None and self.duration <= 0:
            raise ManifestError("duration must be positive when supplied")
        if self.fps is not None and self.fps <= 0:
            raise ManifestError("fps must be positive when supplied")
        if self.width is not None and self.width <= 0:
            raise ManifestError("width must be positive when supplied")
        if self.height is not None and self.height <= 0:
            raise ManifestError("height must be positive when supplied")
        if self.split is not None and self.split not in {"train", "val", "test"}:
            raise ManifestError("split must be one of train, val, test")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ManifestRecord":
        missing = cls.REQUIRED.difference(value)
        if missing:
            raise ManifestError(f"missing required fields: {sorted(missing)}")
        known_values = {key: value.get(key) for key in cls._KNOWN if key != "extra"}
        supplied_extra = value.get("extra", {})
        if not isinstance(supplied_extra, dict):
            raise ManifestError("extra must be an object")
        unknown = {key: item for key, item in value.items() if key not in cls._KNOWN}
        known_values["extra"] = {**supplied_extra, **unknown}
        return cls(**known_values)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        extra = value.pop("extra")
        return {key: item for key, item in {**value, **extra}.items() if item is not None}


def iter_manifest(path: str | Path) -> Iterator[ManifestRecord]:
    """Yield records without loading a potentially multi-GB manifest into RAM."""

    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ManifestError("record must be a JSON object")
                yield ManifestRecord.from_dict(raw)
            except (json.JSONDecodeError, ManifestError) as error:
                raise ManifestError(f"{manifest_path}:{line_number}: {error}") from error


def write_manifest(records: Iterator[ManifestRecord] | list[ManifestRecord], path: str | Path) -> None:
    """Write a portable JSONL manifest, creating only its immediate parent."""

    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")

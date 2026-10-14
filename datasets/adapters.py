"""Dataset adapter registrations.

Native dataset annotation formats vary by release. These adapters deliberately
require an explicit mapping/normalisation implementation before conversion;
they never guess labels, signer IDs, licences, or regional meanings.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from .base import DatasetAdapter, DatasetMetadata, register_adapter
from .manifest import ManifestRecord

logger = logging.getLogger(__name__)


class _MappingRequiredAdapter(DatasetAdapter):
    def build_records(self, raw_root: Path) -> Iterable[ManifestRecord]:
        raise NotImplementedError(
            f"{self.metadata.name} conversion needs a release-specific mapping. "
            "Document the annotation release and licence in docs/datasets before implementing it."
        )


# ---------------------------------------------------------------------------
# WLASL — fully implemented adapter
# ---------------------------------------------------------------------------

@register_adapter
class WLASLAdapter(DatasetAdapter):
    """Parse the official WLASL JSON annotation (v0.3 schema).

    Expected layout under ``raw_root``:
        raw_root/
            WLASL_v0.3.json        (or wlasl_class_list.txt)
            videos/
                00001.mp4
                00002.mp4
                ...

    The annotation is a JSON array where each element describes one *gloss*
    (sign class) and contains an ``instances`` list of video samples::

        [
          {
            "gloss": "book",
            "instances": [
              {
                "video_id": "00001",
                "signer_id": 12,
                "source": "ASL_that",
                "split": "train",
                "bbox": [x, y, w, h],
                "frame_start": 0,
                "frame_end": -1,
                "url": "..."
              },
              ...
            ]
          },
          ...
        ]

    The ``subset`` parameter controls which WLASL split to build:
    - ``100``: glosses with index < 100 (WLASL-100 benchmark subset)
    - ``300``: glosses with index < 300 (WLASL-300 benchmark subset)
    - ``2000``: all glosses (WLASL-2000, the full set)
    - ``None`` (default): all available glosses
    """

    metadata = DatasetMetadata(
        "wlasl",
        "Official WLASL distribution",
        "Verify current terms before use",
        "ASL",
        "WLASL_v0.3.json",
        citation="Li et al., Word-level Deep Sign Language Recognition from Video, WACV 2020",
    )

    def __init__(self, subset: int | None = None, video_extensions: tuple[str, ...] = (".mp4", ".avi", ".mov")) -> None:
        self.subset = subset
        self.video_extensions = video_extensions

    def build_records(self, raw_root: Path) -> Iterable[ManifestRecord]:
        """Yield ManifestRecords from the WLASL annotation JSON."""
        annotation_path = self._find_annotation(raw_root)
        with annotation_path.open("r", encoding="utf-8") as handle:
            annotation = json.load(handle)

        if not isinstance(annotation, list):
            raise ValueError(f"WLASL annotation must be a JSON array, got {type(annotation).__name__}")

        video_dir = self._find_video_dir(raw_root)
        skipped_missing = 0
        emitted = 0

        for gloss_index, gloss_entry in enumerate(annotation):
            if not isinstance(gloss_entry, dict) or "gloss" not in gloss_entry:
                raise ValueError(f"annotation entry {gloss_index} missing 'gloss' key")

            # Apply subset filter
            if self.subset is not None and gloss_index >= self.subset:
                break

            gloss = str(gloss_entry["gloss"]).strip()
            instances = gloss_entry.get("instances", [])

            for instance in instances:
                video_id = str(instance.get("video_id", "")).strip()
                if not video_id:
                    continue

                # Locate the video file
                video_path = self._resolve_video(video_dir, video_id)
                if video_path is None:
                    skipped_missing += 1
                    continue

                # Map WLASL split names to our convention
                raw_split = str(instance.get("split", "")).strip().lower()
                split = {"train": "train", "val": "val", "test": "test"}.get(raw_split)

                signer_id = instance.get("signer_id")
                signer_str = f"wlasl_signer_{signer_id}" if signer_id is not None else None

                # Preserve raw provenance in extra
                extra: dict = {}
                if "bbox" in instance:
                    extra["bbox"] = instance["bbox"]
                if "frame_start" in instance:
                    extra["frame_start"] = instance["frame_start"]
                if "frame_end" in instance:
                    extra["frame_end"] = instance["frame_end"]
                if "source" in instance:
                    extra["source"] = instance["source"]
                if "url" in instance:
                    extra["url"] = instance["url"]
                extra["gloss_index"] = gloss_index

                yield ManifestRecord(
                    sample_id=f"wlasl_{video_id}",
                    dataset="wlasl",
                    video_path=str(video_path),
                    signer_id=signer_str,
                    label=gloss,
                    gloss=gloss,
                    language="ASL",
                    split=split,
                    extra=extra,
                )
                emitted += 1

        if skipped_missing > 0:
            logger.warning("WLASL: skipped %d instances with missing video files", skipped_missing)
        if emitted == 0:
            raise ValueError(
                f"No valid WLASL records produced from {annotation_path}. "
                f"Ensure video files exist under {video_dir}."
            )

    def _find_annotation(self, raw_root: Path) -> Path:
        """Locate the WLASL annotation JSON file."""
        candidates = [
            raw_root / "WLASL_v0.3.json",
            raw_root / "wlasl_v0.3.json",
            raw_root / "WLASL.json",
            raw_root / "wlasl.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        # Also check one level up in case raw_root is the videos directory
        for candidate in candidates:
            parent_candidate = raw_root.parent / candidate.name
            if parent_candidate.exists():
                return parent_candidate
        raise FileNotFoundError(
            f"Could not find WLASL annotation JSON in {raw_root}. "
            f"Expected one of: {', '.join(c.name for c in candidates)}"
        )

    def _find_video_dir(self, raw_root: Path) -> Path:
        """Locate the directory containing WLASL video files."""
        candidates = [
            raw_root / "videos",
            raw_root / "Videos",
            raw_root,  # videos might be directly in raw_root
        ]
        for candidate in candidates:
            if candidate.is_dir():
                # Check if it actually contains video files
                for ext in self.video_extensions:
                    if any(candidate.glob(f"*{ext}")):
                        return candidate
        return raw_root / "videos"  # default path for error messages

    def _resolve_video(self, video_dir: Path, video_id: str) -> Path | None:
        """Try to find a video file by its ID, with multiple extensions."""
        for ext in self.video_extensions:
            candidate = video_dir / f"{video_id}{ext}"
            if candidate.exists():
                return candidate
        # Try without zero-padding
        stripped = video_id.lstrip("0") or "0"
        for ext in self.video_extensions:
            candidate = video_dir / f"{stripped}{ext}"
            if candidate.exists():
                return candidate
        return None


# ---------------------------------------------------------------------------
# Stub adapters for datasets requiring release-specific mapping
# ---------------------------------------------------------------------------

@register_adapter
class MSASLAdapter(_MappingRequiredAdapter):
    metadata = DatasetMetadata("msasl", "Official MS-ASL distribution", "Verify current terms before use", "ASL", "release-specific JSON")


@register_adapter
class How2SignAdapter(_MappingRequiredAdapter):
    metadata = DatasetMetadata("how2sign", "Official How2Sign distribution", "Verify current terms before use", "ASL", "release-specific CSV/JSON")


@register_adapter
class AUTSLAdapter(_MappingRequiredAdapter):
    metadata = DatasetMetadata("autsl", "Official AUTSL distribution", "Verify current terms before use", "Turkish Sign Language", "release-specific annotations")


@register_adapter
class Phoenix2014TAdapter(_MappingRequiredAdapter):
    metadata = DatasetMetadata("phoenix2014t", "Official RWTH-PHOENIX-Weather 2014T distribution", "Verify current terms before use", "German Sign Language", "release-specific annotations")


@register_adapter
class CSLAdapter(_MappingRequiredAdapter):
    metadata = DatasetMetadata("csl", "Official Chinese Sign Language distribution", "Verify current terms before use", "Chinese Sign Language", "release-specific annotations")

"""Validate universal JSONL manifests before model training."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.manifest import ManifestError, iter_manifest
from preprocessing.video import VideoLoader


def validate_manifest(manifest: str | Path, decode: bool = False, num_frames: int = 2) -> dict[str, int]:
    """Validate source existence, IDs, labels and optionally actual decoding."""
    seen_ids: set[str] = set()
    labels: Counter[str] = Counter()
    signers: set[str] = set()
    missing_paths: list[str] = []
    decoder = VideoLoader(num_frames, (32, 32)) if decode else None
    count = 0
    try:
        for record in iter_manifest(manifest):
            count += 1
            if record.sample_id in seen_ids:
                raise ManifestError(f"duplicate sample_id: {record.sample_id}")
            seen_ids.add(record.sample_id)
            if not Path(record.video_path).exists():
                missing_paths.append(record.video_path)
            elif decoder:
                decoder.load(record.video_path)
            if record.label:
                labels[record.label] += 1
            if record.signer_id:
                signers.add(record.signer_id)
    except (ManifestError, OSError, ValueError, RuntimeError) as error:
        raise ManifestError(str(error)) from error
    if missing_paths:
        raise ManifestError(f"{len(missing_paths)} missing sources; first: {missing_paths[0]}")
    return {"samples": count, "labels": len(labels), "signers": len(signers)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--decode", action="store_true", help="Decode each source; potentially expensive on large datasets")
    parser.add_argument("--num-frames", type=int, default=2)
    arguments = parser.parse_args()
    try:
        summary = validate_manifest(arguments.manifest, arguments.decode, arguments.num_frames)
    except ManifestError as error:
        raise SystemExit(f"INVALID: {error}") from error
    print(f"VALID: samples={summary['samples']} labels={summary['labels']} signers={summary['signers']} manifest={arguments.manifest}")


if __name__ == "__main__":
    main()

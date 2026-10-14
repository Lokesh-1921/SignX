"""Prepare an adapter-backed dataset in resumable shards."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import datasets.adapters  # Registers public adapter names.
from datasets.base import adapter_registry
from datasets.manifest import write_manifest


def parse_shard(value: str) -> tuple[int, int]:
    try:
        current, total = (int(part) for part in value.split("/", 1))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shard must use CURRENT/TOTAL, e.g. 0/20") from error
    if total < 1 or not 0 <= current < total:
        raise argparse.ArgumentTypeError("shard must satisfy 0 <= CURRENT < TOTAL")
    return current, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(adapter_registry))
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--output", default="data/manifests")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shard", type=parse_shard, default=(0, 1))
    arguments = parser.parse_args()
    adapter = adapter_registry[arguments.dataset]()
    current, total = arguments.shard
    record_count = 0

    def selected_records():
        nonlocal record_count
        for index, record in enumerate(adapter.build_records(Path(arguments.raw_root) / arguments.dataset)):
            record_count += 1
            if index % total == current:
                yield record

    target = Path(arguments.output) / arguments.dataset / f"shard-{current:05d}-of-{total:05d}.jsonl"
    write_manifest(selected_records(), target)
    print(f"Prepared shard {current}/{total} from {record_count} streamed samples with {arguments.workers} requested workers: {target}")


if __name__ == "__main__":
    main()

"""Leakage-resistant dataset split helpers."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Iterable

from .manifest import ManifestRecord


def signer_independent_split(
    records: Iterable[ManifestRecord], train_fraction: float = 0.8, val_fraction: float = 0.1, seed: int = 42
) -> dict[str, list[ManifestRecord]]:
    """Split complete signer groups deterministically; never split one signer."""

    if not 0 < train_fraction < 1 or not 0 <= val_fraction < 1 or train_fraction + val_fraction >= 1:
        raise ValueError("fractions must be positive and leave a non-empty test split")
    groups: dict[str, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        if not record.signer_id:
            raise ValueError(f"sample {record.sample_id} has no signer_id; cannot make signer-independent split")
        groups[record.signer_id].append(record)
    if len(groups) < 3:
        raise ValueError("at least three distinct signers are required for train/val/test splitting")

    ordered = sorted(groups, key=lambda signer: hashlib.sha256(f"{seed}:{signer}".encode()).hexdigest())
    signer_count = len(ordered)
    train_count = min(max(1, round(signer_count * train_fraction)), signer_count - 2)
    remaining = signer_count - train_count
    val_count = min(max(1, round(signer_count * val_fraction)), remaining - 1)
    val_stop = train_count + val_count
    split_signers = {"train": ordered[:train_count], "val": ordered[train_count:val_stop], "test": ordered[val_stop:]}
    return {
        split: [record for signer in signers for record in groups[signer]]
        for split, signers in split_signers.items()
    }

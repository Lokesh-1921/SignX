"""Modular learning-rate scheduler factory.

Supports:
- ``cosine_warmup``: linear warm-up followed by cosine annealing to zero.
- ``step``: decay by a factor at specified epoch milestones.
- ``none`` / ``constant``: fixed learning rate (legacy default).
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, LambdaLR


class CosineWarmupScheduler(LambdaLR):
    """Linear warm-up for ``warmup_epochs``, then cosine decay to ``min_lr_ratio``."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")
        if total_epochs < 1:
            raise ValueError("total_epochs must be at least 1")
        if not 0.0 <= min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")

        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_ratio = min_lr_ratio

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                # Linear warm-up from 0 to 1
                return (epoch + 1) / max(warmup_epochs, 1)
            # Cosine decay from 1 to min_lr_ratio
            progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)


class StepDecayScheduler(LambdaLR):
    """Decay by ``gamma`` at each milestone epoch."""

    def __init__(
        self,
        optimizer: Optimizer,
        milestones: list[int],
        gamma: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        if gamma <= 0 or gamma > 1:
            raise ValueError("gamma must be in (0, 1]")

        sorted_milestones = sorted(milestones)
        self._milestones = sorted_milestones
        self._gamma = gamma

        def lr_lambda(epoch: int) -> float:
            factor = 1.0
            for milestone in sorted_milestones:
                if epoch >= milestone:
                    factor *= gamma
            return factor

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)


def build_scheduler(
    optimizer: Optimizer,
    config: Mapping[str, Any],
    total_epochs: int,
) -> LRScheduler | None:
    """Factory: build a scheduler from the ``training.scheduler`` config section.

    Config examples::

        # Cosine with warm-up
        scheduler:
          type: cosine_warmup
          warmup_epochs: 5
          min_lr_ratio: 0.01

        # Step decay
        scheduler:
          type: step
          milestones: [30, 60, 90]
          gamma: 0.1

        # No scheduler (constant LR)
        scheduler:
          type: none
    """
    scheduler_type = str(config.get("type", "none")).lower().strip()

    if scheduler_type in {"none", "constant", ""}:
        return None

    if scheduler_type == "cosine_warmup":
        return CosineWarmupScheduler(
            optimizer,
            warmup_epochs=int(config.get("warmup_epochs", 5)),
            total_epochs=total_epochs,
            min_lr_ratio=float(config.get("min_lr_ratio", 0.0)),
        )

    if scheduler_type == "step":
        milestones = config.get("milestones", [])
        if not milestones:
            raise ValueError("step scheduler requires a non-empty 'milestones' list")
        return StepDecayScheduler(
            optimizer,
            milestones=[int(m) for m in milestones],
            gamma=float(config.get("gamma", 0.1)),
        )

    raise ValueError(f"unknown scheduler type: {scheduler_type!r}")

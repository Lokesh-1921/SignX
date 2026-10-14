"""Tests for learning-rate schedulers."""

import torch

from training.scheduler import CosineWarmupScheduler, StepDecayScheduler, build_scheduler


def _make_optimizer() -> torch.optim.SGD:
    """Simple optimizer for testing."""
    model = torch.nn.Linear(4, 2)
    return torch.optim.SGD(model.parameters(), lr=0.1)


def test_cosine_warmup_curve():
    """LR should warm up linearly then decay via cosine."""
    optimizer = _make_optimizer()
    scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=3, total_epochs=10)

    lrs = []
    for epoch in range(10):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    # During warmup, LR should increase
    assert lrs[0] < lrs[1] < lrs[2]
    # After warmup, LR should decrease
    assert lrs[3] > lrs[7]
    # Final LR should be near zero (min_lr_ratio=0.0)
    assert lrs[-1] < lrs[3] * 0.5


def test_cosine_warmup_min_lr_ratio():
    """LR should not decay below min_lr_ratio × base_lr."""
    optimizer = _make_optimizer()
    scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=0, total_epochs=20, min_lr_ratio=0.1)

    for _ in range(20):
        scheduler.step()

    final_lr = optimizer.param_groups[0]["lr"]
    assert final_lr >= 0.1 * 0.1 * 0.95  # allow small floating point margin


def test_step_decay_at_milestones():
    """LR should drop by gamma at each milestone."""
    optimizer = _make_optimizer()
    scheduler = StepDecayScheduler(optimizer, milestones=[5, 10], gamma=0.1)

    lrs = []
    for epoch in range(15):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    # Before first milestone: base LR
    assert abs(lrs[0] - 0.1) < 1e-6
    assert abs(lrs[4] - 0.1) < 1e-6
    # After first milestone: 0.1 * 0.1 = 0.01
    assert abs(lrs[5] - 0.01) < 1e-6
    # After second milestone: 0.1 * 0.1 * 0.1 = 0.001
    assert abs(lrs[10] - 0.001) < 1e-6


def test_build_scheduler_none():
    """'none' type should return None."""
    optimizer = _make_optimizer()
    result = build_scheduler(optimizer, {"type": "none"}, 50)
    assert result is None


def test_build_scheduler_cosine():
    """Factory should create a CosineWarmupScheduler."""
    optimizer = _make_optimizer()
    scheduler = build_scheduler(optimizer, {"type": "cosine_warmup", "warmup_epochs": 5}, 50)
    assert isinstance(scheduler, CosineWarmupScheduler)


def test_build_scheduler_step():
    """Factory should create a StepDecayScheduler."""
    optimizer = _make_optimizer()
    scheduler = build_scheduler(optimizer, {"type": "step", "milestones": [10, 20]}, 50)
    assert isinstance(scheduler, StepDecayScheduler)

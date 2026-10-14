"""Hardware abstraction with CUDA → MPS → CPU priority."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    device: torch.device
    kind: str
    amp_supported: bool


def select_device() -> DeviceInfo:
    if torch.cuda.is_available():
        return DeviceInfo(torch.device("cuda"), "cuda", True)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return DeviceInfo(torch.device("mps"), "mps", False)
    return DeviceInfo(torch.device("cpu"), "cpu", False)

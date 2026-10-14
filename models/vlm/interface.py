"""Interface for optional open VLM integrations (Qwen-VL, LLaVA, etc.)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class VisionLanguageModel(ABC):
    @abstractmethod
    def encode_visual(self, frames: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def encode_video(self, frames: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str: ...

    def translate(self, prompt: str, **kwargs: Any) -> str:
        return self.generate(prompt, **kwargs)

    def reason(self, prompt: str, **kwargs: Any) -> str:
        return self.generate(prompt, **kwargs)

"""Small structured tracker with an append-only global experiment registry.

Optionally integrates with TensorBoard when the ``tensorboard`` package is
available.  All structured events are always written to JSONL regardless.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
from typing import Any
from uuid import uuid4


class ExperimentTracker:
    def __init__(self, name: str, output_dir: str | Path = "experiments/runs") -> None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.experiment_id = f"{timestamp}_{name}_{uuid4().hex[:8]}"
        self.directory = Path(output_dir) / self.experiment_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.events_path = self.directory / "events.jsonl"

        # Optional TensorBoard integration
        self._tb_writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = self.directory / "tensorboard"
            self._tb_writer = SummaryWriter(log_dir=str(tb_dir))
        except ImportError:
            pass  # tensorboard not installed — JSONL-only tracking

    @staticmethod
    def git_commit() -> str | None:
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    def save_config(self, config: dict[str, Any]) -> None:
        (self.directory / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    def log(self, event: str, **values: Any) -> None:
        payload = {"time": datetime.now(UTC).isoformat(), "event": event, **values}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")

    def log_scalars(self, scalars: dict[str, float], global_step: int) -> None:
        """Write scalars to TensorBoard (if available) and to JSONL."""
        if self._tb_writer is not None:
            for tag, value in scalars.items():
                self._tb_writer.add_scalar(tag, value, global_step)
            self._tb_writer.flush()

    def register(self, summary: dict[str, Any]) -> None:
        registry = self.directory.parent.parent / "registry.json"
        entries = json.loads(registry.read_text(encoding="utf-8")) if registry.exists() else []
        entries.append({"experiment_id": self.experiment_id, "git_commit": self.git_commit(), **summary})
        registry.write_text(json.dumps(entries, indent=2, sort_keys=True, default=str), encoding="utf-8")

    def close(self) -> None:
        """Flush and close the TensorBoard writer if active."""
        if self._tb_writer is not None:
            self._tb_writer.close()
            self._tb_writer = None

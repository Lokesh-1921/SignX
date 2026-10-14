"""Print the exact training invocation recorded for a prior experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id")
    parser.add_argument("--runs-dir", default="experiments/runs")
    arguments = parser.parse_args()
    config_path = Path(arguments.runs_dir) / arguments.experiment_id / "config.json"
    if not config_path.exists():
        raise SystemExit(f"experiment configuration not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    print(f"Recorded experiment: {arguments.experiment_id}")
    print(f"Command: python train.py --config {config_path}")
    print(f"Seed: {config['experiment'].get('seed')}")


if __name__ == "__main__":
    main()

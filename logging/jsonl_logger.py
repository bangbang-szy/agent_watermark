from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from .schemas import StepLog


class JsonlExecutionLogger:
    """Append-only JSONL logger used by the online agent and offline decoder."""

    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, run_id: str) -> Path:
        return self.log_dir / f"{run_id}.jsonl"

    def append(self, step: StepLog) -> Path:
        path = self.path_for(step.run_id)
        with path.open("a", encoding="utf-8") as f:
            f.write(step.model_dump_json() + "\n")
        return path

    @staticmethod
    def read(path: str | Path) -> List[StepLog]:
        steps: List[StepLog] = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    steps.append(StepLog.model_validate(json.loads(line)))
        return steps

    @staticmethod
    def write(path: str | Path, steps: Iterable[StepLog]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            for step in steps:
                f.write(step.model_dump_json() + "\n")

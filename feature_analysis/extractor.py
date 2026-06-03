from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from agent_watermark.logging.jsonl_logger import JsonlExecutionLogger
from agent_watermark.logging.schemas import StepLog
from agent_watermark.watermark.signature import FEATURE_NAMES


class BehaviorFeatureExtractor:
    """Extract watermark statistics only from execution logs."""

    def from_steps(self, steps: List[StepLog]) -> Dict[str, float]:
        if not steps:
            return {name: 0.0 for name in FEATURE_NAMES}
        chosen = [s.chosen_action for s in steps]
        tool_steps = [a for a in chosen if a != "final_answer"]
        transitions = list(zip(chosen[:-1], chosen[1:]))
        first_rank_hits = []
        for step in steps:
            ranked = sorted(step.candidate_actions, key=lambda c: c.raw_probability, reverse=True)
            first_rank_hits.append(1.0 if ranked and ranked[0].name == step.chosen_action else 0.0)
        transition_hash = 0.0
        if transitions:
            counts = Counter(transitions)
            transition_hash = sum((hash(a + "->" + b) % 997) * n for (a, b), n in counts.items()) / (997 * len(transitions))
        return {
            "tool_usage_frequency": len(tool_steps) / len(steps),
            "search_tool_ratio": chosen.count("search") / max(1, len(tool_steps)),
            "average_trajectory_length": float(len(steps)),
            "early_stop_ratio": 1.0 if len(steps) <= 3 and chosen[-1] == "final_answer" else 0.0,
            "tool_transition_pattern": float(transition_hash),
            "action_rank_preference": float(np.mean(first_rank_hits)),
            "database_query_ratio": chosen.count("sqlite_db") / max(1, len(tool_steps)),
        }

    def from_log_path(self, path: str | Path) -> Dict[str, float]:
        return self.from_steps(JsonlExecutionLogger.read(path))

    def dataframe(self, paths: Iterable[str | Path]) -> pd.DataFrame:
        rows = []
        for path in paths:
            features = self.from_log_path(path)
            features["log_path"] = str(path)
            rows.append(features)
        return pd.DataFrame(rows)

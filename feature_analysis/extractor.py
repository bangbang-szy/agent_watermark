from __future__ import annotations

import hashlib
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
        candidate_margins = []
        chosen_gains = []
        for step in steps:
            raw_candidates = [c for c in step.candidate_actions if c.raw_probability is not None]
            if raw_candidates:
                ranked = sorted(raw_candidates, key=lambda c: float(c.raw_probability), reverse=True)
                first_rank_hits.append(1.0 if ranked[0].name == step.chosen_action else 0.0)
            raw_probs = sorted([float(c.raw_probability) for c in raw_candidates], reverse=True)
            if len(raw_probs) >= 2:
                candidate_margins.append(raw_probs[0] - raw_probs[1])
            for candidate in step.candidate_actions:
                if (
                    candidate.name == step.chosen_action
                    and candidate.watermarked_probability is not None
                    and candidate.raw_probability is not None
                ):
                    chosen_gains.append(float(candidate.watermarked_probability) - float(candidate.raw_probability))
        transition_hash = 0.0
        if transitions:
            counts = Counter(transitions)
            transition_hash = sum(
                (int(hashlib.sha256(f"{a}->{b}".encode("utf-8")).hexdigest()[:8], 16) % 997) * n
                for (a, b), n in counts.items()
            ) / (997 * len(transitions))
        tool_counts = Counter(tool_steps)
        tool_probs = np.asarray(list(tool_counts.values()), dtype=float)
        if tool_probs.size:
            tool_probs = tool_probs / tool_probs.sum()
            tool_entropy = float(-(tool_probs * np.log2(np.clip(tool_probs, 1e-12, 1.0))).sum())
        else:
            tool_entropy = 0.0
        error_steps = [
            step
            for step in steps
            if step.tool_call
            and (
                step.tool_call.error
                or (
                    isinstance(step.tool_call.observation, str)
                    and any(marker in step.tool_call.observation for marker in ["search_error", "sql_error", "python_error"])
                )
            )
        ]
        return {
            "tool_usage_frequency": len(tool_steps) / len(steps),
            "search_tool_ratio": chosen.count("search") / max(1, len(tool_steps)),
            "average_trajectory_length": float(len(steps)),
            "early_stop_ratio": 1.0 if len(steps) <= 3 and chosen[-1] == "final_answer" else 0.0,
            "tool_transition_pattern": float(transition_hash),
            "action_rank_preference": float(np.mean(first_rank_hits)) if first_rank_hits else 0.0,
            "database_query_ratio": chosen.count("sqlite_db") / max(1, len(tool_steps)),
            "unique_tool_ratio": len(set(tool_steps)) / max(1, len(tool_steps)),
            "candidate_margin_mean": float(np.mean(candidate_margins)) if candidate_margins else 0.0,
            "chosen_probability_gain": float(np.mean(chosen_gains)) if chosen_gains else 0.0,
            "tool_entropy": tool_entropy,
            "tool_error_ratio": len(error_steps) / max(1, len(tool_steps)),
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

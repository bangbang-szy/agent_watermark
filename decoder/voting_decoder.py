from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from agent_watermark.feature_analysis.extractor import BehaviorFeatureExtractor
from agent_watermark.logging.jsonl_logger import JsonlExecutionLogger
from agent_watermark.watermark.signature import FEATURE_NAMES, SignatureGenerator, WatermarkIdentity


@dataclass
class DecodeResult:
    author_id: str
    timestamp: str
    confidence: float
    votes: Dict[str, float]


class MultiStatisticVotingDecoder:
    """Offline decoder that votes over log-derived behavior statistics and action phis."""

    def __init__(self, candidate_authors: Iterable[str], candidate_timestamps: Iterable[str]):
        self.candidate_authors = list(candidate_authors)
        self.candidate_timestamps = list(candidate_timestamps)
        self.extractor = BehaviorFeatureExtractor()
        self.signature = SignatureGenerator()

    def _observed_direction(self, path: str | Path) -> Dict[str, int]:
        features = self.extractor.from_log_path(path)
        signs = {}
        for name in FEATURE_NAMES:
            value = features.get(name, 0.0)
            signs[name] = 1 if value >= 0.5 else -1
        return signs

    def _phi_alignment(self, path: str | Path, identity: WatermarkIdentity) -> float:
        steps = JsonlExecutionLogger.read(path)
        margins: List[float] = []
        for step in steps:
            for candidate in step.candidate_actions:
                if candidate.name == step.chosen_action:
                    margins.append(candidate.watermark_phi * self.signature.tool_phi(identity, candidate.name))
        return float(np.mean(margins)) if margins else 0.0

    def decode(self, path: str | Path) -> DecodeResult:
        observed = self._observed_direction(path)
        rows = []
        for author in self.candidate_authors:
            for ts in self.candidate_timestamps:
                identity = WatermarkIdentity(author, ts)
                expected = self.signature.feature_signature(identity)
                feature_vote = np.mean([1.0 if observed[k] == expected[k] else 0.0 for k in FEATURE_NAMES])
                phi_vote = (self._phi_alignment(path, identity) + 1.0) / 2.0
                score = 0.65 * feature_vote + 0.35 * phi_vote
                rows.append({"author_id": author, "timestamp": ts, "score": float(score)})
        ranked = pd.DataFrame(rows).sort_values("score", ascending=False)
        top = ranked.iloc[0]
        second = ranked.iloc[1]["score"] if len(ranked) > 1 else 0.0
        confidence = float(max(0.0, min(1.0, top["score"] - second + top["score"])))
        return DecodeResult(
            author_id=str(top["author_id"]),
            timestamp=str(top["timestamp"]),
            confidence=confidence,
            votes={f"{r.author_id}|{r.timestamp}": float(r.score) for r in ranked.itertuples()},
        )

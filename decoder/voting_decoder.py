from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from agent_watermark.feature_analysis.extractor import BehaviorFeatureExtractor
from agent_watermark.logging.jsonl_logger import JsonlExecutionLogger
from agent_watermark.watermark.signature import FEATURE_NAMES, SignatureGenerator, WatermarkIdentity, timestamp_bucket


@dataclass
class DecodeResult:
    author_id: str
    timestamp: str
    confidence: float
    votes: Dict[str, float]
    timestamp_bucket: str | None = None
    margin: float = 0.0
    calibrated_confidence: float = 0.0
    abstained: bool = False
    abstain_reason: str | None = None


class MultiStatisticVotingDecoder:
    """Offline decoder that votes over log-derived behavior statistics and action phis."""

    def __init__(
        self,
        candidate_authors: Iterable[str],
        candidate_timestamps: Iterable[str],
        timestamp_granularity: str = "exact",
        min_margin: float = 0.08,
        min_confidence: float = 0.55,
        action_weight: float = 0.85,
        feature_weight: float = 0.15,
    ):
        self.candidate_authors = list(candidate_authors)
        self.candidate_timestamps = list(candidate_timestamps)
        self.timestamp_granularity = timestamp_granularity
        self.min_margin = min_margin
        self.min_confidence = min_confidence
        total_weight = max(action_weight + feature_weight, 1e-12)
        self.action_weight = action_weight / total_weight
        self.feature_weight = feature_weight / total_weight
        self.extractor = BehaviorFeatureExtractor()
        self.signature = SignatureGenerator()

    def _observed_direction_from_steps(self, steps: Sequence) -> Dict[str, int]:
        features = self.extractor.from_steps(list(steps))
        signs = {}
        for name in FEATURE_NAMES:
            value = features.get(name, 0.0)
            signs[name] = 1 if value >= 0.5 else -1
        return signs

    def _feature_vote(self, observed: Dict[str, int], identity: WatermarkIdentity) -> float:
        """Weak auxiliary vote from aggregate trajectory statistics.

        This vote is intentionally low-weight because one short run can be too
        sparse for stable behavioral frequencies.
        """
        expected = self.signature.feature_signature(identity, self.timestamp_granularity)
        return float(np.mean([1.0 if observed[k] == expected[k] else 0.0 for k in FEATURE_NAMES]))

    def _legacy_phi_alignment(self, path: str | Path, identity: WatermarkIdentity) -> float:
        """Compatibility score for logs that include watermark_phi.

        Newer scoring does not require watermark_phi because that field is a
        diagnostic, not a necessary decoding secret.
        """
        steps = JsonlExecutionLogger.read(path)
        margins: List[float] = []
        for step in steps:
            for candidate in step.candidate_actions:
                if candidate.name == step.chosen_action:
                    margins.append(candidate.watermark_phi * self.signature.tool_phi(identity, candidate.name))
        return float(np.mean(margins)) if margins else 0.0

    def _action_delta_alignment(self, steps: Sequence, identity: WatermarkIdentity) -> float:
        """Recover the watermark by comparing relative log-probability shifts.

        For each action-selection step the embedder applies:

            log p'(a) = log p(a) + lambda * phi(a) - log Z

        The unknown normalizer log Z is constant within the candidate set, so
        centering the observed deltas removes it. This makes the decoder depend
        only on execution logs: raw probabilities, watermarked probabilities,
        and candidate action names.
        """
        similarities: List[float] = []
        for step in steps:
            if len(step.candidate_actions) < 2:
                continue
            observed = []
            expected = []
            for candidate in step.candidate_actions:
                raw = max(candidate.raw_probability, 1e-12)
                watermarked = max(candidate.watermarked_probability, 1e-12)
                observed.append(np.log(watermarked) - np.log(raw))
                expected.append(self.signature.tool_phi(identity, candidate.name, self.timestamp_granularity))
            obs = np.asarray(observed, dtype=float)
            exp = np.asarray(expected, dtype=float)
            obs = obs - obs.mean()
            exp = exp - exp.mean()
            denom = float(np.linalg.norm(obs) * np.linalg.norm(exp))
            if denom > 1e-12:
                similarities.append(float(np.dot(obs, exp) / denom))
        if not similarities:
            return 0.0
        return float((np.mean(similarities) + 1.0) / 2.0)

    def decode(self, path: str | Path) -> DecodeResult:
        steps = JsonlExecutionLogger.read(path)
        observed = self._observed_direction_from_steps(steps)
        rows = []
        for author in self.candidate_authors:
            for ts in self.candidate_timestamps:
                identity = WatermarkIdentity(author, ts)
                feature_vote = self._feature_vote(observed, identity)
                action_vote = self._action_delta_alignment(steps, identity)
                score = self.action_weight * action_vote + self.feature_weight * feature_vote
                rows.append(
                    {
                        "author_id": author,
                        "timestamp": ts,
                        "timestamp_bucket": timestamp_bucket(ts, self.timestamp_granularity),
                        "score": float(score),
                    }
                )
        candidates = pd.DataFrame(rows)
        ranked = (
            candidates.sort_values("score", ascending=False)
            .groupby(["author_id", "timestamp_bucket"], as_index=False)
            .first()
            .sort_values("score", ascending=False)
        )
        top = ranked.iloc[0]
        second = ranked.iloc[1]["score"] if len(ranked) > 1 else 0.0
        margin = float(top["score"] - second)
        confidence = float(max(0.0, min(1.0, top["score"] - second + top["score"])))
        calibrated = float(1.0 / (1.0 + np.exp(-12.0 * (margin - self.min_margin))))
        abstained = bool(margin < self.min_margin or confidence < self.min_confidence)
        reason = None
        if margin < self.min_margin:
            reason = "low_margin"
        elif confidence < self.min_confidence:
            reason = "low_confidence"
        return DecodeResult(
            author_id=str(top["author_id"]),
            timestamp=str(top["timestamp"]),
            confidence=confidence,
            votes={f"{r.author_id}|{r.timestamp_bucket}": float(r.score) for r in ranked.itertuples()},
            timestamp_bucket=str(top["timestamp_bucket"]),
            margin=margin,
            calibrated_confidence=calibrated,
            abstained=abstained,
            abstain_reason=reason,
        )

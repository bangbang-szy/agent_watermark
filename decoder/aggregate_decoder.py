from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from agent_watermark.decoder.voting_decoder import MultiStatisticVotingDecoder


@dataclass
class AggregateDecodeResult:
    author_id: str
    timestamp_bucket: str
    confidence: float
    margin: float
    margin_per_run: float
    abstained: bool
    num_runs: int
    votes: Dict[str, float]


class MultiRunVotingDecoder:
    """Aggregate decoder for deployment settings with multiple logs per author/window."""

    def __init__(
        self,
        candidate_authors: Iterable[str],
        candidate_timestamps: Iterable[str],
        timestamp_granularity: str = "hour",
        min_margin_per_run: float = 0.08,
        min_confidence: float = 0.55,
    ):
        self.single_decoder = MultiStatisticVotingDecoder(
            candidate_authors,
            candidate_timestamps,
            timestamp_granularity=timestamp_granularity,
            min_margin=min_margin_per_run,
            min_confidence=min_confidence,
        )
        self.min_margin_per_run = min_margin_per_run
        self.min_confidence = min_confidence

    def decode_many(self, log_paths: Iterable[str | Path]) -> AggregateDecodeResult:
        vote_sums: Dict[str, float] = {}
        num_runs = 0
        for path in log_paths:
            decoded = self.single_decoder.decode(path)
            num_runs += 1
            votes = decoded.votes
            if isinstance(votes, str):
                votes = ast.literal_eval(votes)
            for candidate, score in votes.items():
                vote_sums[candidate] = vote_sums.get(candidate, 0.0) + float(score)
        if not vote_sums:
            raise ValueError("No logs were decoded.")
        ranked = sorted(vote_sums.items(), key=lambda kv: kv[1], reverse=True)
        top_candidate, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        author_id, timestamp_bucket = top_candidate.split("|", 1)
        margin = float(top_score - second_score)
        margin_per_run = margin / max(1, num_runs)
        confidence = float(1.0 / (1.0 + np.exp(-12.0 * (margin_per_run - self.min_margin_per_run))))
        abstained = margin_per_run < self.min_margin_per_run or confidence < self.min_confidence
        return AggregateDecodeResult(
            author_id=author_id,
            timestamp_bucket=timestamp_bucket,
            confidence=confidence,
            margin=margin,
            margin_per_run=margin_per_run,
            abstained=abstained,
            num_runs=num_runs,
            votes=vote_sums,
        )

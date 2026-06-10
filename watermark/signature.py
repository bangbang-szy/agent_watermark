from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable

import numpy as np


FEATURE_NAMES = [
    "tool_usage_frequency",
    "search_tool_ratio",
    "average_trajectory_length",
    "early_stop_ratio",
    "tool_transition_pattern",
    "action_rank_preference",
    "database_query_ratio",
]


@dataclass(frozen=True)
class WatermarkIdentity:
    """Identity payload encoded as small behavior biases."""

    author_id: str
    timestamp: str

    @classmethod
    def create(cls, author_id: str) -> "WatermarkIdentity":
        return cls(author_id=author_id, timestamp=datetime.now(timezone.utc).isoformat())

    @property
    def payload(self) -> str:
        return f"{self.author_id}|{self.timestamp}"


def timestamp_bucket(timestamp: str, granularity: str = "exact") -> str:
    """Canonicalize timestamps for payload-capacity controlled watermarking."""
    if granularity == "exact":
        return timestamp
    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone(timezone.utc)
    if granularity == "minute":
        return dt.replace(second=0, microsecond=0).isoformat()
    if granularity == "hour":
        return dt.replace(minute=0, second=0, microsecond=0).isoformat()
    if granularity == "day":
        return dt.replace(hour=0, minute=0, second=0, microsecond=0).date().isoformat()
    raise ValueError(f"Unsupported timestamp granularity: {granularity}")


class SignatureGenerator:
    """Deterministically maps identity payloads to feature signs and tool biases."""

    def __init__(self, feature_names: Iterable[str] = FEATURE_NAMES):
        self.feature_names = list(feature_names)

    def identity_payload(self, identity: WatermarkIdentity, timestamp_granularity: str = "exact") -> str:
        return f"{identity.author_id}|{timestamp_bucket(identity.timestamp, timestamp_granularity)}"

    def feature_signature(self, identity: WatermarkIdentity, timestamp_granularity: str = "exact") -> Dict[str, int]:
        digest = hashlib.sha256(self.identity_payload(identity, timestamp_granularity).encode("utf-8")).digest()
        return {
            name: 1 if digest[i % len(digest)] & 1 else -1
            for i, name in enumerate(self.feature_names)
        }

    def tool_phi(self, identity: WatermarkIdentity, tool_name: str, timestamp_granularity: str = "exact") -> float:
        """Return a stable action-level phi in [-1, 1] for soft reweighting."""
        data = f"{self.identity_payload(identity, timestamp_granularity)}|tool|{tool_name}".encode("utf-8")
        integer = int.from_bytes(hashlib.sha256(data).digest()[:8], "big")
        return (integer / (2**64 - 1)) * 2.0 - 1.0

    def author_timestamp_code(
        self,
        identity: WatermarkIdentity,
        bits: int = 32,
        timestamp_granularity: str = "exact",
    ) -> np.ndarray:
        digest = hashlib.sha256(self.identity_payload(identity, timestamp_granularity).encode("utf-8")).digest()
        unpacked = np.unpackbits(np.frombuffer(digest, dtype=np.uint8))
        return np.where(unpacked[:bits] > 0, 1, -1)

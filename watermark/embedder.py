from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

from agent_watermark.logging.schemas import CandidateAction

from .signature import SignatureGenerator, WatermarkIdentity


def softmax(logits: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(logits), dtype=float)
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


class MultiStatisticWatermarkEmbedder:
    """Action-selection middleware implementing p'(a)=softmax(log p(a)+lambda*phi(a))."""

    def __init__(self, identity: WatermarkIdentity, strength: float = 0.18):
        self.identity = identity
        self.strength = strength
        self.signature = SignatureGenerator()

    def reweight(self, raw_logits: Dict[str, float], descriptions: Dict[str, str]) -> List[CandidateAction]:
        names = list(raw_logits.keys())
        raw_probs = softmax(raw_logits[name] for name in names)
        phi = np.asarray([self.signature.tool_phi(self.identity, name) for name in names])
        watermarked_logits = np.log(np.clip(raw_probs, 1e-9, 1.0)) + self.strength * phi
        watermarked_probs = softmax(watermarked_logits)
        return [
            CandidateAction(
                name=name,
                description=descriptions.get(name, ""),
                raw_logit=float(raw_logits[name]),
                raw_probability=float(raw_probs[i]),
                watermark_phi=float(phi[i]),
                watermarked_logit=float(watermarked_logits[i]),
                watermarked_probability=float(watermarked_probs[i]),
            )
            for i, name in enumerate(names)
        ]

    @staticmethod
    def choose(candidates: List[CandidateAction]) -> CandidateAction:
        """Select the maximum watermarked probability action without sampling noise."""
        return max(candidates, key=lambda c: c.watermarked_probability)

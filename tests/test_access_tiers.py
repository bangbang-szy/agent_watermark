from pathlib import Path

import pytest

from agent_watermark.decoder.voting_decoder import MultiStatisticVotingDecoder
from agent_watermark.logging.jsonl_logger import JsonlExecutionLogger
from agent_watermark.logging.schemas import StepLog
from agent_watermark.watermark.embedder import MultiStatisticWatermarkEmbedder
from agent_watermark.watermark.signature import WatermarkIdentity


@pytest.mark.parametrize(
    "tier",
    ["actions_only", "actions_candidates", "watermarked_probabilities", "full_trusted_logs"],
)
def test_decoder_supports_each_access_tier(tmp_path: Path, tier: str):
    identity = WatermarkIdentity("alice", "2026-06-10T12:34:56+00:00")
    candidates = MultiStatisticWatermarkEmbedder(identity, 0.2, timestamp_granularity="hour").reweight(
        {"search": 0.2, "sqlite_db": 0.1, "final_answer": -0.5}, {}
    )
    step = StepLog(
        run_id="r",
        step_index=0,
        author_id="alice",
        watermark_timestamp=identity.timestamp,
        task="q",
        state={},
        reasoning_trace="",
        candidate_actions=candidates,
        chosen_action=max(candidates, key=lambda c: float(c.watermarked_probability)).name,
    )
    path = tmp_path / "run.jsonl"
    JsonlExecutionLogger.write(path, [step])
    result = MultiStatisticVotingDecoder(
        ["alice", "bob"], [identity.timestamp], timestamp_granularity="hour", access_tier=tier
    ).decode(path)
    assert result.author_id in {"alice", "bob"}
    assert 0.0 <= result.confidence <= 1.0

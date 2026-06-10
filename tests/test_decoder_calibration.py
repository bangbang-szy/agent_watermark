from pathlib import Path

from agent_watermark.decoder.voting_decoder import MultiStatisticVotingDecoder
from agent_watermark.logging.jsonl_logger import JsonlExecutionLogger
from agent_watermark.logging.schemas import CandidateAction, StepLog
from agent_watermark.watermark.embedder import MultiStatisticWatermarkEmbedder
from agent_watermark.watermark.signature import WatermarkIdentity


def test_decoder_returns_calibration_fields(tmp_path: Path):
    identity = WatermarkIdentity("alice", "2026-06-10T12:34:56+00:00")
    embedder = MultiStatisticWatermarkEmbedder(identity, 0.2, timestamp_granularity="hour")
    candidates = embedder.reweight({"search": 0.2, "sqlite_db": 0.1, "final_answer": -0.5}, {})
    step = StepLog(
        run_id="r",
        step_index=0,
        author_id="alice",
        watermark_timestamp=identity.timestamp,
        task="q",
        state={},
        reasoning_trace="",
        candidate_actions=candidates,
        chosen_action=max(candidates, key=lambda c: c.watermarked_probability).name,
    )
    path = tmp_path / "run.jsonl"
    JsonlExecutionLogger.write(path, [step])
    result = MultiStatisticVotingDecoder(
        ["alice", "bob"],
        [identity.timestamp],
        timestamp_granularity="hour",
    ).decode(path)
    assert result.timestamp_bucket == "2026-06-10T12:00:00+00:00"
    assert result.margin >= 0.0
    assert 0.0 <= result.calibrated_confidence <= 1.0

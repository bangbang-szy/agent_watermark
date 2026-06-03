from agent_watermark.watermark.embedder import MultiStatisticWatermarkEmbedder
from agent_watermark.watermark.signature import WatermarkIdentity


def test_reweight_probabilities_sum_to_one():
    identity = WatermarkIdentity("alice", "2026-01-01T00:00:00+00:00")
    embedder = MultiStatisticWatermarkEmbedder(identity, 0.2)
    candidates = embedder.reweight({"search": 1.0, "sqlite_db": 0.5}, {"search": "", "sqlite_db": ""})
    assert abs(sum(c.watermarked_probability for c in candidates) - 1.0) < 1e-9
    assert {c.name for c in candidates} == {"search", "sqlite_db"}

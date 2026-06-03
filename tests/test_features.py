from agent_watermark.feature_analysis.extractor import BehaviorFeatureExtractor
from agent_watermark.logging.schemas import CandidateAction, StepLog


def test_feature_extraction_from_real_schema():
    step = StepLog(
        run_id="r",
        step_index=0,
        author_id="alice",
        watermark_timestamp="t",
        task="q",
        state={},
        reasoning_trace="",
        candidate_actions=[
            CandidateAction(name="search", raw_logit=1, raw_probability=0.7, watermarked_logit=1, watermarked_probability=0.8),
            CandidateAction(name="final_answer", raw_logit=0, raw_probability=0.3, watermarked_logit=0, watermarked_probability=0.2),
        ],
        chosen_action="search",
    )
    features = BehaviorFeatureExtractor().from_steps([step])
    assert features["tool_usage_frequency"] == 1.0
    assert features["search_tool_ratio"] == 1.0

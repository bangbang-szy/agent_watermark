from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    """Return a stable UTC timestamp for portable JSONL logs."""
    return datetime.now(timezone.utc).isoformat()


class CandidateAction(BaseModel):
    """One tool/action candidate proposed at the action-selection boundary."""

    name: str
    description: str = ""
    raw_logit: float
    raw_probability: float
    watermark_phi: float = 0.0
    watermarked_logit: float
    watermarked_probability: float


class ToolCallLog(BaseModel):
    """Record of one real tool invocation and its observation."""

    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    observation: Any = None
    error: Optional[str] = None
    started_at: str = Field(default_factory=utc_now_iso)
    finished_at: Optional[str] = None


class StepLog(BaseModel):
    """Full action step emitted by a real agent run."""

    run_id: str
    step_index: int
    author_id: str
    watermark_timestamp: str
    task: str
    state: Dict[str, Any]
    reasoning_trace: str
    candidate_actions: List[CandidateAction]
    chosen_action: str
    chosen_arguments: Dict[str, Any] = Field(default_factory=dict)
    tool_call: Optional[ToolCallLog] = None
    final_answer: Optional[str] = None
    created_at: str = Field(default_factory=utc_now_iso)


class RunManifest(BaseModel):
    """Small manifest for grouping JSONL step logs."""

    run_id: str
    author_id: str
    watermark_timestamp: str
    task: str
    log_path: str
    started_at: str = Field(default_factory=utc_now_iso)
    finished_at: Optional[str] = None

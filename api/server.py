from __future__ import annotations

import os
from pathlib import Path
from typing import List

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from agent_watermark.agent_core.database import init_demo_database
from agent_watermark.agent_core.langgraph_agent import AgentConfig, WatermarkedLangGraphAgent
from agent_watermark.agent_core.tools import build_tools
from agent_watermark.decoder.voting_decoder import MultiStatisticVotingDecoder
from agent_watermark.feature_analysis.extractor import BehaviorFeatureExtractor
from agent_watermark.feature_analysis.independence import BehaviorIndependenceAnalyzer
from agent_watermark.logging.jsonl_logger import JsonlExecutionLogger
from agent_watermark.watermark.signature import WatermarkIdentity


class RunRequest(BaseModel):
    task: str
    author_id: str = "alice-lab"


class DecodeRequest(BaseModel):
    log_path: str
    authors: List[str]
    timestamps: List[str]


class AnalyzeRequest(BaseModel):
    log_paths: List[str]


load_dotenv()
app = FastAPI(title="Agent Watermarking System")
cfg = yaml.safe_load(open(Path(__file__).parents[1] / "configs/default.yaml", "r", encoding="utf-8"))


@app.post("/run")
def run_agent(req: RunRequest):
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required.")
    init_demo_database(cfg["sqlite_path"])
    tools = build_tools(cfg["sqlite_path"], cfg["workspace_dir"])
    identity = WatermarkIdentity.create(req.author_id)
    agent_cfg = AgentConfig(cfg["openai_model"], cfg["temperature"], cfg["max_steps"], cfg["watermark_lambda"])
    agent = WatermarkedLangGraphAgent(tools, JsonlExecutionLogger(cfg["log_dir"]), identity, agent_cfg)
    result = agent.run(req.task)
    return {"run_id": result["run_id"], "log_path": str(Path(cfg["log_dir"]) / f"{result['run_id']}.jsonl"), "timestamp": identity.timestamp, "answer": result["answer"]}


@app.post("/decode")
def decode(req: DecodeRequest):
    return MultiStatisticVotingDecoder(req.authors, req.timestamps).decode(req.log_path).__dict__


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    df = BehaviorFeatureExtractor().dataframe(req.log_paths)
    result = BehaviorIndependenceAnalyzer().analyze(df.drop(columns=["log_path"], errors="ignore"), cfg["analysis_dir"])
    return {"independent_behaviors": result.independent_behaviors, "stability_scores": result.stability_scores}

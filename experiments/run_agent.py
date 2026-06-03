from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from agent_watermark.agent_core.database import init_demo_database
from agent_watermark.agent_core.langgraph_agent import AgentConfig, WatermarkedLangGraphAgent
from agent_watermark.agent_core.tools import build_tools
from agent_watermark.experiments.tasks import all_tasks
from agent_watermark.logging.jsonl_logger import JsonlExecutionLogger
from agent_watermark.watermark.signature import WatermarkIdentity


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parents[1] / "configs/default.yaml"))
    parser.add_argument("--author-id", default="alice-lab")
    parser.add_argument("--task", default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before running the real agent.")
    cfg = load_config(args.config)
    init_demo_database(cfg["sqlite_path"])
    Path(cfg["workspace_dir"]).mkdir(parents=True, exist_ok=True)
    logger = JsonlExecutionLogger(cfg["log_dir"])
    identity = WatermarkIdentity.create(args.author_id)
    tools = build_tools(cfg["sqlite_path"], cfg["workspace_dir"])
    agent_cfg = AgentConfig(cfg["openai_model"], cfg["temperature"], cfg["max_steps"], cfg["watermark_lambda"])
    tasks = all_tasks() if args.all else [args.task or all_tasks()[0]]
    for task in tasks:
        agent = WatermarkedLangGraphAgent(tools, logger, identity, agent_cfg)
        result = agent.run(task)
        print({"run_id": result["run_id"], "answer": result["answer"], "timestamp": identity.timestamp})


if __name__ == "__main__":
    main()

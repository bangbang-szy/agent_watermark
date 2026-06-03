from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from agent_watermark.agent_core.database import init_demo_database
from agent_watermark.agent_core.langchain_agent import WatermarkedLangChainAgent
from agent_watermark.agent_core.langgraph_agent import AgentConfig, WatermarkedLangGraphAgent
from agent_watermark.agent_core.tools import build_tools
from agent_watermark.experiments.tasks import all_tasks
from agent_watermark.logging.jsonl_logger import JsonlExecutionLogger
from agent_watermark.watermark.signature import WatermarkIdentity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parents[1] / "configs/default.yaml"))
    parser.add_argument("--author-id", default="alice-lab")
    parser.add_argument("--framework", choices=["langgraph", "langchain"], default="langgraph")
    args = parser.parse_args()
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before migration experiments.")
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    init_demo_database(cfg["sqlite_path"])
    tools = build_tools(cfg["sqlite_path"], cfg["workspace_dir"])
    logger = JsonlExecutionLogger(cfg["log_dir"])
    identity = WatermarkIdentity.create(args.author_id)
    agent_cfg = AgentConfig(cfg["openai_model"], cfg["temperature"], cfg["max_steps"], cfg["watermark_lambda"])
    cls = WatermarkedLangGraphAgent if args.framework == "langgraph" else WatermarkedLangChainAgent
    agent = cls(tools, logger, identity, agent_cfg)
    result = agent.run(all_tasks()[0])
    print({"framework": args.framework, "run_id": result["run_id"], "timestamp": identity.timestamp})


if __name__ == "__main__":
    main()

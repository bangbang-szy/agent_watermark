"""Generate real held-out controls for open-set watermark evaluation.

The controls call the same LangGraph agent and real tools as the main study.
Use a new author ID with a non-zero lambda for an unknown-watermarked control,
and set --watermark-lambda 0 for a genuinely unwatermarked control.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from agent_watermark.experiments.evaluate_watermark import load_config, run_logs, select_tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate real open-set control trajectories.")
    parser.add_argument("--config", default=str(Path(__file__).parents[1] / "configs/deepseek.yaml"))
    parser.add_argument("--author-id", required=True)
    parser.add_argument("--watermark-lambda", type=float, required=True)
    parser.add_argument("--tasks", type=int, default=0, help="0 runs every built-in task.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    load_dotenv()
    config = dict(load_config(args.config))
    api_key_env = config.get("llm_api_key_env", "OPENAI_API_KEY")
    if not os.getenv(api_key_env):
        raise RuntimeError(f"Set {api_key_env} before generating real controls.")
    if args.tasks < 0:
        raise ValueError("--tasks must be non-negative.")

    args.out.mkdir(parents=True, exist_ok=True)
    config["log_dir"] = str(args.out / "logs")
    task_limit = None if args.tasks == 0 else args.tasks
    manifest = run_logs(
        config,
        [args.author_id],
        select_tasks(task_limit),
        args.repeats,
        [args.watermark_lambda],
    )
    manifest.to_csv(args.out / "run_manifest.csv", index=False)
    print(f"generated {len(manifest)} real trajectories")
    print(f"manifest: {args.out / 'run_manifest.csv'}")


if __name__ == "__main__":
    main()

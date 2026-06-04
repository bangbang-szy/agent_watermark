from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

from agent_watermark.agent_core.database import init_demo_database
from agent_watermark.agent_core.langgraph_agent import AgentConfig, WatermarkedLangGraphAgent
from agent_watermark.agent_core.tools import build_tools
from agent_watermark.decoder.voting_decoder import MultiStatisticVotingDecoder
from agent_watermark.experiments.robustness import crop_log, lightweight_finetune_attack, rewrite_output
from agent_watermark.experiments.tasks import TASKS, all_tasks
from agent_watermark.feature_analysis.extractor import BehaviorFeatureExtractor
from agent_watermark.logging.jsonl_logger import JsonlExecutionLogger
from agent_watermark.watermark.signature import WatermarkIdentity


def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists() and not config_path.is_absolute():
        repo_relative = Path(__file__).parents[1] / config_path
        if repo_relative.exists():
            config_path = repo_relative
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_agent(cfg: dict, author_id: str) -> tuple[WatermarkedLangGraphAgent, WatermarkIdentity]:
    identity = WatermarkIdentity.create(author_id)
    tools = build_tools(cfg["sqlite_path"], cfg["workspace_dir"])
    agent_cfg = AgentConfig(
        model=cfg["openai_model"],
        temperature=cfg["temperature"],
        max_steps=cfg["max_steps"],
        watermark_lambda=cfg["watermark_lambda"],
        api_key_env=cfg.get("llm_api_key_env", "OPENAI_API_KEY"),
        base_url=cfg.get("llm_base_url"),
    )
    return WatermarkedLangGraphAgent(tools, JsonlExecutionLogger(cfg["log_dir"]), identity, agent_cfg), identity


def select_tasks(limit: int | None) -> List[str]:
    tasks = all_tasks()
    return tasks if limit is None else tasks[:limit]


def run_logs(cfg: dict, authors: List[str], tasks: List[str], repeats: int) -> pd.DataFrame:
    rows = []
    init_demo_database(cfg["sqlite_path"])
    Path(cfg["workspace_dir"]).mkdir(parents=True, exist_ok=True)
    for author_id in authors:
        for repeat in range(repeats):
            for task in tasks:
                agent, identity = build_agent(cfg, author_id)
                result = agent.run(task)
                log_path = str(Path(cfg["log_dir"]) / f"{result['run_id']}.jsonl")
                rows.append(
                    {
                        "run_id": result["run_id"],
                        "author_id": author_id,
                        "timestamp": identity.timestamp,
                        "task": task,
                        "repeat": repeat,
                        "log_path": log_path,
                        "answer": result.get("answer"),
                    }
                )
                print(json.dumps(rows[-1], ensure_ascii=False))
    return pd.DataFrame(rows)


def existing_manifest(logs: Iterable[str]) -> pd.DataFrame:
    rows = []
    for path in logs:
        steps = JsonlExecutionLogger.read(path)
        if not steps:
            continue
        first = steps[0]
        rows.append(
            {
                "run_id": first.run_id,
                "author_id": first.author_id,
                "timestamp": first.watermark_timestamp,
                "task": first.task,
                "repeat": 0,
                "log_path": str(path),
                "answer": steps[-1].final_answer,
            }
        )
    return pd.DataFrame(rows)


def decode_clean(manifest: pd.DataFrame, candidate_authors: List[str]) -> pd.DataFrame:
    timestamps = manifest["timestamp"].drop_duplicates().tolist()
    decoder = MultiStatisticVotingDecoder(candidate_authors, timestamps)
    rows = []
    for item in manifest.itertuples():
        decoded = decoder.decode(item.log_path)
        rows.append(
            {
                "run_id": item.run_id,
                "true_author": item.author_id,
                "decoded_author": decoded.author_id,
                "true_timestamp": item.timestamp,
                "decoded_timestamp": decoded.timestamp,
                "confidence": decoded.confidence,
                "correct_author": decoded.author_id == item.author_id,
                "correct_timestamp": decoded.timestamp == item.timestamp,
                "votes": decoded.votes,
            }
        )
    return pd.DataFrame(rows)


def decode_attacks(manifest: pd.DataFrame, candidate_authors: List[str], out_dir: Path) -> pd.DataFrame:
    attack_dir = out_dir / "attacked_logs"
    attack_dir.mkdir(parents=True, exist_ok=True)
    timestamps = manifest["timestamp"].drop_duplicates().tolist()
    decoder = MultiStatisticVotingDecoder(candidate_authors, timestamps)
    rows = []
    for item in manifest.itertuples():
        attack_specs = [("clean", 0.0, Path(item.log_path))]
        for ratio in [0.1, 0.2, 0.3]:
            target = attack_dir / f"{item.run_id}_crop_{int(ratio * 100)}.jsonl"
            crop_log(Path(item.log_path), ratio, target)
            attack_specs.append(("log_crop", ratio, target))
        rewritten = attack_dir / f"{item.run_id}_rewrite.jsonl"
        rewrite_output(Path(item.log_path), rewritten)
        attack_specs.append(("output_rewrite", 0.0, rewritten))
        tuned = attack_dir / f"{item.run_id}_preference_drift.jsonl"
        lightweight_finetune_attack(Path(item.log_path), tuned, preferred_tool="search")
        attack_specs.append(("preference_drift", 0.0, tuned))
        for attack, severity, path in attack_specs:
            decoded = decoder.decode(path)
            rows.append(
                {
                    "run_id": item.run_id,
                    "attack": attack,
                    "severity": severity,
                    "true_author": item.author_id,
                    "decoded_author": decoded.author_id,
                    "confidence": decoded.confidence,
                    "correct_author": decoded.author_id == item.author_id,
                    "log_path": str(path),
                }
            )
    return pd.DataFrame(rows)


def extract_step_table(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in manifest.itertuples():
        for step in JsonlExecutionLogger.read(item.log_path):
            rows.append(
                {
                    "run_id": item.run_id,
                    "author_id": item.author_id,
                    "step_index": step.step_index,
                    "chosen_action": step.chosen_action,
                    "candidate_count": len(step.candidate_actions),
                    "is_final": step.chosen_action == "final_answer",
                }
            )
    return pd.DataFrame(rows)


def votes_table(clean_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in clean_results.itertuples():
        for candidate, score in item.votes.items():
            author, timestamp = candidate.split("|", 1)
            rows.append(
                {
                    "run_id": item.run_id,
                    "candidate_author": author,
                    "candidate_timestamp": timestamp,
                    "score": score,
                    "true_author": item.true_author,
                }
            )
    return pd.DataFrame(rows)


def save_bar(ax, labels, values, title: str, ylabel: str, color: str) -> None:
    ax.bar(labels, values, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)


def plot_metrics(
    manifest: pd.DataFrame,
    clean_results: pd.DataFrame,
    attack_results: pd.DataFrame,
    feature_df: pd.DataFrame,
    step_df: pd.DataFrame,
    vote_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "clean_author_accuracy": float(clean_results["correct_author"].mean()),
        "clean_timestamp_accuracy": float(clean_results["correct_timestamp"].mean()),
        "clean_mean_confidence": float(clean_results["confidence"].mean()),
        "num_runs": int(len(manifest)),
        "num_steps": int(len(step_df)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    accuracy = clean_results.groupby("true_author")["correct_author"].mean()
    save_bar(axes[0, 0], accuracy.index, accuracy.values, "Clean decoding accuracy", "accuracy", "#2f6f9f")

    axes[0, 1].hist(clean_results["confidence"], bins=8, color="#35a77c", edgecolor="white")
    axes[0, 1].set_title("Clean confidence distribution")
    axes[0, 1].set_xlabel("confidence")
    axes[0, 1].set_ylabel("runs")
    axes[0, 1].grid(axis="y", alpha=0.25)

    tool_counts = step_df["chosen_action"].value_counts()
    axes[0, 2].pie(tool_counts.values, labels=tool_counts.index, autopct="%1.0f%%", startangle=90)
    axes[0, 2].set_title("Action/tool usage")

    robust = attack_results.groupby(["attack", "severity"], as_index=False).agg(
        accuracy=("correct_author", "mean"), confidence=("confidence", "mean")
    )
    for attack, part in robust.groupby("attack"):
        axes[1, 0].plot(part["severity"], part["accuracy"], marker="o", label=attack)
    axes[1, 0].set_title("Robustness accuracy")
    axes[1, 0].set_xlabel("severity")
    axes[1, 0].set_ylabel("accuracy")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.25)

    traj = feature_df["average_trajectory_length"]
    axes[1, 1].hist(traj, bins=max(3, min(10, len(traj))), color="#8a63b0", edgecolor="white")
    axes[1, 1].set_title("Trajectory length distribution")
    axes[1, 1].set_xlabel("steps")
    axes[1, 1].set_ylabel("runs")
    axes[1, 1].grid(axis="y", alpha=0.25)

    vote_mean = vote_df.groupby("candidate_author")["score"].mean().sort_values(ascending=False)
    save_bar(axes[1, 2], vote_mean.index, vote_mean.values, "Mean decoding vote score", "score", "#c47a34")

    fig.suptitle("Agent Behavior Watermark Evaluation", fontsize=16)
    fig.tight_layout()
    fig.savefig(plot_dir / "watermark_evaluation_overview.png", dpi=180)
    plt.close(fig)

    numeric_features = feature_df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(12, 5))
    numeric_features.boxplot(ax=ax, rot=30)
    ax.set_title("Behavior statistics extracted from execution logs")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "behavior_statistics_boxplot.png", dpi=180)
    plt.close(fig)

    pivot = vote_df.pivot_table(index="run_id", columns="candidate_author", values="score", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.35)))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="magma")
    ax.set_title("Per-run candidate vote scores")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    fig.colorbar(im, ax=ax, label="vote score")
    fig.tight_layout()
    fig.savefig(plot_dir / "vote_score_heatmap.png", dpi=180)
    plt.close(fig)

    robust.to_csv(out_dir / "robustness_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    for attack, part in robust.groupby("attack"):
        ax.plot(part["severity"], part["confidence"], marker="o", label=attack)
    ax.set_title("Robustness mean confidence")
    ax.set_xlabel("severity")
    ax.set_ylabel("confidence")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "robustness_confidence_curve.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-task watermark evaluation and generate presentation plots.")
    parser.add_argument("--config", default=str(Path(__file__).parents[1] / "configs/deepseek.yaml"))
    parser.add_argument("--authors", nargs="+", default=["alice-lab", "bob-lab", "carol-lab"])
    parser.add_argument("--watermarked-author", default="alice-lab")
    parser.add_argument("--tasks", type=int, default=6, help="Number of built-in tasks to run. Use 0 for all tasks.")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--logs", nargs="*", help="Use existing JSONL logs instead of running the agent.")
    parser.add_argument("--out", default="runtime/evaluation")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    load_dotenv()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)

    if args.logs:
        manifest = existing_manifest(args.logs)
    else:
        api_key_env = cfg.get("llm_api_key_env", "OPENAI_API_KEY")
        if not os.getenv(api_key_env):
            raise RuntimeError(f"Set {api_key_env} before running evaluation.")
        task_limit = None if args.tasks == 0 else args.tasks
        manifest = run_logs(cfg, [args.watermarked_author], select_tasks(task_limit), args.repeats)

    if manifest.empty:
        raise RuntimeError("No logs available for evaluation.")

    manifest.to_csv(out_dir / "run_manifest.csv", index=False)
    clean_results = decode_clean(manifest, args.authors)
    attack_results = decode_attacks(manifest, args.authors, out_dir)
    feature_df = BehaviorFeatureExtractor().dataframe(manifest["log_path"].tolist())
    step_df = extract_step_table(manifest)
    vote_df = votes_table(clean_results)

    clean_results.to_csv(out_dir / "clean_decoding_results.csv", index=False)
    attack_results.to_csv(out_dir / "attack_decoding_results.csv", index=False)
    feature_df.to_csv(out_dir / "behavior_features.csv", index=False)
    step_df.to_csv(out_dir / "step_actions.csv", index=False)
    vote_df.to_csv(out_dir / "vote_scores.csv", index=False)
    plot_metrics(manifest, clean_results, attack_results, feature_df, step_df, vote_df, out_dir)

    print(json.dumps(json.loads((out_dir / "summary.json").read_text(encoding="utf-8")), indent=2))
    print(f"plots: {out_dir / 'plots'}")


if __name__ == "__main__":
    main()

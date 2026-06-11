from __future__ import annotations

import argparse
import ast
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
from agent_watermark.experiments.robustness import (
    crop_log,
    lightweight_finetune_attack,
    probability_noise_attack,
    reorder_log_attack,
    rewrite_output,
    tool_call_deletion_attack,
)
from agent_watermark.experiments.tasks import all_tasks
from agent_watermark.feature_analysis.extractor import BehaviorFeatureExtractor
from agent_watermark.logging.jsonl_logger import JsonlExecutionLogger
from agent_watermark.watermark.signature import WatermarkIdentity, timestamp_bucket


def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists() and not config_path.is_absolute():
        repo_relative = Path(__file__).parents[1] / config_path
        if repo_relative.exists():
            config_path = repo_relative
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_agent(cfg: dict, author_id: str, watermark_lambda: float | None = None) -> tuple[WatermarkedLangGraphAgent, WatermarkIdentity]:
    identity = WatermarkIdentity.create(author_id)
    tools = build_tools(cfg["sqlite_path"], cfg["workspace_dir"])
    agent_cfg = AgentConfig(
        model=cfg["openai_model"],
        temperature=cfg["temperature"],
        max_steps=cfg["max_steps"],
        watermark_lambda=cfg["watermark_lambda"] if watermark_lambda is None else watermark_lambda,
        api_key_env=cfg.get("llm_api_key_env", "OPENAI_API_KEY"),
        base_url=cfg.get("llm_base_url"),
        timestamp_granularity=cfg.get("watermark_timestamp_granularity", "exact"),
    )
    return WatermarkedLangGraphAgent(tools, JsonlExecutionLogger(cfg["log_dir"]), identity, agent_cfg), identity


def select_tasks(limit: int | None) -> List[str]:
    tasks = all_tasks()
    return tasks if limit is None else tasks[:limit]


def task_success(log_path: str, answer: str | None) -> bool:
    """Heuristic task-success indicator for batch research runs."""
    if not answer or not str(answer).strip():
        return False
    steps = JsonlExecutionLogger.read(log_path)
    for step in steps:
        if step.tool_call and step.tool_call.error:
            return False
        if step.tool_call and isinstance(step.tool_call.observation, str):
            observation = step.tool_call.observation
            if any(marker in observation for marker in ["search_error", "sql_error", "python_error"]):
                return False
    return True


def run_logs(cfg: dict, authors: List[str], tasks: List[str], repeats: int, lambda_values: List[float]) -> pd.DataFrame:
    rows = []
    init_demo_database(cfg["sqlite_path"])
    Path(cfg["workspace_dir"]).mkdir(parents=True, exist_ok=True)
    for watermark_lambda in lambda_values:
        for author_id in authors:
            for repeat in range(repeats):
                for task in tasks:
                    agent, identity = build_agent(cfg, author_id, watermark_lambda)
                    result = agent.run(task)
                    log_path = str(Path(cfg["log_dir"]) / f"{result['run_id']}.jsonl")
                    row = {
                        "run_id": result["run_id"],
                        "author_id": author_id,
                        "timestamp": identity.timestamp,
                        "task": task,
                        "repeat": repeat,
                        "watermark_lambda": watermark_lambda,
                        "log_path": log_path,
                        "answer": result.get("answer"),
                    }
                    row["task_success"] = task_success(log_path, row["answer"])
                    rows.append(row)
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
                "watermark_lambda": float("nan"),
                "log_path": str(path),
                "answer": steps[-1].final_answer,
                "task_success": task_success(str(path), steps[-1].final_answer),
            }
        )
    return pd.DataFrame(rows)


def decode_clean(
    manifest: pd.DataFrame,
    candidate_authors: List[str],
    timestamp_granularity: str,
    min_margin: float,
    min_confidence: float,
) -> pd.DataFrame:
    timestamps = manifest["timestamp"].drop_duplicates().tolist()
    decoder = MultiStatisticVotingDecoder(
        candidate_authors,
        timestamps,
        timestamp_granularity=timestamp_granularity,
        min_margin=min_margin,
        min_confidence=min_confidence,
    )
    rows = []
    for item in manifest.itertuples():
        decoded = decoder.decode(item.log_path)
        true_bucket = timestamp_bucket(item.timestamp, timestamp_granularity)
        rows.append(
            {
                "run_id": item.run_id,
                "true_author": item.author_id,
                "decoded_author": decoded.author_id,
                "true_timestamp": item.timestamp,
                "decoded_timestamp": decoded.timestamp,
                "true_timestamp_bucket": true_bucket,
                "decoded_timestamp_bucket": decoded.timestamp_bucket,
                "confidence": decoded.confidence,
                "margin": decoded.margin,
                "calibrated_confidence": decoded.calibrated_confidence,
                "abstained": decoded.abstained,
                "abstain_reason": decoded.abstain_reason,
                "correct_author": decoded.author_id == item.author_id,
                "correct_timestamp": decoded.timestamp == item.timestamp,
                "correct_timestamp_bucket": decoded.timestamp_bucket == true_bucket,
                "correct_author_when_not_abstained": (decoded.author_id == item.author_id) if not decoded.abstained else np.nan,
                "votes": decoded.votes,
            }
        )
    return pd.DataFrame(rows)


def decode_attacks(
    manifest: pd.DataFrame,
    candidate_authors: List[str],
    out_dir: Path,
    timestamp_granularity: str,
    min_margin: float,
    min_confidence: float,
    enabled: bool = True,
) -> pd.DataFrame:
    if not enabled:
        return pd.DataFrame(
            columns=[
                "run_id",
                "attack",
                "severity",
                "true_author",
                "watermark_lambda",
                "decoded_author",
                "confidence",
                "margin",
                "calibrated_confidence",
                "abstained",
                "correct_author",
                "correct_author_when_not_abstained",
                "log_path",
            ]
        )
    attack_dir = out_dir / "attacked_logs"
    attack_dir.mkdir(parents=True, exist_ok=True)
    timestamps = manifest["timestamp"].drop_duplicates().tolist()
    decoder = MultiStatisticVotingDecoder(
        candidate_authors,
        timestamps,
        timestamp_granularity=timestamp_granularity,
        min_margin=min_margin,
        min_confidence=min_confidence,
    )
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
        reordered = attack_dir / f"{item.run_id}_reordered.jsonl"
        reorder_log_attack(Path(item.log_path), reordered)
        attack_specs.append(("log_reorder", 0.0, reordered))
        for sigma in [0.01, 0.03, 0.05]:
            noisy = attack_dir / f"{item.run_id}_prob_noise_{int(sigma * 100)}.jsonl"
            probability_noise_attack(Path(item.log_path), noisy, sigma=sigma)
            attack_specs.append(("probability_noise", sigma, noisy))
        deleted = attack_dir / f"{item.run_id}_tool_call_delete.jsonl"
        tool_call_deletion_attack(Path(item.log_path), deleted, ratio=0.3)
        attack_specs.append(("tool_call_deletion", 0.3, deleted))
        for attack, severity, path in attack_specs:
            decoded = decoder.decode(path)
            rows.append(
                {
                    "run_id": item.run_id,
                    "attack": attack,
                    "severity": severity,
                    "true_author": item.author_id,
                    "watermark_lambda": getattr(item, "watermark_lambda", float("nan")),
                    "decoded_author": decoded.author_id,
                    "confidence": decoded.confidence,
                    "margin": decoded.margin,
                    "calibrated_confidence": decoded.calibrated_confidence,
                    "abstained": decoded.abstained,
                    "correct_author": decoded.author_id == item.author_id,
                    "correct_author_when_not_abstained": (decoded.author_id == item.author_id) if not decoded.abstained else np.nan,
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


def aggregate_runs(clean_results: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Aggregate multiple runs by lambda/repeat/true author for paper-style group decoding."""
    merged = clean_results.merge(
        manifest[["run_id", "watermark_lambda", "repeat", "task_success"]],
        on="run_id",
        how="left",
    )
    rows = []
    for keys, part in merged.groupby(["watermark_lambda", "repeat", "true_author"], dropna=False):
        watermark_lambda, repeat, true_author = keys
        vote_sums: Dict[str, float] = {}
        for votes in part["votes"]:
            if isinstance(votes, str):
                votes = ast.literal_eval(votes)
            for candidate, score in votes.items():
                vote_sums[candidate] = vote_sums.get(candidate, 0.0) + float(score)
        ranked = sorted(vote_sums.items(), key=lambda kv: kv[1], reverse=True)
        if not ranked:
            continue
        top_candidate, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        decoded_author, decoded_timestamp = top_candidate.split("|", 1)
        rows.append(
            {
                "watermark_lambda": watermark_lambda,
                "repeat": repeat,
                "true_author": true_author,
                "decoded_author": decoded_author,
                "decoded_timestamp": decoded_timestamp,
                "num_runs": int(len(part)),
                "task_success_rate": float(part["task_success"].mean()),
                "aggregate_margin": float(top_score - second_score),
                "correct_author": decoded_author == true_author,
            }
        )
    return pd.DataFrame(rows)


def decoder_ablation(
    manifest: pd.DataFrame,
    candidate_authors: List[str],
    timestamp_granularity: str,
    min_margin: float,
    min_confidence: float,
) -> pd.DataFrame:
    timestamps = manifest["timestamp"].drop_duplicates().tolist()
    modes = {
        "full": (0.85, 0.15),
        "action_probability_only": (1.0, 0.0),
        "behavior_feature_only": (0.0, 1.0),
    }
    rows = []
    for mode, (action_weight, feature_weight) in modes.items():
        decoder = MultiStatisticVotingDecoder(
            candidate_authors,
            timestamps,
            timestamp_granularity=timestamp_granularity,
            min_margin=min_margin,
            min_confidence=min_confidence,
            action_weight=action_weight,
            feature_weight=feature_weight,
        )
        for item in manifest.itertuples():
            decoded = decoder.decode(item.log_path)
            rows.append(
                {
                    "mode": mode,
                    "run_id": item.run_id,
                    "true_author": item.author_id,
                    "decoded_author": decoded.author_id,
                    "confidence": decoded.confidence,
                    "margin": decoded.margin,
                    "abstained": decoded.abstained,
                    "correct_author": decoded.author_id == item.author_id,
                }
            )
    return pd.DataFrame(rows)


def save_bar(ax, labels, values, title: str, ylabel: str, color: str) -> None:
    ax.bar(labels, values, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)


def mean_ci(series: pd.Series) -> tuple[float, float, float]:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    if len(values) == 1:
        return mean, 0.0, 0.0
    se = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    ci95 = 1.96 * se
    return mean, se, ci95


def write_statistical_summary(
    manifest: pd.DataFrame,
    clean_results: pd.DataFrame,
    attack_results: pd.DataFrame,
    out_dir: Path,
) -> None:
    rows = []
    merged_clean = clean_results.merge(manifest[["run_id", "watermark_lambda", "task_success"]], on="run_id", how="left")
    for watermark_lambda, part in merged_clean.groupby("watermark_lambda", dropna=False):
        for metric, column in [
            ("task_success", "task_success"),
            ("clean_author_accuracy", "correct_author"),
            ("clean_timestamp_bucket_accuracy", "correct_timestamp_bucket"),
            ("clean_confidence", "confidence"),
            ("clean_margin", "margin"),
            ("clean_abstain_rate", "abstained"),
        ]:
            mean, se, ci95 = mean_ci(part[column])
            rows.append(
                {
                    "condition": f"lambda={watermark_lambda}",
                    "metric": metric,
                    "mean": mean,
                    "standard_error": se,
                    "ci95": ci95,
                    "n": int(part[column].notna().sum()),
                }
            )
    for keys, part in attack_results.groupby(["attack", "severity"], dropna=False):
        attack, severity = keys
        for metric, column in [
            ("attack_author_accuracy", "correct_author"),
            ("attack_accuracy_after_abstain", "correct_author_when_not_abstained"),
            ("attack_confidence", "confidence"),
            ("attack_margin", "margin"),
            ("attack_abstain_rate", "abstained"),
        ]:
            mean, se, ci95 = mean_ci(part[column])
            rows.append(
                {
                    "condition": f"{attack}:{severity}",
                    "metric": metric,
                    "mean": mean,
                    "standard_error": se,
                    "ci95": ci95,
                    "n": int(part[column].notna().sum()),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "statistical_summary.csv", index=False)


def plot_metrics(
    manifest: pd.DataFrame,
    clean_results: pd.DataFrame,
    attack_results: pd.DataFrame,
    feature_df: pd.DataFrame,
    step_df: pd.DataFrame,
    vote_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "clean_author_accuracy": float(clean_results["correct_author"].mean()),
        "clean_timestamp_accuracy": float(clean_results["correct_timestamp"].mean()),
        "clean_timestamp_bucket_accuracy": float(clean_results["correct_timestamp_bucket"].mean()),
        "clean_mean_confidence": float(clean_results["confidence"].mean()),
        "clean_mean_margin": float(clean_results["margin"].mean()),
        "clean_abstain_rate": float(clean_results["abstained"].mean()),
        "clean_author_accuracy_after_abstain": float(clean_results["correct_author_when_not_abstained"].mean()),
        "task_success_rate": float(manifest["task_success"].mean()) if "task_success" in manifest else None,
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

    if attack_results.empty:
        robust = pd.DataFrame(
            columns=["attack", "severity", "accuracy", "accuracy_after_abstain", "abstain_rate", "confidence", "margin"]
        )
    else:
        robust = attack_results.groupby(["attack", "severity"], as_index=False).agg(
            accuracy=("correct_author", "mean"),
            accuracy_after_abstain=("correct_author_when_not_abstained", "mean"),
            abstain_rate=("abstained", "mean"),
            confidence=("confidence", "mean"),
            margin=("margin", "mean"),
        )
    robust_groups = [] if robust.empty else robust.groupby("attack")
    for attack, part in robust_groups:
        axes[1, 0].plot(part["severity"], part["accuracy"], marker="o", label=attack)
    axes[1, 0].set_title("Robustness accuracy")
    axes[1, 0].set_xlabel("severity")
    axes[1, 0].set_ylabel("accuracy")
    axes[1, 0].set_ylim(0, 1.05)
    if not robust.empty:
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
    robust_groups = [] if robust.empty else robust.groupby("attack")
    for attack, part in robust_groups:
        ax.plot(part["severity"], part["confidence"], marker="o", label=attack)
    ax.set_title("Robustness mean confidence")
    ax.set_xlabel("severity")
    ax.set_ylabel("confidence")
    ax.set_ylim(0, 1.05)
    if not robust.empty:
        ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "robustness_confidence_curve.png", dpi=180)
    plt.close(fig)

    thresholds = np.linspace(0.0, 0.30, 16)
    cal_rows = []
    for threshold in thresholds:
        retained = clean_results[clean_results["margin"] >= threshold]
        cal_rows.append(
            {
                "margin_threshold": float(threshold),
                "coverage": float(len(retained) / max(1, len(clean_results))),
                "accuracy": float(retained["correct_author"].mean()) if len(retained) else np.nan,
                "mean_confidence": float(retained["confidence"].mean()) if len(retained) else np.nan,
            }
        )
    calibration = pd.DataFrame(cal_rows)
    calibration.to_csv(out_dir / "calibration_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(calibration["coverage"], calibration["accuracy"], marker="o", label="accuracy")
    ax.plot(calibration["coverage"], calibration["mean_confidence"], marker="o", label="mean confidence")
    ax.set_title("Coverage-accuracy calibration")
    ax.set_xlabel("coverage after abstention")
    ax.set_ylabel("score")
    ax.set_xlim(1.05, -0.05)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "coverage_accuracy_calibration.png", dpi=180)
    plt.close(fig)

    ablation_summary = ablation_df.groupby("mode", as_index=False).agg(
        accuracy=("correct_author", "mean"),
        confidence=("confidence", "mean"),
        margin=("margin", "mean"),
        abstain_rate=("abstained", "mean"),
    )
    ablation_summary.to_csv(out_dir / "decoder_ablation_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(ablation_summary))
    width = 0.25
    ax.bar(x - width, ablation_summary["accuracy"], width=width, label="accuracy", color="#2f6f9f")
    ax.bar(x, ablation_summary["confidence"], width=width, label="confidence", color="#35a77c")
    ax.bar(x + width, ablation_summary["abstain_rate"], width=width, label="abstain rate", color="#c47a34")
    ax.set_xticks(x, ablation_summary["mode"], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Decoder component ablation")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "decoder_ablation.png", dpi=180)
    plt.close(fig)

    if "watermark_lambda" in manifest and manifest["watermark_lambda"].notna().any():
        lambda_manifest = manifest.groupby("watermark_lambda", as_index=False).agg(task_success=("task_success", "mean"))
        lambda_clean = clean_results.merge(manifest[["run_id", "watermark_lambda"]], on="run_id", how="left")
        lambda_clean = lambda_clean.groupby("watermark_lambda", as_index=False).agg(
            clean_accuracy=("correct_author", "mean"),
            clean_accuracy_after_abstain=("correct_author_when_not_abstained", "mean"),
            abstain_rate=("abstained", "mean"),
            clean_confidence=("confidence", "mean"),
            clean_margin=("margin", "mean"),
        )
        if attack_results.empty:
            tradeoff = lambda_manifest.merge(lambda_clean, on="watermark_lambda", how="outer")
        else:
            lambda_attack = attack_results.groupby("watermark_lambda", as_index=False).agg(
                attack_accuracy=("correct_author", "mean"),
                attack_accuracy_after_abstain=("correct_author_when_not_abstained", "mean"),
                attack_confidence=("confidence", "mean"),
                attack_abstain_rate=("abstained", "mean"),
            )
            tradeoff = lambda_manifest.merge(lambda_clean, on="watermark_lambda", how="outer").merge(
                lambda_attack, on="watermark_lambda", how="outer"
            )
        tradeoff.to_csv(out_dir / "lambda_tradeoff.csv", index=False)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(tradeoff["watermark_lambda"], tradeoff["task_success"], marker="o", label="task success")
        ax.plot(tradeoff["watermark_lambda"], tradeoff["clean_accuracy"], marker="o", label="clean decode accuracy")
        ax.plot(tradeoff["watermark_lambda"], tradeoff["clean_confidence"], marker="o", label="clean confidence")
        if "attack_accuracy" in tradeoff:
            ax.plot(tradeoff["watermark_lambda"], tradeoff["attack_accuracy"], marker="o", label="attack decode accuracy")
        ax.plot(tradeoff["watermark_lambda"], tradeoff["abstain_rate"], marker="o", label="clean abstain rate")
        ax.set_title("Watermark strength trade-off")
        ax.set_xlabel("lambda")
        ax.set_ylabel("score")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / "lambda_tradeoff_curve.png", dpi=180)
        plt.close(fig)

    write_statistical_summary(manifest, clean_results, attack_results, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-task watermark evaluation and generate presentation plots.")
    parser.add_argument("--config", default=str(Path(__file__).parents[1] / "configs/deepseek.yaml"))
    parser.add_argument("--authors", nargs="+", default=["alice-lab", "bob-lab", "carol-lab"])
    parser.add_argument("--watermarked-author", default="alice-lab")
    parser.add_argument("--tasks", type=int, default=6, help="Number of built-in tasks to run. Use 0 for all tasks.")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--lambda-values", nargs="+", type=float, default=None)
    parser.add_argument("--timestamp-granularity", choices=["exact", "minute", "hour", "day"], default=None)
    parser.add_argument("--min-margin", type=float, default=0.08)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--logs", nargs="*", help="Use existing JSONL logs instead of running the agent.")
    parser.add_argument("--max-logs", type=int, default=None, help="Limit existing logs for quick offline re-analysis.")
    parser.add_argument("--skip-attacks", action="store_true", help="Only compute clean decoding and plots; skip attack generation.")
    parser.add_argument("--out", default="runtime/evaluation")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    load_dotenv()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    timestamp_granularity = args.timestamp_granularity or cfg.get("watermark_timestamp_granularity", "exact")

    if args.logs:
        manifest = existing_manifest(args.logs)
        if args.max_logs is not None:
            manifest = manifest.head(args.max_logs)
    else:
        api_key_env = cfg.get("llm_api_key_env", "OPENAI_API_KEY")
        if not os.getenv(api_key_env):
            raise RuntimeError(f"Set {api_key_env} before running evaluation.")
        task_limit = None if args.tasks == 0 else args.tasks
        lambda_values = args.lambda_values or [float(cfg["watermark_lambda"])]
        manifest = run_logs(cfg, [args.watermarked_author], select_tasks(task_limit), args.repeats, lambda_values)

    if manifest.empty:
        raise RuntimeError("No logs available for evaluation.")

    manifest.to_csv(out_dir / "run_manifest.csv", index=False)
    clean_results = decode_clean(
        manifest,
        args.authors,
        timestamp_granularity,
        args.min_margin,
        args.min_confidence,
        enabled=not args.skip_attacks,
    )
    attack_results = decode_attacks(
        manifest,
        args.authors,
        out_dir,
        timestamp_granularity,
        args.min_margin,
        args.min_confidence,
    )
    feature_df = BehaviorFeatureExtractor().dataframe(manifest["log_path"].tolist())
    step_df = extract_step_table(manifest)
    vote_df = votes_table(clean_results)
    aggregate_df = aggregate_runs(clean_results, manifest)
    ablation_df = decoder_ablation(
        manifest,
        args.authors,
        timestamp_granularity,
        args.min_margin,
        args.min_confidence,
    )

    clean_results.to_csv(out_dir / "clean_decoding_results.csv", index=False)
    attack_results.to_csv(out_dir / "attack_decoding_results.csv", index=False)
    feature_df.to_csv(out_dir / "behavior_features.csv", index=False)
    step_df.to_csv(out_dir / "step_actions.csv", index=False)
    vote_df.to_csv(out_dir / "vote_scores.csv", index=False)
    aggregate_df.to_csv(out_dir / "aggregate_decoding_results.csv", index=False)
    ablation_df.to_csv(out_dir / "decoder_ablation.csv", index=False)
    plot_metrics(manifest, clean_results, attack_results, feature_df, step_df, vote_df, ablation_df, out_dir)

    print(json.dumps(json.loads((out_dir / "summary.json").read_text(encoding="utf-8")), indent=2))
    print(f"plots: {out_dir / 'plots'}")


if __name__ == "__main__":
    main()

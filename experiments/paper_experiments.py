from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from agent_watermark.decoder.voting_decoder import MultiStatisticVotingDecoder
from agent_watermark.experiments.evaluate_watermark import existing_manifest, load_config, run_logs, select_tasks
from agent_watermark.feature_analysis.extractor import BehaviorFeatureExtractor
from agent_watermark.logging.jsonl_logger import JsonlExecutionLogger
from agent_watermark.watermark.signature import timestamp_bucket


def load_manifest(evaluation_dir: Path | None, logs: List[str] | None) -> pd.DataFrame:
    if evaluation_dir is not None:
        manifest_path = evaluation_dir / "run_manifest.csv"
        if manifest_path.exists():
            return pd.read_csv(manifest_path)
    if logs:
        return existing_manifest(logs)
    raise RuntimeError("Provide --evaluation-dir with run_manifest.csv or --logs.")


def decode_with_authors(
    manifest: pd.DataFrame,
    authors: List[str],
    timestamp_granularity: str,
    min_margin: float,
    min_confidence: float,
) -> pd.DataFrame:
    timestamps = manifest["timestamp"].drop_duplicates().tolist()
    decoder = MultiStatisticVotingDecoder(
        authors,
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
                "true_timestamp_bucket": true_bucket,
                "decoded_timestamp_bucket": decoded.timestamp_bucket,
                "correct_author": decoded.author_id == item.author_id,
                "correct_timestamp_bucket": decoded.timestamp_bucket == true_bucket,
                "confidence": decoded.confidence,
                "margin": decoded.margin,
                "abstained": decoded.abstained,
                "correct_author_when_not_abstained": (decoded.author_id == item.author_id) if not decoded.abstained else np.nan,
                "votes": decoded.votes,
            }
        )
    return pd.DataFrame(rows)


def clean_results_from_dir(
    evaluation_dir: Path | None,
    manifest: pd.DataFrame,
    authors: List[str],
    timestamp_granularity: str,
    min_margin: float,
    min_confidence: float,
) -> pd.DataFrame:
    if evaluation_dir is not None:
        path = evaluation_dir / "clean_decoding_results.csv"
        if path.exists():
            return pd.read_csv(path)
    return decode_with_authors(manifest, authors, timestamp_granularity, min_margin, min_confidence)


def feature_table(evaluation_dir: Path | None, manifest: pd.DataFrame) -> pd.DataFrame:
    if evaluation_dir is not None:
        path = evaluation_dir / "behavior_features.csv"
        if path.exists():
            return pd.read_csv(path)
    return BehaviorFeatureExtractor().dataframe(manifest["log_path"].tolist())


def expand_real_authors(seed_authors: List[str], target_size: int, prefix: str) -> List[str]:
    """Build a real author list for live scaling runs."""
    authors = list(dict.fromkeys(seed_authors))
    index = 1
    while len(authors) < target_size:
        candidate = f"{prefix}-{index:02d}"
        if candidate not in authors:
            authors.append(candidate)
        index += 1
    return authors[:target_size]


def baseline_author_recovery(manifest: pd.DataFrame, clean_results: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    merged = manifest[["run_id", "author_id"]].merge(feature_df, on="run_id", how="left")
    numeric = merged.select_dtypes(include=[np.number]).fillna(0.0)
    y = merged["author_id"].astype(str)
    rows = [
        {
            "method": "full_watermark_decoder",
            "accuracy": float(clean_results["correct_author"].mean()),
            "description": "action-probability delta plus behavior voting",
        },
        {
            "method": "random_guess",
            "accuracy": float(1.0 / max(1, y.nunique())),
            "description": "expected chance accuracy over candidate authors",
        },
        {
            "method": "majority_author",
            "accuracy": float(y.value_counts(normalize=True).max()),
            "description": "always predict the most frequent author",
        },
    ]
    if len(merged) >= 6 and y.nunique() > 1:
        length_only = merged[["average_trajectory_length"]].fillna(0.0)
        tool_columns = [
            c
            for c in [
                "tool_usage_frequency",
                "search_tool_ratio",
                "database_query_ratio",
                "unique_tool_ratio",
                "tool_entropy",
            ]
            if c in merged
        ]
        baselines = [
            ("trajectory_length_knn", length_only, "nearest-neighbor classifier over trajectory length only"),
            ("tool_frequency_knn", merged[tool_columns].fillna(0.0), "nearest-neighbor classifier over aggregate tool frequencies"),
            ("all_behavior_features_logreg", numeric, "logistic regression over all aggregate behavior features"),
        ]
        min_class = int(y.value_counts().min())
        n_splits = min(5, min_class)
        for name, x, description in baselines:
            if x.shape[1] == 0 or n_splits < 2:
                continue
            if name.endswith("logreg"):
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
            else:
                clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1))
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=7)
            pred = cross_val_predict(clf, x, y, cv=cv)
            rows.append({"method": name, "accuracy": float(accuracy_score(y, pred)), "description": description})
    return pd.DataFrame(rows)


def author_scaling(
    manifest: pd.DataFrame,
    real_authors: List[str],
    timestamp_granularity: str,
    min_margin: float,
    min_confidence: float,
    sizes: Iterable[int],
) -> pd.DataFrame:
    rows = []
    base = list(dict.fromkeys(real_authors))
    for size in sizes:
        candidates = base[:size]
        decoys = [f"decoy-lab-{i:02d}" for i in range(max(0, size - len(candidates)))]
        candidates = candidates + decoys
        results = decode_with_authors(manifest, candidates, timestamp_granularity, min_margin, min_confidence)
        rows.append(
            {
                "candidate_authors": len(candidates),
                "accuracy": float(results["correct_author"].mean()),
                "timestamp_bucket_accuracy": float(results["correct_timestamp_bucket"].mean()),
                "mean_margin": float(results["margin"].mean()),
                "abstain_rate": float(results["abstained"].mean()),
                "coverage": float(1.0 - results["abstained"].mean()),
            }
        )
    return pd.DataFrame(rows)


def real_author_scaling(
    manifest: pd.DataFrame,
    ordered_authors: List[str],
    timestamp_granularity: str,
    min_margin: float,
    min_confidence: float,
    sizes: Iterable[int],
) -> pd.DataFrame:
    """Evaluate true author scaling using logs generated by those same authors."""
    rows = []
    for size in sizes:
        candidate_authors = ordered_authors[:size]
        subset = manifest[manifest["author_id"].isin(candidate_authors)].copy()
        if subset.empty:
            continue
        results = decode_with_authors(subset, candidate_authors, timestamp_granularity, min_margin, min_confidence)
        rows.append(
            {
                "real_authors": len(candidate_authors),
                "num_runs": int(len(subset)),
                "num_steps": int(sum(len(JsonlExecutionLogger.read(path)) for path in subset["log_path"])),
                "accuracy": float(results["correct_author"].mean()),
                "timestamp_bucket_accuracy": float(results["correct_timestamp_bucket"].mean()),
                "mean_confidence": float(results["confidence"].mean()),
                "mean_margin": float(results["margin"].mean()),
                "abstain_rate": float(results["abstained"].mean()),
                "coverage": float(1.0 - results["abstained"].mean()),
            }
        )
    return pd.DataFrame(rows)


def timestamp_granularity_study(
    manifest: pd.DataFrame,
    authors: List[str],
    min_margin: float,
    min_confidence: float,
) -> pd.DataFrame:
    rows = []
    for granularity in ["exact", "minute", "hour", "day"]:
        results = decode_with_authors(manifest, authors, granularity, min_margin, min_confidence)
        rows.append(
            {
                "granularity": granularity,
                "author_accuracy": float(results["correct_author"].mean()),
                "timestamp_bucket_accuracy": float(results["correct_timestamp_bucket"].mean()),
                "mean_margin": float(results["margin"].mean()),
                "abstain_rate": float(results["abstained"].mean()),
            }
        )
    return pd.DataFrame(rows)


def stealth_distribution_metrics(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in manifest.itertuples():
        for step in JsonlExecutionLogger.read(item.log_path):
            if len(step.candidate_actions) < 2:
                continue
            raw = np.asarray([c.raw_probability for c in step.candidate_actions], dtype=float)
            watermarked = np.asarray([c.watermarked_probability for c in step.candidate_actions], dtype=float)
            raw = raw / raw.sum()
            watermarked = watermarked / watermarked.sum()
            midpoint = 0.5 * (raw + watermarked)
            kl_raw_wm = float(np.sum(raw * np.log(np.clip(raw, 1e-12, 1.0) / np.clip(watermarked, 1e-12, 1.0))))
            js = 0.5 * float(np.sum(raw * np.log(np.clip(raw, 1e-12, 1.0) / np.clip(midpoint, 1e-12, 1.0))))
            js += 0.5 * float(np.sum(watermarked * np.log(np.clip(watermarked, 1e-12, 1.0) / np.clip(midpoint, 1e-12, 1.0))))
            rows.append(
                {
                    "run_id": item.run_id,
                    "author_id": item.author_id,
                    "step_index": step.step_index,
                    "kl_raw_to_watermarked": kl_raw_wm,
                    "js_divergence": js,
                    "mean_abs_probability_shift": float(np.mean(np.abs(watermarked - raw))),
                    "max_abs_probability_shift": float(np.max(np.abs(watermarked - raw))),
                    "top_action_changed": bool(np.argmax(raw) != np.argmax(watermarked)),
                }
            )
    return pd.DataFrame(rows)


def counterfactual_detectability(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in manifest.itertuples():
        for step in JsonlExecutionLogger.read(item.log_path):
            if len(step.candidate_actions) < 2:
                continue
            for label, field in [(0, "raw_probability"), (1, "watermarked_probability")]:
                probs = np.asarray([getattr(c, field) for c in step.candidate_actions], dtype=float)
                probs = probs / probs.sum()
                sorted_probs = np.sort(probs)[::-1]
                rows.append(
                    {
                        "label": label,
                        "top_probability": float(sorted_probs[0]),
                        "second_probability": float(sorted_probs[1]),
                        "margin": float(sorted_probs[0] - sorted_probs[1]),
                        "entropy": float(-(probs * np.log2(np.clip(probs, 1e-12, 1.0))).sum()),
                        "candidate_count": len(probs),
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty or df["label"].nunique() < 2 or len(df) < 10:
        return pd.DataFrame([{"detector_auc": np.nan, "detector_accuracy": np.nan, "n_samples": len(df)}])
    x = df.drop(columns=["label"])
    y = df["label"]
    n_splits = min(5, int(y.value_counts().min()))
    if n_splits < 2:
        return pd.DataFrame([{"detector_auc": np.nan, "detector_accuracy": np.nan, "n_samples": len(df)}])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=7)
    prob = cross_val_predict(clf, x, y, cv=cv, method="predict_proba")[:, 1]
    pred = (prob >= 0.5).astype(int)
    return pd.DataFrame(
        [
            {
                "detector_auc": float(roc_auc_score(y, prob)),
                "detector_accuracy": float(accuracy_score(y, pred)),
                "n_samples": int(len(df)),
                "note": "counterfactual raw-vs-watermarked distribution detector; lower is stealthier",
            }
        ]
    )


def plot_paper_results(
    baseline_df: pd.DataFrame,
    scaling_df: pd.DataFrame,
    real_scaling_df: pd.DataFrame,
    granularity_df: pd.DataFrame,
    stealth_df: pd.DataFrame,
    detectability_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ordered = baseline_df.sort_values("accuracy", ascending=True)
    ax.barh(ordered["method"], ordered["accuracy"], color="#3f7fb5")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("author recovery accuracy")
    ax.set_title("Decoder vs. non-watermark baselines")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "paper_baseline_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(scaling_df["candidate_authors"], scaling_df["accuracy"], marker="o", label="author accuracy")
    ax.plot(scaling_df["candidate_authors"], scaling_df["coverage"], marker="o", label="coverage")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("candidate author set size")
    ax.set_title("Candidate-set pressure test")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "paper_candidate_author_scaling.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if real_scaling_df.empty:
        ax.text(0.5, 0.5, "run with --run-real-author-scaling", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.plot(real_scaling_df["real_authors"], real_scaling_df["accuracy"], marker="o", label="author accuracy")
        ax.plot(real_scaling_df["real_authors"], real_scaling_df["coverage"], marker="o", label="coverage")
        ax.plot(real_scaling_df["real_authors"], real_scaling_df["timestamp_bucket_accuracy"], marker="o", label="timestamp bucket")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("real watermarked authors")
        ax.set_ylabel("score")
        ax.grid(alpha=0.25)
        ax.legend()
    ax.set_title("Real author scalability")
    fig.tight_layout()
    fig.savefig(plot_dir / "paper_real_author_scaling.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(granularity_df["granularity"], granularity_df["timestamp_bucket_accuracy"], marker="o", label="timestamp bucket")
    ax.plot(granularity_df["granularity"], granularity_df["author_accuracy"], marker="o", label="author")
    ax.set_ylim(0, 1.05)
    ax.set_title("Timestamp-capacity trade-off")
    ax.set_xlabel("timestamp granularity")
    ax.set_ylabel("accuracy")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "paper_timestamp_granularity.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    if stealth_df.empty:
        ax.text(0.5, 0.5, "no action-distribution rows", ha="center", va="center")
        ax.set_axis_off()
    else:
        stealth_summary = stealth_df[
            ["js_divergence", "mean_abs_probability_shift", "max_abs_probability_shift", "top_action_changed"]
        ].copy()
        stealth_summary["top_action_changed"] = stealth_summary["top_action_changed"].astype(float)
        stealth_summary.boxplot(ax=ax, rot=20)
        ax.grid(axis="y", alpha=0.25)
    ax.set_title("Action-distribution perturbation")
    fig.tight_layout()
    fig.savefig(plot_dir / "paper_stealth_distribution_shift.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    values = detectability_df.iloc[0]
    ax.bar(["AUC", "Accuracy"], [values["detector_auc"], values["detector_accuracy"]], color=["#8464b7", "#d0833f"])
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance")
    ax.set_ylim(0, 1.05)
    ax.set_title("Counterfactual detectability")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "paper_detectability.png", dpi=180)
    plt.close(fig)


def write_summary(
    baseline_df: pd.DataFrame,
    scaling_df: pd.DataFrame,
    real_scaling_df: pd.DataFrame,
    granularity_df: pd.DataFrame,
    stealth_df: pd.DataFrame,
    detectability_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    full = baseline_df[baseline_df["method"] == "full_watermark_decoder"]
    chance = baseline_df[baseline_df["method"] == "random_guess"]
    summary = {
        "full_decoder_accuracy": float(full["accuracy"].iloc[0]) if not full.empty else None,
        "random_guess_accuracy": float(chance["accuracy"].iloc[0]) if not chance.empty else None,
        "max_candidate_authors_evaluated": int(scaling_df["candidate_authors"].max()) if not scaling_df.empty else 0,
        "accuracy_at_max_candidate_authors": float(scaling_df.sort_values("candidate_authors").iloc[-1]["accuracy"]) if not scaling_df.empty else None,
        "max_real_authors_evaluated": int(real_scaling_df["real_authors"].max()) if not real_scaling_df.empty else 0,
        "accuracy_at_max_real_authors": float(real_scaling_df.sort_values("real_authors").iloc[-1]["accuracy"]) if not real_scaling_df.empty else None,
        "best_timestamp_granularity": str(granularity_df.sort_values("timestamp_bucket_accuracy", ascending=False).iloc[0]["granularity"])
        if not granularity_df.empty
        else None,
        "mean_js_divergence": float(stealth_df["js_divergence"].mean()) if not stealth_df.empty else None,
        "mean_abs_probability_shift": float(stealth_df["mean_abs_probability_shift"].mean()) if not stealth_df.empty else None,
        "top_action_flip_rate": float(stealth_df["top_action_changed"].mean()) if not stealth_df.empty else None,
        "counterfactual_detector_auc": float(detectability_df["detector_auc"].iloc[0]) if not detectability_df.empty else None,
    }
    (out_dir / "paper_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AAAI-style offline experiments from execution logs.")
    parser.add_argument("--config", default=str(Path(__file__).parents[1] / "configs/deepseek.yaml"))
    parser.add_argument("--evaluation-dir", type=Path, default=None, help="Directory produced by evaluate_watermark.py.")
    parser.add_argument("--logs", nargs="*", default=None, help="JSONL logs to analyze if no evaluation directory is given.")
    parser.add_argument("--authors", nargs="+", required=True)
    parser.add_argument("--timestamp-granularity", choices=["exact", "minute", "hour", "day"], default="hour")
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--candidate-author-sizes", nargs="+", type=int, default=[3, 5, 10, 20])
    parser.add_argument("--run-real-author-scaling", action="store_true", help="Generate fresh logs for true author-scale evaluation.")
    parser.add_argument("--real-author-sizes", nargs="+", type=int, default=None, help="Real author counts to run; defaults to candidate sizes.")
    parser.add_argument("--real-author-prefix", default="paper-lab")
    parser.add_argument("--tasks", type=int, default=8, help="Task count for live real-author scaling. Use 0 for all tasks.")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--lambda-values", nargs="+", type=float, default=None)
    parser.add_argument("--out", type=Path, default=Path("runtime/paper_experiments"))
    args = parser.parse_args()

    load_dotenv()
    args.out.mkdir(parents=True, exist_ok=True)
    real_scaling_df = pd.DataFrame()
    live_manifest = None
    analysis_authors = args.authors
    real_sizes = args.real_author_sizes or args.candidate_author_sizes

    if args.run_real_author_scaling:
        cfg = load_config(args.config)
        api_key_env = cfg.get("llm_api_key_env", "OPENAI_API_KEY")
        if not os.getenv(api_key_env):
            raise RuntimeError(f"Set {api_key_env} before running real author scaling.")
        max_real_authors = max(real_sizes)
        real_authors = expand_real_authors(args.authors, max_real_authors, args.real_author_prefix)
        analysis_authors = real_authors
        task_limit = None if args.tasks == 0 else args.tasks
        lambda_values = args.lambda_values or [float(cfg["watermark_lambda"])]
        live_manifest = run_logs(cfg, real_authors, select_tasks(task_limit), args.repeats, lambda_values)
        live_manifest.to_csv(args.out / "real_author_scaling_manifest.csv", index=False)
        real_scaling_df = real_author_scaling(
            live_manifest,
            real_authors,
            args.timestamp_granularity,
            args.min_margin,
            args.min_confidence,
            real_sizes,
        )

    manifest = live_manifest if live_manifest is not None else load_manifest(args.evaluation_dir, args.logs)
    clean_results = clean_results_from_dir(
        None if live_manifest is not None else args.evaluation_dir,
        manifest,
        analysis_authors,
        args.timestamp_granularity,
        args.min_margin,
        args.min_confidence,
    )
    features = feature_table(None if live_manifest is not None else args.evaluation_dir, manifest)
    if "run_id" not in features:
        features["run_id"] = manifest["run_id"].values[: len(features)]

    baseline_df = baseline_author_recovery(manifest, clean_results, features)
    candidate_manifest = manifest
    if live_manifest is not None:
        candidate_manifest = manifest[manifest["author_id"].isin(args.authors)].copy()
    scaling_df = author_scaling(
        candidate_manifest,
        args.authors,
        args.timestamp_granularity,
        args.min_margin,
        args.min_confidence,
        args.candidate_author_sizes,
    )
    granularity_df = timestamp_granularity_study(manifest, analysis_authors, args.min_margin, args.min_confidence)
    stealth_df = stealth_distribution_metrics(manifest)
    detectability_df = counterfactual_detectability(manifest)

    baseline_df.to_csv(args.out / "paper_baselines.csv", index=False)
    scaling_df.to_csv(args.out / "paper_candidate_author_scaling.csv", index=False)
    real_scaling_df.to_csv(args.out / "paper_real_author_scaling.csv", index=False)
    granularity_df.to_csv(args.out / "paper_timestamp_granularity.csv", index=False)
    stealth_df.to_csv(args.out / "paper_stealth_distribution.csv", index=False)
    detectability_df.to_csv(args.out / "paper_detectability.csv", index=False)
    plot_paper_results(baseline_df, scaling_df, real_scaling_df, granularity_df, stealth_df, detectability_df, args.out)
    write_summary(baseline_df, scaling_df, real_scaling_df, granularity_df, stealth_df, detectability_df, args.out)

    print(json.dumps(json.loads((args.out / "paper_summary.json").read_text(encoding="utf-8")), indent=2))
    print(f"plots: {args.out / 'plots'}")


if __name__ == "__main__":
    main()

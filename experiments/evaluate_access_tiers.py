"""Evaluate attribution under progressively restricted execution-log access.

This experiment deliberately changes the scoring rule at each tier.  It does
not claim that a raw-to-watermarked probability delta is available after the
corresponding fields have been redacted.

Example:
    python -m agent_watermark.experiments.evaluate_access_tiers \
      --evaluation-dir full_robustness_report --log-root logs \
      --authors alice-lab bob-lab carol-lab --timestamp-granularity hour \
      --out runtime/access_tiers

For an open-set result, provide independently generated unknown-author and
unwatermarked trajectories through --unknown-logs and --unwatermarked-logs.
They are treated as negatives and are never used as candidate identities.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agent_watermark.decoder.voting_decoder import MultiStatisticVotingDecoder
from agent_watermark.experiments.paper_experiments import load_manifest
from agent_watermark.logging.jsonl_logger import JsonlExecutionLogger
from agent_watermark.watermark.signature import timestamp_bucket


TIERS = ["actions_only", "actions_candidates", "watermarked_probabilities", "full_trusted_logs"]


def resolve_log_paths(manifest: pd.DataFrame, log_root: Path | None) -> pd.DataFrame:
    """Map server-side manifest paths to a local flat JSONL archive by basename."""
    resolved = manifest.copy()
    paths: List[str] = []
    missing: List[str] = []
    for value in resolved["log_path"]:
        source = Path(str(value))
        if source.exists():
            paths.append(str(source))
            continue
        local = log_root / source.name if log_root is not None else source
        if local.exists():
            paths.append(str(local))
        else:
            missing.append(str(source))
            paths.append(str(local))
    if missing:
        preview = ", ".join(missing[:3])
        raise FileNotFoundError(f"Could not resolve {len(missing)} JSONL logs. Examples: {preview}")
    resolved["log_path"] = paths
    return resolved


def manifests_from_paths(paths: Iterable[str], label: str) -> pd.DataFrame:
    """Load negative logs without treating their embedded identity as a candidate."""
    rows = []
    for value in paths:
        steps = JsonlExecutionLogger.read(value)
        if not steps:
            continue
        first = steps[0]
        rows.append({"run_id": first.run_id, "log_path": str(value), "negative_type": label})
    return pd.DataFrame(rows)


def grouped_split(frame: pd.DataFrame, fraction: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministically split trajectories; authors remain balanced when available."""
    if frame.empty:
        return frame.copy(), frame.copy()
    rng = np.random.default_rng(random_state)
    train_indices: List[int] = []
    group_column = "author_id" if "author_id" in frame.columns else None
    groups = frame.groupby(group_column, dropna=False) if group_column else [(None, frame)]
    for _, group in groups:
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        count = max(1, int(round(len(indices) * fraction))) if len(indices) > 1 else 1
        train_indices.extend(indices[:count].tolist())
    train = frame.loc[sorted(set(train_indices))].copy()
    test = frame.drop(index=train.index).copy()
    return train, test


def decode_rows(
    frame: pd.DataFrame,
    decoder: MultiStatisticVotingDecoder,
    timestamp_granularity: str,
    is_positive: bool,
) -> pd.DataFrame:
    rows = []
    for item in frame.itertuples():
        decoded = decoder.decode(item.log_path)
        row = {
            "run_id": item.run_id,
            "log_path": item.log_path,
            "is_positive": is_positive,
            "decoded_author": decoded.author_id,
            "decoded_timestamp_bucket": decoded.timestamp_bucket,
            "confidence": decoded.confidence,
            "margin": decoded.margin,
        }
        if is_positive:
            true_bucket = timestamp_bucket(item.timestamp, timestamp_granularity)
            row.update(
                {
                    "true_author": item.author_id,
                    "true_timestamp_bucket": true_bucket,
                    "correct_author": decoded.author_id == item.author_id,
                    "correct_timestamp_bucket": decoded.timestamp_bucket == true_bucket,
                }
            )
        else:
            row["negative_type"] = getattr(item, "negative_type", "unknown")
            row["correct_author"] = False
            row["correct_timestamp_bucket"] = False
        rows.append(row)
    return pd.DataFrame(rows)


def choose_threshold(
    calibration: pd.DataFrame,
    target_selective_accuracy: float,
    max_negative_fpr: float,
) -> tuple[float, dict]:
    """Pick the lowest margin threshold satisfying held-out safety targets."""
    positives = calibration[calibration["is_positive"]]
    negatives = calibration[~calibration["is_positive"]]
    thresholds = sorted({0.0, *calibration["margin"].dropna().astype(float).tolist()})
    selected: tuple[float, dict] | None = None
    for threshold in thresholds:
        kept_positive = positives[positives["margin"] >= threshold]
        selective_accuracy = float(kept_positive["correct_author"].mean()) if not kept_positive.empty else np.nan
        coverage = float(len(kept_positive) / len(positives)) if len(positives) else 0.0
        negative_fpr = float((negatives["margin"] >= threshold).mean()) if len(negatives) else np.nan
        meets_accuracy = not np.isnan(selective_accuracy) and selective_accuracy >= target_selective_accuracy
        meets_fpr = np.isnan(negative_fpr) or negative_fpr <= max_negative_fpr
        if meets_accuracy and meets_fpr:
            summary = {
                "calibration_selective_accuracy": selective_accuracy,
                "calibration_coverage": coverage,
                "calibration_negative_fpr": negative_fpr,
            }
            selected = (float(threshold), summary)
            break
    if selected is not None:
        return selected
    return float("inf"), {
        "calibration_selective_accuracy": np.nan,
        "calibration_coverage": 0.0,
        "calibration_negative_fpr": np.nan,
    }


def summarize_test(test: pd.DataFrame, threshold: float) -> dict:
    tested = test.copy()
    tested["abstained"] = tested["margin"] < threshold
    positive = tested[tested["is_positive"]]
    negative = tested[~tested["is_positive"]]
    accepted_positive = positive[~positive["abstained"]]
    return {
        "closed_set_author_accuracy": float(positive["correct_author"].mean()) if len(positive) else np.nan,
        "timestamp_bucket_accuracy": float(positive["correct_timestamp_bucket"].mean()) if len(positive) else np.nan,
        "positive_coverage": float((~positive["abstained"]).mean()) if len(positive) else np.nan,
        "selective_author_accuracy": float(accepted_positive["correct_author"].mean()) if len(accepted_positive) else np.nan,
        "negative_false_attribution_rate": float((~negative["abstained"]).mean()) if len(negative) else np.nan,
        "negative_rejection_rate": float(negative["abstained"].mean()) if len(negative) else np.nan,
        "num_positive_test": int(len(positive)),
        "num_negative_test": int(len(negative)),
    }, tested


def plot_results(summary: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    fields = ["closed_set_author_accuracy", "positive_coverage", "selective_author_accuracy", "negative_rejection_rate"]
    available = [field for field in fields if summary[field].notna().any()]
    x = np.arange(len(summary))
    width = 0.8 / max(1, len(available))
    for index, field in enumerate(available):
        ax.bar(x - 0.4 + width / 2 + index * width, summary[field], width, label=field.replace("_", " "))
    ax.set_xticks(x, summary["access_tier"].str.replace("_", " "))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate")
    ax.set_title("Attribution under execution-log access tiers")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "access_tier_results.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate restricted-log decoding and open-set rejection.")
    parser.add_argument("--evaluation-dir", type=Path, default=None)
    parser.add_argument("--logs", nargs="*", default=None)
    parser.add_argument("--log-root", type=Path, default=None)
    parser.add_argument("--authors", nargs="+", required=True)
    parser.add_argument("--timestamp-granularity", choices=["exact", "minute", "hour", "day"], default="hour")
    parser.add_argument("--unknown-logs", nargs="*", default=[])
    parser.add_argument("--unwatermarked-logs", nargs="*", default=[])
    parser.add_argument("--calibration-fraction", type=float, default=0.25)
    parser.add_argument("--target-selective-accuracy", type=float, default=0.95)
    parser.add_argument("--max-negative-fpr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if not 0.0 < args.calibration_fraction < 1.0:
        raise ValueError("--calibration-fraction must be in (0, 1).")
    positive = resolve_log_paths(load_manifest(args.evaluation_dir, args.logs), args.log_root)
    positive = positive[positive["author_id"].isin(args.authors)].copy()
    if positive.empty:
        raise RuntimeError("No positive trajectories remain after filtering to --authors.")
    positive_calibration, positive_test = grouped_split(positive, args.calibration_fraction, args.seed)
    negatives = pd.concat(
        [
            manifests_from_paths(args.unknown_logs, "unknown_watermarked"),
            manifests_from_paths(args.unwatermarked_logs, "unwatermarked"),
        ],
        ignore_index=True,
    )
    negative_calibration, negative_test = grouped_split(negatives, args.calibration_fraction, args.seed + 1)
    candidate_timestamps = positive["timestamp"].drop_duplicates().tolist()

    args.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    all_rows = []
    for tier in TIERS:
        decoder = MultiStatisticVotingDecoder(
            args.authors,
            candidate_timestamps,
            timestamp_granularity=args.timestamp_granularity,
            min_margin=0.0,
            min_confidence=0.0,
            access_tier=tier,
        )
        calibration = pd.concat(
            [
                decode_rows(positive_calibration, decoder, args.timestamp_granularity, True),
                decode_rows(negative_calibration, decoder, args.timestamp_granularity, False),
            ],
            ignore_index=True,
        )
        threshold, calibration_stats = choose_threshold(
            calibration, args.target_selective_accuracy, args.max_negative_fpr
        )
        test = pd.concat(
            [
                decode_rows(positive_test, decoder, args.timestamp_granularity, True),
                decode_rows(negative_test, decoder, args.timestamp_granularity, False),
            ],
            ignore_index=True,
        )
        summary, tested = summarize_test(test, threshold)
        summary.update({"access_tier": tier, "calibrated_min_margin": threshold, **calibration_stats})
        summaries.append(summary)
        tested["access_tier"] = tier
        tested["calibrated_min_margin"] = threshold
        all_rows.append(tested)

    summary_frame = pd.DataFrame(summaries)
    run_frame = pd.concat(all_rows, ignore_index=True)
    summary_frame.to_csv(args.out / "access_tier_summary.csv", index=False)
    run_frame.to_csv(args.out / "access_tier_run_results.csv", index=False)
    plot_results(summary_frame, args.out / "plots")
    print(summary_frame.to_json(orient="records", indent=2))
    print(f"artifacts: {args.out}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


def plot_decoding_accuracy(results_csv: str, out_dir: str) -> None:
    """Plot decoding accuracy over experiment groups."""
    df = pd.read_csv(results_csv)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig = px.line(df, x="condition", y="accuracy", color="author_id", markers=True, title="Watermark decoding accuracy")
    fig.write_html(out / "decoding_accuracy.html")


def plot_robustness(results_json: str, out_dir: str) -> None:
    """Plot confidence under crop/rewrite/fine-tune style attacks."""
    df = pd.read_json(results_json)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig = px.line(df, x="ratio", y="confidence", color="attack", markers=True, title="Robustness curve")
    fig.write_html(out / "robustness_curve.html")


def plot_trajectory_stats(features_csv: str, out_dir: str) -> None:
    """Plot feature distributions from extracted execution logs."""
    df = pd.read_csv(features_csv)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    numeric = df.select_dtypes("number")
    ax = numeric.plot(kind="box", figsize=(10, 5), rot=45)
    ax.set_title("Trajectory behavior statistics")
    plt.tight_layout()
    plt.savefig(out / "trajectory_statistics.png", dpi=180)
    plt.close()

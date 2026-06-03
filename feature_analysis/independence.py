from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.metrics import mutual_info_score


def entropy(values: np.ndarray, bins: int = 10) -> float:
    """Discrete entropy after quantile-style binning."""
    codes = pd.qcut(pd.Series(values).rank(method="first"), q=min(bins, len(values)), duplicates="drop").cat.codes
    _, counts = np.unique(codes, return_counts=True)
    probs = counts / counts.sum()
    return float(-(probs * np.log2(np.clip(probs, 1e-12, 1.0))).sum())


def discretize(values: np.ndarray, bins: int = 10) -> np.ndarray:
    if len(np.unique(values)) <= 1:
        return np.zeros_like(values, dtype=int)
    return pd.qcut(pd.Series(values).rank(method="first"), q=min(bins, len(values)), duplicates="drop").cat.codes.to_numpy()


@dataclass
class IndependenceResult:
    nmi_matrix: pd.DataFrame
    independent_behaviors: List[str]
    stability_scores: Dict[str, float]


class BehaviorIndependenceAnalyzer:
    """Compute MI/NMI, cluster correlated behavior features, and retain stable representatives."""

    def __init__(self, correlation_threshold: float = 0.65):
        self.correlation_threshold = correlation_threshold

    def nmi(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric = df.select_dtypes(include=[np.number])
        names = list(numeric.columns)
        matrix = np.eye(len(names))
        for i, left in enumerate(names):
            for j, right in enumerate(names):
                if i >= j:
                    continue
                x = discretize(numeric[left].to_numpy())
                y = discretize(numeric[right].to_numpy())
                mi = mutual_info_score(x, y)
                h = max(entropy(numeric[left].to_numpy()), entropy(numeric[right].to_numpy()), 1e-9)
                matrix[i, j] = matrix[j, i] = float(mi / h)
        return pd.DataFrame(matrix, index=names, columns=names)

    def stability(self, df: pd.DataFrame) -> Dict[str, float]:
        numeric = df.select_dtypes(include=[np.number])
        scores: Dict[str, float] = {}
        for col in numeric.columns:
            mean = abs(float(numeric[col].mean()))
            std = float(numeric[col].std(ddof=0))
            scores[col] = 1.0 / (1.0 + std / (mean + 1e-6))
        return scores

    def analyze(self, df: pd.DataFrame, output_dir: str | Path | None = None) -> IndependenceResult:
        nmi_matrix = self.nmi(df)
        stability_scores = self.stability(df)
        distance = 1.0 - nmi_matrix.to_numpy()
        condensed = distance[np.triu_indices_from(distance, k=1)]
        if len(condensed) == 0:
            independent = list(nmi_matrix.columns)
            z = None
        else:
            z = linkage(condensed, method="average")
            labels = fcluster(z, t=1.0 - self.correlation_threshold, criterion="distance")
            independent = []
            for label in sorted(set(labels)):
                members = [name for name, lab in zip(nmi_matrix.columns, labels) if lab == label]
                independent.append(max(members, key=lambda m: stability_scores.get(m, 0.0)))
        if output_dir:
            self.plot(nmi_matrix, z, output_dir)
        return IndependenceResult(nmi_matrix, independent, stability_scores)

    def plot(self, nmi_matrix: pd.DataFrame, linkage_matrix, output_dir: str | Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(nmi_matrix.to_numpy(), cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(nmi_matrix.columns)), nmi_matrix.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(nmi_matrix.index)), nmi_matrix.index)
        fig.colorbar(im, ax=ax, label="NMI")
        fig.tight_layout()
        fig.savefig(out / "behavior_nmi_heatmap.png", dpi=180)
        plt.close(fig)
        if linkage_matrix is not None:
            fig, ax = plt.subplots(figsize=(8, 4))
            dendrogram(linkage_matrix, labels=list(nmi_matrix.columns), ax=ax)
            fig.tight_layout()
            fig.savefig(out / "nmi_clustering.png", dpi=180)
            plt.close(fig)

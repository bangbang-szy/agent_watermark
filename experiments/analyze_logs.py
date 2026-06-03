from __future__ import annotations

import argparse
from pathlib import Path

from agent_watermark.feature_analysis.extractor import BehaviorFeatureExtractor
from agent_watermark.feature_analysis.independence import BehaviorIndependenceAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", required=True)
    parser.add_argument("--out", default="runtime/analysis")
    args = parser.parse_args()
    extractor = BehaviorFeatureExtractor()
    df = extractor.dataframe(args.logs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "behavior_features.csv", index=False)
    result = BehaviorIndependenceAnalyzer().analyze(df.drop(columns=["log_path"], errors="ignore"), out)
    result.nmi_matrix.to_csv(out / "nmi_matrix.csv")
    (out / "independent_behaviors.txt").write_text("\n".join(result.independent_behaviors), encoding="utf-8")
    print({"independent_behaviors": result.independent_behaviors, "stability_scores": result.stability_scores})


if __name__ == "__main__":
    main()

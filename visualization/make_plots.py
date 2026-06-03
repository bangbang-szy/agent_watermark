from __future__ import annotations

import argparse

from agent_watermark.visualization.plots import plot_decoding_accuracy, plot_robustness, plot_trajectory_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv")
    parser.add_argument("--robustness-json")
    parser.add_argument("--accuracy-csv")
    parser.add_argument("--out", default="runtime/figures")
    args = parser.parse_args()
    if args.features_csv:
        plot_trajectory_stats(args.features_csv, args.out)
    if args.robustness_json:
        plot_robustness(args.robustness_json, args.out)
    if args.accuracy_csv:
        plot_decoding_accuracy(args.accuracy_csv, args.out)


if __name__ == "__main__":
    main()

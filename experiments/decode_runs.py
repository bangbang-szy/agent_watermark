from __future__ import annotations

import argparse

from agent_watermark.decoder.aggregate_decoder import MultiRunVotingDecoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode one identity from multiple execution logs.")
    parser.add_argument("--logs", nargs="+", required=True)
    parser.add_argument("--authors", nargs="+", required=True)
    parser.add_argument("--timestamps", nargs="+", required=True)
    parser.add_argument("--timestamp-granularity", choices=["exact", "minute", "hour", "day"], default="hour")
    parser.add_argument("--min-margin-per-run", type=float, default=0.08)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    args = parser.parse_args()
    result = MultiRunVotingDecoder(
        args.authors,
        args.timestamps,
        timestamp_granularity=args.timestamp_granularity,
        min_margin_per_run=args.min_margin_per_run,
        min_confidence=args.min_confidence,
    ).decode_many(args.logs)
    print(result)


if __name__ == "__main__":
    main()

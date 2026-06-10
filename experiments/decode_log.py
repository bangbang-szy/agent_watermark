from __future__ import annotations

import argparse

from agent_watermark.decoder.voting_decoder import MultiStatisticVotingDecoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--authors", nargs="+", required=True)
    parser.add_argument("--timestamps", nargs="+", required=True)
    parser.add_argument("--timestamp-granularity", choices=["exact", "minute", "hour", "day"], default="exact")
    parser.add_argument("--min-margin", type=float, default=0.08)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    args = parser.parse_args()
    result = MultiStatisticVotingDecoder(
        args.authors,
        args.timestamps,
        timestamp_granularity=args.timestamp_granularity,
        min_margin=args.min_margin,
        min_confidence=args.min_confidence,
    ).decode(args.log)
    print(result)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse

from agent_watermark.decoder.voting_decoder import MultiStatisticVotingDecoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--authors", nargs="+", required=True)
    parser.add_argument("--timestamps", nargs="+", required=True)
    args = parser.parse_args()
    result = MultiStatisticVotingDecoder(args.authors, args.timestamps).decode(args.log)
    print(result)


if __name__ == "__main__":
    main()

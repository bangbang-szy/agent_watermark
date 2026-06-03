from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

from agent_watermark.decoder.voting_decoder import MultiStatisticVotingDecoder
from agent_watermark.logging.jsonl_logger import JsonlExecutionLogger


def crop_log(input_path: Path, ratio: float, output_path: Path) -> None:
    steps = JsonlExecutionLogger.read(input_path)
    keep = max(1, int(len(steps) * (1.0 - ratio)))
    kept = sorted(random.sample(steps, keep), key=lambda s: s.step_index)
    JsonlExecutionLogger.write(output_path, kept)


def rewrite_output(input_path: Path, output_path: Path) -> None:
    steps = JsonlExecutionLogger.read(input_path)
    for step in steps:
        if step.final_answer:
            step.final_answer = "Rewritten final answer. Behavioral trace preserved."
    JsonlExecutionLogger.write(output_path, steps)


def lightweight_finetune_attack(input_path: Path, output_path: Path, preferred_tool: str = "search") -> None:
    """Simulate prompt/tool preference drift by perturbing logged candidate scores."""
    steps = JsonlExecutionLogger.read(input_path)
    for step in steps:
        step.reasoning_trace = f"[system-prompt-variant] {step.reasoning_trace}"
        logits = []
        for candidate in step.candidate_actions:
            candidate.raw_logit += 0.25 if candidate.name == preferred_tool else -0.05
            logits.append(candidate.watermarked_logit + (0.20 if candidate.name == preferred_tool else 0.0))
        probs = np.exp(logits - np.max(logits))
        probs = probs / probs.sum()
        for candidate, prob, logit in zip(step.candidate_actions, probs, logits):
            candidate.watermarked_logit = float(logit)
            candidate.watermarked_probability = float(prob)
    JsonlExecutionLogger.write(output_path, steps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--authors", nargs="+", required=True)
    parser.add_argument("--timestamps", nargs="+", required=True)
    parser.add_argument("--out", default="runtime/attacks")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    decoder = MultiStatisticVotingDecoder(args.authors, args.timestamps)
    rows = []
    for ratio in [0.1, 0.2, 0.3]:
        attacked = out / f"crop_{int(ratio*100)}.jsonl"
        crop_log(Path(args.log), ratio, attacked)
        decoded = decoder.decode(attacked)
        rows.append({"attack": "log_crop", "ratio": ratio, **decoded.__dict__})
    rewritten = out / "rewrite_output.jsonl"
    rewrite_output(Path(args.log), rewritten)
    decoded = decoder.decode(rewritten)
    rows.append({"attack": "output_rewrite", "ratio": 0.0, **decoded.__dict__})
    tuned = out / "lightweight_finetune.jsonl"
    lightweight_finetune_attack(Path(args.log), tuned)
    decoded = decoder.decode(tuned)
    rows.append({"attack": "lightweight_finetune", "ratio": 0.0, **decoded.__dict__})
    df = pd.DataFrame(rows)
    df.to_json(out / "robustness_results.json", orient="records", force_ascii=False, indent=2)
    print(df[["attack", "ratio", "author_id", "timestamp", "confidence"]])


if __name__ == "__main__":
    main()

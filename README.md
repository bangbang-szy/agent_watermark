# Agent Watermarking System

This is a runnable research prototype for black-box behavioral watermarking of tool-using agents. The watermark is embedded at action selection, not in final text. Offline recovery uses only JSONL execution logs.

## Directory Tree

```text
agent_watermark/
├── agent_core/
│   ├── database.py
│   ├── langchain_agent.py
│   ├── langgraph_agent.py
│   └── tools.py
├── watermark/
│   ├── embedder.py
│   └── signature.py
├── feature_analysis/
│   ├── extractor.py
│   └── independence.py
├── logging/
│   ├── jsonl_logger.py
│   └── schemas.py
├── decoder/
│   └── voting_decoder.py
├── experiments/
│   ├── analyze_logs.py
│   ├── decode_log.py
│   ├── platform_migration.py
│   ├── robustness.py
│   ├── run_agent.py
│   └── tasks.py
├── visualization/
│   ├── make_plots.py
│   └── plots.py
├── api/
│   └── server.py
├── tests/
├── configs/default.yaml
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Setup

```bash
cd agent_watermark
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Set your API key. Do not commit real keys.

```bash
set OPENAI_API_KEY=sk-...
```

PowerShell:

```powershell
$env:OPENAI_API_KEY="sk-..."
```

You can also create `.env`:

```text
OPENAI_API_KEY=sk-...
```

DeepSeek is supported through the OpenAI-compatible API:

```powershell
$env:DEEPSEEK_API_KEY="sk-..."
python -m agent_watermark.experiments.run_agent --config configs/deepseek.yaml --author-id alice-lab --task "Compare average success_rate by framework in the agents table."
```

Default OpenAI model is in `configs/default.yaml` (`gpt-4.1`). DeepSeek defaults are in `configs/deepseek.yaml` (`deepseek-chat`, `https://api.deepseek.com`).

## Run A Real Watermarked Agent

```bash
python -m agent_watermark.experiments.run_agent --author-id alice-lab --task "Query the local database for the paper with the most citations and report its title and year."
```

DeepSeek:

```bash
python -m agent_watermark.experiments.run_agent --config configs/deepseek.yaml --author-id alice-lab --task "Query the local database for the paper with the most citations and report its title and year."
```

Run the task suite:

```bash
python -m agent_watermark.experiments.run_agent --author-id alice-lab --all
```

Each run writes `runtime/logs/<run_id>.jsonl`. Logs include state, reasoning trace, candidate actions, raw logits, raw probabilities, watermarked logits/probabilities, chosen action, tool call, observations, and timestamps.

## Decode From Logs

Use the timestamp printed by the run command:

```bash
python -m agent_watermark.experiments.decode_log ^
  --log runtime/logs/<run_id>.jsonl ^
  --authors alice-lab bob-lab carol-lab ^
  --timestamps 2026-06-03T12:00:00+00:00 other-candidate-time
```

The decoder returns `author_id`, `timestamp`, `confidence`, and all candidate vote scores. It does not call the model and does not require hidden states.

## Feature Independence Analysis

```bash
python -m agent_watermark.experiments.analyze_logs --logs runtime/logs/*.jsonl --out runtime/analysis
```

Outputs:

- `behavior_features.csv`
- `nmi_matrix.csv`
- `behavior_nmi_heatmap.png`
- `nmi_clustering.png`
- `independent_behaviors.txt`

The implementation includes `mutual_info_score`, entropy, normalized mutual information, hierarchical clustering, and stability scoring.

## Robustness Experiments

```bash
python -m agent_watermark.experiments.robustness ^
  --log runtime/logs/<run_id>.jsonl ^
  --authors alice-lab bob-lab carol-lab ^
  --timestamps 2026-06-03T12:00:00+00:00 other-candidate-time
```

Implemented attacks:

- Log cropping at 10%, 20%, 30%
- Final output rewrite
- Lightweight fine-tuning simulation via system-prompt marker and tool-preference score drift

Results are written to `runtime/attacks/robustness_results.json`.

## Presentation Evaluation Plots

Run a compact multi-task evaluation and generate PNG figures for slides or a standup:

```bash
python -m agent_watermark.experiments.evaluate_watermark \
  --config configs/deepseek.yaml \
  --watermarked-author alice-lab \
  --authors alice-lab bob-lab carol-lab \
  --tasks 6 \
  --repeats 1 \
  --out runtime/evaluation
```

Run all built-in tasks:

```bash
python -m agent_watermark.experiments.evaluate_watermark \
  --config configs/deepseek.yaml \
  --watermarked-author alice-lab \
  --authors alice-lab bob-lab carol-lab \
  --tasks 0 \
  --repeats 1 \
  --out runtime/evaluation_all_tasks
```

Sweep watermark strength to measure task-success / watermark-recovery trade-off:

```bash
python -m agent_watermark.experiments.evaluate_watermark \
  --config configs/deepseek.yaml \
  --watermarked-author alice-lab \
  --authors alice-lab bob-lab carol-lab \
  --tasks 6 \
  --repeats 1 \
  --lambda-values 0.00 0.06 0.12 0.18 0.24 0.30 \
  --timestamp-granularity hour \
  --min-margin 0.08 \
  --min-confidence 0.55 \
  --auto-calibrate-threshold \
  --target-selective-accuracy 0.95 \
  --out runtime/evaluation_lambda_sweep
```

Use existing logs without calling the LLM:

```bash
python -m agent_watermark.experiments.evaluate_watermark \
  --logs runtime/logs/*.jsonl \
  --authors alice-lab bob-lab carol-lab \
  --out runtime/evaluation
```

Outputs:

- `runtime/evaluation/plots/watermark_evaluation_overview.png`
- `runtime/evaluation/plots/behavior_statistics_boxplot.png`
- `runtime/evaluation/plots/vote_score_heatmap.png`
- `runtime/evaluation/plots/robustness_confidence_curve.png`
- `runtime/evaluation/plots/lambda_tradeoff_curve.png` when `--lambda-values` is used
- `runtime/evaluation/plots/coverage_accuracy_calibration.png`
- `runtime/evaluation/plots/decoder_ablation.png`
- `runtime/evaluation/plots/aggregate_group_size_curve.png`
- `runtime/evaluation/plots/tool_reliability.png`
- CSV tables for clean decoding, attack decoding, behavior features, tool actions, and vote scores
- `aggregate_decoding_results.csv` for multi-run grouped decoding
- `aggregate_group_size_results.csv` and `aggregate_group_size_summary.csv` for multi-run coverage analysis
- `calibration_curve.csv` for abstention threshold analysis
- `statistical_summary.csv` with mean, standard error, and 95% CI
- `decoder_ablation.csv` and `decoder_ablation_summary.csv` comparing full, action-only, and feature-only decoders
- `tool_diagnostics.csv` for real tool reliability and task-success analysis
- `robustness_accuracy_matrix.png` for attack-by-severity recovery accuracy

For morning-report style experiments that focus on reducing abstention and improving deployment readiness:

```bash
python -m agent_watermark.experiments.evaluate_watermark \
  --config configs/deepseek.yaml \
  --watermarked-author alice-lab \
  --authors alice-lab bob-lab carol-lab \
  --tasks 6 \
  --repeats 1 \
  --lambda-values 0.06 0.12 0.18 0.24 0.30 \
  --timestamp-granularity hour \
  --auto-calibrate-threshold \
  --target-selective-accuracy 0.95 \
  --min-confidence 0.55 \
  --out runtime/morning_report_v3
```

For a larger robustness run with multiple authors and a broader attack matrix:

```bash
python -m agent_watermark.experiments.evaluate_watermark \
  --config configs/deepseek.yaml \
  --watermarked-authors alice-lab bob-lab carol-lab \
  --authors alice-lab bob-lab carol-lab \
  --tasks 0 \
  --repeats 2 \
  --lambda-values 0.06 0.12 0.18 \
  --timestamp-granularity hour \
  --auto-calibrate-threshold \
  --target-selective-accuracy 0.95 \
  --min-confidence 0.55 \
  --crop-ratios 0.1 0.2 0.3 0.4 0.5 \
  --noise-sigmas 0.01 0.03 0.05 0.08 0.10 \
  --tool-deletion-ratios 0.3 0.5 0.7 \
  --preference-tools search sqlite_db python_repl \
  --out runtime/full_robustness_report
```

This full setting runs `3 authors x 16 tasks x 2 repeats x 3 lambdas = 288` real agent trajectories before attacks.
For a cheaper version, use `--tasks 8 --repeats 1`.

The summary now separates `task_success_rate` from `strict_task_success_rate`. The first checks whether
the agent produced a usable final answer; the second treats any intermediate tool error as a failed trajectory,
which is useful for diagnosing tool-chain reliability separately from watermark recovery.

Multi-run deployment decoding is available through:

```bash
python -m agent_watermark.experiments.decode_runs \
  --logs runtime/logs/*.jsonl \
  --authors alice-lab bob-lab carol-lab \
  --timestamps 2026-07-01T12:00:00+00:00 \
  --timestamp-granularity hour
```

The evaluation script includes clean decoding plus robustness attacks:

- 10%, 20%, 30% log cropping
- final-output rewrite
- lightweight tool-preference drift
- local log reordering
- candidate probability noise
- tool-call observation deletion

It also records a heuristic `task_success` signal, defined as a non-empty final answer with no tool-call errors in the execution log.

For short trajectories, exact microsecond timestamp recovery is too high-capacity. Use `--timestamp-granularity hour`
or the DeepSeek config default `watermark_timestamp_granularity: hour` to encode a recoverable timestamp bucket.
The decoder reports both exact timestamp and timestamp-bucket accuracy.

The decoder also reports:

- `margin`: top candidate score minus runner-up score
- `calibrated_confidence`: sigmoid calibration of the margin
- `abstained`: whether evidence is too weak under `--min-margin` / `--min-confidence`
- `correct_author_when_not_abstained`: selective accuracy after refusing weak evidence

## Platform Migration

LangGraph:

```bash
python -m agent_watermark.experiments.platform_migration --config configs/deepseek.yaml --framework langgraph --author-id alice-lab
```

LangChain-style loop:

```bash
python -m agent_watermark.experiments.platform_migration --config configs/deepseek.yaml --framework langchain --author-id alice-lab
```

Both use the same LangChain tools and watermark middleware, which is the intended migration boundary.

## API

```bash
uvicorn agent_watermark.api.server:app --reload --port 8000
```

Endpoints:

- `POST /run`
- `POST /decode`
- `POST /analyze`

Example:

```bash
curl -X POST http://127.0.0.1:8000/run ^
  -H "Content-Type: application/json" ^
  -d "{\"task\":\"Compare average success_rate by framework in the agents table.\",\"author_id\":\"alice-lab\"}"
```

## Visualizations

```bash
python -m agent_watermark.visualization.make_plots ^
  --features-csv runtime/analysis/behavior_features.csv ^
  --robustness-json runtime/attacks/robustness_results.json ^
  --out runtime/figures
```

This generates trajectory statistics and robustness plots. NMI heatmap and clustering plots are generated by `analyze_logs.py`.

## Tests

```bash
pytest
```

The tests cover core math and log-derived feature extraction without requiring an API key. Full agent tests require `OPENAI_API_KEY` and live search/network access.

## Notes

The planner asks the LLM to score candidate actions with utility logits before watermarking. This is necessary because standard hosted tool-calling APIs generally do not expose internal tool-selection logits. The middleware then applies:

```text
p'(a) = softmax(log p(a) + lambda * phi(a))
```

No model parameters are modified, no meaningless tool steps are inserted, and the final text is not used as the carrier.

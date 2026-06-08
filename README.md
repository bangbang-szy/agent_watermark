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
- CSV tables for clean decoding, attack decoding, behavior features, tool actions, and vote scores

The evaluation script includes clean decoding plus robustness attacks:

- 10%, 20%, 30% log cropping
- final-output rewrite
- lightweight tool-preference drift
- local log reordering
- candidate probability noise
- tool-call observation deletion

It also records a heuristic `task_success` signal, defined as a non-empty final answer with no tool-call errors in the execution log.

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

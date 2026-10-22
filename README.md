# agent-eval-harness

A pytest-like evaluation harness for LLM agents. Define a dataset of cases,
run your agent (mock or live), score the results with deterministic checks,
trajectory assertions, a judge, and performance budgets, and gate CI on the
outcome.

## Features

- **Datasets** as YAML — inputs, expected outputs, tags, priorities, budgets.
- **Mock + live modes** — run end-to-end without a provider, then switch to
  real agents (Ollama, LangGraph, or your own adapter).
- **Scoring** — deterministic, judge (LLM-as-judge), trajectory, performance,
  with configurable weights.
- **Regression detection** — compare a run against a named baseline and fail on
  score drops, latency regressions, or new errors.
- **Reports** — JSON, Markdown, and a self-contained HTML report (no server).
- **Storage** — SQLite-backed run history; in-memory for tests.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"            # core + dev
# optional extras:
pip install -e ".[ollama,langgraph,dashboard]"
```

Requires Python 3.11–3.13.

## Quick start

```bash
# Validate config + dataset
agent-eval validate

# Run the evaluation (mock mode by default)
agent-eval run

# Re-render a report for the latest stored run
agent-eval report --format html

# Compare against a saved baseline
agent-eval run --save-baseline gold
agent-eval run --compare --baseline gold
```

## Project layout

```
src/agent_eval/      # library + CLI (typer)
evals/dataset.yaml   # example dataset
examples/            # runnable examples
scripts/             # smoke checks
tests/               # pytest suite
agent-eval.yaml      # project config
```

## Development

```bash
pytest -q          # 723+ tests
ruff check src tests
mypy src
```

## License

MIT

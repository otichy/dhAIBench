# LLM Linguistic Classification Benchmark Agent

Python tooling for benchmarking large language models on linguistic classification tasks. The agent loads semicolon-delimited datasets, queries an OpenAI-compatible endpoint for each example, and writes predictions, metrics, charts, and prompt logs.

- End-to-end benchmark runs across one or many CSV inputs
- Metrics-only recompute from existing output files
- Provider-aware auth and request controls
- Static browser GUI for building CLI commands
- Static browser dashboard for exploring metrics artifacts
- Optional external validator protocol for large label spaces

## Quick Start

Install the required packages:

```bash
python -m pip install openai matplotlib
```

Prepare credentials in `.env` or your shell environment. By default the agent reads `OPENAI_API_KEY`, and it can also infer custom provider variables such as `REQUESTY_API_KEY` and `REQUESTY_BASE_URL`.

Run a benchmark:

```bash
python benchmark_agent.py \
  --input example_input.csv \
  --model gpt-4o-mini \
  --temperature 0.0 \
  --top_p 1.0 \
  --output data/output/
```

On PowerShell, replace `\` with `^` for line continuation.

## Documentation

Start here depending on what you need:

- [Docs Home](docs/README.md)
- [Getting Started](docs/getting-started.md)
- [Common Examples](docs/examples.md)
- [Providers and Authentication](docs/providers.md)
- [GUI Command Builder](docs/gui.md)
- [Metrics Dashboard](docs/metrics-dashboard.md)
- [Validators](docs/validators.md)
- [Outputs and Metrics](docs/outputs-and-metrics.md)
- [CLI Reference](docs/cli-reference.md)

## Repository Layout

- `benchmark_agent.py`: main benchmark runner and evaluation logic
- `config_gui.html`: static command builder for the CLI
- `config_models.js`: model catalog consumed by the GUI
- `validators/`: reference validator implementations
- `web/`: static metrics dashboard
- `scripts/`: maintenance and analysis helpers

## Notes

- Input files are semicolon-delimited CSVs with at least `ID` and `node`.
- Additional columns are preserved in the output CSV even if they are not used in the prompt.
- Metrics artifacts are written under `data/metrics`, and prompt logs under `data/logs`.
- Set `DHAIBENCH_DATA_ROOT` to move the default `data/` root.

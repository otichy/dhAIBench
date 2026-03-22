# Getting Started

## Requirements

- Python 3.9+
- `openai`
- `matplotlib` for confusion heatmaps and optional calibration plots

Install dependencies:

```bash
python -m pip install openai matplotlib
```

## Repository Basics

- `benchmark_agent.py` is the main entry point.
- `example_input.csv` is the smallest reference dataset in the repo.
- `config_gui.html` builds CLI commands in the browser.
- `web/` contains the static metrics dashboard.

## Dataset Format

Input files must be semicolon-delimited CSVs.

| Column | Required | Description |
| --- | --- | --- |
| `ID` | Yes | Unique example identifier |
| `leftContext` | No | Text before the node token |
| `node` | Yes | Target word to classify |
| `rightContext` | No | Text after the node token |
| `info` | No | Supplemental instructions or metadata |
| `truth` | No | Gold label, unless supplied via `--labels` |

When you use `--labels`, provide a separate CSV with `ID;truth`. Extra input columns are carried into the output CSV.

## Credentials

By default the agent reads `OPENAI_API_KEY`. For non-default providers it can infer environment variable names from the provider slug:

- `<PROVIDER>_API_KEY` or `<PROVIDER>_ACCESS_TOKEN`
- `<PROVIDER>_BASE_URL`

Examples:

- `REQUESTY_API_KEY`
- `REQUESTY_BASE_URL`
- `VERTEX_ACCESS_TOKEN`
- `VERTEX_BASE_URL`

You can override discovery with `--api_key_var` and `--api_base_var`.

## First Run

```bash
python benchmark_agent.py \
  --input example_input.csv \
  --model gpt-4o-mini \
  --temperature 0.0 \
  --top_p 1.0 \
  --output data/output/
```

On PowerShell, replace `\` with `^`.

If `--output` is omitted, the agent writes:

- `data/output/<input_basename>__<provider>__<model>__<timestamp>.csv`
- `data/metrics/<output_basename>__metrics.json`
- `data/logs/<output_basename>.log`

Set `DHAIBENCH_DATA_ROOT` if you want a different root for `data/`.

## What To Read Next

- [Common Examples](examples.md) for recurring workflows
- [Providers and Authentication](providers.md) for Vertex, Gemini, Requesty, and custom endpoints
- [Outputs and Metrics](outputs-and-metrics.md) for artifact layout

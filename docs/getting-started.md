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
- `example_input.csv` contains 100 fully labeled examples from the prepositions task.
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

The bundled `example_input.csv` copies the first 100 data rows of `data/input/prepositions.csv`, retaining their original IDs, text, labels, and `speakerID` metadata. The task is to return the correct preposition in a transcribed learner dialogue, or `0` when it should be omitted. All 100 sampled nodes are `in`: 83 gold labels are `in` and 17 are `0`. This is a small introductory sample, not a representative selection of the full task.

| Example ID | Node | Gold label (`truth`) | Expected decision |
| --- | --- | --- | --- |
| `1` | `in` | `in` | Keep the preposition |
| `21` | `in` | `0` | Omit the preposition |

`truth` is used for evaluation and is not included in the target prompt. `speakerID` is carried into the output as an extra column.

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

Use the prepositions task's system prompt, shown on one line below. It asks for the corrected preposition itself as the label, rather than a label such as `correct` or `incorrect`.

```bash
python benchmark_agent.py \
  --input example_input.csv \
  --system_prompt "You are a meticulous annotator focusing on grammatical errors in the use of prepositions. You will receive a short excerpt of a transcribed dialogue where the node word is a preposition. Decide whether it is a correct preposition given the context. Return the correct preposition (even if it already is correct). If the preposition should be omitted (i.e. the grammatical correct solution is leaving it out), return 0." \
  --model gpt-4o-mini \
  --temperature 0.0 \
  --top_p 1.0 \
  --output data/output/
```

Every model run requires a task-specific system prompt defining what to classify and the allowed labels. Use `--system_prompt` or `--system_prompt_b64`, or restore a saved prompt via `--resume` or `--load_params`. Missing, blank, invalid encoded prompts and the former generic default are rejected before any API requests. Metrics-only, model-catalog updates, and log summaries do not require a prompt.

The command uses POSIX `\` line continuation. In PowerShell, use a backtick (`` ` ``) instead, or put the command on one line.

If `--output` is omitted, the agent writes:

- `data/output/<input_basename>__<provider>__<model>__<timestamp>.csv`
- `data/metrics/<output_basename>__metrics.json`
- `data/logs/<output_basename>.log`

Set `DHAIBENCH_DATA_ROOT` if you want a different root for `data/`.

## What To Read Next

- [Common Examples](examples.md) for recurring workflows
- [Providers and Authentication](providers.md) for Vertex, Gemini, Requesty, and custom endpoints
- [Outputs and Metrics](outputs-and-metrics.md) for artifact layout

# Common Examples

Examples below use POSIX line continuation. On PowerShell, replace `\` with `^`.

## Basic Benchmark Run

```bash
python benchmark_agent.py \
  --input data/input/input.csv \
  --model gpt-4o-mini \
  --temperature 0.0 \
  --top_p 1.0 \
  --prompt_layout compact \
  --request_interval_ms 250 \
  --threads 4 \
  --system_prompt "You are a linguistic classifier." \
  --enable_cot \
  --few_shot_examples 5 \
  --calibration
```

## Multiple Input Files

```bash
python benchmark_agent.py \
  --input data/input/input.csv data/input/input_extra.csv \
  --model gpt-4o-mini \
  --output data/output/
```

When `--output` points to a directory, each input gets its own timestamped output file.

## Resume An Existing Output

```bash
python benchmark_agent.py \
  --input data/input/input.csv \
  --output data/output/task__openai__gpt4omini__2026-03-20-18-21.csv \
  --resume
```

The agent skips rows whose `ID` already exists in the output CSV and continues in input order.

## Re-run Only `unclassified` Predictions

```bash
python benchmark_agent.py \
  --input data/input/input.csv \
  --output data/output/task__openai__gpt4omini__2026-03-20-18-21.csv \
  --resume \
  --unclassified
```

To keep iterating until unresolved rows stabilize:

```bash
python benchmark_agent.py \
  --input data/input/input.csv \
  --output data/output/task__openai__gpt4omini__2026-03-20-18-21.csv \
  --resume \
  --repeat_unclassified
```

## Metrics-Only Recompute

Refresh only the aggregate agreement files from existing `*_metrics.json` artifacts:

```bash
python benchmark_agent.py --metrics_only
```

Recompute metrics for an existing output CSV:

```bash
python benchmark_agent.py \
  --metrics_only \
  --input data/output/task__openai__gpt4omini__2026-03-20-18-21.csv
```

Override truth labels from a separate file:

```bash
python benchmark_agent.py \
  --metrics_only \
  --input data/output/task__openai__gpt4omini__2026-03-20-18-21.csv \
  --labels data/new_truth_labels.csv
```

## Refresh The GUI Model Catalog

```bash
python benchmark_agent.py --update-models
```

To update only selected providers:

```bash
python benchmark_agent.py --update-models --models-providers openai requesty vertex
```

## Summarize Prompt-Log Errors

```bash
python benchmark_agent.py \
  --summarize-log-errors data/logs/task__openai__gpt4omini__2026-03-20-18-21.log
```

## Timeout Diagnosis

```bash
python benchmark_agent.py \
  --input data/input/input.csv \
  --model gpt-4o-mini \
  --timeout_probe
```

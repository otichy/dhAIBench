# Common Examples

Examples below use POSIX `\` line continuation. In PowerShell, use a backtick (`` ` ``) instead, or put the command on one line.

The runnable examples use `example_input.csv`, the first 100 fully labeled rows of the prepositions task, and its corresponding system prompt. Return the corrected preposition as the label, or `0` to omit it. This sample contains 83 `in` labels and 17 `0` labels; see [Getting Started](getting-started.md#dataset-format) for the column meanings. Replace both the dataset and task prompt together when benchmarking a different task.

## Basic Benchmark Run

```bash
python benchmark_agent.py \
  --input example_input.csv \
  --model gpt-4o-mini \
  --temperature 0.0 \
  --top_p 1.0 \
  --prompt_layout compact \
  --request_interval_ms 250 \
  --threads 4 \
  --system_prompt "You are a meticulous annotator focusing on grammatical errors in the use of prepositions. You will receive a short excerpt of a transcribed dialogue where the node word is a preposition. Decide whether it is a correct preposition given the context. Return the correct preposition (even if it already is correct). If the preposition should be omitted (i.e. the grammatical correct solution is leaving it out), return 0." \
  --enable_cot \
  --few_shot_examples 5 \
  --calibration
```

## Multiple Input Files

```bash
python benchmark_agent.py \
  --input example_input.csv example_input_extra.csv \
  --system_prompt "You are a meticulous annotator focusing on grammatical errors in the use of prepositions. You will receive a short excerpt of a transcribed dialogue where the node word is a preposition. Decide whether it is a correct preposition given the context. Return the correct preposition (even if it already is correct). If the preposition should be omitted (i.e. the grammatical correct solution is leaving it out), return 0." \
  --model gpt-4o-mini \
  --output data/output/
```

Provide your own `example_input_extra.csv` with additional examples of the same preposition-correction task. When `--output` points to a directory, each input gets its own timestamped output file.

## Resume An Existing Output

```bash
python benchmark_agent.py \
  --input data/input/input.csv \
  --output data/output/task__openai__gpt4omini__2026-03-20-18-21.csv \
  --resume
```

The agent skips rows whose `ID` already exists in the output CSV and continues in input order.

Resume and `--load_params` reuse the saved task-specific system prompt. If the artifacts do not contain one, or contain the former generic default, supply `--system_prompt` explicitly.

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

Reuse the parameters from an earlier metrics file while overriding the model and input:

```bash
python benchmark_agent.py \
  --load_params data/metrics/previous_run__metrics.json \
  --model gpt-5.6 \
  --input data/input/task.csv
```

Only the earlier file's reusable `run_config` parameters are loaded. Explicit CLI
arguments take precedence. The new metrics artifact records both the resolved values
used by the run and the `load_params` source path for provenance; that provenance field
is not itself loaded by a subsequent run.

Run a quick benchmark without writing run metrics or refreshing agreement artifacts:

```bash
python benchmark_agent.py \
  --input example_input.csv \
  --system_prompt "You are a meticulous annotator focusing on grammatical errors in the use of prepositions. You will receive a short excerpt of a transcribed dialogue where the node word is a preposition. Decide whether it is a correct preposition given the context. Return the correct preposition (even if it already is correct). If the preposition should be omitted (i.e. the grammatical correct solution is leaving it out), return 0." \
  --output data/output/sample_test.csv \
  --model gpt-4.1-mini \
  --no_metrics
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
  --input example_input.csv \
  --model gpt-4o-mini \
  --timeout_probe \
  --system_prompt "You are a meticulous annotator focusing on grammatical errors in the use of prepositions. You will receive a short excerpt of a transcribed dialogue where the node word is a preposition. Decide whether it is a correct preposition given the context. Return the correct preposition (even if it already is correct). If the preposition should be omitted (i.e. the grammatical correct solution is leaving it out), return 0."
```

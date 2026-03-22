# Outputs And Metrics

## Output Files

A benchmark run can create several artifacts:

- predictions CSV: `data/output/<output_basename>.csv`
- metrics JSON: `data/metrics/<output_basename>__metrics.json`
- confusion heatmap: `data/metrics/<output_basename>__heatmap.png`
- calibration plot: `data/metrics/<output_basename>__calibration.png`
- prompt log: `data/logs/<output_basename>.log`

If `--output` is omitted, the predictions CSV name is inferred from input, provider, model, and timestamp.

## Predictions CSV

The output CSV contains:

- original context columns
- prediction
- explanation, unless `--no_explanation` is used
- confidence when the model returns a valid value
- token usage fields
- preserved pass-through columns from the original dataset

When `--resume` is used, existing rows are reused by `ID`.

## Metrics JSON

The metrics JSON includes:

- accuracy
- macro F1
- per-label precision, recall, and F1
- confusion matrix
- `task_name`
- `task_description`
- `tags`

It also includes summaries for request-control acceptance and provider-reported cache metadata when those features are used.

## Charts

- Confusion heatmap generation is on by default. Disable it with `--no-confusion_heatmap`.
- Calibration plots are optional and enabled with `--calibration`.

`matplotlib` is required for both chart types.

## Prompt Logs

Prompt logs are stored in NDJSON format, one JSON object per line.

They capture:

- run metadata
- reconstructed run command
- per-attempt request and response details
- retry and validator metadata

Legacy JSON-array logs are auto-migrated to NDJSON on resume, with a backup written as `<output_basename>.log.legacy.json`.

## Metrics-Only Mode

Use `--metrics_only` to recompute metrics from an existing predictions CSV without calling the model API again.

Truth labels are taken from:

1. the `truth` column in the output CSV
2. overrides from `--labels`, when supplied

In this mode, `--output` is ignored and metrics artifacts are written to `data/metrics`.

## Dashboard Consumption

The static dashboard in `web/` reads the `*_metrics.json` artifacts produced here. See [GUI and Metrics Dashboard](gui.md) for browser usage.

## Operational Tips

- Use `--request_interval_ms` to reduce 429 bursts.
- Increase `--max_retries` or `--retry_delay` when a provider is rate-limiting.
- Use `--no_explanation` when explanations are not needed.
- Leave `--temperature` and `--top_p` unset if you want provider defaults instead of explicit values.
- Leave `--logprobs` disabled unless token-level probabilities are required.

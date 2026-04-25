# Outputs And Metrics

## Output Files

A benchmark run can create several artifacts:

- predictions CSV: `data/output/<output_basename>.csv`
- metrics JSON: `data/metrics/<output_basename>__metrics.json`
- agreement summary JSON: `data/metrics/agreement_summary.json`
- agreement clustering JSON: `data/metrics/agreement_clusters.json`
- confusion heatmap: `data/metrics/<output_basename>__heatmap.png`
- calibration plot: `data/metrics/<output_basename>__calibration.png`
- prompt log: `data/logs/<output_basename>.log`
- session log: `data/logs/sessions/benchmark_agent_<timestamp>.log`

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
- Cohen's Kappa
- macro F1
- per-label precision, recall, and F1
- confusion matrix
- `task_name`
- `task_description`
- `tags`

It also includes summaries for request-control acceptance and provider-reported cache metadata when those features are used.

## Agreement Summary JSON

`data/metrics/agreement_summary.json` is rebuilt after normal runs and `--metrics_only`.

It stores only aggregate statistics, not per-example predictions:

- repeat-run Krippendorff's alpha for repeated runs of the same provider/model on the same comparable task variant
- cross-model Krippendorff's alpha for the same task variant using one representative run per provider/model
- both `latest` and `best_accuracy` representative policies for cross-model agreement
- overlap counts, run/model counts, and run references needed by the dashboard

Comparable task variants are matched automatically from the output rows plus normalized tags, so task renames can still be linked when the underlying data are the same.
The comparison key ignores the `open source` tag so that publication-status variants do not split otherwise comparable runs.

## Agreement Clustering JSON

`data/metrics/agreement_clusters.json` is rebuilt alongside the agreement summary.

It stores privacy-safe similarity data for the Agreement tab:

- repeat-run trees for repeated runs of the same provider/model on the same comparable task variant
- one representative run per provider/model for each comparable task variant and representative policy
- pairwise disagreement distances over overlapping rated items
- average-linkage dendrogram merges for the full visible group

The dashboard can use the stored pairwise distances to redraw the dendrogram for a filtered subset of visible runs or models without requiring access to the original output CSVs.

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

## Session Logs

Session logs capture the benchmark runner's own process logging and are written separately under `data/logs/sessions`.

## Metrics-Only Mode

Use `--metrics_only` to recompute metrics from an existing predictions CSV without calling the model API again. If you omit `--input`, metrics-only mode refreshes only `agreement_summary.json` and `agreement_clusters.json` from existing run metrics.

Truth labels are taken from:

1. the `truth` column in the output CSV
2. overrides from `--labels`, when supplied

In this mode, `--output` is ignored and metrics artifacts are written to `data/metrics`.

## Dashboard Consumption

The static dashboard in `web/` reads the `*_metrics.json` artifacts produced here and, when available, `agreement_summary.json`. See [Metrics Dashboard](metrics-dashboard.md) for loading modes, navigation, and feature details.

## Operational Tips

- Use `--request_interval_ms` to reduce 429 bursts.
- Increase `--max_retries` or `--retry_delay` when a provider is rate-limiting.
- Use `--no_explanation` when explanations are not needed.
- Leave `--temperature` and `--top_p` unset if you want provider defaults instead of explicit values.
- Leave `--logprobs` disabled unless token-level probabilities are required.

# Metrics Dashboard

The static dashboard in `web/` is for exploring `*_metrics.json` artifacts produced by benchmark runs. It is read-only: it does not run benchmarks, edit results, or need a backend service of its own.

## What It Includes

The dashboard loads metrics files and derives a run catalogue with:

- task, model, provider, timestamp, and tags
- accuracy, Cohen's Kappa, macro F1, macro precision, macro recall, and calibration metrics
- repeat-run Krippendorff's alpha when an agreement summary is available
- token and request summaries
- estimated cost and pricing metadata when available
- links back to sibling artifacts such as the heatmap, calibration chart, prompt log, output CSV, input CSV, and raw metrics JSON

The main screen includes:

- a filter sidebar for task, model, tags, time range, and missing-accuracy filtering
- KPI cards for total runs, total tasks, best accuracy, and total requests
- a leaderboard area with multiple views
- an agreement area for repeated-run and cross-model alpha
- a prompt token profile panel
- a runs table
- a run-detail modal with links and previews

## Starting The Dashboard

### Local-Only Mode

Use this when you open the dashboard directly from disk with `file://`.

1. Put your `*_metrics.json` files under `data/metrics/`.
2. Open `web/index.html` in a browser.
3. Click `Open Metrics Folder` and choose the folder that contains the metrics files.

In `file://` mode the dashboard cannot auto-scan local folders, so the one manual folder-selection step is required by the browser.

### Server Mode

From repository root:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000/web/`.

In server mode the `Auto (Server)` source attempts, in order:

1. `web/metrics-manifest.json`
2. fallback directory discovery from `../data/metrics/`

The `Reload` button refreshes the current source.

When `data/metrics/agreement_summary.json` is present, the dashboard loads it alongside the run metrics and enables the Agreement tab inside `Leaderboard & Agreement`. When `data/metrics/agreement_clusters.json` is also present, the Agreement tab can render cross-model similarity trees.

Pricing metadata for the scatterplot and run details is loaded from `web/config_prices.js`.
If you deploy the dashboard under a rewritten root or any setup that exposes only `web/`, make sure that file is published there as well.
The price update flow writes a mirrored dashboard copy automatically when it generates the root `config_prices.js`.

### Manifest Notes

`web/generate_metrics_manifest.py` is optional but useful for larger collections.

- `metrics_files` is authoritative when present
- if a listed file 404s, the dashboard retries by filename in common metrics directories
- `metrics_base_dirs` can extend those retry directories
- if manifest loading fails, the dashboard falls back to directory discovery when possible

## Loading Modes And Source Status

The header source panel exposes:

- `Auto (Server)`: load from server-hosted metrics discovery
- `Open Metrics Folder`: choose a local folder through the browser file-system picker
- `Reload`: reload the active source

The status line reports:

- current mode, such as `server` or `folder`
- number of loaded files
- warning count

Warnings are also summarized below the status line so you can spot malformed or skipped files quickly.

## Navigating The Dashboard

The dashboard is built around a simple loop: narrow the run set in the sidebar, inspect the summary cards, compare runs in the leaderboard, and open individual runs for deeper details.

## Filters

The left sidebar controls the active subset of runs:

- `Task`: multi-select task filter
- `Model`: multi-select model filter
- `Time Ranges (OR)`: one or more timestamp windows
- `Hide runs without accuracy`: hide runs that do not expose an accuracy metric
- `Tags`: clickable chips derived from semicolon-delimited run tags

Useful behavior:

- desktop multi-select supports Ctrl/Cmd and Shift
- time ranges are additive, not exclusive
- `Reset All Filters` clears the current selection
- the sidebar can be collapsed, reopened, and used from the mobile filter drawer

The dashboard persists most UI state in browser storage, including filters, selected tab, grouping mode, and theme.

## KPIs

The KPI strip gives a fast summary of the current filtered view:

- `Total Runs`
- `Total Tasks`
- `Best Accuracy`
- `Total Requests`

These values update immediately when filters change.

## Leaderboard

The leaderboard is the main analysis area. `Main Metric` changes the ranking basis for the views below:

- Accuracy
- Cohen's Kappa
- Macro F1
- Macro Precision
- Macro Recall
- Calibration ECE

Available tabs:

- `Chart`: ranked bars for the current metric
- `Scatter`: either metric vs price or metric vs time
- `Table`: sortable metric table
- `Radar`: model profiles across tasks or tags

### Chart Tab

The chart tab shows ranked runs or grouped summaries.

- `Group By` supports `None`, `Model`, and `Task`
- grouped rows show averages; when grouped by model or task, repeated runs are averaged within each task/model first so each task/model contributes once to the grouped summary
- a `TOP` badge marks the best individual run for the current metric when that distinction is relevant
- `Best run per task` switches to a compact task-leader view
- clicking a row opens the run-detail modal

For metrics where lower is better, such as calibration error, the dashboard labels that explicitly.

For accuracy-like metrics, it can also draw approximate 95% confidence intervals derived from the evaluated sample size.

### Scatter Tab

The scatter tab has two x-axis modes:

- `Price`: compare the current metric against estimated total cost or average cost per prediction
- `Time`: compare the current metric over run timestamps

Shared controls include:

- `Group By` for `None`, `Model`, and `Task`
- grouped price points use the same balanced averaging rule as the chart tab when grouping by model or task
- `CI` toggle when the metric supports approximate confidence intervals
- `Labels` toggle for point labels
- `Reset Zoom`

You can zoom by dragging a selection over the plotted area. Clicking a point opens the run-detail modal.

### Table Tab

The leaderboard table gives a dense sortable comparison across runs.

- click a column header to sort
- highlighted cells mark the preferred value for that metric in the current selection
- `Repeat α` is filled only for runs that belong to a repeated same-model agreement group
- row labels may include selected tag badges
- clicking a row opens the run-detail modal

On narrow screens or wide metric sets, the table can be scrolled horizontally.

## Agreement

The `Agreement` tab reads the precomputed `agreement_summary.json` artifact.

- the `Agreement` switch selects `Same model` or `Cross-Model`
- `Same model` shows Krippendorff's alpha across repeated runs of one provider/model on the same comparable task variant
- `Cross-Model` shows Krippendorff's alpha across one representative run per provider/model on the same comparable task variant
- `Compare by` appears only in `Cross-Model` mode and switches between the `Latest` and `Best Accuracy` representative policies
- the tab only shows groups fully represented inside the current filter, so restrictive time/model filters can hide otherwise valid agreement groups
- when `agreement_clusters.json` is present, `Cross-Model` also shows similarity trees built from pairwise disagreement distances
- those trees are recomputed in the browser for the currently visible representative models, so model filters can redraw the clustering even when the full-group alpha row is hidden

### Radar Tab

The radar view compares model profiles across multiple tasks or tags.

- `Group By` becomes an axis selector: `Task` or `Tag`
- the chart plots average metric values for each model across the selected axes
- `Scale` switches between `Linear` and `Contrast`
- at least three axes are required to render the radar
- when many models are present, the dashboard shows the top subset first and lets you load more

For lower-is-better metrics, smaller shapes represent better values.

## Prompt Token Profile

The `Prompt Token Profile` panel shows average tokens per prediction by run, split into:

- input
- cached input
- output
- thinking

Each row also shows prediction count and estimated cost. Clicking a row opens the corresponding run-detail modal.

## Runs Table

The runs table is the quickest raw list view. It includes:

- task
- model
- timestamp
- accuracy
- Cohen's Kappa
- macro F1
- calibration ECE
- requests
- cached input tokens
- source filename

Clicking a row opens the run-detail modal.

## Run Detail Modal

The modal opens from leaderboard rows, scatter points, token-profile rows, and table rows.

It includes:

- run metadata such as task, model, provider, tags, timestamp, and reasoning settings
- metric values and sample counts
- token usage, runtime, pricing, and request totals
- links to metrics JSON, heatmap, calibration plot, log file, output CSV, and input CSV
- chart previews for heatmap and calibration artifacts when present
- an expandable raw JSON view of the loaded metrics file

## Tips

- If `file://` mode appears empty, use `Open Metrics Folder`; auto-loading only works in server mode.
- If a run is missing from the dashboard, check the warning summary first.
- Use tags plus time ranges together when comparing experimental slices.
- Switch `Main Metric` before interpreting rankings; the same filtered dataset can look very different by accuracy versus calibration error.

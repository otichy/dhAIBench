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

When `data/metrics/agreement_summary.json` is present, the dashboard loads it alongside the run metrics and enables the Agreement tab inside `Leaderboard & Agreement`. When `data/metrics/agreement_clusters.json` is also present, the Agreement tab can render same-model and cross-model similarity trees.

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

The share button in the source toolbar copies a URL for the current view. The
URL stores filters, selected tags, leaderboard tab, agreement mode, open run or
agreement-tree detail modals, and scatter/time-series zoom windows. Shared links
restore fully when the recipient loads the same published server metrics. For
locally selected folders or files, the recipient must load the same local metrics
source first because browsers cannot reopen local files from a URL.

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
- `Custom Leaderboard`: task-weighted model ranking and cost comparison

### Custom Leaderboard Tab

Choose tasks from within the sidebar task filter using the tab's checkboxes, then
enter relative weights (for example, 3 and 1 become 75% and 25%). Deselected tasks
retain their weights. `Clear` disables all tasks; `Equal weights` sets weights to
1 without changing the checkboxes. The tab has its own Accuracy/Macro F1 selector.
Model, tag and time filters restrict eligible runs, but cannot silently remove a
required task from the calculation.

`Select shared tasks` finds tasks for which every currently filtered model has an
eligible result for the selected scoring metric. It honors model, tag, date and
missing-accuracy filters while ignoring the current task filter, so it can broaden
an existing task selection. It updates both the task-weight checkboxes and the
sidebar task filter, preserves positive weights, and changes a selected zero weight
to 1. The button is disabled when the filtered models have no eligible task in
common; its tooltip reports the number of shared tasks and models.

Scores are averaged within each model and task first, then combined
as `sum(weight * task_score) / sum(weight)`. Repeat counts and dataset sizes do not
change task importance. Runs with the same model name are grouped across providers,
service tiers and generation settings, matching `Group By: Model` in the Chart
tab. Provider names are listed under each model; individual run details retain
the provider and configuration. Cost uses each run's own provider pricing before
averaging within tasks.

Only models with a score on every enabled, positively weighted task
receive a rank. Incomplete models appear below them with task-count and
weighted coverage. Expand a model row to inspect task scores, contributions,
evaluated sample counts and individual runs. Known stopped/partial runs are
excluded. Different known input filenames or system prompts among the eligible
runs being averaged for the same model/task are flagged as ambiguous; narrow
filters before ranking them. Older artifacts may lack enough
metadata to verify dataset or prompt comparability.
Dataset filenames used by other models or excluded runs do not remove a model's
task coverage. Grouping by task name does not verify that different models used
identical datasets; use the run details and filters to choose comparable results.

The scatterplot compares weighted score with **estimated USD per 1,000
predictions**. Each run's estimated cost is divided by its recorded prediction
count and multiplied by 1,000. These rates are averaged within tasks, then combined
with the same task weights used for scoring. This describes a mix whose task
shares follow the chosen weights; it is not the total cost of the benchmark files.
Missing costs or prediction counts make the aggregate cost unknown rather than
zero or a partial average. Such models remain in the table but are omitted from
the numeric chart. Zero-cost estimates remain valid chart points. Select a point
with the mouse or keyboard to open its model breakdown.
The `Model names` switch adds the model name after each point's rank number
(for example, `#1 kimi-k3`). It defaults to off and is saved with the view and
included in share links. Labels are placed automatically around their points,
prioritizing the highest-ranked models and trying progressively more distant
positions to avoid other labels, point markers and chart edges. A connector line
appears when a label has to move farther from its point. Very dense plots can still
have overlaps when no free position is available.

Each model has a matching color and shape in the chart, model legend
and leaderboard table. These markers remain stable when weights, rankings or
filters change within the loaded catalogue. Hover or focus a point, legend entry
or model row to highlight its counterparts. Select a legend entry to open the
task breakdown. The legend contains only models plotted in the chart: incomplete
models and models with unknown cost appear only in the table. If no model can be
plotted, both the chart and legend are hidden.
The table emphasizes model names and scores, with secondary provider text,
alternating row backgrounds and matching score bars. Full configuration details
remain available through the individual runs in the expanded model row.

Weights and the metric persist in browser storage and the existing share URL.
Saved tasks that disappear from the source remain required until deselected.
`Export CSV` includes task contributions, coverage, pricing date and underlying
run references. Shared views require the same metrics source, and updated metrics
or catalogue prices can change the result. Composite confidence intervals are
not displayed because repeated runs may evaluate the same examples.

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
- when `agreement_clusters.json` is present, both `Same model` and `Cross-Model` also show similarity trees built from pairwise disagreement distances
- same-model trees cluster repeated runs of one provider/model on the same comparable task variant
- cross-model trees cluster one representative run per provider/model, and they are recomputed in the browser for the currently visible representative models so filters can redraw the clustering even when the full-group alpha row is hidden

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

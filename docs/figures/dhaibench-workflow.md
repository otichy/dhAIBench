# Figure: dhAIBench workflow

Files: [SVG](dhaibench-workflow.svg), [PDF](dhaibench-workflow.pdf),
[PNG preview](dhaibench-workflow.png), and [editable generator](generate_workflow.py).
Both publication formats use vector graphics; SVG text remains editable and PDF
fonts are embedded. The page is 7.5 × 10.5625 inches; use at approximately full
text width in a two-column paper, or on a dedicated figure page.

## Suggested caption

**End-to-end dhAIBench workflow.** Linguistic examples and a run configuration
are supplied to the Python benchmark runner through a browser-based command
builder or the command line. For each pending example, the runner constructs a
task prompt, queries the selected model, and parses the response. Step 3 is an
optional validation stage, with an explicit bypass when validation is disabled.
An external validator can normalize a prediction or request another attempt with
additional instructions or candidate labels. Retry limits and failure policies
determine whether a flagged result is retained or processing stops. Predictions
and provenance are saved. An optional resume branch returns pending IDs or
previously unclassified predictions to annotation. In step 4, the runner
calculates metrics against available gold labels and cross-run agreement, then
saves the results as JSON files, with optional plots. In step 5, the browser
dashboard loads those saved JSON files for interactive comparison and inspection.

## Scope and interpretation

- Validation checks the response contract and optional task constraints;
  evaluation against gold labels is a separate step. Gold labels are not shown
  as inputs to the model prompt.
- Dashed arrows denote auxiliary or optional paths: gold-label input, bypassing
  validation, resuming saved output, and continuing after a validation failure
  when the failure policy allows it.
- Missing gold labels prevent supervised label scoring; prediction and operational
  artifacts can still be saved. Calibration and explanation are configurable;
  span verification, confidence semantics and other capabilities vary by provider.
- Accepted annotations are saved during the run; the final evaluation occurs
  after processing. `--resume` reuses existing predictions by ID and processes
  pending IDs; adding `--unclassified` also retries saved unclassified rows.
  `--repeat_unclassified` can automate repeated passes until its stopping
  conditions are met. Metrics-only mode can repeat evaluation from saved
  predictions without model calls; this entry point is omitted for readability.
- Step 4 writes per-run `*__metrics.json`, `agreement_summary.json`, and
  `agreement_clusters.json`; step 5 loads those files. Metrics generation can be
  disabled with `--no_metrics`.
- The command builder and dashboard are distinct static browser interfaces.
  The Python runner executes model calls and generates benchmark artifacts.
- The dashboard supports filtering by task, model, tags and time. Comparison
  eligibility rules, including the exclusion of explicit temperature overrides
  from general comparisons, are described in the dashboard documentation.

## Rebuild

From the repository root, with `matplotlib` installed:

```sh
python docs/figures/generate_workflow.py
```

The generator checks that all node text fits inside its boxes before saving.
Content was checked against `benchmark_agent.py`, `docs/getting-started.md`,
`docs/gui.md`, `docs/validators.md`, `docs/outputs-and-metrics.md` and
`docs/metrics-dashboard.md`.

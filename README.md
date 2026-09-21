### 🔗 [Check our LLM benchmarks on linguistic annotation tasks](https://dhaibench.silent3.ff.cuni.cz/)
- If you want your linguistic annotation benchmarked, either send us the dataset ([ondrej.tichy@ff.cuni.cz](ondrej.tichy@ff.cuni.cz)) or use this tool.
- In either case, we would love to show the results as part of the results above to help others assess what is (im)possible with LLMs.

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
- Metrics artifacts are written under `data/metrics`, prompt logs under `data/logs`, and session logs under `data/logs/sessions`.
- Set `DHAIBENCH_DATA_ROOT` to move the default `data/` root.


### Jev classification through Requesty

Select Requesty and a `typesafe/jev-*` model (for reproducible runs, use a versioned model such as `typesafe/jev-1.13.0`). The GUI changes **System Prompt** to **Choice question** and displays a **Choices** editor. Enter the label and description in each row; use **+ Add choice** or **?** to add or remove rows. At least two unique, nonempty labels and descriptions are required. Include `unclassified` explicitly if the task permits abstention. The choices are a task definition, not labels inferred from evaluation answers.

The GUI automatically encodes choices in `--decision_criteria_b64`, a base64-encoded UTF-8 JSON object mapping labels to descriptions. Existing `--system_prompt` / `--system_prompt_b64` options supply the question. Requests use `REQUESTY_API_KEY` and the existing Requesty endpoint. The question goes into `response_format.questions.classification.instructions`; the example is a text user message. No system-role message is sent. The question and criteria are saved for browser reloads, metrics configuration imports, `--load_params`, and `--resume`.

Jev currently supports one example per request, optional few-shot examples, and the existing concurrent execution and evaluation. Prompt batching, external validators, timeout probes, explanations, CoT, token log probabilities, generation controls, and explicit cache controls are unavailable for this integration. The GUI disables incompatible controls. The CLI automatically leaves explanations empty and rejects unsupported controls.

The output `confidence` is the selected choice's probability, rather than Jev's distribution-summary `confidence`. Run metadata records `confidence_source=choice_probability`; original answers, including all probabilities and Jev's own confidence, are retained in response/parsed-answer logs. Token-level `labelProbability` remains empty. Jev cannot echo the target span: echo fields remain empty and metadata records `span_verification=unavailable`. Accuracy and confidence calibration use the existing evaluation pipeline. Jev cannot produce unrestricted normalization or lemma strings without a defined candidate inventory.

Requesty's decision interface is experimental: see the [Decisions documentation](https://docs.requesty.ai/features/decisions) and [TypeSafe confidence semantics](https://docs.typesafe.ai/confidence).

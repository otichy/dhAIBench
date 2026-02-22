# LLM Linguistic Classification Benchmark Agent

Python tooling for benchmarking large language models on linguistic classification tasks. The agent loads semicolon-delimited datasets, queries an OpenAI-compatible model for each example, and scores the predictions against gold labels. A companion browser-based configurator helps compose the command-line invocation.

- **End-to-end evaluation**: process one or many input CSVs, merge optional label files, gather predictions, explanations, confidences, and token usage, then export results plus metrics.
- **Prompt controls**: tune temperature, top-p, top-k, optional logprobs collection, chain-of-thought prompting, provider-specific reasoning/thinking effort and verbosity levels, few-shot demonstrations, and custom system prompts.
- **Provider-aware defaults**: look up API credentials from `.env` or the environment (OpenAI by default, other OpenAI-compatible endpoints supported via `--provider`).
- **Calibration utilities**: optionally fit reliability diagrams when `matplotlib` is available.
- **Metadata preservation**: the optional `info` column feeds the model extra guidance, and any additional columns are carried through to the output file.
- **GUI command builder**: `config_gui.html` generates CLI strings for the Python agent entirely client-side.

## Repository Layout

- `benchmark_agent.py`: main benchmarking script and evaluation logic.
- `config_gui.html`: static GUI for configuring runs and copying the resulting command. Pulls model lists
  from the generated `config_models.js`.
- `config_models.js`: auto-generated model catalog for the GUI (`python benchmark_agent.py --update-models`).
- `example_input.csv`: sample semicolon-separated dataset with embedded truth labels.
- `.env`: optional environment file (not tracked) for API keys such as `OPENAI_API_KEY`.

## Requirements

- Python 3.9+.
- Packages: [`openai`](https://pypi.org/project/openai/) (required), `matplotlib` (needed for confusion heatmaps and optional calibration plots). Install with:

```bash
python -m pip install openai matplotlib
```

The agent will prompt to install `matplotlib` the first time you request plotting (confusion heatmaps or calibration).

## Dataset Format

Input files must be semicolon-delimited CSVs with headers:

| Column        | Description                                            |
|---------------|--------------------------------------------------------|
| `ID`          | Unique example identifier.                             |
| `leftContext` | Text immediately preceding the node token.             |
| `node`        | Target word to classify.                               |
| `rightContext`| Text immediately following the node token.             |
| `info`        | (Optional) supplemental instructions or metadata.      |
| `truth`       | (Optional) gold label; omit when using `--labels`.     |

When `--labels` is provided, supply a second CSV with `ID;truth` columns. The script merges this file with the examples before scoring. Any extra columns are ignored by the prompt but carried through to the output CSV for convenience.

## Usage

1. Populate `.env` or export environment variables so the agent can locate your API key. By default it reads `OPENAI_API_KEY` (and `OPENAI_BASE_URL` when targeting non-default endpoints). For custom providers, the agent also detects `<PROVIDER>_API_KEY` and `<PROVIDER>_BASE_URL` automatically (for example `REQUESTY_API_KEY` / `REQUESTY_BASE_URL`). You can always override the variable names with `--api_key_var` and `--api_base_var`.
2. Prepare an input dataset (see `example_input.csv` for reference).
3. Run the benchmark (append as many CSVs as you like after `--input`; omit `--output` to use the automatic filenames):

```bash
python benchmark_agent.py ^
  --input data/input.csv data/input_extra.csv ^
  --output runs/ ^
  --model gpt-4o-mini ^
  --temperature 0.0 ^
  --top_p 1.0 ^
  --reasoning_effort medium ^
  --request_interval_ms 250 ^
  --system_prompt "You are a linguistic classifier..." ^
  --enable_cot ^
  --few_shot_examples 5 ^
  --calibration
```

On POSIX shells replace the PowerShell line-continuation (`^`) with `\`.

When `--output` is omitted, each input file produces a sibling result named `<input_basename>_out_<model>_<timestamp>.csv` (plus matching metrics and optional calibration artifacts).
When `--output` points to an existing CSV file, the run resumes: rows whose `ID` is already present in that file are skipped, and processing continues from the first missing `ID` in input order.

### System Prompts and Encoding

- `--system_prompt` accepts regular single-line strings or anything that your shell can quote safely.
- For multi-line prompts, Markdown tables, or text that mixes quotes/backslashes, prefer the base64 form: `--system_prompt_b64 <encoded>`. This avoids quoting issues on both PowerShell and POSIX shells.
- You can produce the encoded string with a tiny helper script (replace the triple-quoted text with your prompt):

  ```bash
  python - <<'PY'
  import base64

  prompt = """You are a meticulous linguistic classifier...
  (rest of your prompt here)
  """

  print(base64.b64encode(prompt.encode("utf-8")).decode())
  PY
  ```

- The GUI in `config_gui.html` chooses automatically: if your prompt fits on one line it emits `--system_prompt`, otherwise it switches to `--system_prompt_b64`. A hint below the System Prompt box reminds you that multi-line instructions will be encoded before reaching the CLI. The Python agent now understands both flags transparently.

### Reasoning/Thinking Effort Controls

Use provider-specific controls when you need tighter control over model "thinking" token budgets:

- `--reasoning_effort {low,medium,high,xhigh}` for GPT/OpenAI-style reasoning (`reasoning.effort` in the request payload).
- `--verbosity {low,medium,high}` for GPT output detail (`verbosity` in Chat Completions, `text.verbosity` in Responses API).
- `--thinking_level {low,medium,high}` for Gemini-style thinking (mapped to `reasoning_effort` on Gemini OpenAI-compatible targets).
- `--effort {low,medium,high,max}` for Claude-style effort (`effort` in the request payload).

These flags are optional and can be omitted to leave model defaults unchanged.
Use `--strict_control_acceptance` to fail examples when requested controls are rejected or stripped from the final successful request payload.

### Command Builder GUI

Open `config_gui.html` in any modern browser. Adjust the sliders, toggles, and text inputs to match your experiment, then copy the generated CLI command into your terminal. The page runs entirely locally and does not submit data over the network.

Model suggestions in the datalist are populated from `config_models.js`. Regenerate that catalog whenever your
provider offerings change:

```bash
python benchmark_agent.py --update-models
```

The command reads provider credentials from `.env` (or your environment), queries each `/models` endpoint, and writes
the catalog to `config_models.js`. Use `--models-providers` to update a subset or `--models-output` to control the path.

## Validation contract (optional)

Some tasks have a very large label space (e.g., lemmatization across a full lexicon), making it impractical to ship all valid labels in every prompt. The agent can instead delegate post-checks to an external **NDJSON validator** that can:

- accept a prediction (optionally normalizing the label),
- request a retry and provide a small `allowed_labels` candidate set,
- abort the run with a clear reason.

### How it works

- The agent starts the validator once and keeps it alive for the whole run.
- The agent sends one JSON object per attempt to the validator over stdin and expects one JSON object back on stdout.
- Validators **must write protocol messages to stdout only**; any logs must go to stderr.

### Example: lemmatization validator

This repository includes a reference validator at `validators/lemmatization_validator.py` (lexicon: one lemma per line, UTF-8). Example invocation:

```bash
python benchmark_agent.py ^
  --input data/input.csv ^
  --model gpt-4o-mini ^
  --validator_cmd validators/lemmatization_validator.py ^
  --validator_args "--lexicon data/lemmata.txt --max_distance 2 --max_suggestions 30"
```

#### Using `info` metadata (e.g., part-of-speech)

The agent forwards the dataset `info` column to the validator unchanged. The reference lemmatization validator can optionally use it to restrict candidates by POS:

- Put POS into `info` as e.g. `pos=NOUN` (or `part-of-speech:VERB`).
- Provide a lexicon with an optional POS in a second column, e.g. `lemma<TAB>POS` (also supports `lemma;POS` or `lemma POS` with `--lexicon_field_sep`).

Example:

```bash
python benchmark_agent.py ^
  --input data/input.csv ^
  --model gpt-4o-mini ^
  --validator_cmd validators/lemmatization_validator.py ^
  --validator_args "--lexicon data/lemmata_with_pos.tsv --lexicon_field_sep tab --use_pos"
```

### Relevant CLI flags

- `--request_interval_ms`: minimum delay between outgoing API requests in milliseconds (0 disables pacing).
- `--validator_cmd`: enable validation and retries driven by the validator.
- `--validator_args`: extra validator args as a single quoted string (supports quoting).
- `--validator_timeout`: per-request validator timeout (seconds).
- `--validator_prompt_max_candidates`: cap how many candidates are appended to the retry prompt.
- `--validator_prompt_max_chars`: cap how large the retry instruction can get.
- `--validator_exhausted_policy`: what to do if the validator keeps requesting retries but `--max_retries` is exhausted (`accept_blank_confidence` / `unclassified` / `error`).

When enabled, the output CSV gains two extra columns: `validatorStatus` and `validatorReason`. Each attempt in the JSON prompt log also includes `validator_request` and `validator_result` fields.

## Outputs

Running the agent creates:

- A semicolon-separated predictions file (`--output`) containing the original context, predicted label, explanation (when requested), and token usage statistics. Confidence is included when the model supplies a valid value; entries that violated the span contract omit it.
- A JSON metrics report (`<output_basename>_metrics.json`) with accuracy, macro F1, per-label precision/recall/F1, and a confusion matrix.
  The metrics JSON also includes `request_control_summary` with run-level acceptance/rejection counts for `reasoning_effort`, `thinking_level`, `effort`, and `verbosity` controls when used.
- A dual-panel confusion heatmap (`<output_basename>_confusion_heatmap.png`) showing absolute counts alongside row-normalized percentages.
- Optionally, a calibration plot (`<output_basename>_calibration.png`) summarizing confidence reliability.
- A JSON prompt log (`<output_basename>.log`) capturing every prompt/response attempt per example for auditability.

Logs streamed to stdout include prompt snapshots, raw responses, retries, and aggregate token totals to aid debugging.

## Tips

- Increase `--max_retries` or `--retry_delay` if your provider rate-limits requests.
- Use `--request_interval_ms` (for example `200` to `1000`) to proactively pace requests and reduce 429 bursts.
- Use `--no_explanation` to reduce token usage when explanations are unnecessary.
- Few-shot examples are drawn from the start of the dataset; place high-quality labeled instances there to guide the model.
- When targeting non-OpenAI services, ensure the endpoint is API-compatible and provide the correct base URL via `.env` or `--api_base_var`.
- `--temperature` and `--top_p` are optional; when omitted, they are not sent and the provider default is used.
- Use `--no-logprobs` when you do not need token-level probability estimates.

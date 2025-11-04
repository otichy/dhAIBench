# LLM Linguistic Classification Benchmark Agent

Python tooling for benchmarking large language models on linguistic classification tasks. The agent loads semicolon-delimited datasets, queries an OpenAI-compatible model for each example, and scores the predictions against gold labels. A companion browser-based configurator helps compose the command-line invocation.

- **End-to-end evaluation**: merge datasets with optional label files, gather predictions, explanations, confidences, and token usage, then export results plus metrics.
- **Prompt controls**: tune temperature, top-p, top-k, chain-of-thought prompting, few-shot demonstrations, and custom system prompts.
- **Provider-aware defaults**: look up API credentials from `.env` or the environment (OpenAI by default, other OpenAI-compatible endpoints supported via `--provider`).
- **Calibration utilities**: optionally fit reliability diagrams when `matplotlib` is available.
- **GUI command builder**: `config_gui.html` generates CLI strings for the Python agent entirely client-side.

## Repository Layout

- `benchmark_agent.py`: main benchmarking script and evaluation logic.
- `config_gui.html`: static GUI for configuring runs and copying the resulting command.
- `example_input.csv`: sample semicolon-separated dataset with embedded truth labels.
- `models.txt`: curated list of model identifiers for reference.
- `.env`: optional environment file (not tracked) for API keys such as `OPENAI_API_KEY`.

## Requirements

- Python 3.9+.
- Packages: [`openai`](https://pypi.org/project/openai/) (required), `matplotlib` (only when generating calibration plots). Install with:

```bash
python -m pip install openai matplotlib
```

The agent can prompt to install `matplotlib` the first time you enable calibration.

## Dataset Format

Input files must be semicolon-delimited CSVs with headers:

| Column        | Description                                            |
|---------------|--------------------------------------------------------|
| `ID`          | Unique example identifier.                             |
| `leftContext` | Text immediately preceding the node token.             |
| `node`        | Target word to classify.                               |
| `rightContext`| Text immediately following the node token.             |
| `truth`       | (Optional) gold label; omit when using `--labels`.     |

When `--labels` is provided, supply a second CSV with `ID;truth` columns. The script merges this file with the examples before scoring.

## Usage

1. Populate `.env` or export environment variables so the agent can locate your API key. By default it reads `OPENAI_API_KEY` (and `OPENAI_BASE_URL` when targeting non-default endpoints). You can override the variable names with `--api_key_var` and `--api_base_var`.
2. Prepare an input dataset (see `example_input.csv` for reference).
3. Run the benchmark:

```bash
python benchmark_agent.py ^
  --input data/input.csv ^
  --output runs/results.csv ^
  --model gpt-4o-mini ^
  --temperature 0.0 ^
  --top_p 1.0 ^
  --system_prompt "You are a linguistic classifier..." ^
  --enable_cot ^
  --few_shot_examples 5 ^
  --calibration
```

On POSIX shells replace the PowerShell line-continuation (`^`) with `\`.

### Command Builder GUI

Open `config_gui.html` in any modern browser. Adjust the sliders, toggles, and text inputs to match your experiment, then copy the generated CLI command into your terminal. The page runs entirely locally and does not submit data over the network.

## Outputs

Running the agent creates:

- A semicolon-separated predictions file (`--output`) containing the original context, predicted label, explanation (when requested), confidence, and token usage statistics.
- A JSON metrics report (`<output_basename>_metrics.json`) with accuracy, macro F1, per-label precision/recall/F1, and a confusion matrix.
- Optionally, a calibration plot (`<output_basename>_calibration.png`) summarizing confidence reliability.

Logs streamed to stdout include prompt snapshots, raw responses, retries, and aggregate token totals to aid debugging.

## Tips

- Increase `--max_retries` or `--retry_delay` if your provider rate-limits requests.
- Use `--no_explanation` to reduce token usage when explanations are unnecessary.
- Few-shot examples are drawn from the start of the dataset; place high-quality labeled instances there to guide the model.
- When targeting non-OpenAI services, ensure the endpoint is API-compatible and provide the correct base URL via `.env` or `--api_base_var`.

# GUI Command Builder

## Command Builder GUI

Open `config_gui.html` in a modern browser. The page runs locally and does not submit your dataset or prompts to a server.

Use it to configure:

- model and sampling settings
- prompt controls such as CoT and system prompt
- input, labels, and output paths
- plotting options
- validator path plus dedicated validator lexicon, max distance, retry increment, and max suggestions fields
- provider-specific controls that map to the CLI

The GUI generates a `python benchmark_agent.py ...` command that you can paste into a terminal.

In Metrics only mode, leave the existing output CSV field empty to generate a command that refreshes only `agreement_summary.json` and `agreement_clusters.json`.

## System Prompt Handling

Run and Run & Validate require a task-specific system prompt defining the classification task and allowed labels. Until one is entered, the GUI shows an error and disables copying the command. Resume recovers the saved prompt from the run artifacts; the CLI rejects resumes without a usable prompt. Metrics only does not require a prompt.

The GUI chooses the prompt flag automatically:

- single-line prompt -> `--system_prompt`
- multiline prompt -> `--system_prompt_b64`

That keeps generated commands portable across shells.

## Model Suggestions

The model datalist is populated from `config_models.js`.

Refresh it when provider offerings change:

```bash
python benchmark_agent.py --update-models
```

For the separate metrics browser, see [Metrics Dashboard](metrics-dashboard.md).

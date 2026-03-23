# GUI Command Builder

## Command Builder GUI

Open `config_gui.html` in a modern browser. The page runs locally and does not submit your dataset or prompts to a server.

Use it to configure:

- model and sampling settings
- prompt controls such as CoT and system prompt
- input, labels, and output paths
- plotting options
- provider-specific controls that map to the CLI

The GUI generates a `python benchmark_agent.py ...` command that you can paste into a terminal.

## System Prompt Handling

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

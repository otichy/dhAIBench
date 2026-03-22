# GUI And Metrics Dashboard

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

## Metrics Dashboard

The `web/` directory contains a static dashboard for browsing `*_metrics.json` artifacts.

### Local-Only Mode

1. Put your metrics files into `data/metrics/`.
2. Open `web/index.html` directly in a browser.
3. Use either `Open Metrics Folder` or `Open Metrics Files` in the dashboard.

Browsers require a user action before a local `file://` page can access files.

### Optional Server Mode

From repository root:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000/web/`.

In server mode the dashboard tries, in order:

1. `web/metrics-manifest.json`
2. directory discovery from `../data/metrics/`

`web/generate_metrics_manifest.py` is optional and can improve startup on larger collections.

## Folder-Local Notes

The dashboard folder also keeps a short local note in [web/README.md](../web/README.md).

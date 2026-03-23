# Documentation

This repository now uses a short root README plus versioned topic guides in `docs/`.

## Choose A Path

- New to the project: [Getting Started](getting-started.md)
- Need runnable commands: [Common Examples](examples.md)
- Working with non-default providers or auth: [Providers and Authentication](providers.md)
- Using the command builder: [GUI Command Builder](gui.md)
- Exploring benchmark results in the browser: [Metrics Dashboard](metrics-dashboard.md)
- Validating predictions against an external lexicon or rule set: [Validators](validators.md)
- Inspecting outputs and metrics artifacts: [Outputs and Metrics](outputs-and-metrics.md)
- Looking for the full flag surface: [CLI Reference](cli-reference.md)

## Maintenance

- Regenerate the CLI reference after flag changes with `python scripts/generate_cli_reference.py`.
- Keep the root [README](../README.md) focused on overview and quickstart.
- Prefer adding new operational detail here in `docs/` rather than growing the top-level README.

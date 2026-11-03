# Pricing tools

Run these commands from the repository root. The runner also uses
`pricing_catalog.py` to add missing model entries to the pricing catalog.

```bash
python scripts/pricing/update_model_prices.py --help
python scripts/pricing/update_model_prices.py
python scripts/pricing/update_model_prices.py --providers openai google
node scripts/pricing/audit_metric_pricing.cjs
```

The updater reads `config_models.js`, fetches provider pricing, applies the
reviewed entries in `benchmark_price_overrides.json`, and writes `config_prices.js`
plus its dashboard copy in `web/config_prices.js`. The pricing updater uses the
Python standard library. Review generated catalog changes before committing them.

The audit uses Node.js and the dashboard pricing engine to check local
`data/metrics/` files. Pass an optional output JSON path to save its findings.

Additional tools operate on local benchmark data:

- `analyze_output_model_pricing.py`: writes output cost analyses and optional
  Matplotlib figures to `scripts/analysis/model_pricing/`.
- `fetch_public_pricing.py`: downloads a dated source snapshot into
  `scripts/analysis/pricing_sources/`.
- `publish_pricing_report.py`: rebuilds a historical reconciliation report from
  existing audit JSON files in `scripts/analysis/`; it requires those local files
  and updates the local reconciliation details.

Analysis inputs, downloaded sources, reports, and other support helpers stay
local. Only `scripts/pricing/` is shared from the scripts directory.

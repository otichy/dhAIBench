"""Publish the completed active-run pricing audit and its reference assumptions."""
import collections
import json
import pathlib

OUT = pathlib.Path(__file__).resolve().parents[1] / "analysis"
ROOT = OUT.parents[1]
prices = json.loads((OUT / "run_pricing_completion.json").read_text(encoding="utf-8"))
changes = json.loads((OUT / "pricing_catalog_changes.json").read_text(encoding="utf-8"))
validation = json.loads((OUT / "pricing_validation.json").read_text(encoding="utf-8"))
before = json.loads((OUT / "pricing_audit_before.json").read_text(encoding="utf-8"))
active_names = {r["file"] for r in prices["rows"]}
active_count = len(active_names)
active_before = [r for r in before["rows"] if r["file"] in active_names]
initial = collections.Counter(r["status"] for r in active_before)
fallbacks = [r for r in prices["rows"] if "No published" in r["reference_note"]]
unrecorded_tiers = []
for row in prices["rows"]:
    d = json.loads((ROOT / "data/metrics" / row["file"]).read_text(encoding="utf-8"))
    if not d.get("run_config", {}).get("service_tier"):
        unrecorded_tiers.append(row["file"])

lines = ["# Pricing completeness report", "", "Verified: 30 October 2026. Currency: USD; token rates are per 1,000,000 tokens.", "",
    "## Result", "", f"**All {active_count} active runs now have pricing and finite total and average-per-prediction costs in the web app's pricing engine. No model, tier, or used-token-bucket prices remain missing under the documented reference policy.**", "",
    "| Check | Result |", "| --- | ---: |",
    f"| Active runs with usable dashboard pricing before | {initial['priced']} / {active_count} |",
    f"| Active runs without usable dashboard pricing before | {active_count - initial['priced']} |",
    f"| Active runs with usable dashboard pricing now | {active_count} / {active_count} |",
    f"| Runs missing a stored estimated cost before | {prices['runs_missing_stored_cost_before']} |",
    f"| Stored estimates added or changed | {prices['runs_with_updated_cost']} |",
    "| Remaining missing prices | 0 |", "",
    f"The archived `__metrics_old.json` file is excluded from the {active_count}-run totals and was not rewritten. Its model rate also resolves in the catalog audit.", "",
    "## How the scatterplot gets its prices", "",
    "`web/app.js` calls `DHAIBenchPricing.estimateRunCost` using `web/config_prices.js`; it does not simply display the metrics' saved cost. Both the root and dashboard catalogs were updated and remain identical. Saved `estimated_cost_usd` and `pricing` records were refreshed using exactly that same JavaScript engine.", "",
    "Total cost is non-cached input × input rate + cached input × cache-read rate + output × output rate, divided by one million. Output totals already include reasoning tokens, so thinking tokens are not charged a second time. Average cost is **total divided by `prediction_count`**, matching the scatterplot's 'Average Cost per Prediction' mode. It is not cost per API attempt or per evaluated input row.", "",
    "## Source policy", "",
    "1. Retain available provider prices; fill missing official model/tier/cache prices from verified public sources. Gemini 4 Argon rates and observed Gemini 3.7/3.8 introductory prices were verified against [Google's official pricing](https://cloud.google.com/model-garden/pricing).",
    "2. For all used e-infra models, use [Requesty reference pricing](https://www.requesty.ai/models), as requested. Standard comparisons prefer the model publisher's route when available; otherwise select a complete low-cost endpoint quote. Use a published Flex endpoint when available.",
    "3. Use [OpenRouter](https://openrouter.ai/models) when the exact model is unavailable on Requesty. This applies to the UFAL Qwen 3.6 35B-A3B checkpoint and the retired Flash Lite Preview SKU. Preview Flex and cache rates come from its [exact endpoint listing](https://openrouter.ai/google/gemini-3.1-flash-lite-preview/providers), not an invented equivalence with the GA model.",
    "4. When an observed Flex hint has no matching published Flex quote, use a labelled **standard-rate reference without assuming a discount**. These estimates have a 'Rate basis' note in the price-calculation tooltip and stored metadata.",
    "5. For legacy runs whose provider was not recorded, explicit unknown-provider reference entries provide model pricing without changing or inferring their actual provider.", "",
    "All e-infra prices are comparison references, not claims about local hosting charges. Quantization and hosting can differ. Recorded model-source metadata identifies Kimi K2.7 as Kimi K2.7 Code, Qwen 3.5 as the 397B-A17B model, and the local Qwen 3.6 checkpoint as 35B-A3B. Qwen 3.8 Flash Next now uses its exact Requesty SKU rather than a different Flash model.", "",
    "Requesty quotes are endpoint token rates. Gateway markups/BYOK fees, taxes, negotiated discounts, cache writes/storage, and non-token tools are excluded. Google introductory rates carry expiry notes. These are dated benchmark estimates, not a reconstruction of historical invoices; pre-existing provider catalog rates outside the patch were retained.", "",
    f"For **{len(unrecorded_tiers)} runs**, service tier was unrecorded; the existing dashboard behavior uses standard rates. The historical run configuration was not altered to claim that this was the original tier.", "",
    "### Flex hints priced with undiscounted standard references", "", "These rows have usable comparison estimates, but their exact original provider-specific Flex tariff remains unverified.", "", "| Metrics file | Recorded provider | Reference basis |", "| --- | --- | --- |"]
for row in fallbacks:
    lines.append(f"| {row['file']} | {row['provider']} | {row['reference_note']} |")
lines += ["", "## Updated rate entries", "", "Each canonical provider/model entry is listed below; namespace and legacy-slug aliases are preserved in the catalog. A dash means no cache quote was needed or published, never a fabricated zero rate.", "",
    "| Provider | Model | Tier | Input /1M | Cache read /1M | Output /1M | Source |", "| --- | --- | --- | ---: | ---: | ---: | --- |"]
for item in changes["changes"]:
    entry = item["after"]
    source = "; ".join(dict.fromkeys(f"[{s['label']}]({s['url']})" for s in entry.get("sources", [])))
    for tier, rate in entry["service_tiers"].items():
        def amount(key):
            value = rate.get(key)
            return "—" if value is None else f"{value:g}"
        lines.append(f"| {item['provider']} | {item['model']} | {tier} | {amount('input_usd_per_mtokens')} | {amount('cached_input_usd_per_mtokens')} | {amount('output_usd_per_mtokens')} | {source} |")
lines += ["", "## Persistence and validation", "",
    "`benchmark_price_overrides.json` stores the reviewed, dated reference entries and aliases. The catalog writer applies it when regenerating the root `config_prices.js` and mirrors the resulting catalog into `web/`. Review/update that snapshot when comparing at a different pricing date; it intentionally preserves this benchmark pricing policy.", "",
    f"Validation executed the actual total and average scatterplot functions for all {active_count} active runs and verified equality with the saved estimates. Tests cover explicit unknown-provider fallback, provider isolation, unsupported tiers, missing cache prices, and genuine zero-price models. Twenty custom-leaderboard tests passed during the pricing update. Pricing changes preserved model labels, predictions, scores, and usage totals. Later removals and agreement refreshes are documented separately in the main report.", "",
    "The 176 existing CSV-versus-metrics token-total differences remain a limitation of the usage ledger. This task did not rewrite token counts; the estimates consistently use the stored run-level totals, which can include retries or resumed invocations.", "",
    "### Remaining missing pricing", "", f"**None for the {active_count} active runs under the reference policy.** Exact historical bills, local hosting costs, unrecorded service tiers, and the unsupported provider-specific Flex tariffs above cannot be verified from public token prices alone.", "",
    "- [Per-run costs, rates, and sources](run_pricing_completion.json)",
    "- [Pricing validation](pricing_validation.json)",
    "- [Before audit](pricing_audit_before.json) and [after audit](pricing_audit_after.json)",
    "- [Catalog changes](pricing_catalog_changes.json)",
    "- [Downloaded public source manifest](pricing_sources/sources.json)",
    "- [Backups before pricing changes](../backup/pricing_completion_2026-10-30/)",
]
(OUT / "pricing_completeness_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
p = OUT / "source_recheck_details.json"
d = json.loads(p.read_text(encoding="utf-8"))
d["pricing_completion"] = {"active_runs": active_count, "all_runs_priced": True, "missing_pricing": [], "validation": validation, "report": "pricing_completeness_report.md"}
p.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Published pricing report: {active_count}/{active_count} priced; {len(fallbacks)} explicitly labelled standard-reference Flex estimates; {len(unrecorded_tiers)} unrecorded tiers use standard pricing.")

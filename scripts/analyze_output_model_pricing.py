from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


OFFICIAL_PRICE_OVERRIDES: dict[tuple[str, str], dict[str, Any]] = {
    # Official Google pricing pages checked 2026-05-26:
    # Gemini API pricing: https://ai.google.dev/gemini-api/docs/pricing
    # Vertex AI pricing: https://cloud.google.com/vertex-ai/generative-ai/pricing
    ("google", "gemini-3.5-flash"): {
        "service_tiers": {
            "standard": {
                "input_usd_per_mtokens": 1.50,
                "cached_input_usd_per_mtokens": 0.15,
                "output_usd_per_mtokens": 9.00,
            },
            "flex": {
                "input_usd_per_mtokens": 0.75,
                "cached_input_usd_per_mtokens": 0.08,
                "output_usd_per_mtokens": 4.50,
            },
            "batch": {
                "input_usd_per_mtokens": 0.75,
                "cached_input_usd_per_mtokens": 0.15,
                "output_usd_per_mtokens": 4.50,
            },
        },
        "sources": [
            {"label": "Gemini API Pricing", "url": "https://ai.google.dev/gemini-api/docs/pricing"},
            {
                "label": "Vertex AI Pricing",
                "url": "https://cloud.google.com/vertex-ai/generative-ai/pricing",
            },
        ],
        "notes": ["Manual override because the current catalog updater did not parse Gemini 3.5 Flash."],
    },
    # Requesty model pages checked 2026-05-27. These cover Requesty-backed outputs
    # whose base model names are present in run metadata but absent from config_prices.js.
    ("requesty", "claude-haiku-4-5"): {
        "service_tiers": {
            "standard": {
                "input_usd_per_mtokens": 1.00,
                "cached_input_usd_per_mtokens": 0.10,
                "output_usd_per_mtokens": 5.00,
            }
        },
        "sources": [
            {"label": "Requesty Model Page", "url": "https://www.requesty.ai/models/anthropic/claude-haiku-4-5"},
            {"label": "Anthropic Claude Pricing", "url": "https://platform.claude.com/docs/en/about-claude/pricing"},
        ],
        "notes": ["Anthropic direct route on Requesty."],
    },
    ("requesty", "claude-sonnet-4-6"): {
        "service_tiers": {
            "standard": {
                "input_usd_per_mtokens": 3.00,
                "cached_input_usd_per_mtokens": 0.30,
                "output_usd_per_mtokens": 15.00,
            }
        },
        "sources": [
            {"label": "Requesty Model Page", "url": "https://www.requesty.ai/models/anthropic/claude-sonnet-4-6"},
            {"label": "Anthropic Claude Pricing", "url": "https://platform.claude.com/docs/en/about-claude/pricing"},
        ],
        "notes": ["Anthropic direct route on Requesty."],
    },
    ("requesty", "claude-opus-4-6"): {
        "service_tiers": {
            "standard": {
                "input_usd_per_mtokens": 5.00,
                "cached_input_usd_per_mtokens": 0.50,
                "output_usd_per_mtokens": 25.00,
            }
        },
        "sources": [
            {"label": "Requesty Model Page", "url": "https://www.requesty.ai/models/anthropic/claude-opus-4-6"},
            {"label": "Anthropic Claude Pricing", "url": "https://platform.claude.com/docs/en/about-claude/pricing"},
        ],
        "notes": ["Anthropic direct route on Requesty."],
    },
    ("requesty", "deepseek-v3.2"): {
        "service_tiers": {
            "standard": {
                "input_usd_per_mtokens": 0.27,
                "cached_input_usd_per_mtokens": 0.13,
                "output_usd_per_mtokens": 0.40,
            }
        },
        "sources": [{"label": "Requesty Model Page", "url": "https://www.requesty.ai/models/novita/deepseek-deepseek-v3-2"}],
        "notes": ["Novita route inferred from output basename."],
    },
    ("requesty", "glm-4.7"): {
        "service_tiers": {
            "standard": {
                "input_usd_per_mtokens": 0.40,
                "cached_input_usd_per_mtokens": None,
                "output_usd_per_mtokens": 2.00,
            }
        },
        "sources": [{"label": "Requesty Model Page", "url": "https://www.requesty.ai/solution/llm-routing/models/nebius/zai-org-glm-4-7"}],
        "notes": ["Nebius Z AI route inferred from output basename."],
    },
    ("requesty", "gpt-5.4-pro"): {
        "service_tiers": {
            "standard": {
                "input_usd_per_mtokens": 30.00,
                "cached_input_usd_per_mtokens": 30.00,
                "output_usd_per_mtokens": 180.00,
            }
        },
        "sources": [{"label": "Requesty Model Page", "url": "https://www.requesty.ai/models/openai/gpt-5-4-pro"}],
        "notes": ["OpenAI route on Requesty."],
    },
    ("requesty", "kimi-k2.5"): {
        "service_tiers": {
            "standard": {
                "input_usd_per_mtokens": 0.60,
                "cached_input_usd_per_mtokens": None,
                "output_usd_per_mtokens": 3.00,
            }
        },
        "sources": [{"label": "Requesty Models Page", "url": "https://www.requesty.ai/models/moonshot"}],
        "notes": ["Moonshot AI route on Requesty."],
    },
}


@dataclass
class ModelUse:
    provider: str
    model: str
    output_count: int
    metrics_count: int
    service_tiers: Counter[str]


def load_catalog(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return json.loads(text[text.index("{") : text.rfind("}") + 1])


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_run_metadata(log_path: Path) -> dict[str, Any] | None:
    if not log_path.exists():
        return None
    try:
        with log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("record_type") == "run_metadata":
                    return record
    except Exception:
        return None
    return None


def fallback_from_basename(basename: str) -> tuple[str, str]:
    parts = basename.split("__")
    if len(parts) >= 3:
        return parts[1], parts[2]
    return "", basename


def normalize_provider(provider: str | None) -> str:
    provider = (provider or "").strip()
    if provider == "einfra":
        return "e-infra"
    if provider == "vertex":
        return "google"
    return provider


def canonical_model(provider: str, model: str) -> str:
    model = model.strip()
    if provider == "google":
        model = model.removeprefix("models/")
        aliases = {
            "gemini3flashpreview": "gemini-3-flash-preview",
            "gemini31propreview": "gemini-3.1-pro-preview",
            "gemini31flashlitepreview": "gemini-3.1-flash-lite-preview",
            "gemini35flash": "gemini-3.5-flash",
        }
        return aliases.get(model, model)
    return model


def config_from_artifact(root: Path, basename: str) -> tuple[str, str, str, bool]:
    metrics_path = root / "data" / "metrics" / f"{basename}__metrics.json"
    log_path = root / "data" / "logs" / f"{basename}.log"
    metrics = read_json(metrics_path) if metrics_path.exists() else None
    metadata = read_run_metadata(log_path) if metrics is None else None

    source = metrics or metadata or {}
    config = source.get("run_config", {}) or {}
    details = source.get("model_details", {}) or {}

    provider = normalize_provider(config.get("provider") or details.get("provider"))
    model = config.get("model") or details.get("model_requested") or details.get("model_for_requests")
    service_tier = config.get("service_tier") or "standard"

    if not model:
        provider, model = fallback_from_basename(basename)
        provider = normalize_provider(provider)

    model = canonical_model(provider, str(model))
    return provider, model, str(service_tier), bool(metrics)


def collect_model_uses(root: Path) -> list[ModelUse]:
    uses: dict[tuple[str, str], ModelUse] = {}
    output_basenames = sorted(path.stem for path in (root / "data" / "output").glob("*.csv"))
    for basename in output_basenames:
        provider, model, service_tier, has_metrics = config_from_artifact(root, basename)
        key = (provider, model)
        if key not in uses:
            uses[key] = ModelUse(provider, model, 0, 0, Counter())
        uses[key].output_count += 1
        uses[key].metrics_count += int(has_metrics)
        uses[key].service_tiers[service_tier] += 1

    return sorted(uses.values(), key=lambda use: (use.provider, use.model))


def candidate_model_keys(provider: str, model: str) -> list[str]:
    keys = [model]
    if provider == "google" and not model.startswith("models/"):
        keys.append(f"models/{model}")
        keys.append(model)
    if model.startswith("models/"):
        keys.append(model.removeprefix("models/"))
    slug_aliases = {
        "gpt5": "gpt-5",
        "gpt5mini": "gpt-5-mini",
        "gpt51": "gpt-5.1",
        "gpt52pro": "gpt-5.2-pro",
    }
    if model in slug_aliases:
        keys.append(slug_aliases[model])
    return list(dict.fromkeys(keys))


def resolve_entry(catalog: dict[str, Any], provider: str, model: str) -> tuple[dict[str, Any] | None, str | None]:
    override = OFFICIAL_PRICE_OVERRIDES.get((provider, model))
    if override is not None:
        return override, model
    models = ((catalog.get("providers") or {}).get(provider) or {}).get("models") or {}
    if provider == "google":
        vertex_models = ((catalog.get("providers") or {}).get("vertex") or {}).get("models") or {}
        models = {**models, **vertex_models}
    visited: set[str] = set()
    for key in candidate_model_keys(provider, model):
        current = key
        while current and current not in visited:
            visited.add(current)
            entry = models.get(current)
            if entry is None:
                break
            ref = entry.get("pricing_ref")
            if ref:
                current = str(ref)
                continue
            return entry, current
    return None, None


def tiers_to_emit(entry: dict[str, Any] | None) -> list[tuple[str, str]]:
    tiers = (entry or {}).get("service_tiers") or {}
    emitted = [(tier, "") for tier in ("standard", "flex") if tier in tiers]
    if emitted:
        return emitted
    return [("", "missing_standard_or_flex_tier")]


def price_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0:
        return None
    return number


def rows_for_uses(catalog: dict[str, Any], uses: list[ModelUse]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for use in uses:
        entry, pricing_key = resolve_entry(catalog, use.provider, use.model)
        for tier, tier_note in tiers_to_emit(entry):
            tier_prices = ((entry or {}).get("service_tiers") or {}).get(tier) or {}
            input_price = price_value(tier_prices.get("input_usd_per_mtokens"))
            cached_input_price = price_value(tier_prices.get("cached_input_usd_per_mtokens"))
            output_price = price_value(tier_prices.get("output_usd_per_mtokens"))
            has_price = input_price is not None or cached_input_price is not None or output_price is not None
            notes = []
            if entry is None:
                notes.append("missing_pricing_entry")
            elif not has_price:
                notes.append(entry.get("reason") or "no_usable_price_values")
            if tier and tier not in use.service_tiers:
                notes.append("tier_not_observed_in_outputs")
            if tier_note:
                notes.append(tier_note)
            rows.append(
                {
                    "provider": use.provider,
                    "model": use.model,
                    "pricing_key": pricing_key or "",
                    "service_tier_used_for_price": tier,
                    "observed_service_tiers": "; ".join(f"{tier}:{count}" for tier, count in use.service_tiers.most_common()),
                    "output_count": use.output_count,
                    "metrics_count": use.metrics_count,
                    "input_usd_per_mtokens": input_price,
                    "cached_input_usd_per_mtokens": cached_input_price,
                    "output_usd_per_mtokens": output_price,
                    "priced": has_price,
                    "notes": " | ".join(str(note) for note in notes if note),
                }
            )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "provider",
        "model",
        "pricing_key",
        "service_tier_used_for_price",
        "observed_service_tiers",
        "output_count",
        "metrics_count",
        "input_usd_per_mtokens",
        "cached_input_usd_per_mtokens",
        "output_usd_per_mtokens",
        "priced",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) < 0.01:
            return f"{value:.4f}"
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def display_label(row: dict[str, Any]) -> str:
    tier = row.get("service_tier_used_for_price")
    prefix = "◎ " if row.get("provider") == "e-infra" else ""
    return f"{prefix}{row['model']} [{tier or 'unpriced'}]"


def priced_rows_for_plot(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priced = [
        row
        for row in rows
        if row["priced"] and row["input_usd_per_mtokens"] is not None and row["output_usd_per_mtokens"] is not None
    ]
    tier_rank = {"flex": 0, "standard": 1}
    group_totals = defaultdict(float)
    for row in priced:
        key = (row["provider"], row["model"])
        group_totals[key] = max(
            group_totals[key],
            float(row["output_usd_per_mtokens"] or 0) + float(row["input_usd_per_mtokens"] or 0),
        )
    priced.sort(
        key=lambda row: (
            group_totals[(row["provider"], row["model"])],
            row["provider"],
            row["model"],
            tier_rank.get(str(row["service_tier_used_for_price"]), 99),
        )
    )
    return priced


def plot_prices(rows: list[dict[str, Any]], path: Path, *, landscape: bool = False) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    priced = priced_rows_for_plot(rows)

    height = 9.5 if landscape else max(8, len(priced) * 0.34)
    width = 18 if landscape else 13
    bar_height = 0.44 if landscape else 0.58
    label_fontsize = 7 if landscape else 8
    value_fontsize = 6 if landscape else 7
    sum_fontsize = 5.5 if landscape else 6
    fig, ax = plt.subplots(figsize=(width, height))
    y_positions = list(range(len(priced)))
    input_prices = [float(row["input_usd_per_mtokens"] or 0) for row in priced]
    output_prices = [float(row["output_usd_per_mtokens"] or 0) for row in priced]

    tier_colors = {
        "standard": ("#B45F24", "#3F70AE"),
        "flex": ("#D9953D", "#6E9ED6"),
    }
    for tier in ("standard", "flex"):
        positions = [pos for pos, row in zip(y_positions, priced) if row["service_tier_used_for_price"] == tier]
        if not positions:
            continue
        tier_output = [float(priced[pos]["output_usd_per_mtokens"] or 0) for pos in positions]
        tier_input = [float(priced[pos]["input_usd_per_mtokens"] or 0) for pos in positions]
        output_color, input_color = tier_colors[tier]
        ax.barh(positions, tier_output, height=bar_height, color=output_color, label=f"{tier} output")
        ax.barh(positions, tier_input, left=tier_output, height=bar_height, color=input_color, label=f"{tier} input")

    for pos, row in zip(y_positions, priced):
        input_price = float(row["input_usd_per_mtokens"] or 0)
        output_price = float(row["output_usd_per_mtokens"] or 0)
        total_price = output_price + input_price
        if output_price > 0.09:
            ax.text(output_price / 2, pos, f"out {fmt(output_price)}", va="center", ha="center", fontsize=value_fontsize, color="white")
        else:
            ax.text(output_price * 1.06 if output_price else 0.032, pos, f"out {fmt(output_price)}", va="center", ha="left", fontsize=value_fontsize)
        if input_price > 0.09:
            ax.text(output_price + (input_price / 2), pos, f"in {fmt(input_price)}", va="center", ha="center", fontsize=value_fontsize, color="white")
        else:
            ax.text(total_price * 1.05, pos, f"in {fmt(input_price)}", va="center", ha="left", fontsize=value_fontsize)
        ax.text(total_price * 1.05, pos + (0.2 if landscape else 0.22), f"sum {fmt(total_price)}", va="center", ha="left", fontsize=sum_fontsize, color="#555555")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([display_label(row) for row in priced], fontsize=label_fontsize)
    ax.set_xlabel("Stacked USD per 1M tokens: output first, then input (logarithmic scale)")
    ax.set_title("Configured prices for models with benchmark outputs - stacked, logarithmic x-axis")
    ax.set_xscale("log")
    ax.set_xlim(0.03, max(out + inp for out, inp in zip(output_prices, input_prices)) * 1.55)
    ax.grid(axis="x", alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="none", edgecolor="none"))
    labels.append("◎ open source")
    ax.legend(handles, labels, frameon=False)
    fig.text(0.5, 0.01, "Provider names are omitted from y-axis labels.", ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_prices_vertical_axis(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    priced = priced_rows_for_plot(rows)
    x_positions = list(range(len(priced)))
    input_prices = [float(row["input_usd_per_mtokens"] or 0) for row in priced]
    output_prices = [float(row["output_usd_per_mtokens"] or 0) for row in priced]
    tier_colors = {
        "standard": ("#B45F24", "#3F70AE"),
        "flex": ("#D9953D", "#6E9ED6"),
    }

    fig, ax = plt.subplots(figsize=(18, 10))
    for tier in ("standard", "flex"):
        positions = [pos for pos, row in zip(x_positions, priced) if row["service_tier_used_for_price"] == tier]
        if not positions:
            continue
        tier_output = [float(priced[pos]["output_usd_per_mtokens"] or 0) for pos in positions]
        tier_input = [float(priced[pos]["input_usd_per_mtokens"] or 0) for pos in positions]
        output_color, input_color = tier_colors[tier]
        ax.bar(positions, tier_output, width=0.72, color=output_color, label=f"{tier} output")
        ax.bar(positions, tier_input, bottom=tier_output, width=0.72, color=input_color, label=f"{tier} input")

    for pos, row in zip(x_positions, priced):
        input_price = float(row["input_usd_per_mtokens"] or 0)
        output_price = float(row["output_usd_per_mtokens"] or 0)
        total_price = output_price + input_price
        ax.text(pos, total_price * 1.08, fmt(total_price), ha="center", va="bottom", fontsize=6, color="#555555")

    ax.set_xticks(x_positions)
    ax.set_xticklabels([display_label(row) for row in priced], rotation=58, ha="right", fontsize=7)
    ax.set_ylabel("Stacked USD per 1M tokens: output first, then input (logarithmic scale)")
    ax.set_title("Configured prices for models with benchmark outputs - vertical price axis")
    ax.set_yscale("log")
    ax.set_ylim(0.03, max(out + inp for out, inp in zip(output_prices, input_prices)) * 1.9)
    ax.grid(axis="y", alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="none", edgecolor="none"))
    labels.append("â—Ž open source")
    ax.legend(handles, labels, frameon=False, ncol=3, loc="upper left")
    fig.text(0.5, 0.01, "Provider names are omitted from x-axis labels.", ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_markdown(rows: list[dict[str, Any]], catalog: dict[str, Any], path: Path) -> None:
    priced = [row for row in rows if row["priced"]]
    unpriced = [row for row in rows if not row["priced"]]
    lines = [
        "# Model Pricing For Existing Outputs",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Pricing catalog updated_at: {catalog.get('updated_at', '')}",
        "",
        "Prices are from the refreshed local `config_prices.js` catalog and are shown as USD per 1M tokens. The chart emits both `standard` and `flex` prices whenever those tiers are configured, even if only one tier appears in the existing outputs. Bars are stacked as output price first, then input price, and the x-axis is logarithmic.",
        "",
        "Google API and Vertex AI Gemini outputs are displayed together as `google` and merged when the underlying model name matches. Gemini 3.5 Flash is added as a sourced manual override because the refreshed catalog did not yet parse that model from the official pricing pages.",
        "",
        "Online checks for the Google missing-price cases used the official Gemini API and Vertex AI pricing pages. Gemini 3.5 Flash has usable official per-token prices; the refreshed catalog still reports no compatible per-token pricing for Gemini 3.1 Flash Lite Preview and Gemini 3 Pro Preview.",
        "",
        "Requesty overrides were added from Requesty model pages for Claude Haiku 4.5, Claude Sonnet 4.6, Claude Opus 4.6, Novita DeepSeek V3.2, Nebius GLM-4.7, OpenAI GPT-5.4 Pro, and Moonshot Kimi K2.5. Claude prices were cross-checked against Anthropic's Claude pricing page.",
        "",
        "## Files",
        "",
        "- `model_prices_for_outputs.png`: stacked output/input price chart for priced models",
        "- `model_prices_for_outputs_horizontal.png`: landscape-oriented horizontal version of the same chart",
        "- `model_prices_for_outputs_vertical_axis.png`: bars arranged horizontally with price on the vertical axis",
        "- `model_prices_for_outputs.csv`: all provider/model pairs found in outputs, including unpriced models",
        "",
        "## Coverage",
        "",
        f"- Provider/model/tier rows represented: {len(rows)}",
        f"- Rows with usable configured prices: {len(priced)}",
        f"- Rows missing usable prices: {len(unpriced)}",
        "",
        "## Missing Or Unusable Prices",
        "",
    ]
    if unpriced:
        lines.extend(["| provider | model | outputs | observed tiers | note |", "|---|---|---:|---|---|"])
        for row in unpriced:
            lines.append(
                f"| {row['provider']} | {row['model']} | {row['output_count']} | {row['observed_service_tiers']} | {row['notes']} |"
            )
    else:
        lines.append("No missing prices.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "data" / "analysis" / "model_pricing"
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog(root / "config_prices.js")
    uses = collect_model_uses(root)
    rows = rows_for_uses(catalog, uses)
    write_csv(rows, out_dir / "model_prices_for_outputs.csv")
    write_markdown(rows, catalog, out_dir / "README.md")
    plot_prices(rows, out_dir / "model_prices_for_outputs.png")
    plot_prices(rows, out_dir / "model_prices_for_outputs_horizontal.png", landscape=True)
    plot_prices_vertical_axis(rows, out_dir / "model_prices_for_outputs_vertical_axis.png")
    print(f"Wrote pricing analysis to {out_dir}")
    print(f"Provider/model/tier rows: {len(rows)}; priced: {sum(bool(row['priced']) for row in rows)}; unpriced: {sum(not bool(row['priced']) for row in rows)}")


if __name__ == "__main__":
    main()

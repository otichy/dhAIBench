from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


TASK_ROWS = 115
START_DATE = "2026-05-14"
END_DATE = "2026-05-19"


@dataclass
class Run:
    basename: str
    timestamp: str
    provider: str | None
    model: str | None
    temperature: float | None
    output_rows: int | None
    metrics_total_examples: int | None
    accuracy: float | None
    macro_f1: float | None
    cohen_kappa: float | None
    has_metrics: bool
    has_log: bool
    status: str
    fail_reason: str


def parse_timestamp(name: str) -> str:
    match = re.search(r"__(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})(?:__metrics)?", name)
    if not match:
        return ""
    return match.group(1)


def timestamp_in_window(timestamp: str) -> bool:
    if not timestamp:
        return False
    day = timestamp[:10]
    return START_DATE <= day <= END_DATE


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


def count_output_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = handle.readline()
            delimiter = ";" if header.count(";") >= header.count(",") else ","
            handle.seek(0)
            return sum(1 for _ in csv.DictReader(handle, delimiter=delimiter))
    except Exception:
        return None


def read_predictions(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = handle.readline()
        delimiter = ";" if header.count(";") >= header.count(",") else ","
        handle.seek(0)
        reader = csv.DictReader(handle, delimiter=delimiter)
        predictions: dict[str, str] = {}
        for row in reader:
            example_id = str(row.get("ID") or row.get("id") or "").strip()
            prediction = str(row.get("prediction") or "").strip()
            if example_id:
                predictions[example_id] = prediction
        return predictions


def cohen_kappa_between(left: list[str], right: list[str]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    observed = sum(1 for a, b in zip(left, right) if a == b) / len(left)
    labels = set(left) | set(right)
    expected = 0.0
    for label in labels:
        left_rate = sum(1 for value in left if value == label) / len(left)
        right_rate = sum(1 for value in right if value == label) / len(right)
        expected += left_rate * right_rate
    if math.isclose(1.0, expected):
        return 1.0 if math.isclose(1.0, observed) else None
    return (observed - expected) / (1 - expected)


def pairwise_run_agreement(root: Path, runs: list[Run]) -> dict[str, Any]:
    valid_runs = [run for run in runs if run.status == "valid"]
    if len(valid_runs) < 2:
        return {
            "pair_count": 0,
            "overlap_mean": None,
            "agreement_mean": None,
            "agreement_sd": None,
            "agreement_ci95": None,
            "agreement_ci95_low": None,
            "agreement_ci95_high": None,
            "kappa_mean": None,
            "kappa_sd": None,
            "kappa_ci95": None,
            "kappa_ci95_low": None,
            "kappa_ci95_high": None,
        }

    prediction_cache: dict[str, dict[str, str]] = {}
    for run in valid_runs:
        output_path = root / "data" / "output" / f"{run.basename}.csv"
        prediction_cache[run.basename] = read_predictions(output_path)

    agreements: list[float] = []
    kappas: list[float] = []
    overlaps: list[int] = []
    for left_index, left_run in enumerate(valid_runs):
        for right_run in valid_runs[left_index + 1 :]:
            left_predictions = prediction_cache[left_run.basename]
            right_predictions = prediction_cache[right_run.basename]
            shared_ids = sorted(set(left_predictions) & set(right_predictions))
            if not shared_ids:
                continue
            left_values = [left_predictions[example_id] for example_id in shared_ids]
            right_values = [right_predictions[example_id] for example_id in shared_ids]
            agreements.append(sum(1 for a, b in zip(left_values, right_values) if a == b) / len(shared_ids))
            kappa = cohen_kappa_between(left_values, right_values)
            if kappa is not None:
                kappas.append(kappa)
            overlaps.append(len(shared_ids))

    def stats(values: list[float]) -> tuple[float | None, float | None, float | None, float | None, float | None]:
        if not values:
            return None, None, None, None, None
        mean = statistics.fmean(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        t_critical = t_critical_95(len(values))
        ci95 = t_critical * sd / math.sqrt(len(values)) if t_critical is not None else None
        low = max(0.0, mean - ci95) if ci95 is not None else None
        high = min(1.0, mean + ci95) if ci95 is not None else None
        return mean, sd, ci95, low, high

    agreement_mean, agreement_sd, agreement_ci95, agreement_low, agreement_high = stats(agreements)
    kappa_mean, kappa_sd, kappa_ci95, kappa_low, kappa_high = stats(kappas)
    return {
        "pair_count": len(agreements),
        "overlap_mean": statistics.fmean(overlaps) if overlaps else None,
        "agreement_mean": agreement_mean,
        "agreement_sd": agreement_sd,
        "agreement_ci95": agreement_ci95,
        "agreement_ci95_low": agreement_low,
        "agreement_ci95_high": agreement_high,
        "kappa_mean": kappa_mean,
        "kappa_sd": kappa_sd,
        "kappa_ci95": kappa_ci95,
        "kappa_ci95_low": kappa_low,
        "kappa_ci95_high": kappa_high,
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.{digits}f}"
    return str(value)


def status_for(
    output_rows: int | None,
    has_metrics: bool,
    metrics_total_examples: int | None,
    accuracy: float | None,
) -> tuple[str, str]:
    reasons: list[str] = []
    if output_rows != TASK_ROWS:
        reasons.append(f"output_rows={output_rows}")
    if not has_metrics:
        reasons.append("missing_metrics")
    if has_metrics and metrics_total_examples != TASK_ROWS:
        reasons.append(f"metrics_total_examples={metrics_total_examples}")
    if has_metrics and accuracy is None:
        reasons.append("missing_accuracy")
    if reasons:
        return "fail", "; ".join(reasons)
    return "valid", ""


def t_critical_95(sample_size: int) -> float | None:
    if sample_size < 2:
        return None
    by_df = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
    }
    return by_df.get(sample_size - 1, 1.96)


def load_runs(root: Path) -> list[Run]:
    data_dir = root / "data"
    output_dir = data_dir / "output"
    metrics_dir = data_dir / "metrics"
    logs_dir = data_dir / "logs"

    basenames: set[str] = set()
    for output_path in output_dir.glob("like_interrater__*__2026-05-1*.csv"):
        timestamp = parse_timestamp(output_path.name)
        if timestamp_in_window(timestamp):
            basenames.add(output_path.stem)
    for metrics_path in metrics_dir.glob("like_interrater__*__2026-05-1*__metrics.json"):
        timestamp = parse_timestamp(metrics_path.name)
        if timestamp_in_window(timestamp):
            basenames.add(metrics_path.name.removesuffix("__metrics.json"))

    runs: list[Run] = []
    for basename in sorted(basenames, key=lambda item: parse_timestamp(item)):
        output_path = output_dir / f"{basename}.csv"
        metrics_path = metrics_dir / f"{basename}__metrics.json"
        log_path = logs_dir / f"{basename}.log"

        metrics = read_json(metrics_path) if metrics_path.exists() else None
        metadata = read_run_metadata(log_path)
        config = {}
        details = {}
        if metrics:
            config = metrics.get("run_config", {}) or {}
            details = metrics.get("model_details", {}) or {}
        elif metadata:
            config = metadata.get("run_config", {}) or {}
            details = metadata.get("model_details", {}) or {}

        provider = config.get("provider") or details.get("provider")
        model = config.get("model") or details.get("model_requested")
        temperature = config.get("temperature")
        if temperature is not None:
            temperature = float(temperature)

        output_rows = count_output_rows(output_path)
        metrics_total = metrics.get("total_examples") if metrics else None
        accuracy = metrics.get("accuracy") if metrics else None
        macro_f1 = metrics.get("macro_f1") if metrics else None
        cohen_kappa = metrics.get("cohen_kappa") if metrics else None
        status, fail_reason = status_for(output_rows, bool(metrics), metrics_total, accuracy)

        runs.append(
            Run(
                basename=basename,
                timestamp=parse_timestamp(basename),
                provider=provider,
                model=model,
                temperature=temperature,
                output_rows=output_rows,
                metrics_total_examples=metrics_total,
                accuracy=accuracy,
                macro_f1=macro_f1,
                cohen_kappa=cohen_kappa,
                has_metrics=bool(metrics),
                has_log=log_path.exists(),
                status=status,
                fail_reason=fail_reason,
            )
        )
    return runs


def write_runs_csv(runs: list[Run], path: Path) -> None:
    fields = [
        "basename",
        "timestamp",
        "provider",
        "model",
        "temperature",
        "status",
        "fail_reason",
        "output_rows",
        "metrics_total_examples",
        "accuracy",
        "macro_f1",
        "cohen_kappa",
        "has_metrics",
        "has_log",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            writer.writerow({field: getattr(run, field) for field in fields})


def grouped_summary(runs: list[Run], root: Path) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[Run]] = defaultdict(list)
    for run in runs:
        if run.model is not None and run.temperature is not None:
            groups[(run.model, run.temperature)].append(run)

    rows: list[dict[str, Any]] = []
    for (model, temperature), group in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        valid = [run for run in group if run.status == "valid"]
        failed = [run for run in group if run.status == "fail"]
        accuracies = [run.accuracy for run in valid if run.accuracy is not None]
        macro_f1s = [run.macro_f1 for run in valid if run.macro_f1 is not None]
        kappas = [run.cohen_kappa for run in valid if run.cohen_kappa is not None]
        accuracy_mean = statistics.fmean(accuracies) if accuracies else None
        accuracy_sd = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0 if accuracies else None
        t_critical = t_critical_95(len(accuracies))
        accuracy_ci95 = (
            t_critical * accuracy_sd / math.sqrt(len(accuracies))
            if accuracy_mean is not None and accuracy_sd is not None and t_critical is not None
            else None
        )
        cohen_kappa_mean = statistics.fmean(kappas) if kappas else None
        cohen_kappa_sd = statistics.stdev(kappas) if len(kappas) > 1 else 0.0 if kappas else None
        kappa_t_critical = t_critical_95(len(kappas))
        cohen_kappa_ci95 = (
            kappa_t_critical * cohen_kappa_sd / math.sqrt(len(kappas))
            if cohen_kappa_mean is not None and cohen_kappa_sd is not None and kappa_t_critical is not None
            else None
        )
        repeat = pairwise_run_agreement(root, valid)
        rows.append(
            {
                "model": model,
                "temperature": temperature,
                "runs_total": len(group),
                "runs_valid": len(valid),
                "runs_failed": len(failed),
                "accuracy_mean": accuracy_mean,
                "accuracy_sd": accuracy_sd,
                "accuracy_ci95": accuracy_ci95,
                "accuracy_ci95_low": max(0.0, accuracy_mean - accuracy_ci95) if accuracy_ci95 is not None else None,
                "accuracy_ci95_high": min(1.0, accuracy_mean + accuracy_ci95) if accuracy_ci95 is not None else None,
                "accuracy_min": min(accuracies) if accuracies else None,
                "accuracy_max": max(accuracies) if accuracies else None,
                "macro_f1_mean": statistics.fmean(macro_f1s) if macro_f1s else None,
                "macro_f1_sd": statistics.stdev(macro_f1s) if len(macro_f1s) > 1 else 0.0 if macro_f1s else None,
                "cohen_kappa_mean": cohen_kappa_mean,
                "cohen_kappa_sd": cohen_kappa_sd,
                "cohen_kappa_ci95": cohen_kappa_ci95,
                "cohen_kappa_ci95_low": max(0.0, cohen_kappa_mean - cohen_kappa_ci95) if cohen_kappa_ci95 is not None else None,
                "cohen_kappa_ci95_high": min(1.0, cohen_kappa_mean + cohen_kappa_ci95) if cohen_kappa_ci95 is not None else None,
                "repeat_pair_count": repeat["pair_count"],
                "repeat_overlap_mean": repeat["overlap_mean"],
                "repeat_agreement_mean": repeat["agreement_mean"],
                "repeat_agreement_sd": repeat["agreement_sd"],
                "repeat_agreement_ci95": repeat["agreement_ci95"],
                "repeat_agreement_ci95_low": repeat["agreement_ci95_low"],
                "repeat_agreement_ci95_high": repeat["agreement_ci95_high"],
                "repeat_kappa_mean": repeat["kappa_mean"],
                "repeat_kappa_sd": repeat["kappa_sd"],
                "repeat_kappa_ci95": repeat["kappa_ci95"],
                "repeat_kappa_ci95_low": repeat["kappa_ci95_low"],
                "repeat_kappa_ci95_high": repeat["kappa_ci95_high"],
                "best_run": max(valid, key=lambda run: run.accuracy or -1).basename if valid else "",
            }
        )
    return rows


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "model",
        "temperature",
        "runs_total",
        "runs_valid",
        "runs_failed",
        "accuracy_mean",
        "accuracy_sd",
        "accuracy_ci95",
        "accuracy_ci95_low",
        "accuracy_ci95_high",
        "accuracy_min",
        "accuracy_max",
        "macro_f1_mean",
        "macro_f1_sd",
        "cohen_kappa_mean",
        "cohen_kappa_sd",
        "cohen_kappa_ci95",
        "cohen_kappa_ci95_low",
        "cohen_kappa_ci95_high",
        "repeat_pair_count",
        "repeat_overlap_mean",
        "repeat_agreement_mean",
        "repeat_agreement_sd",
        "repeat_agreement_ci95",
        "repeat_agreement_ci95_low",
        "repeat_agreement_ci95_high",
        "repeat_kappa_mean",
        "repeat_kappa_sd",
        "repeat_kappa_ci95",
        "repeat_kappa_ci95_low",
        "repeat_kappa_ci95_high",
        "best_run",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(runs: list[Run], summaries: list[dict[str, Any]], metric: str, ylabel: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted({run.model for run in runs if run.model})
    colors = {
        "deepseek-v4-pro-thinking": "#3566A8",
        "gemini-3.1-flash-lite-preview": "#1E8A5A",
        "glm-5.1": "#A65F2B",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for model in models:
        rows = [row for row in summaries if row["model"] == model and row[f"{metric}_mean"] is not None]
        rows.sort(key=lambda row: row["temperature"])
        if rows:
            x_values = [row["temperature"] for row in rows]
            y_values = [row[f"{metric}_mean"] for row in rows]
            ax.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=2,
                color=colors.get(model),
                label=model,
            )
            if metric in {"accuracy", "cohen_kappa"}:
                ci_rows = [
                    row
                    for row in rows
                    if row.get(f"{metric}_ci95_low") is not None and row.get(f"{metric}_ci95_high") is not None
                ]
                if ci_rows:
                    ax.fill_between(
                        [row["temperature"] for row in ci_rows],
                        [row[f"{metric}_ci95_low"] for row in ci_rows],
                        [row[f"{metric}_ci95_high"] for row in ci_rows],
                        color=colors.get(model),
                        alpha=0.14,
                        linewidth=0,
                    )
        points = [run for run in runs if run.model == model and run.status == "valid" and getattr(run, metric) is not None]
        for index, run in enumerate(points):
            jitter = ((index % 3) - 1) * 0.018
            ax.scatter(run.temperature + jitter, getattr(run, metric), s=34, alpha=0.55, color=colors.get(model))

    if metric in {"accuracy", "cohen_kappa"}:
        failure_groups: dict[tuple[str, float], list[Run]] = defaultdict(list)
        for run in runs:
            if run.status == "fail" and run.model is not None and run.temperature is not None:
                failure_groups[(run.model, run.temperature)].append(run)
        model_offsets = {
            model: offset
            for model, offset in zip(models, [-0.035, 0.0, 0.035, -0.07, 0.07])
        }
        label_offsets = {
            models[0]: (-26, 16, "right") if len(models) > 0 else (0, 16, "center"),
            models[1]: (0, 16, "center") if len(models) > 1 else (0, 16, "center"),
            models[2]: (26, 16, "left") if len(models) > 2 else (0, 16, "center"),
        }
        for (model, temperature), failed_runs in sorted(failure_groups.items()):
            x_position = temperature + model_offsets.get(model, 0.0)
            fail_y = 0.612 if metric == "accuracy" else 0.562
            ax.scatter(
                x_position,
                fail_y,
                marker="x",
                s=72,
                color=colors.get(model, "#B94A48"),
                linewidths=2,
                zorder=5,
            )
            short_reasons = []
            for run in failed_runs:
                if run.output_rows != TASK_ROWS:
                    short_reasons.append(str(run.output_rows))
                elif not run.has_metrics:
                    short_reasons.append("no metrics")
                else:
                    short_reasons.append("metrics")
            reason_text = "/".join(short_reasons[:3])
            if len(short_reasons) > 3:
                reason_text += "/..."
            x_offset, y_offset, ha = label_offsets.get(model, (0, 16, "center"))
            ax.annotate(
                f"{len(failed_runs)} fail\nrows: {reason_text}",
                xy=(x_position, fail_y),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                ha=ha,
                va="bottom",
                fontsize=7,
                color=colors.get(model, "#B94A48"),
            )

    ax.set_title(f"like_interrater performance by temperature ({ylabel})")
    ax.set_xlabel("Temperature")
    ax.set_ylabel(ylabel)
    if metric == "accuracy":
        ax.set_ylim(0.6, 1.02)
        ax.set_xlim(-0.12, 2.25)
        break_kwargs = dict(transform=ax.transAxes, color="black", clip_on=False, linewidth=1.4)
        ax.plot((-0.012, 0.012), (-0.006, 0.026), **break_kwargs)
        ax.plot((-0.012, 0.012), (0.026, 0.058), **break_kwargs)
        ax.annotate(
            "axis cut: 0-0.60 omitted",
            xy=(0, 0.6),
            xycoords=("axes fraction", "data"),
            xytext=(10, 8),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=8,
            color="#333333",
        )
    elif metric == "cohen_kappa":
        ax.set_ylim(0.55, 0.92)
        ax.set_xlim(-0.12, 2.25)
        break_kwargs = dict(transform=ax.transAxes, color="black", clip_on=False, linewidth=1.4)
        ax.plot((-0.012, 0.012), (-0.006, 0.026), **break_kwargs)
        ax.plot((-0.012, 0.012), (0.026, 0.058), **break_kwargs)
        ax.annotate(
            "axis cut: 0-0.55 omitted",
            xy=(0, 0.55),
            xycoords=("axes fraction", "data"),
            xytext=(10, 8),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=8,
            color="#333333",
        )
    else:
        ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    if metric == "accuracy":
        fig.text(
            0.5,
            0.01,
            "Broken y-axis: 0-0.60 omitted. Shaded bands are 95% CIs for mean accuracy; x markers show failed/incomplete outputs.",
            ha="center",
            fontsize=8,
            color="#555555",
        )
        fig.tight_layout(rect=(0, 0.035, 1, 1))
    elif metric == "cohen_kappa":
        fig.text(
            0.5,
            0.01,
            "Broken y-axis: 0-0.55 omitted. Shaded bands are 95% CIs for mean Cohen's kappa; x markers show failed/incomplete outputs.",
            ha="center",
            fontsize=8,
            color="#555555",
        )
        fig.tight_layout(rect=(0, 0.035, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_completion(runs: list[Run], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups: dict[tuple[str, float], dict[str, int]] = defaultdict(lambda: {"valid": 0, "fail": 0})
    for run in runs:
        if run.model is None or run.temperature is None:
            continue
        groups[(run.model, run.temperature)][run.status] += 1

    labels = [f"{model}\nT={temperature:g}" for model, temperature in sorted(groups)]
    valid_counts = [groups[key]["valid"] for key in sorted(groups)]
    fail_counts = [groups[key]["fail"] for key in sorted(groups)]
    x = range(len(labels))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, valid_counts, color="#4A7C59", label="valid")
    ax.bar(x, fail_counts, bottom=valid_counts, color="#B94A48", label="fail")
    ax.set_title("Run completeness by model and temperature")
    ax.set_ylabel("Runs")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylim(0, max([a + b for a, b in zip(valid_counts, fail_counts)] + [1]) + 0.5)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_repeat_agreement(runs: list[Run], summaries: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted({run.model for run in runs if run.model})
    colors = {
        "deepseek-v4-pro-thinking": "#3566A8",
        "gemini-3.1-flash-lite-preview": "#1E8A5A",
        "glm-5.1": "#A65F2B",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for model in models:
        rows = [row for row in summaries if row["model"] == model and row["repeat_kappa_mean"] is not None]
        rows.sort(key=lambda row: row["temperature"])
        if not rows:
            continue
        ax.plot(
            [row["temperature"] for row in rows],
            [row["repeat_kappa_mean"] for row in rows],
            marker="o",
            linewidth=2,
            color=colors.get(model),
            label=model,
        )
        ci_rows = [
            row
            for row in rows
            if row.get("repeat_kappa_ci95_low") is not None and row.get("repeat_kappa_ci95_high") is not None
        ]
        if ci_rows:
            ax.fill_between(
                [row["temperature"] for row in ci_rows],
                [row["repeat_kappa_ci95_low"] for row in ci_rows],
                [row["repeat_kappa_ci95_high"] for row in ci_rows],
                color=colors.get(model),
                alpha=0.14,
                linewidth=0,
            )
        for row in rows:
            ax.annotate(
                f"n={row['repeat_pair_count']}",
                xy=(row["temperature"], row["repeat_kappa_mean"]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=colors.get(model),
            )

    ax.set_title("like_interrater repeat-run agreement by temperature")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Mean pairwise Cohen's kappa between runs")
    ax.set_ylim(0.55, 1.02)
    ax.set_xlim(-0.12, 2.25)
    break_kwargs = dict(transform=ax.transAxes, color="black", clip_on=False, linewidth=1.4)
    ax.plot((-0.012, 0.012), (-0.006, 0.026), **break_kwargs)
    ax.plot((-0.012, 0.012), (0.026, 0.058), **break_kwargs)
    ax.annotate(
        "axis cut: 0-0.55 omitted",
        xy=(0, 0.55),
        xycoords=("axes fraction", "data"),
        xytext=(10, 8),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#333333",
    )
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.text(
        0.5,
        0.01,
        "Agreement is computed between repeated model runs over shared example IDs, not against gold labels. n is the number of run pairs.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_markdown(runs: list[Run], summaries: list[dict[str, Any]], path: Path) -> None:
    valid = [run for run in runs if run.status == "valid"]
    failed = [run for run in runs if run.status == "fail"]
    best = max(valid, key=lambda run: run.accuracy or -1) if valid else None

    lines = [
        "# like_interrater temperature overview",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Window: {START_DATE} to {END_DATE}",
        f"Expected task rows: {TASK_ROWS}",
        "",
        "A run is counted as valid for model-performance comparisons only when its output CSV parses to 115 rows, it has a metrics JSON file, the metrics file reports 115 total examples, and accuracy is present.",
        "",
        "## Headline",
        "",
        f"- Outputs found: {len(runs)}",
        f"- Valid runs: {len(valid)}",
        f"- Failed/incomplete runs: {len(failed)}",
    ]
    if best:
        lines.append(
            f"- Best valid single run: {best.model} at T={best.temperature:g}, accuracy={fmt(best.accuracy)}, macro F1={fmt(best.macro_f1)} ({best.basename})"
        )

    lines.extend(
        [
            "",
            "## Mean performance by model and temperature",
            "",
            "| model | temp | valid/total | accuracy mean | accuracy 95% CI | macro F1 mean | gold kappa mean | run-pair kappa mean | run-pair agreement | run pairs |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        total = row["runs_total"]
        valid_count = row["runs_valid"]
        accuracy_ci = (
            f"{fmt(row['accuracy_ci95_low'])}-{fmt(row['accuracy_ci95_high'])}"
            if row.get("accuracy_ci95_low") is not None
            else ""
        )
        kappa_ci = (
            f"{fmt(row['cohen_kappa_ci95_low'])}-{fmt(row['cohen_kappa_ci95_high'])}"
            if row.get("cohen_kappa_ci95_low") is not None
            else ""
        )
        lines.append(
            "| {model} | {temperature:g} | {valid}/{total} | {acc} | {ci} | {f1} | {kappa} | {repeat_kappa} | {repeat_agreement} | {repeat_pairs} |".format(
                model=row["model"],
                temperature=row["temperature"],
                valid=valid_count,
                total=total,
                acc=fmt(row["accuracy_mean"]),
                ci=accuracy_ci,
                f1=fmt(row["macro_f1_mean"]),
                kappa=fmt(row["cohen_kappa_mean"]),
                repeat_kappa=fmt(row["repeat_kappa_mean"]),
                repeat_agreement=fmt(row["repeat_agreement_mean"]),
                repeat_pairs=fmt(row["repeat_pair_count"], 0),
            )
        )

    lines.extend(["", "## Failed or incomplete outputs", ""])
    if failed:
        lines.extend(["| timestamp | model | temp | output rows | metrics rows | reason |", "|---|---|---:|---:|---:|---|"])
        for run in failed:
            lines.append(
                f"| {run.timestamp} | {run.model or ''} | {fmt(run.temperature, 1)} | {fmt(run.output_rows, 0)} | {fmt(run.metrics_total_examples, 0)} | {run.fail_reason} |"
            )
    else:
        lines.append("No failed or incomplete outputs found.")

    lines.extend(
        [
            "",
            "## Visualizations",
            "",
            "- `like_interrater_temperature_accuracy.png`",
            "- `like_interrater_temperature_interrater_agreement.png`",
            "- `like_interrater_temperature_repeat_run_agreement.png`",
            "- `like_interrater_temperature_macro_f1.png`",
            "- `like_interrater_temperature_completion.png`",
            "",
            "The accuracy and interrater-agreement charts use truncated y-axes, 95% confidence intervals for repeated valid runs, and x markers near the baseline for failed/incomplete outputs.",
            "The repeat-run agreement chart compares predictions between repeated runs of the same model at the same temperature over shared example IDs. It does not use gold labels.",
            "",
            "## Data files",
            "",
            "- `like_interrater_temperature_runs.csv`: one row per output/metrics basename",
            "- `like_interrater_temperature_summary.csv`: grouped model-temperature summary",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "data" / "analysis" / "like_interrater_temperature"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(root)
    summaries = grouped_summary(runs, root)

    write_runs_csv(runs, out_dir / "like_interrater_temperature_runs.csv")
    write_summary_csv(summaries, out_dir / "like_interrater_temperature_summary.csv")
    write_markdown(runs, summaries, out_dir / "README.md")

    try:
        plot_metric(runs, summaries, "accuracy", "Accuracy", out_dir / "like_interrater_temperature_accuracy.png")
        plot_metric(
            runs,
            summaries,
            "cohen_kappa",
            "Cohen's kappa (interrater agreement)",
            out_dir / "like_interrater_temperature_interrater_agreement.png",
        )
        plot_repeat_agreement(runs, summaries, out_dir / "like_interrater_temperature_repeat_run_agreement.png")
        plot_metric(runs, summaries, "macro_f1", "Macro F1", out_dir / "like_interrater_temperature_macro_f1.png")
        plot_completion(runs, out_dir / "like_interrater_temperature_completion.png")
    except ImportError as exc:
        print(f"Skipping plots because matplotlib is unavailable: {exc}")

    print(f"Wrote analysis to {out_dir}")
    print(f"Runs: {len(runs)}; valid: {sum(run.status == 'valid' for run in runs)}; failed: {sum(run.status == 'fail' for run in runs)}")


if __name__ == "__main__":
    main()

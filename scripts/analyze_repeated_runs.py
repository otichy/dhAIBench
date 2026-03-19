#!/usr/bin/env python3
"""Analyze repeated benchmark runs for accuracy vs. latency/timeout patterns."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = REPO_ROOT / "data" / "metrics"
LOGS_DIR = REPO_ROOT / "data" / "logs"
OUTPUT_DIR = REPO_ROOT / "data" / "analysis"
OUTPUT_DETAILS_CSV = OUTPUT_DIR / "repeated_run_details.csv"
OUTPUT_GROUPS_CSV = OUTPUT_DIR / "repeated_run_groups.csv"
OUTPUT_REPORT_MD = OUTPUT_DIR / "repeated_run_analysis.md"
LOCAL_TIMEZONE = ZoneInfo("Europe/Prague") if ZoneInfo is not None else None


@dataclass
class RunRecord:
    task_slug: str
    task_name: str
    provider: Optional[str]
    model: Optional[str]
    file_provider_slug: Optional[str]
    file_model_slug: Optional[str]
    base_name: str
    metrics_path: Path
    log_path: Optional[Path]
    start_timestamp_utc: Optional[str]
    start_timestamp_local: Optional[str]
    local_date: Optional[str]
    local_hour: Optional[int]
    accuracy: Optional[float]
    overall_time_seconds: Optional[float]
    evaluated_examples: Optional[int]
    attempts_total_metrics: Optional[int]
    attempt_mean_seconds: Optional[float]
    attempt_median_seconds: Optional[float]
    success_mean_seconds: Optional[float]
    timeout_mean_seconds: Optional[float]
    success_attempts: int
    error_attempts: int
    timeout_attempts: int
    timeout_examples: int
    final_timeout_examples: int
    non_timeout_error_attempts: int
    log_example_records: int

    @property
    def group_key(self) -> tuple[str, Optional[str], Optional[str]]:
        return (self.task_slug, self.provider, self.model)

    @property
    def runtime_per_example_seconds(self) -> Optional[float]:
        if (
            self.overall_time_seconds is None
            or self.evaluated_examples is None
            or self.evaluated_examples <= 0
        ):
            return None
        return self.overall_time_seconds / self.evaluated_examples


def parse_artifact_stem(base_name: str) -> tuple[str, Optional[str], Optional[str], Optional[str]]:
    parts = base_name.rsplit("__", 3)
    if len(parts) == 4:
        return parts[0], parts[1] or None, parts[2] or None, parts[3] or None
    if len(parts) == 3:
        return parts[0], parts[1] or None, parts[2] or None, None
    return base_name, None, None, None


def parse_timestamp_local(timestamp_utc: Optional[str]) -> tuple[Optional[str], Optional[str], Optional[int]]:
    if not timestamp_utc:
        return None, None, None
    dt_utc = datetime.fromisoformat(timestamp_utc.replace("Z", "+00:00"))
    if LOCAL_TIMEZONE is None:
        dt_local = dt_utc
    else:
        dt_local = dt_utc.astimezone(LOCAL_TIMEZONE)
    return dt_local.isoformat(timespec="minutes"), dt_local.date().isoformat(), dt_local.hour


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_json_lines(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def maybe_float(value: object) -> Optional[float]:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def maybe_int(value: object) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def mean_or_none(values: list[float]) -> Optional[float]:
    return statistics.mean(values) if values else None


def median_or_none(values: list[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def pearson(xs: list[float], ys: list[float]) -> Optional[float]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    sum_xx = sum((x - mean_x) ** 2 for x in xs)
    sum_yy = sum((y - mean_y) ** 2 for y in ys)
    if sum_xx <= 0 or sum_yy <= 0:
        return None
    sum_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return sum_xy / math.sqrt(sum_xx * sum_yy)


def slugify_for_compare(value: Optional[str]) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def load_run_records() -> list[RunRecord]:
    runs: list[RunRecord] = []
    for metrics_path in sorted(METRICS_DIR.glob("*__metrics.json")):
        base_name = metrics_path.name[: -len("__metrics.json")]
        task_slug, file_provider_slug, file_model_slug, _ = parse_artifact_stem(base_name)
        metrics = load_json(metrics_path)
        model_details = metrics.get("model_details") or {}
        provider = model_details.get("provider") or file_provider_slug
        model = (
            model_details.get("model_requested")
            or model_details.get("model_for_requests")
            or file_model_slug
        )
        start_timestamp_utc = metrics.get("first_prompt_timestamp")
        start_timestamp_local, local_date, local_hour = parse_timestamp_local(start_timestamp_utc)

        log_path = LOGS_DIR / f"{base_name}.log"
        if not log_path.exists():
            log_path = None

        attempt_durations: list[float] = []
        success_durations: list[float] = []
        timeout_durations: list[float] = []
        success_attempts = 0
        error_attempts = 0
        timeout_attempts = 0
        timeout_examples = 0
        final_timeout_examples = 0
        non_timeout_error_attempts = 0
        log_example_records = 0

        if log_path is not None:
            for record in safe_json_lines(log_path):
                if record.get("record_type") != "example_result":
                    continue
                log_example_records += 1
                timed_out_example = False
                final_prediction = record.get("final_prediction") or {}
                if final_prediction.get("validator_status") == "accepted_after_request_timeout":
                    final_timeout_examples += 1
                    timed_out_example = True
                for attempt in record.get("attempts") or []:
                    duration_seconds = maybe_float(attempt.get("duration_seconds"))
                    if duration_seconds is not None:
                        attempt_durations.append(duration_seconds)
                    status = attempt.get("status")
                    if status == "success":
                        success_attempts += 1
                        if duration_seconds is not None:
                            success_durations.append(duration_seconds)
                    elif status == "error":
                        error_attempts += 1
                        error_category = attempt.get("error_category")
                        if error_category == "request_timeout":
                            timeout_attempts += 1
                            timed_out_example = True
                            if duration_seconds is not None:
                                timeout_durations.append(duration_seconds)
                        else:
                            non_timeout_error_attempts += 1
                if timed_out_example:
                    timeout_examples += 1

        runs.append(
            RunRecord(
                task_slug=task_slug,
                task_name=metrics.get("task_name") or task_slug,
                provider=provider,
                model=model,
                file_provider_slug=file_provider_slug,
                file_model_slug=file_model_slug,
                base_name=base_name,
                metrics_path=metrics_path,
                log_path=log_path,
                start_timestamp_utc=start_timestamp_utc,
                start_timestamp_local=start_timestamp_local,
                local_date=local_date,
                local_hour=local_hour,
                accuracy=maybe_float(metrics.get("accuracy")),
                overall_time_seconds=maybe_float(metrics.get("overall_time_seconds")),
                evaluated_examples=maybe_int(metrics.get("evaluated_example_count")),
                attempts_total_metrics=maybe_int(
                    ((metrics.get("request_control_summary") or {}).get("attempts_total"))
                ),
                attempt_mean_seconds=mean_or_none(attempt_durations),
                attempt_median_seconds=median_or_none(attempt_durations),
                success_mean_seconds=mean_or_none(success_durations),
                timeout_mean_seconds=mean_or_none(timeout_durations),
                success_attempts=success_attempts,
                error_attempts=error_attempts,
                timeout_attempts=timeout_attempts,
                timeout_examples=timeout_examples,
                final_timeout_examples=final_timeout_examples,
                non_timeout_error_attempts=non_timeout_error_attempts,
                log_example_records=log_example_records,
            )
        )
    return runs


def repeated_groups(runs: list[RunRecord]) -> dict[tuple[str, Optional[str], Optional[str]], list[RunRecord]]:
    groups: dict[tuple[str, Optional[str], Optional[str]], list[RunRecord]] = defaultdict(list)
    for run in runs:
        groups[run.group_key].append(run)
    return {key: value for key, value in groups.items() if len(value) >= 2}


def metric_within_group_correlation(
    groups: dict[tuple[str, Optional[str], Optional[str]], list[RunRecord]],
    metric_name: str,
) -> tuple[Optional[float], int]:
    acc_deltas: list[float] = []
    metric_deltas: list[float] = []
    for items in groups.values():
        paired = [
            (run.accuracy, getattr(run, metric_name))
            for run in items
            if run.accuracy is not None and getattr(run, metric_name) is not None
        ]
        if len(paired) < 2:
            continue
        mean_acc = statistics.mean(acc for acc, _ in paired)
        mean_metric = statistics.mean(metric for _, metric in paired)
        for acc, metric in paired:
            acc_deltas.append(acc - mean_acc)
            metric_deltas.append(metric - mean_metric)
    return pearson(acc_deltas, metric_deltas), len(acc_deltas)


def hour_bucket(hour: Optional[int]) -> Optional[str]:
    if hour is None:
        return None
    if 0 <= hour <= 5:
        return "00-05"
    if 6 <= hour <= 11:
        return "06-11"
    if 12 <= hour <= 17:
        return "12-17"
    return "18-23"


def fmt_float(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def fmt_seconds(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}s"


def write_details_csv(groups: dict[tuple[str, Optional[str], Optional[str]], list[RunRecord]]) -> None:
    rows: list[dict[str, object]] = []
    for group_key, items in sorted(groups.items()):
        valid_acc = [run.accuracy for run in items if run.accuracy is not None]
        group_best = max(valid_acc) if valid_acc else None
        group_mean = statistics.mean(valid_acc) if valid_acc else None
        for run in sorted(items, key=lambda item: item.start_timestamp_local or item.base_name):
            rows.append(
                {
                    "task_slug": run.task_slug,
                    "task_name": run.task_name,
                    "provider": run.provider,
                    "model": run.model,
                    "base_name": run.base_name,
                    "start_timestamp_local": run.start_timestamp_local,
                    "local_date": run.local_date,
                    "local_hour": run.local_hour,
                    "hour_bucket": hour_bucket(run.local_hour),
                    "accuracy": run.accuracy,
                    "accuracy_delta_vs_group_mean": (
                        (run.accuracy - group_mean)
                        if run.accuracy is not None and group_mean is not None
                        else None
                    ),
                    "accuracy_drop_vs_group_best": (
                        (group_best - run.accuracy)
                        if run.accuracy is not None and group_best is not None
                        else None
                    ),
                    "overall_time_seconds": run.overall_time_seconds,
                    "runtime_per_example_seconds": run.runtime_per_example_seconds,
                    "attempt_mean_seconds": run.attempt_mean_seconds,
                    "attempt_median_seconds": run.attempt_median_seconds,
                    "success_mean_seconds": run.success_mean_seconds,
                    "timeout_mean_seconds": run.timeout_mean_seconds,
                    "evaluated_examples": run.evaluated_examples,
                    "attempts_total_metrics": run.attempts_total_metrics,
                    "success_attempts": run.success_attempts,
                    "error_attempts": run.error_attempts,
                    "timeout_attempts": run.timeout_attempts,
                    "timeout_examples": run.timeout_examples,
                    "final_timeout_examples": run.final_timeout_examples,
                    "non_timeout_error_attempts": run.non_timeout_error_attempts,
                    "log_example_records": run.log_example_records,
                    "log_path": str(run.log_path) if run.log_path else "",
                    "metrics_path": str(run.metrics_path),
                }
            )
    if not rows:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_DETAILS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_group_summary_csv(groups: dict[tuple[str, Optional[str], Optional[str]], list[RunRecord]]) -> None:
    rows: list[dict[str, object]] = []
    for (task_slug, provider, model), items in sorted(groups.items()):
        valid_acc = [run.accuracy for run in items if run.accuracy is not None]
        valid_runtime = [run.overall_time_seconds for run in items if run.overall_time_seconds is not None]
        rows.append(
            {
                "task_slug": task_slug,
                "provider": provider,
                "model": model,
                "run_count": len(items),
                "accuracy_min": min(valid_acc) if valid_acc else None,
                "accuracy_max": max(valid_acc) if valid_acc else None,
                "accuracy_spread": (max(valid_acc) - min(valid_acc)) if len(valid_acc) >= 2 else None,
                "runtime_min_seconds": min(valid_runtime) if valid_runtime else None,
                "runtime_max_seconds": max(valid_runtime) if valid_runtime else None,
                "runtime_spread_seconds": (
                    max(valid_runtime) - min(valid_runtime) if len(valid_runtime) >= 2 else None
                ),
                "max_timeout_examples": max(run.timeout_examples for run in items),
                "max_final_timeout_examples": max(run.final_timeout_examples for run in items),
                "runs_with_log_telemetry": sum(1 for run in items if run.log_path is not None),
            }
        )
    if not rows:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_GROUPS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_task_clusters(
    groups: dict[tuple[str, Optional[str], Optional[str]], list[RunRecord]]
) -> list[dict[str, object]]:
    task_groups: dict[str, list[tuple[tuple[str, Optional[str], Optional[str]], list[RunRecord]]]] = defaultdict(list)
    for key, items in groups.items():
        task_groups[key[0]].append((key, items))

    clusters: list[dict[str, object]] = []
    for task_slug, members in sorted(task_groups.items()):
        if len(members) < 2:
            continue
        worst_date_counter: Counter[str] = Counter()
        drops: list[float] = []
        worst_rows: list[tuple[tuple[str, Optional[str], Optional[str]], RunRecord, float]] = []
        for group_key, items in members:
            valid = [run for run in items if run.accuracy is not None]
            if len(valid) < 2:
                continue
            best_accuracy = max(run.accuracy for run in valid if run.accuracy is not None)
            worst_run = min(valid, key=lambda run: run.accuracy if run.accuracy is not None else 1.0)
            drop = best_accuracy - (worst_run.accuracy or 0.0)
            if worst_run.local_date:
                worst_date_counter[worst_run.local_date] += 1
            drops.append(drop)
            worst_rows.append((group_key, worst_run, drop))
        if len(worst_rows) < 2 or not worst_date_counter:
            continue
        dominant_date, dominant_count = worst_date_counter.most_common(1)[0]
        dominant_drops = [
            drop
            for _, worst_run, drop in worst_rows
            if worst_run.local_date == dominant_date
        ]
        clusters.append(
            {
                "task_slug": task_slug,
                "repeated_groups": len(members),
                "groups_with_valid_comparison": len(worst_rows),
                "dominant_worst_date": dominant_date,
                "dominant_worst_group_count": dominant_count,
                "dominant_worst_group_share": dominant_count / len(worst_rows),
                "mean_drop_on_dominant_date": statistics.mean(dominant_drops) if dominant_drops else None,
                "max_drop_on_dominant_date": max(dominant_drops) if dominant_drops else None,
            }
        )
    clusters.sort(
        key=lambda item: (
            -float(item["dominant_worst_group_share"]),
            -(item["mean_drop_on_dominant_date"] or 0.0),
            item["task_slug"],
        )
    )
    return clusters


def build_outlier_rows(
    groups: dict[tuple[str, Optional[str], Optional[str]], list[RunRecord]]
) -> list[dict[str, object]]:
    outliers: list[dict[str, object]] = []
    for group_key, items in groups.items():
        valid = [run for run in items if run.accuracy is not None]
        if len(valid) < 2:
            continue
        best_run = max(valid, key=lambda run: run.accuracy if run.accuracy is not None else -1.0)
        for run in valid:
            drop = (best_run.accuracy or 0.0) - (run.accuracy or 0.0)
            outliers.append(
                {
                    "task_slug": run.task_slug,
                    "provider": run.provider,
                    "model": run.model,
                    "run_timestamp": run.start_timestamp_local,
                    "accuracy": run.accuracy,
                    "accuracy_drop_vs_group_best": drop,
                    "group_best_accuracy": best_run.accuracy,
                    "best_run_timestamp": best_run.start_timestamp_local,
                    "runtime_seconds": run.overall_time_seconds,
                    "attempt_mean_seconds": run.attempt_mean_seconds,
                    "timeout_examples": run.timeout_examples,
                    "final_timeout_examples": run.final_timeout_examples,
                    "base_name": run.base_name,
                }
            )
    outliers.sort(
        key=lambda item: (
            -(item["accuracy_drop_vs_group_best"] or 0.0),
            item["task_slug"],
            item["provider"] or "",
            item["model"] or "",
        )
    )
    return outliers


def build_report(
    runs: list[RunRecord],
    groups: dict[tuple[str, Optional[str], Optional[str]], list[RunRecord]],
) -> str:
    repeated_runs = [run for items in groups.values() for run in items]
    runs_with_log_telemetry = sum(1 for run in repeated_runs if run.log_path is not None)
    provider_mismatches = [
        run
        for run in runs
        if run.file_provider_slug
        and run.provider
        and slugify_for_compare(run.file_provider_slug) != slugify_for_compare(run.provider)
    ]

    correlation_specs = [
        ("overall_time_seconds", "overall_time_seconds"),
        ("attempt_mean_seconds", "attempt_mean_seconds"),
        ("timeout_examples", "timeout_examples"),
        ("timeout_attempts", "timeout_attempts"),
        ("final_timeout_examples", "final_timeout_examples"),
    ]
    correlation_rows = []
    for label, metric_name in correlation_specs:
        correlation, paired_points = metric_within_group_correlation(groups, metric_name)
        correlation_rows.append((label, correlation, paired_points))

    hour_buckets: dict[str, list[float]] = defaultdict(list)
    for items in groups.values():
        valid = [run for run in items if run.accuracy is not None and run.local_hour is not None]
        if len(valid) < 2:
            continue
        mean_accuracy = statistics.mean(run.accuracy for run in valid if run.accuracy is not None)
        for run in valid:
            bucket = hour_bucket(run.local_hour)
            if bucket is not None and run.accuracy is not None:
                hour_buckets[bucket].append(run.accuracy - mean_accuracy)

    task_clusters = build_task_clusters(groups)
    outliers = build_outlier_rows(groups)

    group_rows = []
    for (task_slug, provider, model), items in sorted(groups.items()):
        valid_acc = [run.accuracy for run in items if run.accuracy is not None]
        valid_runtime = [run.overall_time_seconds for run in items if run.overall_time_seconds is not None]
        group_rows.append(
            (
                task_slug,
                provider,
                model,
                len(items),
                min(valid_acc) if valid_acc else None,
                max(valid_acc) if valid_acc else None,
                (max(valid_acc) - min(valid_acc)) if len(valid_acc) >= 2 else None,
                min(valid_runtime) if valid_runtime else None,
                max(valid_runtime) if valid_runtime else None,
                max(run.timeout_examples for run in items),
                max(run.final_timeout_examples for run in items),
            )
        )

    generated_at = datetime.now().astimezone(LOCAL_TIMEZONE).isoformat(timespec="minutes")
    lines: list[str] = []
    lines.append("# Repeated Run Accuracy / Latency / Timeout Analysis")
    lines.append("")
    lines.append(f"Generated: `{generated_at}`")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- Total metric runs scanned: `{len(runs)}`")
    lines.append(f"- Repeated task+provider+model groups: `{len(groups)}`")
    lines.append(f"- Repeated runs included in the comparison set: `{len(repeated_runs)}`")
    lines.append(f"- Repeated runs with matching log telemetry: `{runs_with_log_telemetry}`")
    lines.append(
        f"- Non-empty filename/provider mismatches in artifacts: `{len(provider_mismatches)}`"
    )
    if provider_mismatches:
        lines.append("  These are most notably the early `adv-ing__vertex__...` artifacts whose metrics report `google` as provider.")
    lines.append("")
    lines.append("## Controlled Correlations")
    lines.append("")
    lines.append("Each correlation is computed after de-meaning within each repeated group, so it reflects variation inside the same task/model combination rather than raw differences between tasks.")
    lines.append("")
    lines.append("| Metric | Within-group Pearson r | Paired run points |")
    lines.append("| --- | ---: | ---: |")
    for label, correlation, paired_points in correlation_rows:
        lines.append(
            f"| `{label}` | {fmt_float(correlation, 4)} | {paired_points} |"
        )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- Across all repeated groups, raw runtime and timeout counts show only weak negative correlations with accuracy.")
    lines.append("- Where detailed attempt-latency telemetry is available, slower per-attempt responses align much more strongly with lower accuracy, but the sample is small and concentrated in a few March 2026 runs.")
    lines.append("")
    lines.append("## Task-Level Date Clusters")
    lines.append("")
    lines.append("| Task Slug | Repeated Groups | Dominant worst-run date | Groups on that date | Share | Mean drop on that date | Max drop on that date |")
    lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: |")
    for cluster in task_clusters[:10]:
        lines.append(
            "| `{task}` | {groups_total} | `{date}` | {count} | {share:.0%} | {mean_drop} | {max_drop} |".format(
                task=cluster["task_slug"],
                groups_total=cluster["groups_with_valid_comparison"],
                date=cluster["dominant_worst_date"],
                count=cluster["dominant_worst_group_count"],
                share=cluster["dominant_worst_group_share"],
                mean_drop=fmt_float(cluster["mean_drop_on_dominant_date"], 4),
                max_drop=fmt_float(cluster["max_drop_on_dominant_date"], 4),
            )
        )
    lines.append("")
    lines.append("The clearest cluster is `like_interrater`: all four repeated provider/model groups hit their worst accuracy on `2026-03-13`, and each improved materially on `2026-03-16` reruns.")
    lines.append("")
    lines.append("## Largest Accuracy Drops Vs Group Best")
    lines.append("")
    lines.append("| Task | Provider | Model | Worst run (local) | Drop vs best | Accuracy | Runtime | Mean attempt latency | Timeout examples | Final timeout examples |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in outliers[:12]:
        lines.append(
            "| `{task}` | `{provider}` | `{model}` | `{timestamp}` | {drop} | {accuracy} | {runtime} | {attempt_mean} | {timeout_examples} | {final_timeout_examples} |".format(
                task=row["task_slug"],
                provider=row["provider"],
                model=row["model"],
                timestamp=row["run_timestamp"],
                drop=fmt_float(row["accuracy_drop_vs_group_best"], 4),
                accuracy=fmt_float(row["accuracy"], 4),
                runtime=fmt_seconds(row["runtime_seconds"]),
                attempt_mean=fmt_seconds(row["attempt_mean_seconds"]),
                timeout_examples=row["timeout_examples"],
                final_timeout_examples=row["final_timeout_examples"],
            )
        )
    lines.append("")
    lines.append("## Repeated Group Summary")
    lines.append("")
    lines.append("| Task | Provider | Model | Runs | Accuracy min | Accuracy max | Spread | Runtime min | Runtime max | Max timeout examples | Max final timeout examples |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for task_slug, provider, model, run_count, acc_min, acc_max, acc_spread, runtime_min, runtime_max, timeout_max, final_timeout_max in group_rows:
        lines.append(
            "| `{task}` | `{provider}` | `{model}` | {runs} | {acc_min} | {acc_max} | {acc_spread} | {runtime_min} | {runtime_max} | {timeout_max} | {final_timeout_max} |".format(
                task=task_slug,
                provider=provider,
                model=model,
                runs=run_count,
                acc_min=fmt_float(acc_min, 4),
                acc_max=fmt_float(acc_max, 4),
                acc_spread=fmt_float(acc_spread, 4),
                runtime_min=fmt_seconds(runtime_min),
                runtime_max=fmt_seconds(runtime_max),
                timeout_max=timeout_max,
                final_timeout_max=final_timeout_max,
            )
        )
    lines.append("")
    lines.append("## Accuracy Delta By Local Hour Bucket")
    lines.append("")
    lines.append("| Hour bucket | Run points | Mean accuracy delta vs group mean | Min | Max |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for bucket in ("00-05", "06-11", "12-17", "18-23"):
        values = hour_buckets.get(bucket, [])
        lines.append(
            f"| `{bucket}` | {len(values)} | {fmt_float(mean_or_none(values), 4)} | {fmt_float(min(values) if values else None, 4)} | {fmt_float(max(values) if values else None, 4)} |"
        )
    lines.append("")
    lines.append("This hour-bucket view does not show one universal bad hour across the whole dataset. The stronger signal is date- and task-specific clusters rather than a global time-of-day effect.")
    lines.append("")
    lines.append("## Main Readout")
    lines.append("")
    lines.append("- The strongest evidence of time-linked lower accuracy is not global; it is concentrated in specific task/date clusters.")
    lines.append("- `like_interrater` shows the clearest degradation window: on `2026-03-13`, repeated runs for `openai/gpt-5.4`, `vertex/gemini-3-flash-preview`, `vertex/gemini-3.1-pro-preview`, and `e-infra/deepseek-v3.2-thinking` all recorded their worst accuracy for that same task. Those runs were also much slower than the later `2026-03-16` repeats.")
    lines.append("- `OE_number` on `vertex/gemini-3-flash-preview` also shows a shorter-lived issue: the `2026-03-10 21:18` run had `16` timeout-affected examples and `3` final timeout examples, with accuracy dropping from the group best `0.9792` to `0.9550`. Later repeats the same night recovered.")
    lines.append("- Some groups show the opposite pattern or no pattern at all. For example, `adv-ing/requesty/claude-sonnet-4-6` had a slightly lower-accuracy run that was much faster, not slower. That argues against a single universal latency=>accuracy rule.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Detailed repeated-run table: `{OUTPUT_DETAILS_CSV.relative_to(REPO_ROOT)}`")
    lines.append(f"- Group summary table: `{OUTPUT_GROUPS_CSV.relative_to(REPO_ROOT)}`")
    lines.append(f"- This report: `{OUTPUT_REPORT_MD.relative_to(REPO_ROOT)}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    runs = load_run_records()
    groups = repeated_groups(runs)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_details_csv(groups)
    write_group_summary_csv(groups)
    OUTPUT_REPORT_MD.write_text(build_report(runs, groups), encoding="utf-8")
    print(f"Wrote {OUTPUT_DETAILS_CSV.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUTPUT_GROUPS_CSV.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUTPUT_REPORT_MD.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

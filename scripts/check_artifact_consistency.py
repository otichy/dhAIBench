#!/usr/bin/env python3
"""Check consistency between metrics JSON, output CSV, and prompt log artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS_DIR = REPO_ROOT / "data" / "metrics"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "output"
DEFAULT_LOGS_DIR = REPO_ROOT / "data" / "logs"
DEFAULT_BACKUP_DIR = REPO_ROOT / "data" / "backup"
DEFAULT_ANALYSIS_DIR = REPO_ROOT / "data" / "analysis"
METRICS_SUFFIXES = ("__metrics.json", "_metrics.json")
OUTPUT_TOKENS_DEFINITION = (
    "total_tokens - prompt_tokens (or completion_tokens + thinking_tokens fallback)"
)


@dataclass
class CsvTokenSummary:
    row_count: int = 0
    rows_with_token_usage: int = 0
    rows_with_output_tokens: int = 0
    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0
    total_tokens_total: int = 0
    output_tokens_total: int = 0


@dataclass
class LogSummary:
    example_records: int = 0
    attempts_total: int = 0
    attempts_with_usage_metadata: int = 0
    success_attempts: int = 0
    provider_values: set[str] = field(default_factory=set)
    model_requested_values: set[str] = field(default_factory=set)


@dataclass
class Issue:
    code: str
    detail: str


@dataclass
class ArtifactReport:
    metrics_path: Path
    output_path: Optional[Path]
    log_path: Optional[Path]
    expected_output_path: Path
    expected_log_path: Path
    csv_summary: Optional[CsvTokenSummary]
    log_summary: Optional[LogSummary]
    metrics_provider: str
    metrics_model_requested: str
    prediction_count: Optional[int]
    request_attempts_total: Optional[int]
    usage_attempts_with_usage_metadata: Optional[int]
    issues: list[Issue]


@dataclass
class RepairOutcome:
    metrics_path: Path
    backup_path: Optional[Path] = None
    token_totals_repaired: bool = False
    attempt_totals_repaired: bool = False
    notes: list[str] = field(default_factory=list)


def _as_int(value: object) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
        if not math.isfinite(number):
            return None
        return int(number)
    return None


def _metrics_base_name(metrics_path: Path) -> str:
    name = metrics_path.name
    for suffix in METRICS_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return metrics_path.stem


def _preferred_output_path(metrics_path: Path, payload: dict) -> Path:
    source_output_csv = str(payload.get("source_output_csv") or "").strip()
    if source_output_csv:
        return Path(source_output_csv)
    return DEFAULT_OUTPUT_DIR / f"{_metrics_base_name(metrics_path)}.csv"


def _iter_metrics_paths(inputs: list[str]) -> Iterable[Path]:
    if not inputs:
        inputs = [str(DEFAULT_METRICS_DIR)]

    seen: set[Path] = set()
    for raw in inputs:
        path = Path(raw)
        candidates: list[Path] = []
        if path.is_dir():
            for suffix in METRICS_SUFFIXES:
                candidates.extend(sorted(path.glob(f"*{suffix}")))
        elif path.is_file():
            candidates = [path]
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield candidate


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _resolve_output_path(metrics_path: Path, payload: dict) -> Optional[Path]:
    candidate = _preferred_output_path(metrics_path, payload)
    return candidate if candidate.exists() else None


def _preferred_log_path(metrics_path: Path, payload: dict) -> Path:
    output_candidate = _preferred_output_path(metrics_path, payload)
    stem = output_candidate.stem if output_candidate.suffix else _metrics_base_name(metrics_path)
    return DEFAULT_LOGS_DIR / f"{stem}.log"


def _resolve_log_path(metrics_path: Path, payload: dict) -> Optional[Path]:
    candidate = _preferred_log_path(metrics_path, payload)
    return candidate if candidate.exists() else None


def _detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = handle.readline()
    if header.count(";") >= max(header.count(","), header.count("\t")):
        return ";"
    if header.count("\t") > header.count(","):
        return "\t"
    return ","


def load_output_csv_summary(path: Path) -> CsvTokenSummary:
    delimiter = _detect_delimiter(path)
    summary = CsvTokenSummary()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row in reader:
            summary.row_count += 1
            prompt_tokens = _as_int(row.get("promptTokens"))
            completion_tokens = _as_int(row.get("completionTokens"))
            total_tokens = _as_int(row.get("totalTokens"))

            if prompt_tokens is not None:
                summary.prompt_tokens_total += prompt_tokens
            if completion_tokens is not None:
                summary.completion_tokens_total += completion_tokens
            if total_tokens is not None:
                summary.total_tokens_total += total_tokens

            if any(value is not None for value in (prompt_tokens, completion_tokens, total_tokens)):
                summary.rows_with_token_usage += 1

            output_tokens = None
            if total_tokens is not None and prompt_tokens is not None:
                output_tokens = max(total_tokens - prompt_tokens, 0)
            elif completion_tokens is not None:
                output_tokens = completion_tokens

            if output_tokens is not None:
                summary.output_tokens_total += output_tokens
                if output_tokens > 0:
                    summary.rows_with_output_tokens += 1

    return summary


def _iter_log_records(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def load_log_summary(path: Path) -> LogSummary:
    summary = LogSummary()
    for record in _iter_log_records(path):
        if record.get("record_type") == "run_metadata":
            model_details = record.get("model_details") or {}
            provider = str(model_details.get("provider") or "").strip()
            model_requested = str(model_details.get("model_requested") or "").strip()
            if provider:
                summary.provider_values.add(provider)
            if model_requested:
                summary.model_requested_values.add(model_requested)

        attempts = record.get("attempts")
        if not isinstance(attempts, list):
            continue
        if record.get("record_type") == "example_result":
            summary.example_records += 1
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            summary.attempts_total += 1
            if attempt.get("status") == "success":
                summary.success_attempts += 1
            response_payload = attempt.get("response")
            if isinstance(response_payload, dict) and isinstance(
                response_payload.get("usage_metadata"), dict
            ):
                summary.attempts_with_usage_metadata += 1
    return summary


def build_backfilled_token_usage_totals(existing_totals: Optional[dict], csv_summary: CsvTokenSummary) -> dict:
    existing = dict(existing_totals or {})
    cached_input_tokens_total = _as_int(existing.get("cached_input_tokens_total")) or 0
    thinking_tokens_total = _as_int(existing.get("thinking_tokens_total")) or 0
    attempts_with_cached_input_tokens = _as_int(existing.get("attempts_with_cached_input_tokens")) or 0
    attempts_with_thinking_tokens = _as_int(existing.get("attempts_with_thinking_tokens")) or 0

    backfilled = dict(existing)
    if _as_int(existing.get("attempts_total")) is not None:
        backfilled["attempts_total"] = _as_int(existing.get("attempts_total"))
    backfilled["attempts_with_token_usage"] = csv_summary.rows_with_token_usage
    backfilled["attempts_with_output_tokens"] = csv_summary.rows_with_output_tokens
    backfilled["attempts_with_cached_input_tokens"] = attempts_with_cached_input_tokens
    backfilled["attempts_with_thinking_tokens"] = attempts_with_thinking_tokens
    backfilled["input_tokens_total"] = csv_summary.prompt_tokens_total
    backfilled["cached_input_tokens_total"] = cached_input_tokens_total
    backfilled["non_cached_input_tokens_total"] = max(
        csv_summary.prompt_tokens_total - cached_input_tokens_total,
        0,
    )
    backfilled["output_tokens_total"] = csv_summary.output_tokens_total
    backfilled["thinking_tokens_total"] = thinking_tokens_total
    backfilled["output_tokens_definition"] = (
        str(existing.get("output_tokens_definition") or "").strip() or OUTPUT_TOKENS_DEFINITION
    )
    return backfilled


def update_metrics_token_totals_from_csv(metrics_path: Path) -> tuple[bool, Optional[Path], Optional[CsvTokenSummary]]:
    metrics_payload = _load_json(metrics_path)
    output_path = _resolve_output_path(metrics_path, metrics_payload)
    if output_path is None:
        return False, None, None

    csv_summary = load_output_csv_summary(output_path)
    existing_totals = metrics_payload.get("token_usage_totals")
    replacement = build_backfilled_token_usage_totals(existing_totals, csv_summary)
    if replacement == existing_totals:
        return False, output_path, csv_summary

    metrics_payload["token_usage_totals"] = replacement
    _write_json(metrics_path, metrics_payload)
    return True, output_path, csv_summary


def _maybe_add_numeric_mismatch(
    issues: list[Issue],
    code: str,
    label: str,
    metrics_value: object,
    expected_value: object,
) -> None:
    actual = _as_int(metrics_value)
    expected = _as_int(expected_value)
    if actual is None or expected is None:
        return
    if actual != expected:
        issues.append(Issue(code, f"{label}: metrics={actual}, derived={expected}"))


def analyze_metrics_artifact(metrics_path: Path) -> ArtifactReport:
    metrics_payload = _load_json(metrics_path)
    output_path = _resolve_output_path(metrics_path, metrics_payload)
    log_path = _resolve_log_path(metrics_path, metrics_payload)
    expected_output_path = _preferred_output_path(metrics_path, metrics_payload)
    expected_log_path = _preferred_log_path(metrics_path, metrics_payload)
    csv_summary = load_output_csv_summary(output_path) if output_path is not None else None
    log_summary = load_log_summary(log_path) if log_path is not None else None

    issues: list[Issue] = []
    token_usage_totals = metrics_payload.get("token_usage_totals") or {}
    request_control_summary = metrics_payload.get("request_control_summary") or {}
    usage_metadata_summary = metrics_payload.get("usage_metadata_summary") or {}
    model_details = metrics_payload.get("model_details") or {}
    metrics_provider = str(model_details.get("provider") or "").strip()
    metrics_model_requested = str(model_details.get("model_requested") or "").strip()
    prediction_count = _as_int(metrics_payload.get("prediction_count"))
    request_attempts_total = _as_int(request_control_summary.get("attempts_total"))
    usage_attempts_with_usage_metadata = _as_int(
        usage_metadata_summary.get("attempts_with_usage_metadata")
    )

    if output_path is None:
        issues.append(Issue("missing_output_csv", "No output CSV could be resolved for this metrics file."))
    if log_path is None:
        issues.append(Issue("missing_prompt_log", "No prompt log could be resolved for this metrics file."))

    if csv_summary is not None:
        _maybe_add_numeric_mismatch(
            issues,
            "prediction_count_mismatch_csv_rows",
            "prediction_count vs output rows",
            metrics_payload.get("prediction_count"),
            csv_summary.row_count,
        )
        _maybe_add_numeric_mismatch(
            issues,
            "attempts_with_token_usage_mismatch_csv",
            "attempts_with_token_usage vs rows with token columns",
            token_usage_totals.get("attempts_with_token_usage"),
            csv_summary.rows_with_token_usage,
        )
        _maybe_add_numeric_mismatch(
            issues,
            "input_tokens_total_mismatch_csv",
            "input_tokens_total vs CSV promptTokens sum",
            token_usage_totals.get("input_tokens_total"),
            csv_summary.prompt_tokens_total,
        )
        _maybe_add_numeric_mismatch(
            issues,
            "output_tokens_total_mismatch_csv",
            "output_tokens_total vs CSV-derived output token sum",
            token_usage_totals.get("output_tokens_total"),
            csv_summary.output_tokens_total,
        )

    if log_summary is not None:
        _maybe_add_numeric_mismatch(
            issues,
            "attempts_total_mismatch_log",
            "request_control_summary.attempts_total vs prompt-log attempts",
            request_control_summary.get("attempts_total"),
            log_summary.attempts_total,
        )
        _maybe_add_numeric_mismatch(
            issues,
            "usage_attempts_mismatch_log",
            "usage_metadata_summary.attempts_with_usage_metadata vs prompt-log attempts with usage",
            usage_metadata_summary.get("attempts_with_usage_metadata"),
            log_summary.attempts_with_usage_metadata,
        )

        if metrics_provider and log_summary.provider_values and metrics_provider not in log_summary.provider_values:
            issues.append(
                Issue(
                    "provider_mismatch_log",
                    f"metrics provider={metrics_provider}, log providers={sorted(log_summary.provider_values)}",
                )
            )
        if (
            metrics_model_requested
            and log_summary.model_requested_values
            and metrics_model_requested not in log_summary.model_requested_values
        ):
            issues.append(
                Issue(
                    "model_requested_mismatch_log",
                    "metrics model_requested="
                    f"{metrics_model_requested}, log models={sorted(log_summary.model_requested_values)}",
                )
            )

    return ArtifactReport(
        metrics_path=metrics_path,
        output_path=output_path,
        log_path=log_path,
        expected_output_path=expected_output_path,
        expected_log_path=expected_log_path,
        csv_summary=csv_summary,
        log_summary=log_summary,
        metrics_provider=metrics_provider,
        metrics_model_requested=metrics_model_requested,
        prediction_count=prediction_count,
        request_attempts_total=request_attempts_total,
        usage_attempts_with_usage_metadata=usage_attempts_with_usage_metadata,
        issues=issues,
    )


def _relative_backup_path(metrics_path: Path) -> Path:
    try:
        return metrics_path.resolve().relative_to(DEFAULT_METRICS_DIR.resolve())
    except ValueError:
        return Path(metrics_path.name)


def ensure_metrics_backup(metrics_path: Path, backup_run_dir: Path) -> Path:
    backup_path = backup_run_dir / _relative_backup_path(metrics_path)
    if not backup_path.exists():
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(metrics_path, backup_path)
    return backup_path


def _should_repair_token_totals_from_csv(existing_totals: dict, replacement_totals: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    comparable_fields = [
        "attempts_with_token_usage",
        "attempts_with_output_tokens",
        "input_tokens_total",
        "non_cached_input_tokens_total",
        "output_tokens_total",
    ]
    for field_name in comparable_fields:
        existing_value = _as_int(existing_totals.get(field_name))
        replacement_value = _as_int(replacement_totals.get(field_name))
        if existing_value is None or replacement_value is None:
            continue
        if replacement_value < existing_value:
            reasons.append(f"{field_name}: metrics={existing_value}, csv={replacement_value}")
    return bool(reasons), reasons


def _repair_attempt_totals_from_log(metrics_payload: dict, log_summary: LogSummary) -> list[str]:
    reasons: list[str] = []
    fields = [
        ("request_control_summary", "attempts_total"),
        ("usage_metadata_summary", "attempts_total"),
        ("token_usage_totals", "attempts_total"),
    ]
    for section_name, field_name in fields:
        section = metrics_payload.get(section_name)
        if not isinstance(section, dict):
            continue
        existing_value = _as_int(section.get(field_name))
        if existing_value is None or log_summary.attempts_total >= existing_value:
            continue
        section[field_name] = log_summary.attempts_total
        reasons.append(
            f"{section_name}.{field_name}: metrics={existing_value}, prompt_log={log_summary.attempts_total}"
        )
    return reasons


def repair_metrics_artifact(
    metrics_path: Path,
    backup_run_dir: Path,
    *,
    repair_lower_token_totals_from_csv: bool = False,
    repair_lower_attempt_totals_from_log: bool = False,
) -> RepairOutcome:
    metrics_payload = _load_json(metrics_path)
    output_path = _resolve_output_path(metrics_path, metrics_payload)
    log_path = _resolve_log_path(metrics_path, metrics_payload)

    outcome = RepairOutcome(metrics_path=metrics_path)
    modified = False

    if repair_lower_token_totals_from_csv and output_path is not None:
        csv_summary = load_output_csv_summary(output_path)
        existing_totals = metrics_payload.get("token_usage_totals") or {}
        replacement_totals = build_backfilled_token_usage_totals(existing_totals, csv_summary)
        should_repair, reasons = _should_repair_token_totals_from_csv(existing_totals, replacement_totals)
        if should_repair:
            metrics_payload["token_usage_totals"] = replacement_totals
            outcome.token_totals_repaired = True
            outcome.notes.extend(reasons)
            modified = True

    if repair_lower_attempt_totals_from_log and log_path is not None:
        log_summary = load_log_summary(log_path)
        reasons = _repair_attempt_totals_from_log(metrics_payload, log_summary)
        if reasons:
            outcome.attempt_totals_repaired = True
            outcome.notes.extend(reasons)
            modified = True

    if modified:
        outcome.backup_path = ensure_metrics_backup(metrics_path, backup_run_dir)
        _write_json(metrics_path, metrics_payload)

    return outcome


def _escape_md(text: object) -> str:
    value = str(text if text is not None else "")
    return value.replace("|", "\\|").replace("\n", " ")


def _render_markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    if not rows:
        return ["_None._"]
    header_line = "| " + " | ".join(_escape_md(header) for header in headers) + " |"
    separator_line = "| " + " | ".join("---" for _ in headers) + " |"
    body_lines = [
        "| " + " | ".join(_escape_md(cell) for cell in row) + " |"
        for row in rows
    ]
    return [header_line, separator_line, *body_lines]


def write_markdown_report(
    reports: list[ArtifactReport],
    repair_outcomes: list[RepairOutcome],
    report_path: Path,
    backup_run_dir: Optional[Path],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    repaired_rows = [
        [
            outcome.metrics_path.name,
            "yes" if outcome.token_totals_repaired else "",
            "yes" if outcome.attempt_totals_repaired else "",
            str(outcome.backup_path) if outcome.backup_path is not None else "",
            "; ".join(outcome.notes),
        ]
        for outcome in repair_outcomes
        if outcome.token_totals_repaired or outcome.attempt_totals_repaired
    ]
    model_mismatch_rows = []
    missing_artifact_rows = []
    usage_mismatch_rows = []
    prediction_mismatch_rows = []

    for report in reports:
        if any(issue.code == "model_requested_mismatch_log" for issue in report.issues):
            log_models = sorted(report.log_summary.model_requested_values) if report.log_summary else []
            model_mismatch_rows.append(
                [
                    report.metrics_path.name,
                    report.metrics_model_requested,
                    "; ".join(log_models),
                ]
            )

        missing_csv = report.output_path is None
        missing_log = report.log_path is None
        if missing_csv or missing_log:
            missing_artifact_rows.append(
                [
                    report.metrics_path.name,
                    "yes" if missing_csv else "",
                    str(report.expected_output_path),
                    "yes" if missing_log else "",
                    str(report.expected_log_path),
                ]
            )

        if any(issue.code == "usage_attempts_mismatch_log" for issue in report.issues):
            usage_mismatch_rows.append(
                [
                    report.metrics_path.name,
                    report.usage_attempts_with_usage_metadata
                    if report.usage_attempts_with_usage_metadata is not None
                    else "",
                    report.log_summary.attempts_with_usage_metadata if report.log_summary else "",
                ]
            )

        if any(issue.code == "prediction_count_mismatch_csv_rows" for issue in report.issues):
            prediction_mismatch_rows.append(
                [
                    report.metrics_path.name,
                    report.prediction_count if report.prediction_count is not None else "",
                    report.csv_summary.row_count if report.csv_summary is not None else "",
                ]
            )

    lines = [
        "# Artifact Consistency Report",
        "",
        f"- Checked metrics files: {len(reports)}",
        f"- Metrics files repaired: {sum(1 for row in repaired_rows)}",
        f"- Backup directory: {backup_run_dir if backup_run_dir is not None else ''}",
        "",
        "## Repairs Applied",
        "",
        *_render_markdown_table(
            ["Metrics File", "Token Totals Repaired", "Attempt Totals Repaired", "Backup JSON", "Notes"],
            repaired_rows,
        ),
        "",
        "## model_requested Disagreements",
        "",
        *_render_markdown_table(
            ["Metrics File", "metrics.model_requested", "prompt_log model_requested values"],
            model_mismatch_rows,
        ),
        "",
        "## Missing CSVs And Logs",
        "",
        *_render_markdown_table(
            ["Metrics File", "Missing CSV", "Expected CSV Path", "Missing Log", "Expected Log Path"],
            missing_artifact_rows,
        ),
        "",
        "## Usage-Attempt Disagreements",
        "",
        *_render_markdown_table(
            ["Metrics File", "metrics attempts_with_usage_metadata", "prompt_log attempts_with_usage_metadata"],
            usage_mismatch_rows,
        ),
        "",
        "## Prediction Count Vs Output Rows",
        "",
        *_render_markdown_table(
            ["Metrics File", "metrics prediction_count", "output CSV rows"],
            prediction_mismatch_rows,
        ),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _format_report(report: ArtifactReport) -> str:
    lines = [str(report.metrics_path)]
    if report.output_path is not None:
        lines.append(f"  output_csv: {report.output_path}")
    if report.log_path is not None:
        lines.append(f"  prompt_log: {report.log_path}")
    for issue in report.issues:
        lines.append(f"  - {issue.code}: {issue.detail}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check metrics/output/log consistency and optionally backfill token totals from output CSVs."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Metrics JSON files or directories. Defaults to data/metrics.",
    )
    parser.add_argument(
        "--write-token-totals-from-csv",
        action="store_true",
        help="Overwrite token_usage_totals using the paired output CSV token columns where available.",
    )
    parser.add_argument(
        "--repair-lower-token-totals-from-csv",
        action="store_true",
        help=(
            "When CSV-derived token totals/counts are lower than the metrics JSON values, "
            "rewrite token_usage_totals from the CSV and back up the original metrics JSON."
        ),
    )
    parser.add_argument(
        "--repair-lower-attempt-totals-from-log",
        action="store_true",
        help=(
            "When prompt-log attempts_total is lower than metrics attempts_total, "
            "rewrite the metrics attempts_total fields from the prompt log and back up the original metrics JSON."
        ),
    )
    parser.add_argument(
        "--backup-dir",
        default=str(DEFAULT_BACKUP_DIR),
        help="Root directory for backup copies of metrics JSONs modified by repair operations.",
    )
    parser.add_argument(
        "--report-path",
        default=str(DEFAULT_ANALYSIS_DIR / "artifact_consistency_report.md"),
        help="Markdown report path for mismatch tables.",
    )
    parser.add_argument(
        "--only-mismatched",
        action="store_true",
        help="Only print reports for metrics files with at least one mismatch.",
    )
    args = parser.parse_args(argv)

    metrics_paths = list(_iter_metrics_paths(args.paths))
    if not metrics_paths:
        parser.error("No metrics JSON files found.")

    updated_count = 0
    repair_outcomes: list[RepairOutcome] = []
    backup_run_dir: Optional[Path] = None
    if args.repair_lower_token_totals_from_csv or args.repair_lower_attempt_totals_from_log:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        backup_run_dir = Path(args.backup_dir) / f"artifact_consistency_{timestamp}"
    reports: list[ArtifactReport] = []
    for metrics_path in metrics_paths:
        if args.write_token_totals_from_csv:
            updated, _, _ = update_metrics_token_totals_from_csv(metrics_path)
            if updated:
                updated_count += 1
        if backup_run_dir is not None:
            outcome = repair_metrics_artifact(
                metrics_path,
                backup_run_dir,
                repair_lower_token_totals_from_csv=args.repair_lower_token_totals_from_csv,
                repair_lower_attempt_totals_from_log=args.repair_lower_attempt_totals_from_log,
            )
            if outcome.token_totals_repaired or outcome.attempt_totals_repaired:
                updated_count += 1
            repair_outcomes.append(outcome)
        reports.append(analyze_metrics_artifact(metrics_path))

    mismatched = [report for report in reports if report.issues]
    selected = mismatched if args.only_mismatched else reports
    report_path = Path(args.report_path)
    write_markdown_report(reports, repair_outcomes, report_path, backup_run_dir)

    print(
        f"Checked {len(reports)} metrics file(s); "
        f"{len(mismatched)} mismatched; {updated_count} updated; report={report_path}."
    )
    for report in selected:
        print()
        print(_format_report(report))

    return 1 if mismatched else 0


if __name__ == "__main__":
    raise SystemExit(main())

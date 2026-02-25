#!/usr/bin/env python3
"""LLM Linguistic Classification Benchmark Agent.

This script loads a semicolon-delimited dataset of classification examples,
queries an OpenAI-compatible large language model, and evaluates the model's
predictions against ground-truth labels. The tool also supports optional label
files, calibration plots, and command generation via a companion HTML/JS GUI.
"""

from __future__ import annotations

import argparse
import base64
import codecs
import copy
import csv
import json
import logging
import math
import os
import re
import shlex
import sys
import time
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple

from validator_client import ValidatorClient, ValidatorError, ValidatorRunInfo


NODE_MARKER_LEFT = "âź¦"
NODE_MARKER_RIGHT = "âź§"
SPAN_SOURCE_NODE = "node"
SPAN_SOURCE_MARKED_SUBSPAN = "marked_subspan"
MANDATORY_SYSTEM_APPEND = (
    "Classify ONLY the text that is explicitly wrapped inside âź¦ âź§ (the 'node' or its marked sub-span). "
    "Use the surrounding context as supporting evidence, but never change the focus away from the highlighted text. "
    'If you cannot determine the class/label for the node, return "unclassified".'
)
EMPTY_RESPONSE_RETRY_DELAY_SECONDS = 120.0
MAX_CACHE_PADDING_TOKENS = 200_000
TOKEN_CHAR_ESTIMATE_RATIO = 4.0
CACHE_PADDING_PREFIX = "Cache-normalization filler block. Ignore this block for classification.\nCACHE_PAD_BEGIN"
CACHE_PADDING_TOKEN = " cachepad"
CACHE_PADDING_SUFFIX = "\nCACHE_PAD_END"
VERTEX_DEFAULT_ACCESS_TOKEN_COMMAND = "gcloud auth application-default print-access-token"
VERTEX_DEFAULT_ACCESS_TOKEN_REFRESH_SECONDS = 3300
VERTEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 300
VERTEX_DEFAULT_ADC_LOGIN_COMMAND = "gcloud auth application-default login"


# --------------------------- Utilities ------------------------------------- #


def parse_env_file(path: str) -> Dict[str, str]:
    """Parse KEY=VALUE entries from a .env-style file into a dictionary."""
    values: Dict[str, str] = {}
    if not path:
        return values

    if not os.path.exists(path):
        logging.debug("Env file %s does not exist; skipping.", path)
        return values

    with open(path, "r", encoding="utf-8") as env_file:
        for line in env_file:
            striped = line.strip()
            if not striped or striped.startswith("#"):
                continue
            if "=" not in striped:
                logging.warning("Skipping malformed env line: %s", striped)
                continue
            key, value = striped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                values[key] = value
    return values


def resolve_env_value(key: Optional[str], env_map: Dict[str, str]) -> Optional[str]:
    """Resolve an env var from .env values first, then process environment."""
    if not key:
        return None
    if key in env_map:
        return env_map[key]
    return os.environ.get(key)


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp with a trailing Z."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc_timestamp(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp into a timezone-aware UTC datetime."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    normalized = f"{text[:-1]}+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_duration_human(total_seconds: float) -> str:
    """Format duration in a compact human-readable form like '2h 03m 04s'."""
    seconds_int = int(round(max(0.0, total_seconds)))
    days, remainder = divmod(seconds_int, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m {seconds:02d}s"
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def compute_prompt_time_window(log_records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute first/last prompt timestamps and elapsed seconds across prompt attempts."""
    first_prompt: Optional[datetime] = None
    last_prompt: Optional[datetime] = None

    for record in log_records:
        if not isinstance(record, dict):
            continue
        attempts = record.get("attempts")
        if not isinstance(attempts, list):
            continue
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            attempt_time = parse_utc_timestamp(attempt.get("timestamp"))
            if attempt_time is None:
                continue
            if first_prompt is None or attempt_time < first_prompt:
                first_prompt = attempt_time
            if last_prompt is None or attempt_time > last_prompt:
                last_prompt = attempt_time

    if first_prompt is None or last_prompt is None:
        return {
            "first_prompt_timestamp": None,
            "last_prompt_timestamp": None,
            "overall_time_seconds": None,
            "overall_time_human": None,
        }

    elapsed_seconds = max(0.0, (last_prompt - first_prompt).total_seconds())
    return {
        "first_prompt_timestamp": first_prompt.isoformat().replace("+00:00", "Z"),
        "last_prompt_timestamp": last_prompt.isoformat().replace("+00:00", "Z"),
        "overall_time_seconds": elapsed_seconds,
        "overall_time_human": format_duration_human(elapsed_seconds),
    }


def compute_request_control_summary(
    log_records: Iterable[Dict[str, Any]],
    configured_controls: Dict[str, Any],
) -> Dict[str, Any]:
    """Aggregate request-control acceptance telemetry from prompt logs."""
    control_keys = (
        "reasoning_effort",
        "thinking_level",
        "effort",
        "verbosity",
        "prompt_cache_key",
        "gemini_cached_content",
        "requesty_auto_cache",
    )
    per_control: Dict[str, Dict[str, Any]] = {
        key: {
            "configured_value": configured_controls.get(key),
            "requested_attempts": 0,
            "sent_attempts": 0,
            "accepted_attempts": 0,
            "rejected_attempts": 0,
            "missing_from_final_request_attempts": 0,
            "acceptance_rate": None,
            "rejected_reasons": {},
            "rejected_example_ids": [],
        }
        for key in control_keys
    }

    attempts_total = 0
    attempts_with_telemetry = 0

    for record in log_records:
        if not isinstance(record, dict):
            continue
        example_id = str(record.get("example_id", "")).strip()
        attempts = record.get("attempts")
        if not isinstance(attempts, list):
            continue

        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            attempts_total += 1

            request_controls = attempt.get("request_controls")
            if not isinstance(request_controls, dict):
                continue
            attempts_with_telemetry += 1

            requested = request_controls.get("requested")
            sent = request_controls.get("sent")
            rejected = request_controls.get("rejected")
            requested_map = requested if isinstance(requested, dict) else {}
            sent_map = sent if isinstance(sent, dict) else {}
            rejected_map = rejected if isinstance(rejected, dict) else {}

            for key in control_keys:
                control_stats = per_control[key]
                requested_flag = key in requested_map and requested_map.get(key) not in (None, "")
                sent_flag = key in sent_map and sent_map.get(key) not in (None, "")
                rejected_reason = rejected_map.get(key)

                if requested_flag:
                    control_stats["requested_attempts"] += 1
                if sent_flag:
                    control_stats["sent_attempts"] += 1
                if requested_flag and not sent_flag:
                    control_stats["missing_from_final_request_attempts"] += 1

                if rejected_reason is not None:
                    control_stats["rejected_attempts"] += 1
                    reason_text = str(rejected_reason).strip() or "unknown"
                    reasons = control_stats["rejected_reasons"]
                    reasons[reason_text] = int(reasons.get(reason_text, 0)) + 1
                    rejected_examples = control_stats["rejected_example_ids"]
                    if (
                        example_id
                        and example_id not in rejected_examples
                        and len(rejected_examples) < 50
                    ):
                        rejected_examples.append(example_id)

                if requested_flag and sent_flag and rejected_reason is None:
                    control_stats["accepted_attempts"] += 1

    for key in control_keys:
        control_stats = per_control[key]
        requested_attempts = int(control_stats["requested_attempts"])
        if requested_attempts > 0:
            control_stats["acceptance_rate"] = (
                control_stats["accepted_attempts"] / requested_attempts
            )

    configured_non_null = {
        key: value for key, value in configured_controls.items() if value is not None
    }
    return {
        "configured": configured_non_null,
        "attempts_total": attempts_total,
        "attempts_with_control_telemetry": attempts_with_telemetry,
        "per_control": per_control,
    }


def _flatten_numeric_metrics(
    value: Any,
    prefix: str = "",
    out: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Flatten nested numeric values into a dot-path map."""
    if out is None:
        out = {}
    if isinstance(value, dict):
        for raw_key, nested in value.items():
            key = str(raw_key)
            next_prefix = f"{prefix}.{key}" if prefix else key
            _flatten_numeric_metrics(nested, next_prefix, out)
        return out
    if isinstance(value, list):
        for nested in value:
            next_prefix = f"{prefix}[]" if prefix else "[]"
            _flatten_numeric_metrics(nested, next_prefix, out)
        return out
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            out[prefix or "value"] = out.get(prefix or "value", 0.0) + number
    return out


def _normalize_metric_number(value: float) -> Any:
    """Render whole-number floats as ints in metrics output."""
    rounded = round(value)
    if math.isfinite(value) and abs(value - rounded) < 1e-9:
        return int(rounded)
    return value


def compute_usage_metadata_summary(log_records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate usage-metadata/caching signals from prompt logs."""
    attempts_total = 0
    attempts_with_usage_metadata = 0
    attempts_with_cached_token_signals = 0
    cached_tokens_total_estimate = 0.0
    cache_read_tokens_total = 0.0
    cache_write_tokens_total = 0.0
    cache_token_fields_totals: Dict[str, float] = {}
    attempts_with_gemini_cached_content_token_signals = 0
    gemini_cached_content_token_count_total = 0.0
    gemini_cached_content_token_fields_totals: Dict[str, float] = {}

    for record in log_records:
        if not isinstance(record, dict):
            continue
        attempts = record.get("attempts")
        if not isinstance(attempts, list):
            continue
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            attempts_total += 1
            response_payload = attempt.get("response")
            if not isinstance(response_payload, dict):
                continue
            usage_metadata = response_payload.get("usage_metadata")
            if not isinstance(usage_metadata, dict):
                continue
            attempts_with_usage_metadata += 1

            flattened = _flatten_numeric_metrics(usage_metadata)
            attempt_cache_values: List[float] = []
            attempt_has_gemini_cached_content_tokens = False
            for field_path, value in flattened.items():
                lowered = field_path.lower()
                if ("cache" in lowered or "cached" in lowered) and "token" in lowered:
                    cache_token_fields_totals[field_path] = (
                        cache_token_fields_totals.get(field_path, 0.0) + value
                    )
                    attempt_cache_values.append(value)
                    if "read" in lowered or "hit" in lowered:
                        cache_read_tokens_total += value
                    elif "cached" in lowered and "write" not in lowered and "creation" not in lowered and "create" not in lowered:
                        # Fields like prompt_tokens_details.cached_tokens (OpenAI) and
                        # usageMetadata.cachedContentTokenCount (Gemini) represent cache reads
                        # but don't contain "read" or "hit" in their name.
                        cache_read_tokens_total += value
                    if "write" in lowered or "creation" in lowered or "create" in lowered:
                        cache_write_tokens_total += value
                canonical = re.sub(r"[^a-z0-9]", "", lowered)
                if "cachedcontenttokencount" in canonical:
                    gemini_cached_content_token_fields_totals[field_path] = (
                        gemini_cached_content_token_fields_totals.get(field_path, 0.0) + value
                    )
                    gemini_cached_content_token_count_total += value
                    attempt_has_gemini_cached_content_tokens = True

            if attempt_cache_values:
                attempts_with_cached_token_signals += 1
                cached_tokens_total_estimate += max(attempt_cache_values)
            if attempt_has_gemini_cached_content_tokens:
                attempts_with_gemini_cached_content_token_signals += 1

    normalized_fields = {
        field_path: _normalize_metric_number(value)
        for field_path, value in sorted(cache_token_fields_totals.items())
    }
    normalized_gemini_fields = {
        field_path: _normalize_metric_number(value)
        for field_path, value in sorted(gemini_cached_content_token_fields_totals.items())
    }
    return {
        "attempts_total": attempts_total,
        "attempts_with_usage_metadata": attempts_with_usage_metadata,
        "attempts_with_cached_token_signals": attempts_with_cached_token_signals,
        "cached_tokens_total_estimate": _normalize_metric_number(cached_tokens_total_estimate),
        "cache_read_tokens_total": _normalize_metric_number(cache_read_tokens_total),
        "cache_write_tokens_total": _normalize_metric_number(cache_write_tokens_total),
        "cache_token_fields_totals": normalized_fields,
        "attempts_with_gemini_cached_content_token_signals": attempts_with_gemini_cached_content_token_signals,
        "gemini_cached_content_token_count_total": _normalize_metric_number(
            gemini_cached_content_token_count_total
        ),
        "gemini_cached_content_token_fields_totals": normalized_gemini_fields,
    }


def is_placeholder_value(value: Optional[str]) -> bool:
    """Return True if a value looks like an unresolved placeholder token."""
    if value is None:
        return True
    normalized = str(value).strip().lower()
    if not normalized:
        return True
    placeholder_prefixes = ("your-", "replace-", "changeme", "<")
    if normalized.startswith(placeholder_prefixes):
        return True
    if "your-" in normalized and "api-key" in normalized:
        return True
    return False


def decode_cli_system_prompt(raw_prompt: Optional[str]) -> Optional[str]:
    """Decode GUI-escaped newline/tab/backslash sequences back into the original text."""
    if raw_prompt is None or "\\" not in raw_prompt:
        return raw_prompt
    try:
        return codecs.decode(raw_prompt, "unicode_escape")
    except Exception:
        logging.debug("Failed to decode system prompt escapes; using raw value.")
        replacements = (
            ("\\r\\n", "\n"),
            ("\\n", "\n"),
            ("\\r", "\r"),
            ("\\t", "\t"),
            ("\\\\", "\\"),
        )
        decoded = raw_prompt
        for pattern, replacement in replacements:
            decoded = decoded.replace(pattern, replacement)
        return decoded


def decode_system_prompt_b64(encoded_prompt: Optional[str]) -> Optional[str]:
    """Decode a base64-encoded system prompt produced by the GUI."""
    if not encoded_prompt:
        return None
    try:
        decoded_bytes = base64.b64decode(encoded_prompt, validate=True)
        return decoded_bytes.decode("utf-8")
    except Exception:
        logging.error("Unable to decode system prompt from base64; using default system prompt instead.")
        return None


def safe_float(value: Any, default: float = float("nan")) -> float:
    """Convert a value to float, returning default on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def is_quota_or_rate_limit_error(exc: BaseException) -> bool:
    """Best-effort detection for provider quota/rate-limit failures."""
    for attr_name in ("status_code", "status", "http_status"):
        status_value = getattr(exc, attr_name, None)
        if status_value is None:
            continue
        status_text = str(status_value).strip()
        if status_text == "429":
            return True

    text = str(exc).strip().lower()
    indicators = (
        "429",
        "too many requests",
        "rate limit",
        "resource has been exhausted",
        "resource exhausted",
        "insufficient_quota",
        "exceeded your current quota",
        "quota",
    )
    return any(marker in text for marker in indicators)


def is_empty_model_response_error(exc: BaseException) -> bool:
    """Best-effort detection for empty textual model responses."""
    text = str(exc).strip().lower()
    if not text:
        return False
    markers = (
        "empty model response",
        "empty response",
        "response was empty",
        "no output text",
    )
    return any(marker in text for marker in markers)


def ensure_directory(path: str) -> None:
    """Create the directory for path if it does not yet exist."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def parse_optional_int(value: Any) -> Optional[int]:
    """Parse an optional integer-like CSV value."""
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        try:
            number = float(text)
        except ValueError:
            return None
        if math.isfinite(number) and number.is_integer():
            return int(number)
    return None


def parse_optional_float(value: Any) -> Optional[float]:
    """Parse an optional floating-point CSV value."""
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def parse_optional_bool(value: Any) -> Optional[bool]:
    """Parse an optional boolean-like value."""
    text = "" if value is None else str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


class AccessTokenProvider(Protocol):
    """Protocol for providers capable of returning short-lived auth tokens."""

    def get_token(self, force_refresh: bool = False) -> str:
        ...


class VertexAccessTokenProvider:
    """Refresh Vertex OAuth access tokens via a shell command."""

    def __init__(
        self,
        initial_token: Optional[str],
        refresh_command: Optional[str] = None,
        refresh_interval_seconds: int = VERTEX_DEFAULT_ACCESS_TOKEN_REFRESH_SECONDS,
        auto_adc_login: bool = True,
        adc_login_command: Optional[str] = None,
    ) -> None:
        normalized_initial = (initial_token or "").strip()
        self._token: Optional[str] = normalized_initial or None
        # Unknown age for externally supplied token: force an early refresh attempt.
        self._expires_at_epoch: Optional[float] = 0.0 if self._token else None
        self._refresh_command = (
            (refresh_command or VERTEX_DEFAULT_ACCESS_TOKEN_COMMAND).strip()
            or VERTEX_DEFAULT_ACCESS_TOKEN_COMMAND
        )
        self._refresh_interval_seconds = max(60, int(refresh_interval_seconds))
        self._auto_adc_login = bool(auto_adc_login)
        self._adc_login_command = (
            (adc_login_command or VERTEX_DEFAULT_ADC_LOGIN_COMMAND).strip()
            or VERTEX_DEFAULT_ADC_LOGIN_COMMAND
        )
        self._attempted_adc_login = False
        self._warned_static_fallback = False

    def _extract_subprocess_error_text(self, exc: subprocess.CalledProcessError) -> str:
        stdout = str(exc.stdout or "").strip()
        stderr = str(exc.stderr or "").strip()
        pieces = [piece for piece in (stderr, stdout, str(exc)) if piece]
        return " | ".join(pieces).lower()

    def _looks_like_missing_adc(self, exc: subprocess.CalledProcessError) -> bool:
        text = self._extract_subprocess_error_text(exc)
        indicators = (
            "default credentials were not found",
            "application default credentials",
            "run `gcloud auth application-default login`",
            "run 'gcloud auth application-default login'",
            "gcloud.auth.application-default.print-access-token",
        )
        return any(marker in text for marker in indicators)

    def _can_attempt_interactive_adc_login(self) -> bool:
        return bool(self._auto_adc_login and sys.stdin.isatty() and sys.stdout.isatty())

    def _attempt_adc_login(self) -> bool:
        if self._attempted_adc_login:
            return False
        if not self._can_attempt_interactive_adc_login():
            return False
        self._attempted_adc_login = True
        logging.warning(
            "Vertex Application Default Credentials are missing. Launching interactive login: %s",
            self._adc_login_command,
        )
        try:
            subprocess.run(self._adc_login_command, shell=True, check=True)
            logging.info("Vertex ADC login completed. Retrying access-token refresh.")
            return True
        except Exception as login_exc:  # noqa: BLE001
            logging.error("Vertex ADC interactive login failed: %s", login_exc)
            return False

    def _refresh_from_command(self) -> str:
        completed = subprocess.run(
            self._refresh_command,
            shell=True,
            capture_output=True,
            text=True,
            check=True,
        )
        stdout = (completed.stdout or "").strip()
        if not stdout:
            stderr = (completed.stderr or "").strip()
            raise RuntimeError(
                "Vertex token refresh command returned no token output."
                + (f" stderr: {stderr}" if stderr else "")
            )
        token = stdout.splitlines()[-1].strip()
        if not token:
            raise RuntimeError("Vertex token refresh command returned an empty token line.")
        self._token = token
        self._expires_at_epoch = time.time() + self._refresh_interval_seconds
        return token

    def get_token(self, force_refresh: bool = False) -> str:
        now = time.time()
        if not force_refresh and self._token:
            if self._expires_at_epoch is None:
                return self._token
            if now < (self._expires_at_epoch - VERTEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS):
                return self._token

        try:
            return self._refresh_from_command()
        except subprocess.CalledProcessError as exc:
            if self._looks_like_missing_adc(exc) and self._attempt_adc_login():
                try:
                    return self._refresh_from_command()
                except Exception:  # noqa: BLE001
                    pass
            if self._token:
                self._expires_at_epoch = None
                if not self._warned_static_fallback:
                    logging.warning(
                        "Unable to refresh Vertex access token automatically (%s). "
                        "Continuing with the currently configured token.",
                        exc,
                    )
                    self._warned_static_fallback = True
                return self._token
            raise RuntimeError(
                "Unable to obtain a Vertex access token. "
                "Run `gcloud auth application-default login` and ensure the refresh "
                f"command works: {self._refresh_command}"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            if self._token:
                self._expires_at_epoch = None
                if not self._warned_static_fallback:
                    logging.warning(
                        "Unable to refresh Vertex access token automatically (%s). "
                        "Continuing with the currently configured token.",
                        exc,
                    )
                    self._warned_static_fallback = True
                return self._token
            raise RuntimeError(
                "Unable to obtain a Vertex access token. "
                "Run `gcloud auth application-default login` and ensure the refresh "
                f"command works: {self._refresh_command}"
            ) from exc


def extract_json_object(text: str) -> Dict[str, Any]:
    """Extract and parse the first JSON object from a string."""
    text = text.strip()
    if not text:
        raise ValueError("Empty model response")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def sanitize_model_identifier(model: str) -> str:
    """Return a filesystem-friendly model identifier."""
    slug = re.sub(r"[^0-9A-Za-z]+", "", model.lower())
    return slug or "model"


def split_validator_args(raw_args: Optional[str]) -> List[str]:
    """Split a validator args string into argv tokens (supports quoting)."""
    if not raw_args:
        return []
    raw = str(raw_args).strip()
    if not raw:
        return []
    try:
        return shlex.split(raw, posix=(os.name != "nt"))
    except ValueError as exc:
        raise ValueError(f"Unable to parse --validator_args: {exc}") from exc


def build_validator_command(validator_cmd: str, raw_validator_args: Optional[str]) -> List[str]:
    """Build argv for the validator process, using sys.executable for .py scripts."""
    cmd = (validator_cmd or "").strip()
    if not cmd:
        raise ValueError("validator_cmd is empty")
    extra = split_validator_args(raw_validator_args)
    if cmd.lower().endswith(".py"):
        return [sys.executable, cmd, *extra]
    return [cmd, *extra]


def render_validator_retry_message(
    allowed_labels: Iterable[Any],
    instruction: str,
    max_candidates: int,
    max_chars: int,
) -> str:
    """Build a deterministic retry instruction appended as an extra user message."""
    cleaned: List[str] = []
    seen: set[str] = set()
    for item in allowed_labels or []:
        value = str(item).strip()
        if not value or value in seen:
            continue
        cleaned.append(value)
        seen.add(value)

    if max_candidates > 0:
        cleaned = cleaned[:max_candidates]

    base_instruction = (instruction or "").strip() or "Choose the correct label from allowed_labels."

    def build_text(labels: List[str], truncated_from: Optional[int] = None) -> str:
        note = ""
        if truncated_from is not None and truncated_from > len(labels):
            note = f"\n\n(Note: allowed_labels truncated from {truncated_from} to {len(labels)} item(s) to fit limits.)"
        serialized = json.dumps(labels, ensure_ascii=False, indent=2)
        return (
            "External validator rejected the previous label.\n\n"
            f"{base_instruction}\n\n"
            'You MUST set "label" to exactly one item in allowed_labels (case-sensitive). '
            'If none fit, return "unclassified".\n\n'
            f"allowed_labels:\n{serialized}{note}"
        )

    if max_chars <= 0:
        return build_text(cleaned)

    full = build_text(cleaned)
    if len(full) <= max_chars:
        return full

    original_count = len(cleaned)
    labels = cleaned
    while len(labels) > 1:
        labels = labels[: max(1, len(labels) // 2)]
        candidate = build_text(labels, truncated_from=original_count)
        if len(candidate) <= max_chars:
            return candidate

    # Fall back to a compact format (no indentation) if still too large.
    compact = (
        "External validator rejected the previous label.\n\n"
        f"{base_instruction}\n\n"
        'You MUST set "label" to exactly one item in allowed_labels (case-sensitive). '
        'If none fit, return "unclassified".\n\n'
        f"allowed_labels: {json.dumps(labels, ensure_ascii=False)}"
    )
    if len(compact) <= max_chars:
        return compact

    return compact[: max_chars].rstrip()


def build_default_output_filename(
    input_path: str,
    provider: str,
    model: str,
    timestamp_tag: str,
) -> str:
    """Construct the default output filename for an input dataset."""
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    provider_slug = sanitize_model_identifier(provider)
    model_slug = sanitize_model_identifier(model)
    return f"{base_name}_out_{provider_slug}_{model_slug}_{timestamp_tag}.csv"


def resolve_output_path(
    input_path: str,
    provider: str,
    model: str,
    output_argument: Optional[str],
    timestamp_tag: str,
    multiple_inputs: bool,
) -> str:
    """Determine the output path for an input file."""
    filename = build_default_output_filename(input_path, provider, model, timestamp_tag)
    if output_argument:
        resolved_argument = os.path.expanduser(output_argument)
        treat_as_directory = (
            multiple_inputs
            or resolved_argument.endswith(os.sep)
            or os.path.isdir(resolved_argument)
            or not os.path.splitext(resolved_argument)[1]
        )
        if treat_as_directory:
            os.makedirs(resolved_argument, exist_ok=True)
            return os.path.join(os.path.abspath(resolved_argument), filename)
        return resolved_argument
    return os.path.join(os.path.dirname(os.path.abspath(input_path)), filename)


def mark_node_in_context(left: str, node: str, right: str) -> str:
    """Return the combined context, adding node markers only when needed."""
    left_part = left.rstrip()
    right_part = right.lstrip()

    left_sep = "" if not left_part else " " if not left.endswith((" ", "\n")) else ""
    right_sep = "" if not right_part else " " if not right.startswith((" ", "\n")) else ""

    # Allow users to pre-mark parts of the node; if the input already contains markers,
    # keep it as-is instead of adding another pair.
    if NODE_MARKER_LEFT in node or NODE_MARKER_RIGHT in node:
        marked_node = node
    else:
        marked_node = f"{NODE_MARKER_LEFT}{node}{NODE_MARKER_RIGHT}"

    combined = f"{left_part}{left_sep}{marked_node}{right_sep}{right_part}"
    return combined.strip()


def extract_marked_spans(node: str) -> List[str]:
    """Return all substrings that have been explicitly wrapped in marker glyphs."""
    spans: List[str] = []
    search_start = 0
    marker_left = NODE_MARKER_LEFT
    marker_right = NODE_MARKER_RIGHT
    while True:
        left_idx = node.find(marker_left, search_start)
        if left_idx == -1:
            break
        span_start = left_idx + len(marker_left)
        right_idx = node.find(marker_right, span_start)
        if right_idx == -1:
            break
        span = node[span_start:right_idx].strip()
        if span:
            spans.append(span)
        search_start = right_idx + len(marker_right)
    return spans


def resolve_span_contract(node: str) -> Tuple[str, str]:
    """Derive the node_echo text and expected span_source for validation."""
    spans = extract_marked_spans(node)
    if spans:
        expected = " ".join(spans)
        return expected, SPAN_SOURCE_MARKED_SUBSPAN
    return node.strip(), SPAN_SOURCE_NODE


def estimate_token_count_from_chars(char_count: int) -> int:
    """Estimate token count from character count using a fixed heuristic."""
    safe_chars = max(0, int(char_count))
    if safe_chars <= 0:
        return 0
    return max(1, int(round(safe_chars / TOKEN_CHAR_ESTIMATE_RATIO)))


def estimate_token_count_from_text(text: str) -> int:
    """Estimate token count for free-form text."""
    return estimate_token_count_from_chars(len(text or ""))


def estimate_cache_padding_tokens(padding_tokens: int) -> int:
    """Estimate token contribution of a cache padding block."""
    safe_tokens = max(0, min(int(padding_tokens), MAX_CACHE_PADDING_TOKENS))
    if safe_tokens <= 0:
        return 0
    char_count = (
        len(CACHE_PADDING_PREFIX)
        + (len(CACHE_PADDING_TOKEN) * safe_tokens)
        + len(CACHE_PADDING_SUFFIX)
    )
    return estimate_token_count_from_chars(char_count)


def estimate_required_cache_padding_tokens(
    shared_prefix_tokens: int, target_shared_prefix_tokens: int
) -> int:
    """Return the minimum cache-padding units required to reach target shared-prefix length."""
    target = max(0, int(target_shared_prefix_tokens))
    baseline = max(0, int(shared_prefix_tokens))
    if target <= 0 or baseline >= target:
        return 0

    low = 0
    high = 1
    while high < MAX_CACHE_PADDING_TOKENS and (
        baseline + estimate_cache_padding_tokens(high)
    ) < target:
        high *= 2
    high = min(high, MAX_CACHE_PADDING_TOKENS)
    if (baseline + estimate_cache_padding_tokens(high)) < target:
        return MAX_CACHE_PADDING_TOKENS

    while low < high:
        mid = (low + high) // 2
        if baseline + estimate_cache_padding_tokens(mid) >= target:
            high = mid
        else:
            low = mid + 1
    return low


def build_cache_padding_text(padding_tokens: int) -> str:
    """Build deterministic filler text to increase prompt length for cache thresholds."""
    safe_tokens = max(0, int(padding_tokens))
    if safe_tokens <= 0:
        return ""
    if safe_tokens > MAX_CACHE_PADDING_TOKENS:
        logging.warning(
            "Requested cache padding tokens (%d) exceed safe cap (%d); truncating.",
            safe_tokens,
            MAX_CACHE_PADDING_TOKENS,
        )
        safe_tokens = MAX_CACHE_PADDING_TOKENS
    filler = CACHE_PADDING_TOKEN * safe_tokens
    return f"{CACHE_PADDING_PREFIX}{filler}{CACHE_PADDING_SUFFIX}"


# --------------------------- Data Loading ---------------------------------- #


@dataclass
class Example:
    example_id: str
    left_context: str
    node: str
    right_context: str
    info: str = ""
    truth: Optional[str] = None
    extras: Dict[str, str] = field(default_factory=dict)


@dataclass
class CompletionResult:
    text: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    token_logprobs: Optional[List[Dict[str, Any]]] = None
    usage_metadata: Optional[Dict[str, Any]] = None
    request_controls_requested: Dict[str, str] = field(default_factory=dict)
    request_controls_sent: Dict[str, str] = field(default_factory=dict)
    request_controls_rejected: Dict[str, str] = field(default_factory=dict)


@dataclass
class PromptBuildArtifacts:
    messages: List[Dict[str, str]]
    shared_prefix_text: str
    variable_payload_text: str
    shared_prefix_tokens_estimate: int
    variable_payload_tokens_estimate: int


PROVIDER_DEFAULTS: Dict[str, Dict[str, str]] = {
    "openai": {"api_key_var": "OPENAI_API_KEY", "api_base_var": "OPENAI_BASE_URL"},
    "anthropic": {"api_key_var": "ANTHROPIC_API_KEY", "api_base_var": "ANTHROPIC_BASE_URL"},
    "cohere": {"api_key_var": "COHERE_API_KEY", "api_base_var": "COHERE_BASE_URL"},
    "google": {"api_key_var": "GOOGLE_API_KEY", "api_base_var": "GOOGLE_BASE_URL"},
    "huggingface": {"api_key_var": "HF_API_KEY", "api_base_var": "HF_BASE_URL"},
    "e-infra": {"api_key_var": "E-INFRA_API_KEY", "api_base_var": "E-INFRA_BASE_URL"},
    "requesty": {"api_key_var": "REQUESTY_API_KEY", "api_base_var": "REQUESTY_BASE_URL"},
    "vertex": {"api_key_var": "VERTEX_ACCESS_TOKEN", "api_base_var": "VERTEX_BASE_URL"},
}

PROVIDER_BASE_FALLBACKS: Dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "requesty": "https://router.requesty.ai/v1",
}


def provider_slug_to_env_prefix(provider_slug: str) -> str:
    """Convert provider slug to an env-var prefix."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", (provider_slug or "").strip()).strip("_")
    upper = cleaned.upper()
    return upper or "PROVIDER"


def infer_provider_defaults(provider_slug: str) -> Dict[str, str]:
    """Infer API key/base env-var names from provider slug."""
    prefix = provider_slug_to_env_prefix(provider_slug)
    return {
        "api_key_var": f"{prefix}_API_KEY",
        "api_base_var": f"{prefix}_BASE_URL",
    }


def discover_provider_defaults(env_map: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    """Discover provider env-var mappings from known defaults plus *_API_KEY/*_ACCESS_TOKEN keys."""
    discovered: Dict[str, Dict[str, str]] = {
        slug: dict(defaults) for slug, defaults in PROVIDER_DEFAULTS.items()
    }

    key_pattern = re.compile(r"^([A-Z0-9][A-Z0-9_-]*)_(API_KEY|ACCESS_TOKEN)$")
    candidate_keys = set(env_map.keys()) | set(os.environ.keys())
    for key in candidate_keys:
        match = key_pattern.match(str(key))
        if not match:
            continue
        prefix = match.group(1)
        slug = re.sub(r"[^a-z0-9]+", "-", prefix.lower()).strip("-")
        if not slug:
            continue
        discovered.setdefault(
            slug,
            {"api_key_var": key, "api_base_var": f"{prefix}_BASE_URL"},
        )

    return discovered


def resolve_vertex_bootstrap_token(
    api_key_var: Optional[str],
    env_map: Dict[str, str],
) -> Optional[str]:
    """Resolve Vertex token value from configured var, then legacy fallback vars."""
    primary = resolve_env_value(api_key_var, env_map)
    if not is_placeholder_value(primary):
        return primary
    legacy = resolve_env_value("VERTEX_API_KEY", env_map)
    if not is_placeholder_value(legacy):
        logging.info(
            "Using legacy VERTEX_API_KEY value as Vertex access token bootstrap."
        )
        return legacy
    alternate = resolve_env_value("VERTEX_ACCESS_TOKEN", env_map)
    if not is_placeholder_value(alternate):
        return alternate
    return None


def resolve_vertex_token_refresh_interval_seconds(
    env_map: Dict[str, str],
    override_seconds: Optional[int] = None,
) -> int:
    """Resolve Vertex token refresh interval from env with sane defaults."""
    if override_seconds is not None:
        return max(60, int(override_seconds))
    raw_value = (
        resolve_env_value("VERTEX_ACCESS_TOKEN_REFRESH_SECONDS", env_map)
        or resolve_env_value("VERTEX_TOKEN_REFRESH_SECONDS", env_map)
        or resolve_env_value("VERTEX_ACCESS_TOKEN_TTL_SECONDS", env_map)
    )
    parsed = parse_optional_int(raw_value)
    if parsed is None or parsed <= 0:
        return VERTEX_DEFAULT_ACCESS_TOKEN_REFRESH_SECONDS
    return parsed


def resolve_vertex_auto_adc_login(
    env_map: Dict[str, str],
    override_value: Optional[bool] = None,
) -> bool:
    """Resolve whether missing ADC should trigger interactive gcloud login."""
    if override_value is not None:
        return bool(override_value)
    raw_value = resolve_env_value("VERTEX_AUTO_ADC_LOGIN", env_map)
    parsed = parse_optional_bool(raw_value)
    if parsed is None:
        return True
    return parsed


def build_vertex_access_token_provider(
    env_map: Dict[str, str],
    initial_token: Optional[str],
    auto_adc_login_override: Optional[bool] = None,
    refresh_interval_seconds_override: Optional[int] = None,
) -> VertexAccessTokenProvider:
    """Create a Vertex token provider configured via env vars."""
    refresh_command = resolve_env_value("VERTEX_ACCESS_TOKEN_COMMAND", env_map)
    if is_placeholder_value(refresh_command):
        refresh_command = None
    adc_login_command = resolve_env_value("VERTEX_ADC_LOGIN_COMMAND", env_map)
    if is_placeholder_value(adc_login_command):
        adc_login_command = None
    refresh_interval_seconds = resolve_vertex_token_refresh_interval_seconds(
        env_map,
        override_seconds=refresh_interval_seconds_override,
    )
    auto_adc_login = resolve_vertex_auto_adc_login(
        env_map,
        override_value=auto_adc_login_override,
    )
    return VertexAccessTokenProvider(
        initial_token=initial_token,
        refresh_command=refresh_command,
        refresh_interval_seconds=refresh_interval_seconds,
        auto_adc_login=auto_adc_login,
        adc_login_command=adc_login_command,
    )


def normalize_api_base(provider: str, api_base: Optional[str]) -> Optional[str]:
    """Ensure the API base ends with a version segment."""
    candidate = (api_base or PROVIDER_BASE_FALLBACKS.get(provider, "")).strip()
    if not candidate:
        return None
    trimmed = candidate.rstrip("/")
    if provider == "vertex":
        if re.search(r"/endpoints/openapi$", trimmed, re.IGNORECASE):
            return trimmed
        if re.search(r"/projects/[^/]+/locations/[^/]+/endpoints$", trimmed, re.IGNORECASE):
            return f"{trimmed}/openapi"
        if re.search(r"/projects/[^/]+/locations/[^/]+$", trimmed, re.IGNORECASE):
            return f"{trimmed}/endpoints/openapi"
        return trimmed
    if provider == "google":
        if re.search(r"/openai$", trimmed, re.IGNORECASE):
            return trimmed
        if re.search(r"/v\d+(?:beta\d*)?$", trimmed, re.IGNORECASE):
            return f"{trimmed}/openai"
        return f"{trimmed}/v1beta/openai"
    if trimmed.endswith("/openapi"):
        return trimmed
    if not re.search(r"/v\d+(?:beta\d*)?$", trimmed, re.IGNORECASE):
        trimmed = f"{trimmed}/v1"
    return trimmed


def _parse_model_payload(payload: Dict[str, Any], provider: str, endpoint: str) -> Tuple[List[str], Optional[str]]:
    """Normalize provider model payloads into a list of model IDs."""
    if provider == "vertex":
        model_items: Any = payload.get("data")
        if not isinstance(model_items, list):
            model_items = payload.get("publisherModels")
        if not isinstance(model_items, list):
            model_items = payload.get("models")
        if isinstance(model_items, list):
            models: List[str] = []
            for item in model_items:
                if isinstance(item, str):
                    candidate = item.strip()
                    if candidate:
                        models.append(candidate)
                    continue
                if not isinstance(item, dict):
                    continue
                identifier = item.get("id")
                if not isinstance(identifier, str) or not identifier.strip():
                    identifier = item.get("name")
                if isinstance(identifier, str) and identifier.strip():
                    normalized = identifier.strip()
                    # Vertex publisher list returns values like
                    # "publishers/google/models/gemini-2.5-pro"; keep the short model id.
                    marker = "/models/"
                    lowered = normalized.lower()
                    marker_index = lowered.rfind(marker)
                    if marker_index >= 0:
                        normalized = normalized[marker_index + len(marker) :]
                    normalized = normalized.strip("/")
                    if normalized:
                        models.append(normalized)
            models = sorted(set(models))
            if not models:
                logging.warning("Provider %s returned an empty model list.", provider)
            else:
                logging.info("Fetched %d models for provider %s.", len(models), provider)
            return models, None

    items = payload.get("data")
    if isinstance(items, list):
        models: List[str] = []
        for item in items:
            if isinstance(item, str):
                models.append(item)
            elif isinstance(item, dict):
                identifier = item.get("id")
                if isinstance(identifier, str):
                    models.append(identifier)
        models = sorted(set(models))
        if not models:
            logging.warning("Provider %s returned an empty model list.", provider)
        else:
            logging.info("Fetched %d models for provider %s.", len(models), provider)
        return models, None

    logging.error("Unexpected payload when fetching models for provider %s: %r", provider, payload)
    return [], "Unexpected response schema"


def _extract_vertex_project_id(api_base: str) -> Optional[str]:
    """Extract Vertex project id from api base URL."""
    match = re.search(r"/projects/([^/]+)/", api_base)
    if not match:
        return None
    project_id = match.group(1).strip()
    return project_id or None


def _vertex_publisher_models_endpoint(api_base: str) -> str:
    """Build Vertex publisher models endpoint from current api base."""
    parsed = urllib.parse.urlparse(api_base)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc or "aiplatform.googleapis.com"
    return f"{scheme}://{netloc}/v1beta1/publishers/google/models"


def _fetch_vertex_publisher_models(
    api_base: str,
    auth_token: str,
) -> Tuple[List[str], Optional[str]]:
    """Fetch Vertex publisher models via the v1beta1 endpoint."""
    endpoint = _vertex_publisher_models_endpoint(api_base)
    project_id = _extract_vertex_project_id(api_base)
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json",
    }
    if project_id:
        headers["x-goog-user-project"] = project_id

    models: List[str] = []
    next_page_token: Optional[str] = None
    max_pages = 20
    pages_fetched = 0

    while pages_fetched < max_pages:
        page_endpoint = endpoint
        if next_page_token:
            separator = "&" if "?" in endpoint else "?"
            page_endpoint = f"{endpoint}{separator}pageToken={urllib.parse.quote(next_page_token)}"
        request = urllib.request.Request(page_endpoint, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
            message = f"HTTP {exc.code} {exc.reason or ''} {detail}".strip()
            logging.error(
                "Failed Vertex publisher model listing at %s: %s",
                page_endpoint,
                message,
            )
            return [], message
        except urllib.error.URLError as exc:
            message = str(exc)
            logging.error("Connection error for Vertex publisher model listing: %s", message)
            return [], message
        except json.JSONDecodeError as exc:
            logging.error("Malformed JSON from Vertex publisher model listing (%s): %s", page_endpoint, exc)
            return [], "Invalid JSON response"

        page_models, parse_error = _parse_model_payload(payload, "vertex", page_endpoint)
        if parse_error:
            return [], parse_error
        models.extend(page_models)

        token_value = payload.get("nextPageToken") if isinstance(payload, dict) else None
        if isinstance(token_value, str) and token_value.strip():
            next_page_token = token_value.strip()
            pages_fetched += 1
            continue
        break

    return sorted(set(models)), None


def _fetch_models_with_curl(
    endpoint: str,
    api_key: str,
    provider: str,
) -> Tuple[List[str], Optional[str]]:
    """Fallback to curl when Python lacks SSL support for HTTPS."""
    headers = [
        ("Authorization", f"Bearer {api_key}"),
        ("Content-Type", "application/json"),
    ]
    errors: List[str] = []
    for binary in ("curl", "curl.exe"):
        cmd = [binary, "-sS", "--fail", "--max-time", "60"]
        for name, value in headers:
            cmd.extend(["-H", f"{name}: {value}"])
        cmd.append(endpoint)
        try:
            completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except FileNotFoundError:
            errors.append(f"{binary} not found")
            continue
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            errors.append(f"{binary} exit {exc.returncode}: {detail}")
            continue

        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            errors.append(f"{binary} invalid JSON: {exc}")
            continue

        return _parse_model_payload(payload, provider, endpoint)

    combined_error = "; ".join(errors) if errors else "curl unavailable"
    logging.error("Curl fallback failed for provider %s: %s", provider, combined_error)
    return [], combined_error


def fetch_provider_models(
    provider: str,
    api_key: str,
    api_base: str,
    token_provider: Optional[AccessTokenProvider] = None,
) -> Tuple[List[str], Optional[str]]:
    """Fetch available models for a provider using raw HTTP."""
    endpoint = f"{api_base.rstrip('/')}/models"

    def maybe_fetch_vertex_fallback(auth_token: str) -> Optional[Tuple[List[str], Optional[str]]]:
        if provider != "vertex":
            return None
        logging.info(
            "Falling back to Vertex publisher model listing endpoint (v1beta1/publishers/google/models)."
        )
        return _fetch_vertex_publisher_models(api_base, auth_token)

    max_attempts = 2 if token_provider is not None else 1
    for attempt_index in range(max_attempts):
        auth_token = api_key
        if token_provider is not None:
            auth_token = token_provider.get_token(force_refresh=(attempt_index > 0))
        request = urllib.request.Request(
            endpoint,
            headers={
                "Authorization": f"Bearer {auth_token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return _parse_model_payload(payload, provider, endpoint)
        except urllib.error.HTTPError as exc:
            if provider == "vertex" and exc.code == 404:
                fallback_result = maybe_fetch_vertex_fallback(auth_token)
                if fallback_result is not None:
                    return fallback_result
            if token_provider is not None and exc.code in {401, 403} and attempt_index == 0:
                logging.warning(
                    "Authentication failed while fetching models for provider %s; refreshing token and retrying once.",
                    provider,
                )
                continue
            detail = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
            message = f"HTTP {exc.code} {exc.reason or ''} {detail}".strip()
            logging.error("Failed to fetch models for provider %s: %s", provider, message)
            if provider == "vertex" and exc.code == 403 and "quota project" in message.lower():
                logging.error(
                    "Vertex ADC requires a quota project. Run: gcloud auth application-default set-quota-project <PROJECT_ID>"
                )
            return [], message
        except urllib.error.URLError as exc:
            message = str(exc)
            logging.error("Connection error while fetching models for provider %s: %s", provider, message)
            # If Python lacks SSL support, urllib cannot handle HTTPS. Fall back to curl if available.
            if "unknown url type: https" in message.lower():
                logging.warning(
                    "Python SSL support appears to be missing; trying curl fallback for provider %s.", provider
                )
                return _fetch_models_with_curl(endpoint, auth_token, provider)
            return [], message
        except json.JSONDecodeError as exc:
            logging.error("Malformed JSON response from provider %s (%s): %s", provider, endpoint, exc)
            return [], "Invalid JSON response"

    return [], "Unable to fetch models after token refresh retries"


def _gemini_caching_base_url(api_base_url: Optional[str]) -> str:
    """Derive the Gemini REST caching API base URL from the OpenAI-compat base URL.

    Google AI Studio's OpenAI-compat endpoint is:
        https://generativelanguage.googleapis.com/v1beta/openai
    The caching REST API lives at the parent path:
        https://generativelanguage.googleapis.com/v1beta
    """
    fallback = "https://generativelanguage.googleapis.com/v1beta"
    if not api_base_url:
        return fallback
    trimmed = api_base_url.rstrip("/")
    if trimmed.lower().endswith("/openai"):
        return trimmed[: -len("/openai")]
    return trimmed


def _extract_vertex_project_location_prefix(api_base_url: Optional[str]) -> Optional[str]:
    """Extract Vertex project/location prefix from API base."""
    if not api_base_url:
        return None
    trimmed = api_base_url.rstrip("/")
    match = re.search(r"/v\d+(?:beta\d*)?/(projects/[^/]+/locations/[^/]+)", trimmed)
    if match:
        value = match.group(1).strip().strip("/")
        return value or None
    return None


def _is_vertex_gemini_caching_target(api_base_url: Optional[str]) -> bool:
    """Return True when api_base_url looks like a Vertex OpenAI endpoint."""
    if not api_base_url:
        return False
    trimmed = api_base_url.rstrip("/")
    lowered = trimmed.lower()
    if "/endpoints/openapi" in lowered and "/projects/" in lowered and "/locations/" in lowered:
        return True
    return False


def _vertex_gemini_caching_root_url(api_base_url: Optional[str]) -> str:
    """Convert Vertex OpenAI base URL into Vertex Gemini cache root URL."""
    if not api_base_url:
        raise RuntimeError("Vertex cache root URL requires an API base URL.")
    trimmed = api_base_url.rstrip("/")
    lowered = trimmed.lower()
    marker = "/endpoints/openapi"
    marker_index = lowered.rfind(marker)
    if marker_index >= 0:
        return trimmed[:marker_index]
    return trimmed


def _extract_vertex_quota_project(api_base_url: Optional[str]) -> Optional[str]:
    """Extract Vertex project id for x-goog-user-project header."""
    if not api_base_url:
        return None
    match = re.search(r"/projects/([^/]+)/", api_base_url.rstrip("/"))
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def _normalize_vertex_cache_model(api_base_url: Optional[str], model: str) -> str:
    """Normalize model name for Vertex cache API (expects full resource path)."""
    normalized_model = str(model or "").strip()
    if not normalized_model:
        return normalized_model
    if normalized_model.startswith("projects/"):
        return normalized_model

    prefix = _extract_vertex_project_location_prefix(api_base_url)
    if not prefix:
        return normalized_model

    if normalized_model.startswith("publishers/"):
        return f"{prefix}/{normalized_model}"
    if normalized_model.startswith("models/"):
        return f"{prefix}/publishers/google/{normalized_model}"
    return f"{prefix}/publishers/google/models/{normalized_model}"


def normalize_model_for_provider(provider: str, model: str) -> str:
    """Normalize provider-specific model aliases into API-accepted identifiers."""
    normalized = str(model or "").strip()
    if not normalized:
        return normalized
    provider_slug = (provider or "").strip().lower()
    if provider_slug != "vertex":
        return normalized

    # Vertex OpenAPI endpoint expects "<publisher>/<model>".
    if normalized.startswith("projects/"):
        match = re.search(r"/publishers/([^/]+)/models/([^/]+)$", normalized)
        if match:
            return f"{match.group(1)}/{match.group(2)}"
        return normalized
    if "/" not in normalized:
        return f"google/{normalized}"
    return normalized


def build_run_model_details(
    provider: str,
    requested_model: str,
    api_base_url: Optional[str],
    api_key_var: Optional[str],
    api_base_var: Optional[str],
    gemini_cached_content: Optional[str],
) -> Dict[str, Any]:
    """Build run-level model metadata stored in log and metrics outputs."""
    provider_slug = (provider or "").strip().lower()
    normalized_request_model = normalize_model_for_provider(provider_slug, requested_model)
    details: Dict[str, Any] = {
        "provider": provider_slug,
        "model_requested": str(requested_model or "").strip(),
        "model_for_requests": normalized_request_model,
        "api_base_url": api_base_url or "",
        "api_key_var": str(api_key_var or "").strip(),
        "api_base_var": str(api_base_var or "").strip(),
        "chat_completions_endpoint": (
            f"{api_base_url.rstrip('/')}/chat/completions" if api_base_url else ""
        ),
    }
    if provider_slug == "vertex":
        details["vertex_cache_model"] = _normalize_vertex_cache_model(
            api_base_url, requested_model
        )
    if gemini_cached_content:
        details["gemini_cached_content"] = str(gemini_cached_content).strip()
    return details


def upsert_prompt_log_run_metadata(
    log_records: List[Dict[str, Any]],
    model_details: Dict[str, Any],
) -> None:
    """Insert or update a run-metadata record in the prompt log list."""
    metadata_record = {
        "record_type": "run_metadata",
        "timestamp": utc_timestamp(),
        "model_details": model_details,
    }
    for index, entry in enumerate(log_records):
        if isinstance(entry, dict) and entry.get("record_type") == "run_metadata":
            log_records[index] = metadata_record
            return
    log_records.insert(0, metadata_record)


def create_gemini_cached_content(
    api_key: str,
    api_base_url: Optional[str],
    model: str,
    system_prompt: str,
    ttl_seconds: int,
) -> str:
    """Create a Gemini CachedContent resource from a system prompt.

    Uses the Gemini REST API directly (no additional SDK required).
    Returns the resource name, e.g. ``cachedContents/abc123``.
    Raises ``RuntimeError`` on API errors.
    """
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    model_for_request = model
    if _is_vertex_gemini_caching_target(api_base_url):
        base = _vertex_gemini_caching_root_url(api_base_url)
        endpoint = f"{base}/cachedContents"
        headers["Authorization"] = f"Bearer {api_key}"
        quota_project = _extract_vertex_quota_project(api_base_url)
        if quota_project:
            headers["x-goog-user-project"] = quota_project
        model_for_request = _normalize_vertex_cache_model(api_base_url, model)
    else:
        base = _gemini_caching_base_url(api_base_url)
        endpoint = f"{base}/cachedContents?key={api_key}"
    body = json.dumps(
        {
            "model": model_for_request,
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "ttl": f"{ttl_seconds}s",
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
        raise RuntimeError(
            f"Gemini CachedContent creation failed: HTTP {exc.code} {exc.reason or ''} {detail}".strip()
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Gemini CachedContent creation failed: {exc}") from exc
    name = payload.get("name")
    if not name:
        raise RuntimeError(
            f"Gemini CachedContent creation response did not include a name: {payload}"
        )
    return str(name)


def delete_gemini_cached_content(
    api_key: str,
    api_base_url: Optional[str],
    cache_name: str,
) -> bool:
    """Delete a Gemini CachedContent resource by name.

    ``cache_name`` should be the resource name returned by
    :func:`create_gemini_cached_content`, e.g. ``cachedContents/abc123``.
    Returns ``True`` on success, ``False`` on failure (logs a warning).
    """
    name_path = cache_name.lstrip("/")
    headers: Dict[str, str] = {}
    if _is_vertex_gemini_caching_target(api_base_url):
        base = _vertex_gemini_caching_root_url(api_base_url)
        if name_path.startswith("projects/"):
            # Build absolute resource URL using the same host/version as base.
            root_match = re.match(r"^(https?://[^/]+)/(v\d+(?:beta\d*)?)/", base.rstrip("/"))
            if root_match:
                endpoint = f"{root_match.group(1)}/{root_match.group(2)}/{name_path}"
            else:
                endpoint = f"{base}/{name_path}"
        else:
            endpoint = f"{base}/{name_path}"
        headers["Authorization"] = f"Bearer {api_key}"
        quota_project = _extract_vertex_quota_project(api_base_url)
        if quota_project:
            headers["x-goog-user-project"] = quota_project
    else:
        base = _gemini_caching_base_url(api_base_url)
        endpoint = f"{base}/{name_path}?key={api_key}"
    request = urllib.request.Request(endpoint, headers=headers, method="DELETE")
    try:
        with urllib.request.urlopen(request, timeout=60):
            pass
        return True
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
        logging.warning(
            "Failed to delete Gemini cache %s: HTTP %s %s %s",
            cache_name,
            exc.code,
            exc.reason or "",
            detail,
        )
        return False
    except urllib.error.URLError as exc:
        logging.warning("Failed to delete Gemini cache %s: %s", cache_name, exc)
        return False


def write_model_catalog_js(catalog: Dict[str, Dict[str, Any]], output_path: str) -> None:
    """Write the catalog as a small JS module consumable by the GUI."""
    ensure_directory(output_path)
    content = (
        "// Auto-generated by benchmark_agent.py --update-models\n"
        "window.MODEL_CATALOG = "
        + json.dumps(catalog, indent=2, ensure_ascii=False)
        + ";\n"
    )
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(content)
    logging.info("Wrote model catalog to %s", output_path)


def update_model_catalog(
    providers: Optional[List[str]],
    output_path: str,
) -> int:
    """Generate a model catalog JS file using credentials sourced from the environment."""
    env_map = parse_env_file(".env")
    available_provider_defaults = discover_provider_defaults(env_map)
    if providers:
        selected = providers
    else:
        selected = []
        for provider_slug, defaults in sorted(available_provider_defaults.items()):
            api_base = normalize_api_base(provider_slug, resolve_env_value(defaults["api_base_var"], env_map))
            if not api_base:
                if provider_slug != "vertex":
                    continue
            if provider_slug == "vertex":
                vertex_models_api_base = normalize_api_base(
                    "vertex", resolve_env_value("VERTEX_MODELS_BASE_URL", env_map)
                )
                if not api_base and not vertex_models_api_base:
                    continue
                selected.append(provider_slug)
                continue
            api_key = resolve_env_value(defaults["api_key_var"], env_map)
            if is_placeholder_value(api_key):
                continue
            selected.append(provider_slug)

    if not selected:
        logging.error(
            "No providers are configured for update. Add provider API keys to .env or pass --models-providers explicitly."
        )
        return 1

    logging.info("Updating model catalog for providers: %s", ", ".join(selected))
    catalog: Dict[str, Dict[str, Any]] = {}
    errors = 0

    for provider in selected:
        provider_slug = provider.lower()
        defaults = available_provider_defaults.get(provider_slug)
        if not defaults:
            defaults = infer_provider_defaults(provider_slug)
            logging.info(
                "Provider %s not in known/discovered defaults; trying inferred env vars %s and %s.",
                provider_slug,
                defaults["api_key_var"],
                defaults["api_base_var"],
            )
        api_key_var = defaults["api_key_var"]
        api_base_var = defaults["api_base_var"]
        api_base = normalize_api_base(provider_slug, resolve_env_value(api_base_var, env_map))
        models_api_base = api_base
        models_api_base_var = api_base_var
        if provider_slug == "vertex":
            vertex_models_api_base = normalize_api_base(
                "vertex", resolve_env_value("VERTEX_MODELS_BASE_URL", env_map)
            )
            if vertex_models_api_base:
                models_api_base = vertex_models_api_base
                models_api_base_var = "VERTEX_MODELS_BASE_URL"
                if api_base and api_base != models_api_base:
                    logging.info(
                        "Using Vertex models base URL from %s for catalog retrieval: %s "
                        "(runtime base in %s remains %s).",
                        models_api_base_var,
                        models_api_base,
                        api_base_var,
                        api_base,
                    )
        if not models_api_base:
            logging.warning(
                "Skipping provider %s; missing API base URL in %s (.env first, env fallback).",
                provider_slug,
                models_api_base_var,
            )
            continue
        token_provider: Optional[AccessTokenProvider] = None
        api_key = resolve_env_value(api_key_var, env_map)
        if provider_slug == "vertex":
            bootstrap_token = resolve_vertex_bootstrap_token(api_key_var, env_map)
            token_provider = build_vertex_access_token_provider(env_map, bootstrap_token)
            try:
                api_key = token_provider.get_token(force_refresh=True)
            except RuntimeError as exc:
                if is_placeholder_value(bootstrap_token):
                    logging.warning(
                        "Skipping provider %s; unable to obtain Vertex access token: %s",
                        provider_slug,
                        exc,
                    )
                    continue
                api_key = bootstrap_token
                logging.warning(
                    "Provider %s will use static bootstrap token because refresh failed: %s",
                    provider_slug,
                    exc,
                )
        if is_placeholder_value(api_key):
            logging.warning("Skipping provider %s; missing API key in %s (.env first, env fallback).", provider_slug, api_key_var)
            continue
        models, error = fetch_provider_models(
            provider_slug, api_key, models_api_base, token_provider=token_provider
        )
        catalog[provider_slug] = {
            "models": models,
            "api_base": api_base or models_api_base,
            "models_api_base": models_api_base,
            "api_key_var": api_key_var,
            "api_base_var": api_base_var,
            "models_api_base_var": models_api_base_var,
            "error": error,
            "timestamp": utc_timestamp(),
        }
        if error or not models:
            errors += 1

    if not catalog:
        logging.error("No providers were updated; aborting catalog write.")
        return 1

    write_model_catalog_js(catalog, output_path)
    if errors:
        logging.warning(
            "Model catalog generated with %d provider(s) reporting errors. See log for details.",
            errors,
        )
    return 0



def read_examples(path: str) -> Tuple[List[Example], List[str]]:
    """Read a semicolon-delimited CSV file into Example records."""
    examples: List[Example] = []
    extra_field_order: List[str] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        required_fields = {"ID", "leftContext", "node", "rightContext"}
        fieldnames = reader.fieldnames or []
        missing = required_fields - set(fieldnames)
        if missing:
            raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

        allowed_fields = {"ID", "leftContext", "node", "rightContext", "truth", "info"}
        extra_field_order = [name for name in fieldnames if name not in allowed_fields]

        for row in reader:
            extras: Dict[str, str] = {}
            for key, value in row.items():
                if key not in allowed_fields:
                    value_str = "" if value is None else str(value).strip()
                    extras[key] = value_str

            info_value = row.get("info")
            info_text = "" if info_value is None else str(info_value).strip()

            example = Example(
                example_id=str(row.get("ID", "")).strip(),
                left_context=row.get("leftContext", "").strip(),
                node=row.get("node", "").strip(),
                right_context=row.get("rightContext", "").strip(),
                info=info_text,
                truth=str(row.get("truth", "")).strip() or None,
                extras=extras,
            )
            if not example.example_id:
                logging.warning("Skipping row with empty ID: %s", row)
                continue
            examples.append(example)
    return examples, extra_field_order


def load_existing_prompt_log(path: str) -> List[Dict[str, Any]]:
    """Load existing prompt log entries for resume mode."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:  # noqa: BLE001
        logging.warning("Unable to parse existing prompt log %s; starting fresh log: %s", path, exc)
        return []
    if isinstance(payload, list):
        return payload
    logging.warning("Prompt log %s is not a JSON list; starting fresh log.", path)
    return []


def load_existing_output_predictions(
    output_path: str,
) -> Tuple[List[str], Dict[str, Prediction], int, int, int]:
    """Load predictions from an existing output CSV to support resume mode."""
    predictions: Dict[str, Prediction] = {}
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_reported_tokens = 0

    with open(output_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        existing_fieldnames = [name for name in (reader.fieldnames or []) if name]
        if not existing_fieldnames:
            return [], predictions, total_prompt_tokens, total_completion_tokens, total_reported_tokens

        required_fields = {"ID", "prediction"}
        missing = required_fields - set(existing_fieldnames)
        if missing:
            raise ValueError(
                f"Cannot resume from {output_path}: missing required output columns {sorted(missing)}."
            )

        for row_index, row in enumerate(reader, start=2):
            example_id = str(row.get("ID", "")).strip()
            if not example_id:
                logging.warning("Ignoring resume row %d in %s because ID is empty.", row_index, output_path)
                continue

            label = str(row.get("prediction", "")).strip()
            confidence = parse_optional_float(row.get("confidence"))
            prompt_tokens = parse_optional_int(row.get("promptTokens"))
            completion_tokens = parse_optional_int(row.get("completionTokens"))
            total_tokens = parse_optional_int(row.get("totalTokens"))
            label_logprob = parse_optional_float(row.get("labelLogProb"))
            label_probability = parse_optional_float(row.get("labelProbability"))

            if prompt_tokens is not None:
                total_prompt_tokens += prompt_tokens
            if completion_tokens is not None:
                total_completion_tokens += completion_tokens
            if total_tokens is not None:
                total_reported_tokens += total_tokens

            if example_id in predictions:
                logging.warning(
                    "Duplicate ID %s in existing output %s; keeping last occurrence.",
                    example_id,
                    output_path,
                )

            predictions[example_id] = Prediction(
                label=label,
                explanation=str(row.get("explanation", "") or ""),
                confidence=confidence,
                raw_response="",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                label_logprob=label_logprob,
                label_probability=label_probability,
                node_echo=str(row.get("nodeEcho", "") or "") or None,
                span_source=str(row.get("spanSource", "") or "") or None,
                validator_status=str(row.get("validatorStatus", "") or "") or None,
                validator_reason=str(row.get("validatorReason", "") or "") or None,
            )

    return existing_fieldnames, predictions, total_prompt_tokens, total_completion_tokens, total_reported_tokens


def read_label_file(path: str) -> Dict[str, str]:
    """Load labels from a semicolon-delimited CSV file keyed by ID."""
    labels: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        required_fields = {"ID", "truth"}
        missing = required_fields - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

        for row in reader:
            example_id = str(row.get("ID", "")).strip()
            truth = str(row.get("truth", "")).strip()
            if not example_id:
                logging.warning("Skipping label row with empty ID.")
                continue
            labels[example_id] = truth
    return labels


def merge_labels(examples: List[Example], labels: Optional[Dict[str, str]]) -> None:
    """Attach labels from a dictionary to the examples in-place."""
    if not labels:
        return
    missing = []
    for example in examples:
        if example.example_id in labels:
            example.truth = labels[example.example_id]
        else:
            missing.append(example.example_id)
    if missing:
        logging.warning("No label found for %d example(s): %s", len(missing), missing[:5])


def select_few_shot_examples(
    examples: List[Example],
    target_id: str,
    count: int,
) -> List[Example]:
    """Return up to `count` labeled examples excluding the current target."""
    if count <= 0:
        return []
    context: List[Example] = []
    for example in examples:
        if example.truth is None or example.example_id == target_id:
            continue
        context.append(example)
        if len(context) >= count:
            break
    return context


# --------------------------- OpenAI Client --------------------------------- #


class OpenAIConnector:
    """Thin wrapper supporting both legacy and modern OpenAI Python SDKs."""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        provider: str = "openai",
        request_interval_ms: int = 0,
        access_token_provider: Optional[AccessTokenProvider] = None,
    ) -> None:
        self.client_type: str
        self._provider = (provider or "openai").strip().lower()
        self._chat_incompatible_models: set[str] = set()
        self._chat_unsupported_params: Dict[str, set[str]] = {}
        self._responses_unsupported_params: Dict[str, set[str]] = {}
        self._logged_vertex_model_normalizations: set[str] = set()
        self._min_request_interval_seconds = max(0, request_interval_ms) / 1000.0
        self._last_request_started_at: Optional[float] = None
        self._access_token_provider = access_token_provider
        self._current_api_key = api_key
        try:
            from openai import OpenAI

            # Newer SDK (>= 1.0)
            kwargs: Dict[str, Any] = {"api_key": self._current_api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self._client = OpenAI(**kwargs)
            if hasattr(self._client, "chat") and hasattr(self._client.chat, "completions"):
                self.client_type = "chat_v1"
            elif hasattr(self._client, "responses"):
                self.client_type = "responses_v1"
            else:
                raise RuntimeError("Unsupported OpenAI client configuration.")
        except ImportError:
            try:
                import openai  # type: ignore

                openai.api_key = self._current_api_key
                if base_url:
                    if hasattr(openai, "base_url"):
                        openai.base_url = base_url  # type: ignore[attr-defined]
                    openai.api_base = base_url  # type: ignore[attr-defined]
                self._client = openai
                self.client_type = "legacy"
            except ImportError as exc:
                raise RuntimeError(
                    "OpenAI Python SDK not installed. Install `openai` package."
                ) from exc

    def _apply_api_key_to_client(self, api_key: str) -> None:
        self._current_api_key = api_key
        if self.client_type in {"chat_v1", "responses_v1"}:
            try:
                setattr(self._client, "api_key", api_key)
            except Exception:  # noqa: BLE001
                logging.debug("Unable to set api_key dynamically on OpenAI client instance.")
        elif self.client_type == "legacy":
            self._client.api_key = api_key

    def _refresh_access_token_if_needed(self, force_refresh: bool = False) -> None:
        if self._access_token_provider is None:
            return
        token = self._access_token_provider.get_token(force_refresh=force_refresh)
        if token != self._current_api_key:
            self._apply_api_key_to_client(token)

    def _normalize_model_for_provider(self, model: str) -> str:
        """Normalize provider-specific model aliases into API-accepted identifiers."""
        return normalize_model_for_provider(self._provider, model)

    def _throttle_request_if_needed(self) -> None:
        """Sleep if needed to maintain minimum spacing between outgoing API requests."""
        if self._min_request_interval_seconds <= 0:
            return

        now = time.perf_counter()
        if self._last_request_started_at is not None:
            elapsed = now - self._last_request_started_at
            remaining = self._min_request_interval_seconds - elapsed
            if remaining > 0:
                logging.debug(
                    "Rate limit pacing active; sleeping %.3fs before next API request.",
                    remaining,
                )
                time.sleep(remaining)
        self._last_request_started_at = time.perf_counter()

    def complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
        verbosity: Optional[str],
        service_tier: Optional[str],
        include_logprobs: bool,
        reasoning_effort: Optional[str],
        thinking_level: Optional[str],
        effort: Optional[str],
        prompt_cache_key: Optional[str],
        gemini_cached_content: Optional[str],
        requesty_auto_cache: Optional[bool],
    ) -> CompletionResult:
        """Dispatch a chat completion request and return the message content."""
        # Top-k is not currently supported in OpenAI Chat API; we log and ignore.
        if top_k is not None:
            logging.debug("top_k is not supported by OpenAI Chat API; ignoring value %s.", top_k)
        request_model = self._normalize_model_for_provider(model)
        if self._provider == "vertex" and request_model != str(model or "").strip():
            normalization_key = f"{model}=>{request_model}"
            if normalization_key not in self._logged_vertex_model_normalizations:
                logging.info(
                    "Vertex model alias normalized from %s to %s for OpenAPI requests.",
                    model,
                    request_model,
                )
                self._logged_vertex_model_normalizations.add(normalization_key)
        model_key = request_model.strip().lower()
        normalized_prompt_cache_key = (
            str(prompt_cache_key).strip() if prompt_cache_key is not None else ""
        ) or None
        normalized_gemini_cached_content = (
            str(gemini_cached_content).strip() if gemini_cached_content is not None else ""
        ) or None
        normalized_requesty_auto_cache: Optional[bool] = None
        if isinstance(requesty_auto_cache, bool):
            normalized_requesty_auto_cache = requesty_auto_cache
        elif requesty_auto_cache is not None:
            normalized_requesty_auto_cache = str(requesty_auto_cache).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        is_gemini_target = self._provider == "google" or "gemini" in model_key
        is_requesty_target = self._provider == "requesty"
        requested_controls: Dict[str, str] = {}
        rejected_controls: Dict[str, str] = {}

        if reasoning_effort:
            requested_controls["reasoning_effort"] = reasoning_effort
        if thinking_level:
            requested_controls["thinking_level"] = thinking_level
        if effort:
            requested_controls["effort"] = effort
        if verbosity:
            requested_controls["verbosity"] = verbosity
        if normalized_prompt_cache_key:
            requested_controls["prompt_cache_key"] = normalized_prompt_cache_key
        if normalized_gemini_cached_content and is_gemini_target:
            requested_controls["gemini_cached_content"] = normalized_gemini_cached_content
        if normalized_requesty_auto_cache is not None and is_requesty_target:
            requested_controls["requesty_auto_cache"] = (
                "true" if normalized_requesty_auto_cache else "false"
            )

        def control_key_for_param(param_name: str) -> Optional[str]:
            mapping = {
                "reasoning": "reasoning_effort",
                "reasoning_effort": "reasoning_effort",
                "thinkingLevel": "thinking_level",
                "thinking_level": "thinking_level",
                "thinking_config": "thinking_level",
                "google_thinking_config": "thinking_level",
                "effort": "effort",
                "verbosity": "verbosity",
                "prompt_cache_key": "prompt_cache_key",
                "cached_content": "gemini_cached_content",
                "gemini_cached_content": "gemini_cached_content",
                "google_cached_content": "gemini_cached_content",
                "requesty_auto_cache": "requesty_auto_cache",
                "requesty.autocache": "requesty_auto_cache",
                "requesty.auto_cache": "requesty_auto_cache",
                "text_verbosity": "verbosity",
            }
            return mapping.get(param_name)

        def mark_control_rejected(param_name: str, reason: str) -> None:
            control_key = control_key_for_param(param_name)
            if not control_key:
                return
            if control_key in requested_controls:
                rejected_controls[control_key] = reason

        def normalize_unsupported_parameter(name: Optional[str]) -> Optional[str]:
            if not isinstance(name, str):
                return None
            normalized = name.strip().lower().replace("-", "_").replace(".", "_")
            alias_map = {
                "servicetier": "service_tier",
                "service_tier": "service_tier",
                "topp": "top_p",
                "top_p": "top_p",
                "temperature": "temperature",
                "verbosity": "verbosity",
                "text_verbosity": "verbosity",
                "reasoning": "reasoning",
                "reasoning_effort": "reasoning_effort",
                "thinkinglevel": "thinkingLevel",
                "thinking_level": "thinkingLevel",
                "thinkingconfig": "thinkingLevel",
                "thinking_config": "thinkingLevel",
                "google_thinking_config": "thinkingLevel",
                "effort": "effort",
                "toplogprobs": "top_logprobs",
                "top_logprobs": "top_logprobs",
                "logprobs": "logprobs",
                "promptcachekey": "prompt_cache_key",
                "prompt_cache_key": "prompt_cache_key",
                "cachedcontent": "cached_content",
                "cached_content": "cached_content",
                "google_cached_content": "cached_content",
                "requestyautocache": "requesty_auto_cache",
                "requesty_auto_cache": "requesty_auto_cache",
                "requesty_autocache": "requesty_auto_cache",
            }
            if normalized in alias_map:
                return alias_map[normalized]
            if "thinking" in normalized and "level" in normalized:
                return "thinkingLevel"
            if "reasoning" in normalized and "effort" in normalized:
                return "reasoning_effort"
            return normalized or None

        def extract_error_text(exc: Exception) -> str:
            error_message = getattr(exc, "message", None)
            if error_message is None and hasattr(exc, "error"):
                error_payload = getattr(exc, "error")
                if isinstance(error_payload, dict):
                    error_message = error_payload.get("message")
                else:
                    error_message = getattr(error_payload, "message", None)
            return str(error_message or exc)

        def warn_logprob_retry(exc: Exception) -> None:
            status_code = getattr(exc, "status_code", None) or getattr(exc, "http_status", None) or getattr(exc, "status", None)
            text = extract_error_text(exc)
            if isinstance(status_code, str) and status_code.isdigit():
                status_code = int(status_code)
            if status_code == 403 or "403" in text:
                logging.warning(
                    "The API rejected the logprobs request with HTTP 403. This model or service tier likely does not support token log probabilities; retrying without logprobs. Details: %s",
                    text,
                )
            else:
                logging.debug("Logprobs unavailable for this client (%s); retrying without logprobs.", exc)

        def should_retry_with_responses(exc: Exception) -> bool:
            text = extract_error_text(exc).lower()
            return any(
                marker in text
                for marker in (
                    "not a chat model",
                    "not supported in the v1/chat/completions endpoint",
                    "did you mean to use v1/completions",
                    "use v1/completions",
                    "use the responses api",
                )
            )

        def extract_unsupported_parameter(exc: Exception) -> Optional[str]:
            param = getattr(exc, "param", None)
            error_message = getattr(exc, "message", None)
            body = getattr(exc, "body", None)

            payload_items: List[Dict[str, Any]] = []
            if isinstance(body, dict):
                payload_items.append(body)
            elif isinstance(body, list):
                payload_items.extend(item for item in body if isinstance(item, dict))

            for payload_item in payload_items:
                error_payload = payload_item.get("error")
                if isinstance(error_payload, dict):
                    if error_message is None:
                        error_message = error_payload.get("message")
                    if not param:
                        param = error_payload.get("param")

            if error_message is None and hasattr(exc, "error"):
                error_payload = getattr(exc, "error")
                if isinstance(error_payload, dict):
                    error_message = error_payload.get("message")
                    if not param:
                        param = error_payload.get("param")
                else:
                    error_message = getattr(error_payload, "message", None)
                    if not param:
                        param = getattr(error_payload, "param", None)

            if isinstance(param, str):
                normalized = normalize_unsupported_parameter(param)
                if normalized:
                    return normalized

            text = str(error_message or exc)
            patterns = (
                r"unsupported parameter:\s*['`\"]?([a-zA-Z0-9_.-]+)['`\"]?",
                r"unknown name\s*['`\"]([a-zA-Z0-9_.-]+)['`\"]\s*:\s*cannot find field",
                r"unrecognized request argument supplied:\s*['`\"]?([a-zA-Z0-9_.-]+)['`\"]?",
                r"unexpected keyword argument\s*['`\"]([a-zA-Z0-9_.-]+)['`\"]",
            )
            for pattern in patterns:
                match = re.search(pattern, text, flags=re.IGNORECASE)
                if match:
                    return normalize_unsupported_parameter(match.group(1))
            return None

        def infer_known_unsupported_parameter(exc: Exception) -> Optional[str]:
            text = extract_error_text(exc).lower()
            if "service_tier" in text or "service tier" in text or "service-tier" in text:
                if any(marker in text for marker in ("unsupported", "unknown", "unrecognized", "invalid")):
                    return "service_tier"
            if "top_p" in text or "top p" in text or "top-p" in text:
                if any(marker in text for marker in ("unsupported", "unknown", "unrecognized", "invalid", "not supported", "not allowed")):
                    return "top_p"
            if "temperature" in text:
                if any(marker in text for marker in ("unsupported", "unknown", "unrecognized", "invalid", "not supported", "not allowed")):
                    return "temperature"
            if "verbosity" in text:
                if any(marker in text for marker in ("unsupported", "unknown", "unrecognized", "invalid", "not supported", "not allowed")):
                    return "verbosity"
            if "reasoning_effort" in text or "reasoning effort" in text:
                if any(marker in text for marker in ("unsupported", "unknown", "unrecognized", "invalid", "not supported", "not allowed")):
                    return "reasoning_effort"
            if "reasoning" in text:
                if any(marker in text for marker in ("unsupported", "unknown", "unrecognized", "invalid", "not supported", "not allowed")):
                    return "reasoning"
            if (
                "thinkinglevel" in text
                or "thinking level" in text
                or "thinking_level" in text
                or "thinking_config" in text
            ):
                if any(marker in text for marker in ("unsupported", "unknown", "unrecognized", "invalid", "not supported", "not allowed")):
                    return "thinkingLevel"
            if "effort" in text:
                if any(marker in text for marker in ("unsupported", "unknown", "unrecognized", "invalid", "not supported", "not allowed")):
                    return "effort"
            if "top_logprobs" in text or "top logprobs" in text:
                if any(marker in text for marker in ("unsupported", "unknown", "unrecognized", "invalid")):
                    return "top_logprobs"
            if "logprobs" in text:
                if any(marker in text for marker in ("unsupported", "unknown", "unrecognized", "invalid", "not allowed")):
                    return "logprobs"
            if "prompt_cache_key" in text or "prompt cache key" in text:
                if any(marker in text for marker in ("unsupported", "unknown", "unrecognized", "invalid", "not allowed")):
                    return "prompt_cache_key"
            if "cached_content" in text or "cached content" in text:
                if any(marker in text for marker in ("unsupported", "unknown", "unrecognized", "invalid", "not allowed")):
                    return "cached_content"
            if ("requesty" in text and ("auto_cache" in text or "auto cache" in text or "autocache" in text)):
                if any(marker in text for marker in ("unsupported", "unknown", "unrecognized", "invalid", "not allowed")):
                    return "requesty_auto_cache"
            return None

        def apply_reasoning_controls(request_args: Dict[str, Any]) -> None:
            extra_body: Dict[str, Any] = {}
            existing_extra = request_args.get("extra_body")
            if isinstance(existing_extra, dict):
                extra_body.update(existing_extra)
            if reasoning_effort:
                if is_gemini_target:
                    request_args["reasoning_effort"] = reasoning_effort
                else:
                    extra_body["reasoning"] = {"effort": reasoning_effort}
            if thinking_level:
                if is_gemini_target:
                    # Google-specific params must be nested under extra_body["extra_body"]["google"]
                    # so they appear as {"extra_body": {"google": {...}}} in the HTTP request body.
                    # thinking_level should be expressed via google.thinking_config, not reasoning_effort.
                    gemini_inner = extra_body.get("extra_body")
                    if not isinstance(gemini_inner, dict):
                        gemini_inner = {}
                    google_payload = gemini_inner.get("google")
                    if not isinstance(google_payload, dict):
                        google_payload = {}
                    thinking_config = google_payload.get("thinking_config")
                    if not isinstance(thinking_config, dict):
                        thinking_config = {}
                    thinking_config["thinking_level"] = thinking_level
                    google_payload["thinking_config"] = thinking_config
                    gemini_inner["google"] = google_payload
                    extra_body["extra_body"] = gemini_inner
                else:
                    extra_body["thinkingLevel"] = thinking_level
            if effort:
                extra_body["effort"] = effort
            if normalized_gemini_cached_content and is_gemini_target:
                # Google-specific params must be nested under extra_body["extra_body"]["google"]
                # so they appear as {"extra_body": {"google": {...}}} in the HTTP request body.
                gemini_inner = extra_body.get("extra_body")
                if not isinstance(gemini_inner, dict):
                    gemini_inner = {}
                google_payload = gemini_inner.get("google")
                if not isinstance(google_payload, dict):
                    google_payload = {}
                google_payload["cached_content"] = normalized_gemini_cached_content
                gemini_inner["google"] = google_payload
                extra_body["extra_body"] = gemini_inner
            if extra_body:
                request_args["extra_body"] = extra_body

        def apply_requesty_controls(request_args: Dict[str, Any]) -> None:
            if not is_requesty_target or normalized_requesty_auto_cache is None:
                return
            extra_body: Dict[str, Any] = {}
            existing_extra = request_args.get("extra_body")
            if isinstance(existing_extra, dict):
                extra_body.update(existing_extra)
            requesty_payload = extra_body.get("requesty")
            if not isinstance(requesty_payload, dict):
                requesty_payload = {}
            requesty_payload["auto_cache"] = bool(normalized_requesty_auto_cache)
            extra_body["requesty"] = requesty_payload
            request_args["extra_body"] = extra_body

        def remove_request_parameter(request_args: Dict[str, Any], param_name: str) -> bool:
            removed = False
            if param_name == "reasoning":
                if "reasoning_effort" in request_args:
                    request_args.pop("reasoning_effort", None)
                    removed = True
            if param_name == "reasoning_effort":
                if "reasoning" in request_args:
                    request_args.pop("reasoning", None)
                    removed = True
            if param_name == "verbosity":
                if "verbosity" in request_args:
                    request_args.pop("verbosity", None)
                    removed = True
            if param_name in request_args:
                request_args.pop(param_name, None)
                removed = True
            extra_body = request_args.get("extra_body")
            if isinstance(extra_body, dict) and param_name in extra_body:
                extra_body.pop(param_name, None)
                removed = True
            if isinstance(extra_body, dict):
                if param_name in {"thinkingLevel", "thinking_level", "thinking_config", "google_thinking_config"}:
                    gemini_inner = extra_body.get("extra_body")
                    if isinstance(gemini_inner, dict):
                        google_payload = gemini_inner.get("google")
                        if isinstance(google_payload, dict):
                            thinking_cfg = google_payload.get("thinking_config")
                            if isinstance(thinking_cfg, dict) and "thinking_level" in thinking_cfg:
                                thinking_cfg.pop("thinking_level", None)
                                removed = True
                                if not thinking_cfg:
                                    google_payload.pop("thinking_config", None)
                                if not google_payload:
                                    gemini_inner.pop("google", None)
                                if not gemini_inner:
                                    extra_body.pop("extra_body", None)
                if param_name in {"cached_content", "gemini_cached_content", "google_cached_content"}:
                    gemini_inner = extra_body.get("extra_body")
                    if isinstance(gemini_inner, dict):
                        google_payload = gemini_inner.get("google")
                        if isinstance(google_payload, dict) and "cached_content" in google_payload:
                            google_payload.pop("cached_content", None)
                            removed = True
                            if not google_payload:
                                gemini_inner.pop("google", None)
                            if not gemini_inner:
                                extra_body.pop("extra_body", None)
                if param_name in {"reasoning", "reasoning_effort"}:
                    if "reasoning" in extra_body:
                        extra_body.pop("reasoning", None)
                        removed = True
                if param_name in {"requesty_auto_cache", "requesty.autocache", "requesty.auto_cache"}:
                    requesty_payload = extra_body.get("requesty")
                    if isinstance(requesty_payload, dict) and "auto_cache" in requesty_payload:
                        requesty_payload.pop("auto_cache", None)
                        removed = True
                        if not requesty_payload:
                            extra_body.pop("requesty", None)
                if not extra_body:
                    request_args.pop("extra_body", None)
            text_payload = request_args.get("text")
            if isinstance(text_payload, dict):
                if param_name in {"verbosity", "text", "text_verbosity"}:
                    if "verbosity" in text_payload:
                        text_payload.pop("verbosity", None)
                        removed = True
                    if not text_payload:
                        request_args.pop("text", None)
            elif param_name == "text" and "text" in request_args:
                request_args.pop("text", None)
                removed = True
            return removed

        def collect_sent_controls(request_args: Dict[str, Any]) -> Dict[str, str]:
            sent: Dict[str, str] = {}

            if request_args.get("reasoning_effort") is not None:
                sent["reasoning_effort"] = str(request_args["reasoning_effort"])

            reasoning_payload = request_args.get("reasoning")
            if isinstance(reasoning_payload, dict):
                effort_value = reasoning_payload.get("effort")
                if effort_value is not None:
                    sent["reasoning_effort"] = str(effort_value)
            elif reasoning_payload is not None:
                sent["reasoning_effort"] = str(reasoning_payload)

            if request_args.get("thinkingLevel") is not None:
                sent["thinking_level"] = str(request_args["thinkingLevel"])

            if request_args.get("effort") is not None:
                sent["effort"] = str(request_args["effort"])

            if request_args.get("verbosity") is not None:
                sent["verbosity"] = str(request_args["verbosity"])
            if request_args.get("prompt_cache_key") is not None:
                sent["prompt_cache_key"] = str(request_args["prompt_cache_key"])
            if request_args.get("cached_content") is not None:
                sent["gemini_cached_content"] = str(request_args["cached_content"])

            extra_body = request_args.get("extra_body")
            if isinstance(extra_body, dict):
                extra_reasoning = extra_body.get("reasoning")
                if isinstance(extra_reasoning, dict):
                    effort_value = extra_reasoning.get("effort")
                    if effort_value is not None:
                        sent["reasoning_effort"] = str(effort_value)
                elif extra_reasoning is not None:
                    sent["reasoning_effort"] = str(extra_reasoning)

                if extra_body.get("thinkingLevel") is not None:
                    sent["thinking_level"] = str(extra_body["thinkingLevel"])
                top_level_thinking_cfg = extra_body.get("thinking_config")
                if isinstance(top_level_thinking_cfg, dict) and top_level_thinking_cfg.get("thinking_level") is not None:
                    sent["thinking_level"] = str(top_level_thinking_cfg["thinking_level"])
                if extra_body.get("cached_content") is not None:
                    sent["gemini_cached_content"] = str(extra_body["cached_content"])
                requesty_payload = extra_body.get("requesty")
                if isinstance(requesty_payload, dict) and requesty_payload.get("auto_cache") is not None:
                    sent["requesty_auto_cache"] = (
                        "true" if bool(requesty_payload["auto_cache"]) else "false"
                    )

                gemini_inner = extra_body.get("extra_body")
                if isinstance(gemini_inner, dict):
                    google_payload = gemini_inner.get("google")
                    if isinstance(google_payload, dict):
                        thinking_cfg = google_payload.get("thinking_config")
                        if isinstance(thinking_cfg, dict) and thinking_cfg.get("thinking_level") is not None:
                            sent["thinking_level"] = str(thinking_cfg["thinking_level"])
                        if google_payload.get("cached_content") is not None:
                            sent["gemini_cached_content"] = str(google_payload["cached_content"])

                if extra_body.get("effort") is not None:
                    sent["effort"] = str(extra_body["effort"])

            text_payload = request_args.get("text")
            if isinstance(text_payload, dict):
                if text_payload.get("verbosity") is not None:
                    sent["verbosity"] = str(text_payload["verbosity"])

            return sent

        def finalize_control_state(
            request_args: Dict[str, Any],
        ) -> Tuple[Dict[str, str], Dict[str, str]]:
            sent_controls = collect_sent_controls(request_args)
            # Rejections can be recorded on an earlier failed attempt and then recovered
            # by retrying with an alternate representation. Strict acceptance should apply
            # to the final successful payload only.
            final_rejected_controls = {
                key: reason
                for key, reason in rejected_controls.items()
                if key not in sent_controls
            }
            return sent_controls, final_rejected_controls

        def collect_logprobs(logprobs_obj: Any) -> Optional[List[Dict[str, Any]]]:
            entries: List[Dict[str, Any]] = []
            if not logprobs_obj:
                return None
            content_attr = getattr(logprobs_obj, "content", None)
            if content_attr:
                for item in content_attr:
                    entries.append(
                        {
                            "token": getattr(item, "token", None),
                            "logprob": getattr(item, "logprob", None),
                            "top_logprobs": [
                                {
                                    "token": getattr(candidate, "token", None),
                                    "logprob": getattr(candidate, "logprob", None),
                                }
                                for candidate in getattr(item, "top_logprobs", []) or []
                            ],
                        }
                    )
                return entries
            if isinstance(logprobs_obj, dict):
                content = logprobs_obj.get("content") or logprobs_obj.get("tokens")
                if content:
                    for item in content:
                        if isinstance(item, dict):
                            entries.append(
                                {
                                    "token": item.get("token") or item.get("text"),
                                    "logprob": item.get("logprob"),
                                    "top_logprobs": item.get("top_logprobs"),
                                }
                            )
                    return entries
            return None

        def usage_metric(usage_obj: Any, key: str) -> Optional[int]:
            if not usage_obj:
                return None
            candidate_keys = [key]
            if key == "prompt_tokens":
                candidate_keys.append("input_tokens")
            elif key == "completion_tokens":
                candidate_keys.append("output_tokens")

            for candidate_key in candidate_keys:
                if isinstance(usage_obj, dict):
                    value = usage_obj.get(candidate_key)
                else:
                    value = getattr(usage_obj, candidate_key, None)
                if isinstance(value, int) and not isinstance(value, bool):
                    return int(value)
            return None

        def serialize_usage_payload(value: Any) -> Any:
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, dict):
                serialized: Dict[str, Any] = {}
                for raw_key, raw_value in value.items():
                    if raw_value is None:
                        continue
                    serialized[str(raw_key)] = serialize_usage_payload(raw_value)
                return serialized
            if isinstance(value, (list, tuple)):
                serialized_items = [
                    serialize_usage_payload(item) for item in value if item is not None
                ]
                return serialized_items
            for method_name in ("model_dump", "to_dict", "dict"):
                method = getattr(value, method_name, None)
                if callable(method):
                    try:
                        dumped = method()
                    except TypeError:
                        continue
                    except Exception:
                        continue
                    return serialize_usage_payload(dumped)
            value_dict = getattr(value, "__dict__", None)
            if isinstance(value_dict, dict):
                return serialize_usage_payload(value_dict)
            return str(value)

        def collect_usage_metadata(response_obj: Any, usage_obj: Any) -> Optional[Dict[str, Any]]:
            usage_payload = serialize_usage_payload(usage_obj)
            usage_metadata_obj: Any = None

            def read_field(value: Any, *keys: str) -> Any:
                if value is None:
                    return None
                for key in keys:
                    if isinstance(value, dict):
                        if key in value and value.get(key) is not None:
                            return value.get(key)
                        continue
                    attr_value = getattr(value, key, None)
                    if attr_value is not None:
                        return attr_value
                    model_extra = getattr(value, "model_extra", None)
                    if isinstance(model_extra, dict) and model_extra.get(key) is not None:
                        return model_extra.get(key)
                return None

            usage_metadata_obj = read_field(response_obj, "usage_metadata", "usageMetadata")
            if usage_metadata_obj is None:
                usage_metadata_obj = read_field(usage_payload, "usage_metadata", "usageMetadata")
            usage_metadata_payload = serialize_usage_payload(usage_metadata_obj)

            metadata: Dict[str, Any] = {}
            if isinstance(usage_payload, dict) and usage_payload:
                metadata["usage"] = usage_payload
            if isinstance(usage_metadata_payload, dict) and usage_metadata_payload:
                metadata["usage_metadata"] = usage_metadata_payload

            # Capture any remaining model_extra fields from the response that were not
            # already extracted as usage_metadata above.  This ensures provider-specific
            # extensions (e.g. Gemini's usageMetadata, thoughtsTokenCount, etc.) are
            # preserved in the log even if their key names differ from what read_field
            # searches for, making the data available for compute_usage_metadata_summary.
            response_model_extra = getattr(response_obj, "model_extra", None)
            if isinstance(response_model_extra, dict):
                already_used = {"usage_metadata", "usageMetadata"}
                extra_fields: Dict[str, Any] = {}
                for key, val in response_model_extra.items():
                    if key in already_used or val is None:
                        continue
                    serialized = serialize_usage_payload(val)
                    if serialized is not None and serialized != {} and serialized != []:
                        extra_fields[key] = serialized
                if extra_fields:
                    metadata["response_extra"] = extra_fields

            return metadata or None

        def complete_with_responses_api() -> CompletionResult:
            request_args = {
                "model": request_model,
                "input": messages,
            }
            if include_logprobs:
                request_args["logprobs"] = True
            if temperature is not None:
                request_args["temperature"] = temperature
            if top_p is not None:
                request_args["top_p"] = top_p
            if verbosity:
                text_payload = request_args.get("text")
                if not isinstance(text_payload, dict):
                    text_payload = {}
                text_payload["verbosity"] = verbosity
                request_args["text"] = text_payload
            if service_tier and service_tier != "standard":
                request_args["service_tier"] = service_tier
            if normalized_prompt_cache_key:
                request_args["prompt_cache_key"] = normalized_prompt_cache_key
            apply_reasoning_controls(request_args)
            apply_requesty_controls(request_args)
            for unsupported in self._responses_unsupported_params.get(model_key, set()):
                if remove_request_parameter(request_args, unsupported):
                    mark_control_rejected(unsupported, "previously_rejected")

            while True:
                try:
                    self._refresh_access_token_if_needed()
                    self._throttle_request_if_needed()
                    response = self._client.responses.create(**request_args)
                    break
                except Exception as exc:  # noqa: BLE001
                    unsupported_param = extract_unsupported_parameter(exc) or infer_known_unsupported_parameter(exc)
                    if unsupported_param in {"model", "input", "messages"}:
                        raise
                    if unsupported_param and remove_request_parameter(request_args, unsupported_param):
                        self._responses_unsupported_params.setdefault(model_key, set()).add(
                            unsupported_param
                        )
                        mark_control_rejected(unsupported_param, "api_rejected")
                        logging.info(
                            "Responses API rejected parameter '%s' for model %s; retrying without it.",
                            unsupported_param,
                            request_model,
                        )
                        continue
                    if "logprobs" in request_args:
                        self._responses_unsupported_params.setdefault(model_key, set()).add("logprobs")
                        warn_logprob_retry(exc)
                        request_args.pop("logprobs", None)
                        continue
                    raise
            usage = getattr(response, "usage", None)
            prompt_tokens = usage_metric(usage, "prompt_tokens")
            completion_tokens = usage_metric(usage, "completion_tokens")
            total_tokens = usage_metric(usage, "total_tokens")
            if total_tokens is None and (
                prompt_tokens is not None or completion_tokens is not None
            ):
                total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

            texts: List[str] = []
            token_logprobs: List[Dict[str, Any]] = []
            output_items = getattr(response, "output", None) or []
            if isinstance(output_items, list):
                for item in output_items:
                    content_segments = (
                        item.get("content")
                        if isinstance(item, dict)
                        else getattr(item, "content", None)
                    ) or []
                    for segment in content_segments:
                        segment_type = (
                            segment.get("type")
                            if isinstance(segment, dict)
                            else getattr(segment, "type", None)
                        )
                        segment_text = (
                            (segment.get("text") if isinstance(segment, dict) else getattr(segment, "text", None))
                            or (
                                segment.get("output_text")
                                if isinstance(segment, dict)
                                else getattr(segment, "output_text", None)
                            )
                            or ""
                        )
                        if segment_type == "output_text" and segment_text:
                            texts.append(segment_text)
                            segment_logprobs = collect_logprobs(
                                segment.get("logprobs")
                                if isinstance(segment, dict)
                                else getattr(segment, "logprobs", None)
                            )
                            if segment_logprobs:
                                token_logprobs.extend(segment_logprobs)

            if not texts:
                fallback_text = getattr(response, "output_text", None)
                if isinstance(fallback_text, list):
                    fallback_text = "".join(str(chunk) for chunk in fallback_text if chunk is not None)
                if isinstance(fallback_text, str) and fallback_text:
                    texts.append(fallback_text)

            usage_metadata = collect_usage_metadata(response, usage)

            sent_controls, final_rejected_controls = finalize_control_state(request_args)
            return CompletionResult(
                text="".join(texts),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                token_logprobs=token_logprobs or None,
                usage_metadata=usage_metadata,
                request_controls_requested=dict(requested_controls),
                request_controls_sent=sent_controls,
                request_controls_rejected=final_rejected_controls,
            )

        if self.client_type == "chat_v1":
            if hasattr(self._client, "responses") and model_key in self._chat_incompatible_models:
                return complete_with_responses_api()
            request_args = {
                "model": request_model,
                "messages": messages,
            }
            if include_logprobs:
                request_args["logprobs"] = True
                request_args["top_logprobs"] = 1
            if temperature is not None:
                request_args["temperature"] = temperature
            if top_p is not None:
                request_args["top_p"] = top_p
            if verbosity:
                request_args["verbosity"] = verbosity
            if service_tier and service_tier != "standard":
                request_args["service_tier"] = service_tier
            if normalized_prompt_cache_key:
                request_args["prompt_cache_key"] = normalized_prompt_cache_key
            apply_reasoning_controls(request_args)
            apply_requesty_controls(request_args)
            for unsupported in self._chat_unsupported_params.get(model_key, set()):
                if remove_request_parameter(request_args, unsupported):
                    mark_control_rejected(unsupported, "previously_rejected")

            while True:
                try:
                    self._refresh_access_token_if_needed()
                    self._throttle_request_if_needed()
                    response = self._client.chat.completions.create(**request_args)
                    break
                except Exception as exc:  # noqa: BLE001
                    if hasattr(self._client, "responses") and should_retry_with_responses(exc):
                        self._chat_incompatible_models.add(model_key)
                        logging.info(
                            "Model %s is not chat-completions compatible; retrying with Responses API.",
                            request_model,
                        )
                        return complete_with_responses_api()
                    unsupported_param = extract_unsupported_parameter(exc) or infer_known_unsupported_parameter(exc)
                    if unsupported_param in {"model", "input", "messages"}:
                        raise
                    if unsupported_param and remove_request_parameter(request_args, unsupported_param):
                        unsupported_params = self._chat_unsupported_params.setdefault(model_key, set())
                        unsupported_params.add(unsupported_param)
                        mark_control_rejected(unsupported_param, "api_rejected")
                        if unsupported_param == "logprobs":
                            unsupported_params.add("top_logprobs")
                        logging.info(
                            "Chat Completions rejected parameter '%s' for model %s; retrying without it.",
                            unsupported_param,
                            request_model,
                        )
                        if unsupported_param == "logprobs":
                            request_args.pop("top_logprobs", None)
                        continue
                    if "logprobs" in request_args or "top_logprobs" in request_args:
                        unsupported_params = self._chat_unsupported_params.setdefault(model_key, set())
                        unsupported_params.add("logprobs")
                        unsupported_params.add("top_logprobs")
                        warn_logprob_retry(exc)
                        request_args.pop("logprobs", None)
                        request_args.pop("top_logprobs", None)
                        continue
                    raise
            message = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
            completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
            total_tokens = getattr(usage, "total_tokens", None) if usage else None
            token_logprobs = collect_logprobs(getattr(response.choices[0], "logprobs", None))
            usage_metadata = collect_usage_metadata(response, usage)
            sent_controls, final_rejected_controls = finalize_control_state(request_args)
            return CompletionResult(
                text=message,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                token_logprobs=token_logprobs,
                usage_metadata=usage_metadata,
                request_controls_requested=dict(requested_controls),
                request_controls_sent=sent_controls,
                request_controls_rejected=final_rejected_controls,
            )
        if self.client_type == "responses_v1":
            return complete_with_responses_api()
        # Legacy SDK path
        try:
            request_args = {
                "model": request_model,
                "messages": messages,
            }
            if include_logprobs:
                request_args["logprobs"] = True
                request_args["top_logprobs"] = 1
            if temperature is not None:
                request_args["temperature"] = temperature
            if top_p is not None:
                request_args["top_p"] = top_p
            if verbosity:
                mark_control_rejected("verbosity", "legacy_sdk_ignored")
                logging.debug("Verbosity control is ignored in legacy OpenAI SDK mode.")
            if service_tier and service_tier != "standard":
                request_args["service_tier"] = service_tier
            if normalized_prompt_cache_key:
                request_args["prompt_cache_key"] = normalized_prompt_cache_key
            if normalized_gemini_cached_content:
                mark_control_rejected("cached_content", "legacy_sdk_ignored")
            if normalized_requesty_auto_cache is not None and is_requesty_target:
                mark_control_rejected("requesty_auto_cache", "legacy_sdk_ignored")
            if reasoning_effort or thinking_level or effort:
                mark_control_rejected("reasoning", "legacy_sdk_ignored")
                mark_control_rejected("thinkingLevel", "legacy_sdk_ignored")
                mark_control_rejected("effort", "legacy_sdk_ignored")
                logging.debug(
                    "Reasoning/thinking effort controls are ignored in legacy OpenAI SDK mode."
                )
            self._refresh_access_token_if_needed()
            self._throttle_request_if_needed()
            response = self._client.ChatCompletion.create(**request_args)
        except Exception as exc:  # noqa: BLE001
            if include_logprobs and (
                "logprobs" in request_args or "top_logprobs" in request_args
            ):
                warn_logprob_retry(exc)
                request_args.pop("logprobs", None)
                request_args.pop("top_logprobs", None)
                self._refresh_access_token_if_needed()
                self._throttle_request_if_needed()
                response = self._client.ChatCompletion.create(**request_args)
            else:
                raise
        content = response["choices"][0]["message"]["content"]
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        logprobs_obj = None
        choice = response["choices"][0]
        if isinstance(choice, dict):
            logprobs_obj = choice.get("logprobs")
        token_logprobs = collect_logprobs(logprobs_obj)
        usage_metadata = collect_usage_metadata(response, usage)
        sent_controls, final_rejected_controls = finalize_control_state(request_args)
        return CompletionResult(
            text=content or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            token_logprobs=token_logprobs,
            usage_metadata=usage_metadata,
            request_controls_requested=dict(requested_controls),
            request_controls_sent=sent_controls,
            request_controls_rejected=final_rejected_controls,
        )


# --------------------------- Prompt Builder -------------------------------- #


def build_prompt_artifacts(
    example: Example,
    system_prompt: Optional[str],
    enable_cot: bool,
    include_explanation: bool,
    prompt_layout: str = "standard",
    few_shot_context: Optional[List[Example]] = None,
    cache_padding_text: Optional[str] = None,
    suppress_system_message: bool = False,
) -> PromptBuildArtifacts:
    """Construct chat messages and shared-prefix metadata for cache targeting."""
    layout = (prompt_layout or "standard").strip().lower()
    alias_map = {
        "legacy": "standard",
        "minimal": "compact",
        "aggressive": "compact",
    }
    normalized_layout = alias_map.get(layout, layout)
    if normalized_layout != layout:
        logging.warning(
            "Prompt layout %r is deprecated; using %r instead.",
            layout,
            normalized_layout,
        )
    layout = normalized_layout
    if layout not in {"standard", "compact"}:
        logging.warning(
            "Unknown prompt layout %r; falling back to 'standard'.", prompt_layout
        )
        layout = "standard"

    if system_prompt:
        system_msg = system_prompt.strip()
    else:
        system_msg = (
            "You are a meticulous linguistic classifier. "
            "Classify the highlighted node word according to the task instructions."
        )
    system_msg = f"{system_msg.rstrip()}\n\n{MANDATORY_SYSTEM_APPEND}"

    if layout == "standard":
        user_instructions = [
            "You will receive a text excerpt with separate left/right context fields and a marked example where the node is wrapped as âź¦nodeâź§.",
            "When the node itself contains inner âź¦...âź§ spans, those marked passages are the classification target; the rest of the node and the contexts remain useful evidence only.",
            "Identify the label that best matches the required span according to the task definition.",
            "The payload includes a classification_target helper indicating exactly which text must be classified.",
        ]
    else:
        user_instructions = [
            "You will receive left_context, node, and right_context fields for a text excerpt.",
            "If the node contains inner âź¦...âź§ spans, classify only those marked spans; otherwise classify the full node.",
            "Identify the label that best matches the required span according to the task definition.",
            "The payload includes a classification_target helper indicating exactly which text must be classified.",
        ]

    if enable_cot:
        user_instructions.insert(
            2,
            "Think through the linguistic evidence step-by-step before committing to the label.",
        )

    if include_explanation:
        json_instruction = (
            "Return a JSON object with keys: label (string), explanation (string), confidence (float in [0,1]), "
            "node_echo (string), span_source (string)."
        )
    else:
        json_instruction = (
            "Return a JSON object with keys: label (string), confidence (float in [0,1]), node_echo (string), "
            'span_source (string). Do not include an explanation field.'
        )

    user_instructions.append(json_instruction)
    user_instructions.append(
        'Set span_source to "node" when the entire node is being classified. '
        'If any inner âź¦...âź§ spans exist, set span_source to "marked_subspan" and set node_echo to exactly the marked text '
        "(join multiple marked spans with a single space, in order)."
    )
    user_instructions.append(
        "An additional field named 'info' may provide guidance or metadata relevant to the label; factor it into your decision."
    )
    user_instructions.append(
        "Contract: if node_echo or span_source fail to meet these requirements, the response will be rejected."
    )
    user_instructions.append("Do not include any text outside the JSON object.")

    user_content_prefix = "\n".join(user_instructions)

    def serialize_payload(payload: Any) -> str:
        if layout == "standard":
            return json.dumps(payload, ensure_ascii=False, indent=2)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def build_example_payload(target: Example, include_label: bool) -> Dict[str, Any]:
        span_text, span_focus = resolve_span_contract(target.node)
        marked_example = mark_node_in_context(
            target.left_context, target.node, target.right_context
        )
        classification_target = {
            "focus": span_focus,
            "text": span_text,
        }
        payload: Dict[str, Any]
        if layout == "compact":
            payload = {
                "left_context": target.left_context,
                "node": target.node,
                "right_context": target.right_context,
                "classification_target": classification_target,
            }
            if target.info:
                payload["info"] = target.info
        else:
            classification_note = (
                "Classify only the marked sub-span; use the rest of the node plus contexts as supporting evidence."
                if span_focus == SPAN_SOURCE_MARKED_SUBSPAN
                else "Classify the entire node; left/right contexts simply provide supporting evidence."
            )
            payload = {
                "left_context": target.left_context,
                "node": target.node,
                "right_context": target.right_context,
                "info": target.info,
                "marked_example": marked_example,
                "classification_target": {
                    "focus": span_focus,
                    "text": span_text,
                    "note": classification_note,
                },
            }
        if include_label:
            payload["label"] = target.truth
        return payload

    if few_shot_context:
        samples = [
            build_example_payload(sample, include_label=True)
            for sample in few_shot_context
        ]
        user_content_prefix += (
            f"\n\nHere are {len(samples)} labeled example(s) you should mimic when classifying:\n"
            + serialize_payload(samples)
        )

    normalized_cache_padding = (cache_padding_text or "").strip()
    if normalized_cache_padding:
        system_msg = system_msg.rstrip() + "\n\n" + normalized_cache_padding

    target_payload = build_example_payload(example, include_label=False)
    user_content_prefix += "\n\nNow classify this example:\n"
    variable_payload_text = serialize_payload(target_payload)
    user_content = user_content_prefix + variable_payload_text

    if suppress_system_message:
        messages = [{"role": "user", "content": user_content}]
    else:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ]
    shared_prefix_text = f"{system_msg}\n\n{user_content_prefix}"
    shared_prefix_tokens_estimate = estimate_token_count_from_text(shared_prefix_text)
    variable_payload_tokens_estimate = estimate_token_count_from_text(variable_payload_text)
    return PromptBuildArtifacts(
        messages=messages,
        shared_prefix_text=shared_prefix_text,
        variable_payload_text=variable_payload_text,
        shared_prefix_tokens_estimate=shared_prefix_tokens_estimate,
        variable_payload_tokens_estimate=variable_payload_tokens_estimate,
    )


def build_messages(
    example: Example,
    system_prompt: Optional[str],
    enable_cot: bool,
    include_explanation: bool,
    prompt_layout: str = "standard",
    few_shot_context: Optional[List[Example]] = None,
    cache_padding_text: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Construct chat messages for the classification prompt."""
    return build_prompt_artifacts(
        example=example,
        system_prompt=system_prompt,
        enable_cot=enable_cot,
        include_explanation=include_explanation,
        prompt_layout=prompt_layout,
        few_shot_context=few_shot_context,
        cache_padding_text=cache_padding_text,
    ).messages


# --------------------------- Evaluation ------------------------------------ #


@dataclass
class Prediction:
    label: str
    explanation: str
    confidence: Optional[float]
    raw_response: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    label_logprob: Optional[float] = None
    label_probability: Optional[float] = None
    node_echo: Optional[str] = None
    span_source: Optional[str] = None
    validator_status: Optional[str] = None
    validator_reason: Optional[str] = None
    shared_prefix_tokens_estimate: Optional[int] = None
    variable_prompt_tokens_estimate: Optional[int] = None


class ProviderQuotaExceededError(RuntimeError):
    """Raised when provider quota/rate limit is exhausted after retries."""


class ProviderEmptyResponseError(RuntimeError):
    """Raised when provider repeatedly returns empty model responses."""


class RequestedControlRejectedError(RuntimeError):
    """Raised when requested controls are rejected by the endpoint."""


def compute_metrics(
    truths: Iterable[str],
    preds: Iterable[str],
) -> Dict[str, Any]:
    """Compute accuracy, macro metrics, per-label stats, and confusion matrix."""
    truths = list(truths)
    preds = list(preds)
    if len(truths) != len(preds):
        raise ValueError("Length mismatch between truths and predictions.")

    labels = sorted(set(truths) | set(preds))

    confusion: Dict[str, Dict[str, int]] = {t: {p: 0 for p in labels} for t in labels}
    total = len(truths)
    correct = 0

    for y_true, y_pred in zip(truths, preds):
        confusion[y_true][y_pred] += 1
        if y_true == y_pred:
            correct += 1

    per_label: Dict[str, Dict[str, float]] = {}
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0

    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        precision_sum += precision
        recall_sum += recall
        f1_sum += f1

        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion[label].values()),
        }

    label_count = len(labels) if labels else 1
    metrics = {
        "accuracy": correct / total if total else 0.0,
        "macro_precision": precision_sum / label_count,
        "macro_recall": recall_sum / label_count,
        "macro_f1": f1_sum / label_count,
        "per_label": per_label,
        "confusion_matrix": confusion,
        "labels": labels,
        "total_examples": total,
    }
    return metrics


PLOTTING_DEPS_AVAILABLE: Optional[bool] = None


def ensure_calibration_dependencies(purpose: str = "plotting") -> bool:
    """Verify plotting dependencies are installed, installing if user agrees."""
    global PLOTTING_DEPS_AVAILABLE
    if PLOTTING_DEPS_AVAILABLE is not None:
        return PLOTTING_DEPS_AVAILABLE

    required_packages = ["matplotlib"]
    missing: List[str] = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if not missing:
        PLOTTING_DEPS_AVAILABLE = True
        return PLOTTING_DEPS_AVAILABLE

    message = (
        f"The following packages are required for {purpose} but missing: {', '.join(missing)}.\n"
        "Install them now? [y/N]: "
    )
    try:
        user_reply = input(message)
    except EOFError:
        user_reply = ""

    if user_reply.strip().lower() not in {"y", "yes"}:
        logging.info("Skipping installation; calibration plots will be disabled.")
        return False

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logging.error("Failed to install required packages: %s", exc)
        return False

    # Verify installation succeeded.
    for package in missing:
        try:
            __import__(package)
        except ImportError:
            logging.error("Package %s remains unavailable after installation.", package)
            PLOTTING_DEPS_AVAILABLE = False
            return False

    logging.info("Successfully installed %s dependencies.", purpose)
    PLOTTING_DEPS_AVAILABLE = True
    return PLOTTING_DEPS_AVAILABLE


def configure_matplotlib_backend(purpose: str = "plotting") -> bool:
    """Use a non-interactive backend so plots work in headless/low-memory environments."""
    if not ensure_calibration_dependencies(purpose):
        logging.warning("matplotlib not installed; skipping %s.", purpose)
        return False

    try:
        import matplotlib

        matplotlib.use("Agg")
        return True
    except Exception as exc:  # Defensive: backend selection failed.
        logging.warning("Unable to configure matplotlib backend for %s: %s; skipping plotting.", purpose, exc)
        return False


def generate_calibration_plot(
    confidences: List[float],
    correctness: List[bool],
    output_path: str,
    bin_count: int = 10,
) -> None:
    """Generate a reliability diagram showing calibration performance."""
    if not configure_matplotlib_backend("calibration plots"):
        return
    import matplotlib.pyplot as plt

    if not confidences or not correctness:
        logging.warning("No confidence data available; skipping calibration plot.")
        return

    # Clamp values between 0 and 1 to avoid plotting issues.
    capped = [min(1.0, max(0.0, c)) for c in confidences]
    bins = [i / bin_count for i in range(bin_count + 1)]
    bin_totals = [0] * bin_count
    bin_correct = [0] * bin_count

    for conf, corr in zip(capped, correctness):
        # Last bin includes 1.0
        index = min(bin_count - 1, int(conf * bin_count))
        bin_totals[index] += 1
        bin_correct[index] += 1 if corr else 0

    accuracies = [bin_correct[i] / bin_totals[i] if bin_totals[i] else 0 for i in range(bin_count)]
    confidences = [
        (bins[i] + bins[i + 1]) / 2 for i in range(bin_count)
    ]  # Midpoints for plotting

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.bar(confidences, accuracies, width=1 / bin_count, alpha=0.6, align="center", label="Model")
    plt.xlabel("Predicted confidence")
    plt.ylabel("Empirical accuracy")
    plt.title("Calibration Plot")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()
    logging.info("Saved calibration plot to %s", output_path)


def generate_confusion_heatmap(
    confusion: Dict[str, Dict[str, int]],
    labels: List[str],
    output_path: str,
) -> None:
    """Render a dual-panel confusion matrix heatmap (counts + row percentages)."""
    if not configure_matplotlib_backend("confusion heatmap"):
        return
    import matplotlib.pyplot as plt

    if not confusion or not labels:
        logging.warning("Confusion matrix or label list empty; skipping heatmap.")
        return

    size = len(labels)
    matrix = [
        [float(confusion.get(true_label, {}).get(pred_label, 0)) for pred_label in labels]
        for true_label in labels
    ]
    row_totals = [sum(row) for row in matrix]
    percentage_matrix: List[List[float]] = []
    for total, row in zip(row_totals, matrix):
        if total > 0:
            percentage_row = [(value / total) * 100.0 for value in row]
        else:
            percentage_row = [0.0 for _ in row]
        percentage_matrix.append(percentage_row)

    fig_width = max(8, size * 2.2)
    fig_height = max(4.5, size * 0.8)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), constrained_layout=True)

    specs = [
        ("Confusion Matrix (counts)", matrix, lambda v: f"{int(round(v))}"),
        ("Confusion Matrix (row %)", percentage_matrix, lambda v: f"{v:.1f}%"),
    ]

    for ax, (title, data, formatter) in zip(axes, specs):
        im = ax.imshow(data, cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks(range(size))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(size))
        ax.set_yticklabels(labels)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        data_max = max((max(row) for row in data), default=0) or 1.0
        for i in range(size):
            for j in range(size):
                value = data[i][j]
                display = formatter(value)
                ax.text(
                    j,
                    i,
                    display,
                    ha="center",
                    va="center",
                    color="white" if value > (data_max * 0.6) else "black",
                    fontsize=9,
                )

    fig.suptitle("Confusion Matrix Overview", fontsize=14)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logging.info("Saved confusion heatmap to %s", output_path)


# --------------------------- Logprob Utilities ----------------------------- #


def _extract_target_logprob(
    response_text: str,
    target: str,
    token_logprobs: Optional[List[Dict[str, Any]]],
) -> Tuple[Optional[float], Optional[float]]:
    """Estimate log probability/probability for a target substring in response text."""
    if not token_logprobs or not target:
        return None, None

    if response_text.find(target) == -1:
        return None, None

    combined = "".join(str(entry.get("token", "")) for entry in token_logprobs)
    substring_index = combined.find(target)
    if substring_index == -1:
        # Attempt a whitespace-insensitive match.
        normalized_combined = re.sub(r"\s+", "", combined)
        normalized_target = re.sub(r"\s+", "", target)
        substring_index = normalized_combined.find(normalized_target)
        if substring_index == -1:
            return None, None
        # Unable to map back accurately; fall back to tokens containing pieces of target.
        matching_tokens = [
            entry for entry in token_logprobs if str(entry.get("token", "")) and str(entry.get("token", "")) in target
        ]
    else:
        end_index = substring_index + len(target)
        cumulative = 0
        matching_tokens = []
        for entry in token_logprobs:
            token_text = str(entry.get("token", ""))
            start = cumulative
            end = start + len(token_text)
            cumulative = end
            if end <= substring_index:
                continue
            if start >= end_index:
                break
            matching_tokens.append(entry)

    logprob_sum = 0.0
    valid = False
    for entry in matching_tokens:
        logprob = entry.get("logprob")
        if isinstance(logprob, (int, float)):
            logprob_sum += logprob
            valid = True
        else:
            return None, None

    if not valid:
        return None, None

    probability = math.exp(logprob_sum)
    return logprob_sum, probability


def extract_label_logprob(
    response_text: str,
    label: str,
    token_logprobs: Optional[List[Dict[str, Any]]],
) -> Tuple[Optional[float], Optional[float]]:
    """Estimate the log probability and probability for the label token sequence."""
    if not token_logprobs or not label:
        return None, None

    target = f'"label": "{label}"'
    logprob, probability = _extract_target_logprob(response_text, target, token_logprobs)
    if logprob is not None and probability is not None:
        return logprob, probability
    # Fallback to matching the label string alone (including surrounding quotes).
    fallback_target = f'"{label}"'
    return _extract_target_logprob(response_text, fallback_target, token_logprobs)


def extract_node_echo_logprob(
    response_text: str,
    node_echo: str,
    token_logprobs: Optional[List[Dict[str, Any]]],
) -> Tuple[Optional[float], Optional[float]]:
    """Estimate the log probability/probability for node_echo value in model JSON output."""
    if not token_logprobs or not node_echo:
        return None, None
    target = f'"node_echo": "{node_echo}"'
    return _extract_target_logprob(response_text, target, token_logprobs)


# --------------------------- Main Benchmarking ----------------------------- #


def classify_example(
    connector: OpenAIConnector,
    example: Example,
    model: str,
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    verbosity: Optional[str],
    service_tier: Optional[str],
    include_logprobs: bool,
    reasoning_effort: Optional[str],
    thinking_level: Optional[str],
    effort: Optional[str],
    system_prompt: Optional[str],
    enable_cot: bool,
    include_explanation: bool,
    prompt_layout: str,
    few_shot_context: Optional[List[Example]],
    max_retries: int,
    retry_delay: float,
    validator_client: Optional[ValidatorClient] = None,
    validator_prompt_max_candidates: int = 50,
    validator_prompt_max_chars: int = 8000,
    validator_exhausted_policy: str = "accept_blank_confidence",
    strict_control_acceptance: bool = False,
    cache_padding_text: Optional[str] = None,
    cache_padding_tokens_estimate: int = 0,
    prompt_cache_key: Optional[str] = None,
    gemini_cached_content: Optional[str] = None,
    requesty_auto_cache: Optional[bool] = None,
) -> Tuple[Prediction, List[Dict[str, Any]]]:
    """Query the model and parse the prediction, returning attempt logs."""
    prompt_artifacts = build_prompt_artifacts(
        example=example,
        system_prompt=system_prompt,
        enable_cot=enable_cot,
        include_explanation=include_explanation,
        prompt_layout=prompt_layout,
        few_shot_context=few_shot_context,
        cache_padding_text=cache_padding_text,
        # When using a Gemini CachedContent the system instruction is already stored
        # in the cache; sending it again causes a 400 INVALID_ARGUMENT error.
        suppress_system_message=bool(gemini_cached_content),
    )
    base_messages = prompt_artifacts.messages
    validator_patch_message: Optional[Dict[str, str]] = None
    validator_status: Optional[str] = None
    validator_reason: Optional[str] = None
    last_error: Optional[Exception] = None
    latest_raw_response = ""
    latest_prompt_tokens: Optional[int] = None
    latest_completion_tokens: Optional[int] = None
    latest_total_tokens: Optional[int] = None
    validation_failures = 0
    interaction_logs: List[Dict[str, Any]] = []

    for attempt in range(1, max_retries + 1):
        messages = list(base_messages)
        if validator_patch_message is not None:
            messages.append(validator_patch_message)

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            prompt_snapshot = json.dumps(messages, ensure_ascii=False, indent=2)
            logging.debug(
                "Prompt for example %s (attempt %d/%d):\n%s",
                example.example_id,
                attempt,
                max_retries,
                prompt_snapshot,
            )

        log_entry: Dict[str, Any] = {
            "attempt": attempt,
            "timestamp": utc_timestamp(),
            "request": copy.deepcopy(messages),
            "prompt_padding": {
                "applied": bool(cache_padding_text),
                "padding_tokens_estimate": int(cache_padding_tokens_estimate)
                if cache_padding_tokens_estimate > 0
                else 0,
            },
            "prompt_estimate": {
                "shared_prefix_tokens_estimate": prompt_artifacts.shared_prefix_tokens_estimate,
                "variable_tokens_estimate": prompt_artifacts.variable_payload_tokens_estimate,
            },
        }
        try:
            result = connector.complete(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                verbosity=verbosity,
                service_tier=service_tier,
                include_logprobs=include_logprobs,
                reasoning_effort=reasoning_effort,
                thinking_level=thinking_level,
                effort=effort,
                prompt_cache_key=prompt_cache_key,
                gemini_cached_content=gemini_cached_content,
                requesty_auto_cache=requesty_auto_cache,
            )
            raw = result.text
            latest_raw_response = raw
            latest_prompt_tokens = result.prompt_tokens
            latest_completion_tokens = result.completion_tokens
            latest_total_tokens = result.total_tokens
            log_entry["request_controls"] = {
                "requested": result.request_controls_requested,
                "sent": result.request_controls_sent,
                "rejected": result.request_controls_rejected,
            }
            if result.request_controls_rejected:
                rejection_reasons = set(result.request_controls_rejected.values())
                log_message = (
                    "Request controls were rejected for example %s: %s "
                    "(sent=%s)."
                )
                if rejection_reasons <= {"previously_rejected"}:
                    logging.debug(
                        log_message,
                        example.example_id,
                        result.request_controls_rejected,
                        result.request_controls_sent,
                    )
                else:
                    logging.warning(
                        log_message,
                        example.example_id,
                        result.request_controls_rejected,
                        result.request_controls_sent,
                    )
            if strict_control_acceptance and result.request_controls_requested:
                requested_keys = set(result.request_controls_requested.keys())
                sent_keys = set(result.request_controls_sent.keys())
                rejected_keys = set(result.request_controls_rejected.keys())
                missing_keys = sorted((requested_keys - sent_keys) | rejected_keys)
                if missing_keys:
                    raise RequestedControlRejectedError(
                        "Requested controls were not accepted: "
                        f"{missing_keys}; rejected={result.request_controls_rejected}; "
                        f"sent={result.request_controls_sent}."
                    )
            log_entry["response"] = {
                "text": raw,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
                "usage_metadata": result.usage_metadata,
            }
            payload = extract_json_object(raw)
            label = str(payload.get("label", "")).strip()
            if include_explanation:
                explanation = str(payload.get("explanation", "")).strip()
                if not explanation:
                    logging.warning(
                        "Model omitted explanation for example %s despite it being requested.",
                        example.example_id,
                    )
            else:
                explanation = str(payload.get("explanation", "")).strip()
                if explanation:
                    logging.debug(
                        "Model returned an explanation for example %s even though it was not requested; ignoring.",
                        example.example_id,
                    )
                explanation = ""
            confidence_raw = payload.get("confidence")
            confidence = safe_float(confidence_raw, default=0.0)
            if label == "":
                raise ValueError("Model returned empty label.")
            if not math.isfinite(confidence):
                logging.debug(
                    "Invalid confidence %r received for example %s; forcing to 0.0.",
                    confidence_raw,
                    example.example_id,
                )
                confidence = 0.0
            elif not (0.0 <= confidence <= 1.0):
                logging.debug(
                    "Clamping out-of-range confidence %.4f for example %s to [0,1].",
                    confidence,
                    example.example_id,
                )
                confidence = min(1.0, max(0.0, confidence))

            node_echo = str(payload.get("node_echo", "")).strip()
            span_source = str(payload.get("span_source", "")).strip()
            expected_node_echo, expected_span_source = resolve_span_contract(example.node)
            span_source_normalized = span_source.lower()
            expected_span_source_normalized = expected_span_source.lower()

            if node_echo != expected_node_echo or span_source_normalized != expected_span_source_normalized:
                validation_failures += 1
                logging.warning(
                    "Model referenced an incorrect span for example %s (node_echo=%r, span_source=%r, expected node_echo=%r, expected span_source=%r).",
                    example.example_id,
                    node_echo,
                    span_source,
                    expected_node_echo,
                    expected_span_source,
                )
                log_entry["status"] = "validation_failed"
                log_entry["validation_error"] = {
                    "node_echo": node_echo,
                    "span_source": span_source,
                    "expected_node_echo": expected_node_echo,
                    "expected_span_source": expected_span_source,
                }
                if validation_failures >= 3 or attempt == max_retries:
                    logging.error(
                        "Validation failed %d time(s) for example %s; accepting last response but withholding confidence.",
                        validation_failures,
                        example.example_id,
                    )
                    confidence = None
                    log_entry["status"] = "accepted_after_validation"
                else:
                    raise ValueError("Model failed node/span contract; retrying.")

            if validator_client is not None:
                request_id = f"{example.example_id}:{attempt}"
                validator_request: Dict[str, Any] = {
                    "type": "validate",
                    "schema_version": 1,
                    "request_id": request_id,
                    "attempt": {"index": attempt, "max": max_retries},
                    "example": {
                        "id": example.example_id,
                        "left_context": example.left_context,
                        "node": example.node,
                        "right_context": example.right_context,
                        "info": example.info,
                        "truth": example.truth,
                        "classification_target": {
                            "focus": expected_span_source,
                            "text": expected_node_echo,
                        },
                    },
                    "prediction": {
                        "label": label,
                        "confidence": confidence,
                        "explanation": explanation,
                        "node_echo": node_echo,
                        "span_source": span_source_normalized,
                        "raw_response": raw,
                    },
                }
                log_entry["validator_request"] = validator_request

                try:
                    validator_result = validator_client.validate(validator_request)
                except ValidatorError as exc:
                    raise RuntimeError(f"Validator failed for example {example.example_id}: {exc}") from exc

                log_entry["validator_result"] = validator_result

                action = str(validator_result.get("action", "")).strip().lower()
                reason = str(validator_result.get("reason", "")).strip()

                if action == "accept":
                    normalized = validator_result.get("normalized") or {}
                    normalized_label = str(normalized.get("label", "")).strip()
                    if normalized_label:
                        label = normalized_label
                    validator_status = "accept"
                    validator_reason = reason
                elif action == "abort":
                    log_entry["parsed_payload"] = payload
                    log_entry["status"] = "validator_abort"
                    interaction_logs.append(log_entry)
                    raise RuntimeError(
                        f"Validator aborted example {example.example_id}: {reason or 'no reason provided'}"
                    )
                elif action == "retry":
                    retry_payload = validator_result.get("retry") or {}
                    allowed_labels = retry_payload.get("allowed_labels") or []
                    retry_instruction = str(retry_payload.get("instruction", "")).strip()
                    if attempt >= max_retries:
                        if validator_exhausted_policy == "accept_blank_confidence":
                            confidence = None
                            validator_status = "accepted_after_validator"
                            validator_reason = reason or "validator_retry_exhausted"
                            log_entry["status"] = "accepted_after_validator"
                        elif validator_exhausted_policy == "unclassified":
                            label = "unclassified"
                            confidence = None
                            validator_status = "accepted_after_validator"
                            validator_reason = reason or "validator_retry_exhausted"
                            log_entry["status"] = "accepted_after_validator"
                        else:
                            log_entry["status"] = "validator_abort"
                            interaction_logs.append(log_entry)
                            raise RuntimeError(
                                f"Validator requested retry but attempts exhausted for example {example.example_id}."
                            )
                    else:
                        retry_message = render_validator_retry_message(
                            allowed_labels=allowed_labels,
                            instruction=retry_instruction,
                            max_candidates=validator_prompt_max_candidates,
                            max_chars=validator_prompt_max_chars,
                        )
                        validator_patch_message = {"role": "user", "content": retry_message}
                        log_entry["parsed_payload"] = payload
                        log_entry["status"] = "validator_retry"
                        interaction_logs.append(log_entry)
                        time.sleep(retry_delay)
                        continue
                else:
                    raise RuntimeError(
                        f"Validator returned unknown action={action!r} for example {example.example_id}."
                    )

            total_tokens = result.total_tokens
            if total_tokens is None and result.prompt_tokens is not None and result.completion_tokens is not None:
                total_tokens = result.prompt_tokens + result.completion_tokens

            logging.debug("Response for example %s:\n%s", example.example_id, raw)
            if any(value is not None for value in (result.prompt_tokens, result.completion_tokens, total_tokens)):
                logging.info(
                    "Token usage for example %s -> prompt: %s, completion: %s, total: %s",
                    example.example_id,
                    result.prompt_tokens,
                    result.completion_tokens,
                    total_tokens,
                )

            label_logprob, label_probability = extract_label_logprob(
                raw, label, result.token_logprobs
            )
            if label_logprob is not None and label_probability is not None:
                logging.info(
                    "Label probability for example %s -> logprob: %.4f, probability: %.4f",
                    example.example_id,
                    label_logprob,
                    label_probability,
                )
            node_echo_logprob, node_echo_probability = extract_node_echo_logprob(
                raw, node_echo, result.token_logprobs
            )
            if node_echo_logprob is not None and node_echo_probability is not None:
                logging.info(
                    "Node probability for example %s -> logprob: %.4f, probability: %.4f",
                    example.example_id,
                    node_echo_logprob,
                    node_echo_probability,
                )
            response_log = log_entry.get("response")
            if isinstance(response_log, dict):
                response_log["node_echo_logprob"] = (
                    f"{node_echo_logprob:.6f}" if node_echo_logprob is not None else None
                )
                response_log["node_echo_probability"] = (
                    f"{node_echo_probability:.6f}" if node_echo_probability is not None else None
                )

            log_entry["status"] = log_entry.get("status", "success")
            log_entry["parsed_payload"] = payload
            interaction_logs.append(log_entry)
            return Prediction(
                label=label,
                explanation=explanation,
                confidence=confidence,
                raw_response=raw,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=total_tokens,
                label_logprob=label_logprob,
                label_probability=label_probability,
                node_echo=node_echo or None,
                span_source=span_source or None,
                validator_status=validator_status,
                validator_reason=validator_reason,
                shared_prefix_tokens_estimate=prompt_artifacts.shared_prefix_tokens_estimate,
                variable_prompt_tokens_estimate=prompt_artifacts.variable_payload_tokens_estimate,
            ), interaction_logs
        except Exception as exc:  # noqa: BLE001 - surface API errors to user
            last_error = exc
            strict_control_error = (
                strict_control_acceptance and isinstance(exc, RequestedControlRejectedError)
            )
            empty_response_error = is_empty_model_response_error(exc)
            if isinstance(exc, json.JSONDecodeError) and attempt < max_retries:
                # Give the model deterministic feedback about why the previous output was rejected.
                validator_patch_message = {
                    "role": "user",
                    "content": (
                        "Your previous response was not valid JSON and could not be parsed "
                        f"({exc}).\n"
                        "Return ONLY one syntactically valid JSON object, with properly escaped "
                        'strings and no trailing commas. Do not include Markdown/code fences.'
                    ),
                }
            if "status" not in log_entry:
                log_entry["status"] = "error"
            log_entry["error"] = str(exc)
            interaction_logs.append(log_entry)
            logging.warning(
                "Attempt %d/%d failed for example %s: %s",
                attempt,
                max_retries,
                example.example_id,
                exc,
            )
            if strict_control_error:
                break
            if attempt < max_retries:
                if empty_response_error:
                    logging.warning(
                        "Empty model response detected for example %s; waiting %.0f seconds before retry.",
                        example.example_id,
                        EMPTY_RESPONSE_RETRY_DELAY_SECONDS,
                    )
                    time.sleep(EMPTY_RESPONSE_RETRY_DELAY_SECONDS)
                else:
                    time.sleep(retry_delay)

    assert last_error is not None
    if isinstance(last_error, json.JSONDecodeError):
        logging.error(
            "Unable to parse model output as JSON for example %s after %d attempt(s); "
            "continuing with fallback label='unclassified' and blank confidence.",
            example.example_id,
            max_retries,
        )
        return Prediction(
            label="unclassified",
            explanation="",
            confidence=None,
            raw_response=latest_raw_response,
            prompt_tokens=latest_prompt_tokens,
            completion_tokens=latest_completion_tokens,
            total_tokens=latest_total_tokens,
            label_logprob=None,
            label_probability=None,
            node_echo=None,
            span_source=None,
            validator_status="accepted_after_parse_error",
            validator_reason=str(last_error),
            shared_prefix_tokens_estimate=prompt_artifacts.shared_prefix_tokens_estimate,
            variable_prompt_tokens_estimate=prompt_artifacts.variable_payload_tokens_estimate,
        ), interaction_logs
    if is_quota_or_rate_limit_error(last_error):
        detail = str(last_error).strip() or last_error.__class__.__name__
        raise ProviderQuotaExceededError(
            f"Provider quota/rate limit exhausted for example {example.example_id}: {detail}"
        ) from last_error
    if is_empty_model_response_error(last_error):
        detail = str(last_error).strip() or last_error.__class__.__name__
        raise ProviderEmptyResponseError(
            "Provider returned empty model responses after retries for "
            f"example {example.example_id}: {detail}"
        ) from last_error
    if isinstance(last_error, RequestedControlRejectedError):
        raise last_error
    raise RuntimeError(f"Failed to classify example {example.example_id}") from last_error


def process_dataset(
    connector: OpenAIConnector,
    input_path: str,
    output_path: str,
    args: argparse.Namespace,
    include_explanation: bool,
    calibration_enabled: bool,
    label_map: Optional[Dict[str, str]],
    resolved_api_base_url: Optional[str],
    validator_client: Optional[ValidatorClient] = None,
) -> Tuple[int, int, int, bool]:
    """Run the full evaluation pipeline for a single dataset."""
    logging.info("Loading dataset from %s", input_path)
    examples, extra_field_order = read_examples(input_path)
    logging.info("Loaded %d examples.", len(examples))

    if label_map:
        merge_labels(examples, label_map)

    missing_truth = [ex.example_id for ex in examples if ex.truth is None]
    if missing_truth:
        logging.warning(
            "Found %d example(s) without ground-truth labels. Metrics will skip them.",
            len(missing_truth),
        )

    predictions: Dict[str, Prediction] = {}
    halted_by_quota = False
    few_shot_count = max(0, args.few_shot_examples)
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_reported_tokens = 0
    cache_pad_target_tokens = max(0, int(getattr(args, "cache_pad_target_tokens", 0) or 0))
    cache_padding_tokens_estimate = 0
    cache_padding_text: Optional[str] = None
    cache_padding_calibration_shared_prefix_tokens: Optional[int] = None
    cache_padding_calibration_example_id: Optional[str] = None
    cache_padding_applied_examples = 0
    cache_padding_missing_prefix_warned = False

    # When --create_gemini_cache pre-computed a fixed padding, use it for every request
    # instead of calibrating at runtime.  This ensures the system message in each request
    # matches the system instruction stored in the Gemini CachedContent resource.
    gemini_cache_preset_padding = getattr(args, "_gemini_cache_preset_padding", None)
    if gemini_cache_preset_padding:
        cache_padding_text = gemini_cache_preset_padding
        cache_padding_tokens_estimate = estimate_token_count_from_chars(len(gemini_cache_preset_padding))
        cache_padding_calibration_shared_prefix_tokens = 0  # sentinel: skip calibration
        logging.info(
            "Using pre-computed cache padding from --create_gemini_cache (%d chars, ~%d tokens).",
            len(gemini_cache_preset_padding),
            cache_padding_tokens_estimate,
        )
    elif cache_pad_target_tokens > 0:
        logging.info(
            "Cache padding target enabled at %d shared-prefix tokens. First successful example will calibrate shared-prefix padding.",
            cache_pad_target_tokens,
        )

    ensure_directory(output_path)
    log_path = os.path.splitext(output_path)[0] + ".log"
    ensure_directory(log_path)
    fieldnames = [
        "ID",
        "leftContext",
        "node",
        "rightContext",
        "info",
        "truth",
    ]
    if extra_field_order:
        fieldnames.extend(extra_field_order)
    fieldnames.extend(
        [
            "prediction",
            "explanation",
            "confidence",
            "nodeEcho",
            "spanSource",
            "promptTokens",
            "completionTokens",
            "totalTokens",
            "labelLogProb",
            "labelProbability",
            "validatorStatus",
            "validatorReason",
        ]
    )

    resume_mode = os.path.isfile(output_path) and os.path.getsize(output_path) > 0
    writer_fieldnames = list(fieldnames)
    if resume_mode:
        (
            existing_fieldnames,
            existing_predictions,
            existing_prompt_tokens,
            existing_completion_tokens,
            existing_reported_tokens,
        ) = load_existing_output_predictions(output_path)
        if not existing_fieldnames:
            logging.warning(
                "Existing output file %s has no CSV header; starting a fresh output file.",
                output_path,
            )
            resume_mode = False
        else:
            writer_fieldnames = existing_fieldnames
            predictions.update(existing_predictions)
            total_prompt_tokens += existing_prompt_tokens
            total_completion_tokens += existing_completion_tokens
            total_reported_tokens += existing_reported_tokens

            missing_writer_fields = {"ID", "prediction"} - set(writer_fieldnames)
            if missing_writer_fields:
                raise ValueError(
                    f"Cannot resume writing to {output_path}: missing required columns {sorted(missing_writer_fields)}."
                )

            missing_columns = [name for name in fieldnames if name not in writer_fieldnames]
            if missing_columns:
                logging.warning(
                    "Resuming into an older output schema; new rows cannot populate columns: %s",
                    ", ".join(missing_columns),
                )

    processed_ids = set(predictions.keys())
    processed_in_input = sum(1 for ex in examples if ex.example_id in processed_ids)
    pending_total = len(examples) - processed_in_input
    if resume_mode:
        if pending_total <= 0:
            logging.info(
                "Output already contains all %d input IDs; no new examples to classify.",
                len(examples),
            )
        else:
            first_pending = next((ex for ex in examples if ex.example_id not in processed_ids), None)
            if first_pending is not None:
                logging.info(
                    "Resuming from first unprocessed record ID=%s (%d already processed, %d remaining).",
                    first_pending.example_id,
                    processed_in_input,
                    pending_total,
                )

    if resume_mode:
        logging.info("Resuming predictions in %s", output_path)
    else:
        logging.info("Writing predictions to %s", output_path)
    logging.info("Saving prompt/response log to %s", log_path)

    if resume_mode:
        log_records = load_existing_prompt_log(log_path)
    else:
        log_records = []

    run_model_details = build_run_model_details(
        provider=args.provider,
        requested_model=args.model,
        api_base_url=resolved_api_base_url,
        api_key_var=args.api_key_var,
        api_base_var=args.api_base_var,
        gemini_cached_content=args.gemini_cached_content,
    )
    upsert_prompt_log_run_metadata(log_records, run_model_details)

    def flush_prompt_log() -> None:
        with open(log_path, "w", encoding="utf-8") as log_handle:
            json.dump(log_records, log_handle, ensure_ascii=False, indent=2)

    flush_prompt_log()

    file_mode = "a" if resume_mode else "w"
    with open(output_path, file_mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=writer_fieldnames, delimiter=";")
        if not resume_mode:
            writer.writeheader()
        processed_this_run = 0
        try:
            for example in examples:
                if example.example_id in processed_ids:
                    continue
                processed_this_run += 1
                logging.info(
                    "Classifying example %s (%d/%d)",
                    example.example_id,
                    processed_this_run,
                    pending_total,
                )
                few_shot_context = select_few_shot_examples(examples, example.example_id, few_shot_count)
                padding_active_for_example = bool(cache_padding_text)
                try:
                    prediction, attempt_logs = classify_example(
                        connector=connector,
                        example=example,
                        model=args.model,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        verbosity=args.verbosity,
                        service_tier=args.service_tier,
                        include_logprobs=args.logprobs,
                        reasoning_effort=args.reasoning_effort,
                        thinking_level=args.thinking_level,
                        effort=args.effort,
                        system_prompt=args.system_prompt,
                        enable_cot=args.enable_cot,
                        include_explanation=include_explanation,
                        prompt_layout=args.prompt_layout,
                        few_shot_context=few_shot_context,
                        max_retries=args.max_retries,
                        retry_delay=args.retry_delay,
                        validator_client=validator_client,
                        validator_prompt_max_candidates=args.validator_prompt_max_candidates,
                        validator_prompt_max_chars=args.validator_prompt_max_chars,
                        validator_exhausted_policy=args.validator_exhausted_policy,
                        strict_control_acceptance=args.strict_control_acceptance,
                        cache_padding_text=cache_padding_text,
                        cache_padding_tokens_estimate=cache_padding_tokens_estimate,
                        prompt_cache_key=args.prompt_cache_key,
                        gemini_cached_content=args.gemini_cached_content,
                        requesty_auto_cache=args.requesty_auto_cache,
                    )
                except ProviderQuotaExceededError as exc:
                    halted_by_quota = True
                    logging.error(
                        "%s",
                        exc,
                    )
                    logging.error(
                        "Stopping dataset early due to provider quota/rate limit exhaustion. "
                        "Partial outputs were saved and can be resumed later."
                    )
                    break
                except ProviderEmptyResponseError as exc:
                    halted_by_quota = True
                    logging.error(
                        "%s",
                        exc,
                    )
                    logging.error(
                        "Stopping dataset early due to repeated empty model responses. "
                        "Partial outputs were saved and can be resumed later."
                    )
                    break
                predictions[example.example_id] = prediction
                if padding_active_for_example:
                    cache_padding_applied_examples += 1
                if prediction.prompt_tokens is not None:
                    total_prompt_tokens += prediction.prompt_tokens
                if prediction.completion_tokens is not None:
                    total_completion_tokens += prediction.completion_tokens
                if prediction.total_tokens is not None:
                    total_reported_tokens += prediction.total_tokens

                if cache_pad_target_tokens > 0:
                    if cache_padding_calibration_shared_prefix_tokens is None:
                        shared_prefix_tokens = prediction.shared_prefix_tokens_estimate
                        if shared_prefix_tokens is None:
                            if not cache_padding_missing_prefix_warned:
                                logging.warning(
                                    "Shared-prefix token estimate unavailable; waiting to calibrate cache padding."
                                )
                                cache_padding_missing_prefix_warned = True
                        else:
                            cache_padding_missing_prefix_warned = False
                            cache_padding_calibration_shared_prefix_tokens = shared_prefix_tokens
                            cache_padding_calibration_example_id = example.example_id
                            required_padding_tokens = estimate_required_cache_padding_tokens(
                                shared_prefix_tokens,
                                cache_pad_target_tokens,
                            )
                            if required_padding_tokens > 0:
                                cache_padding_tokens_estimate = required_padding_tokens
                                cache_padding_text = build_cache_padding_text(required_padding_tokens)
                                logging.info(
                                    "Calibrated cache padding from example %s: shared_prefix_tokens~%d, target=%d, applying +%d padding units to subsequent prompts.",
                                    example.example_id,
                                    shared_prefix_tokens,
                                    cache_pad_target_tokens,
                                    cache_padding_tokens_estimate,
                                )
                            else:
                                cache_padding_tokens_estimate = 0
                                cache_padding_text = None
                                logging.info(
                                    "Cache padding target already met on calibration example %s (shared_prefix_tokens~%d, target=%d); no padding will be applied.",
                                    example.example_id,
                                    shared_prefix_tokens,
                                    cache_pad_target_tokens,
                                )

                log_records.append(
                    {
                        "example_id": example.example_id,
                        "attempts": attempt_logs,
                        "final_prediction": {
                            "label": prediction.label,
                            "confidence": prediction.confidence,
                            "explanation": prediction.explanation,
                            "truth": example.truth,
                            "validator_status": prediction.validator_status,
                            "validator_reason": prediction.validator_reason,
                        },
                    }
                )

                confidence_str = (
                    f"{prediction.confidence:.4f}" if prediction.confidence is not None else ""
                )
                row = {
                    "ID": example.example_id,
                    "leftContext": example.left_context,
                    "node": example.node,
                    "rightContext": example.right_context,
                    "info": example.info,
                    "truth": example.truth or "",
                    "prediction": prediction.label,
                    "explanation": prediction.explanation,
                    "confidence": confidence_str,
                    "nodeEcho": prediction.node_echo or "",
                    "spanSource": prediction.span_source or "",
                    "promptTokens": prediction.prompt_tokens if prediction.prompt_tokens is not None else "",
                    "completionTokens": prediction.completion_tokens if prediction.completion_tokens is not None else "",
                    "totalTokens": prediction.total_tokens if prediction.total_tokens is not None else "",
                    "labelLogProb": f"{prediction.label_logprob:.6f}" if prediction.label_logprob is not None else "",
                    "labelProbability": f"{prediction.label_probability:.6f}" if prediction.label_probability is not None else "",
                    "validatorStatus": prediction.validator_status or "",
                    "validatorReason": prediction.validator_reason or "",
                }
                for field in extra_field_order:
                    row[field] = example.extras.get(field, "")
                row_to_write = {field: row.get(field, "") for field in writer_fieldnames}
                writer.writerow(row_to_write)
                handle.flush()
                flush_prompt_log()
                processed_ids.add(example.example_id)
        finally:
            flush_prompt_log()

    logging.info("Saved prompt log to %s", log_path)

    metrics_output = os.path.splitext(output_path)[0] + "_metrics.json"

    missing_predictions = [
        ex.example_id for ex in examples if ex.example_id not in predictions
    ]
    if missing_predictions:
        if halted_by_quota:
            logging.warning(
                "Stopped with %d unprocessed example(s); first pending IDs: %s",
                len(missing_predictions),
                missing_predictions[:5],
            )
        else:
            raise RuntimeError(
                f"Missing predictions for {len(missing_predictions)} example(s), "
                f"including {missing_predictions[:5]}."
            )

    evaluated_examples = [
        ex for ex in examples if ex.truth is not None and ex.example_id in predictions
    ]
    evaluated_truths = [ex.truth for ex in evaluated_examples]
    evaluated_preds = [predictions[ex.example_id].label for ex in evaluated_examples]

    confidences: List[float] = []
    correctness: List[bool] = []
    for example in examples:
        prediction = predictions.get(example.example_id)
        if prediction is None or example.truth is None or prediction.confidence is None:
            continue
        correctness.append(prediction.label == example.truth)
        confidences.append(prediction.confidence)

    prompt_time_window = compute_prompt_time_window(log_records)
    configured_controls = {
        "reasoning_effort": args.reasoning_effort,
        "thinking_level": args.thinking_level,
        "effort": args.effort,
        "verbosity": args.verbosity,
        "prompt_cache_key": args.prompt_cache_key,
        "gemini_cached_content": args.gemini_cached_content,
        "requesty_auto_cache": args.requesty_auto_cache,
    }
    request_control_summary = compute_request_control_summary(log_records, configured_controls)
    usage_metadata_summary = compute_usage_metadata_summary(log_records)
    cache_padding_summary = {
        "enabled": cache_pad_target_tokens > 0,
        "target_shared_prefix_tokens": cache_pad_target_tokens,
        "calibration_shared_prefix_tokens": cache_padding_calibration_shared_prefix_tokens,
        # Backwards-compatible aliases kept for existing dashboards.
        "target_prompt_tokens": cache_pad_target_tokens,
        "calibration_prompt_tokens": cache_padding_calibration_shared_prefix_tokens,
        "calibration_example_id": cache_padding_calibration_example_id,
        "applied_padding_tokens_estimate": cache_padding_tokens_estimate,
        "examples_with_padding_applied": cache_padding_applied_examples,
    }

    metrics: Dict[str, Any] = {}
    if evaluated_truths:
        metrics = compute_metrics(evaluated_truths, evaluated_preds)
        metrics["model_details"] = run_model_details
        metrics["prompt_layout"] = args.prompt_layout
        metrics["cache_padding"] = cache_padding_summary
        metrics.update(prompt_time_window)
        metrics["request_control_summary"] = request_control_summary
        metrics["usage_metadata_summary"] = usage_metadata_summary
        with open(metrics_output, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, ensure_ascii=False)
        logging.info("Saved metrics to %s", metrics_output)
        if metrics.get("overall_time_seconds") is not None:
            logging.info(
                "Prompt window runtime: %.2f seconds (%s -> %s)",
                metrics["overall_time_seconds"],
                metrics["first_prompt_timestamp"],
                metrics["last_prompt_timestamp"],
            )
        confusion = metrics.get("confusion_matrix")
        labels = metrics.get("labels", [])
        if confusion:
            heatmap_path = os.path.splitext(output_path)[0] + "_confusion_heatmap.png"
            generate_confusion_heatmap(confusion, labels, heatmap_path)
    else:
        logging.warning("No ground-truth labels available; skipping metric computation.")

    configured_controls_non_null = request_control_summary.get("configured", {})
    if configured_controls_non_null:
        per_control = request_control_summary.get("per_control", {})
        for control_name, configured_value in configured_controls_non_null.items():
            control_stats = per_control.get(control_name, {})
            logging.info(
                "Control '%s=%s' summary -> requested attempts: %s, accepted: %s, rejected: %s.",
                control_name,
                configured_value,
                control_stats.get("requested_attempts", 0),
                control_stats.get("accepted_attempts", 0),
                control_stats.get("rejected_attempts", 0),
            )

    if usage_metadata_summary.get("attempts_with_usage_metadata", 0):
        logging.info(
            "Usage metadata summary -> attempts with metadata: %s, cache-signaled attempts: %s, cached tokens (estimate): %s.",
            usage_metadata_summary.get("attempts_with_usage_metadata", 0),
            usage_metadata_summary.get("attempts_with_cached_token_signals", 0),
            usage_metadata_summary.get("cached_tokens_total_estimate", 0),
        )
        gemini_cached_token_total = usage_metadata_summary.get(
            "gemini_cached_content_token_count_total", 0
        )
        gemini_cached_attempts = usage_metadata_summary.get(
            "attempts_with_gemini_cached_content_token_signals", 0
        )
        if gemini_cached_attempts or gemini_cached_token_total:
            logging.info(
                "Gemini cache metadata -> attempts with cachedContentTokenCount: %s, total cachedContentTokenCount: %s.",
                gemini_cached_attempts,
                gemini_cached_token_total,
            )
        elif (
            str(getattr(args, "provider", "")).strip().lower() == "google"
            or "gemini" in str(getattr(args, "model", "")).strip().lower()
        ):
            if getattr(args, "gemini_cached_content", None):
                logging.warning(
                    "Gemini cached content was configured but usage metadata did not report cachedContentTokenCount."
                )
            elif cache_pad_target_tokens > 0:
                logging.warning(
                    "Cache padding was applied (target %d tokens) but Gemini did not report any cachedContentTokenCount. "
                    "Implicit caching may not be supported for this model or endpoint, or the cache was not yet warm. "
                    "For reliable caching, create a CachedContent resource and pass it via --gemini_cached_content.",
                    cache_pad_target_tokens,
                )

    if cache_pad_target_tokens > 0:
        logging.info(
            "Cache padding summary -> target shared-prefix: %d, calibration shared-prefix tokens: %s, applied padding units: %d, padded examples: %d.",
            cache_pad_target_tokens,
            cache_padding_calibration_shared_prefix_tokens
            if cache_padding_calibration_shared_prefix_tokens is not None
            else "N/A",
            cache_padding_tokens_estimate,
            cache_padding_applied_examples,
        )

    if calibration_enabled and confidences and correctness:
        calibration_path = os.path.splitext(output_path)[0] + "_calibration.png"
        generate_calibration_plot(confidences, correctness, calibration_path)

    if metrics:
        accuracy = metrics.get("accuracy", 0.0)
        logging.info("Overall accuracy: %.2f%%", accuracy * 100)
        logging.info("Macro F1: %.3f", metrics.get("macro_f1", 0.0))
        logging.info("Label breakdown:")
        for label, stats in metrics["per_label"].items():
            logging.info(
                "  %s -> precision: %.3f, recall: %.3f, f1: %.3f, support: %d",
                label,
                stats["precision"],
                stats["recall"],
                stats["f1"],
                stats["support"],
            )

    if total_prompt_tokens or total_completion_tokens or total_reported_tokens:
        total_token_usage = (
            total_reported_tokens
            if total_reported_tokens
            else total_prompt_tokens + total_completion_tokens
        )
        logging.info(
            "Aggregate token usage -> prompt: %s, completion: %s, total: %s",
            total_prompt_tokens or "N/A",
            total_completion_tokens or "N/A",
            total_token_usage,
        )

    return total_prompt_tokens, total_completion_tokens, total_reported_tokens, halted_by_quota


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark an OpenAI model on a linguistic classification dataset."
    )

    parser.add_argument("--input", nargs="+", help="Path(s) to input CSV file(s) with examples.")
    parser.add_argument(
        "--labels",
        help="Optional path to CSV file that provides ground-truth labels (ID;truth).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output CSV path or directory. When omitted, defaults to "
            "<input>_out_<provider>_<model>_<timestamp>.csv alongside each input file. "
            "If the resolved output CSV already exists, the run resumes from the first ID "
            "not present in that file."
        ),
    )
    parser.add_argument("--model", help="Model name (e.g., gpt-4-turbo).")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature. Omit to let the provider/model use its default.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Nucleus sampling parameter. Omit to let the provider/model use its default.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        help="Top-k sampling (ignored for APIs that do not support it).",
    )
    parser.add_argument(
        "--service_tier",
        choices=["standard", "flex", "priority"],
        default="standard",
        help="Optional service-tier hint for providers that support differentiated throughput.",
    )
    parser.add_argument(
        "--verbosity",
        choices=["low", "medium", "high"],
        default=None,
        help=(
            "Optional output verbosity control for GPT models. "
            "Sent as verbosity (Chat Completions) or text.verbosity (Responses API)."
        ),
    )
    parser.add_argument(
        "--reasoning_effort",
        choices=["low", "medium", "high", "xhigh"],
        default=None,
        help=(
            "Optional reasoning effort level. "
            "Sent as reasoning.effort for OpenAI-style models and as reasoning_effort for Gemini targets."
        ),
    )
    parser.add_argument(
        "--thinking_level",
        choices=["minimal", "low", "medium", "high"],
        default=None,
        help=(
            "Optional Gemini thinking level (minimal applies to Gemini Flash models). "
            "Sent via extra_body.google.thinking_config for Gemini OpenAI-compatible targets."
        ),
    )
    parser.add_argument(
        "--effort",
        choices=["low", "medium", "high", "max"],
        default=None,
        help=(
            "Optional Claude effort level. "
            "Sent as effort when provided."
        ),
    )
    parser.add_argument(
        "--strict_control_acceptance",
        action="store_true",
        help=(
            "Fail an example when requested controls are rejected "
            "or not present in the final successful request payload."
        ),
    )
    parser.add_argument(
        "--provider",
        default="openai",
        help=(
            "Model provider identifier used to look up default credentials. "
            "Known providers are preconfigured; custom providers are inferred from "
            "<PROVIDER>_API_KEY (or <PROVIDER>_ACCESS_TOKEN) and <PROVIDER>_BASE_URL."
        ),
    )
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--system_prompt",
        default="You are a linguistic classifier that excels at semantic disambiguation.",
        help="System prompt injected into the chat completion.",
    )
    prompt_group.add_argument(
        "--system_prompt_b64",
        help="Base64-encoded system prompt (used by the GUI to ensure cross-platform commands).",
    )
    parser.add_argument(
        "--few_shot_examples",
        type=int,
        default=0,
        help="Number of labeled examples to prepend as few-shot demonstrations.",
    )
    parser.add_argument(
        "--prompt_layout",
        choices=["standard", "compact"],
        default="standard",
        help=(
            "Prompt payload layout. "
            "standard preserves the current verbose payload; "
            "compact removes duplicated fields to improve cache reuse."
        ),
    )
    parser.add_argument(
        "--cache_pad_target_tokens",
        type=int,
        default=0,
        help=(
            "Optional shared-prefix token target for cache padding. "
            "If >0, the first successful example calibrates shared-prefix length; "
            "subsequent prompts are padded toward this shared-prefix target."
        ),
    )
    parser.add_argument(
        "--prompt_cache_key",
        default=None,
        help=(
            "Optional provider cache-routing key (when supported) to improve "
            "prompt-cache hit consistency for stable prompt prefixes."
        ),
    )
    parser.add_argument(
        "--gemini_cached_content",
        default=None,
        help=(
            "Optional Gemini context-cache resource name for providers that expose "
            "Gemini OpenAI-compatible caching via extra_body.extra_body.google.cached_content. "
            "Mutually exclusive with --create_gemini_cache."
        ),
    )
    parser.add_argument(
        "--requesty_auto_cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable/disable Requesty automatic caching by sending "
            "extra_body.requesty.auto_cache. "
            "Only used when --provider requesty."
        ),
    )
    parser.add_argument(
        "--vertex_auto_adc_login",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable/disable automatic one-time ADC login for Vertex when credentials "
            "are missing (browser-based gcloud auth flow). "
            "Only used when --provider vertex."
        ),
    )
    parser.add_argument(
        "--vertex_access_token_refresh_seconds",
        type=int,
        default=None,
        help=(
            "Override Vertex access-token refresh interval in seconds. "
            "Only used when --provider vertex."
        ),
    )
    parser.add_argument(
        "--create_gemini_cache",
        action="store_true",
        help=(
            "Auto-create a Gemini CachedContent resource from the system prompt before "
            "the benchmark run and delete it afterward (unless --keep_gemini_cache is set). "
            "Sets --gemini_cached_content automatically. "
            "Mutually exclusive with --gemini_cached_content."
        ),
    )
    parser.add_argument(
        "--gemini_cache_ttl",
        type=int,
        default=3600,
        help=(
            "Time-to-live in seconds for the auto-created Gemini cache "
            "(default: 3600 = 1 hour). Only used when --create_gemini_cache is set."
        ),
    )
    parser.add_argument(
        "--keep_gemini_cache",
        action="store_true",
        help=(
            "Do not delete the auto-created Gemini cache after the run. "
            "The cache resource name is logged so it can be reused via --gemini_cached_content. "
            "Only meaningful when --create_gemini_cache is set."
        ),
    )
    parser.add_argument(
        "--enable_cot",
        action="store_true",
        help="If set, encourages the model to reason step-by-step before answering.",
    )
    parser.add_argument(
        "--no_explanation",
        action="store_true",
        help="Skip requesting explanations to reduce token usage.",
    )
    parser.add_argument(
        "--logprobs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable token log probabilities when supported. "
            "Use --no-logprobs to skip requesting logprobs."
        ),
    )
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="Generate a calibration plot using the model's confidences.",
    )
    parser.add_argument(
        "--api_key_var",
        default=None,
        help="Environment variable name that stores the API key or access token.",
    )
    parser.add_argument(
        "--api_base_var",
        default=None,
        help="Environment variable name that stores the API base URL.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum number of retry attempts per example on API errors.",
    )
    parser.add_argument(
        "--retry_delay",
        type=float,
        default=5.0,
        help="Delay (seconds) between API retries.",
    )
    parser.add_argument(
        "--request_interval_ms",
        type=int,
        default=0,
        help=(
            "Minimum delay in milliseconds between outgoing API requests. "
            "Use 0 to disable request pacing."
        ),
    )
    parser.add_argument(
        "--validator_cmd",
        default=None,
        help=(
            "Optional path to an NDJSON validator executable/script. "
            "When provided, the agent will validate each prediction and may retry with extra constraints. "
            "If the path ends with .py it will be run via the current Python interpreter."
        ),
    )
    parser.add_argument(
        "--validator_args",
        default="",
        help=(
            "Optional extra arguments passed to the validator command as a single string "
            "(supports quoting). Example: \"--lexicon data/lemmas.txt --max_distance 2\"."
        ),
    )
    parser.add_argument(
        "--validator_timeout",
        type=float,
        default=5.0,
        help="Timeout (seconds) for each validator request/response roundtrip.",
    )
    parser.add_argument(
        "--validator_prompt_max_candidates",
        type=int,
        default=50,
        help="Maximum number of allowed_labels candidates rendered into a validator retry prompt.",
    )
    parser.add_argument(
        "--validator_prompt_max_chars",
        type=int,
        default=8000,
        help="Maximum character length of the validator retry instruction appended to the prompt.",
    )
    parser.add_argument(
        "--validator_exhausted_policy",
        choices=["accept_blank_confidence", "unclassified", "error"],
        default="accept_blank_confidence",
        help=(
            "What to do when the validator keeps requesting retry but --max_retries is exhausted. "
            "accept_blank_confidence keeps the last label but blanks confidence; unclassified forces label to "
            "\"unclassified\"; error aborts the run."
        ),
    )
    parser.add_argument(
        "--validator_debug",
        action="store_true",
        help="Log validator NDJSON send/receive payloads at DEBUG level.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    parser.add_argument(
        "--update-models",
        "-updatemodels",
        dest="update_models",
        action="store_true",
        help="If set, fetch available models for configured providers and update config_models.js.",
    )
    parser.add_argument(
        "--models-output",
        default="config_models.js",
        help="Output path for generated model catalog JS when --update-models is used.",
    )
    parser.add_argument(
        "--models-providers",
        nargs="+",
        help=(
            "Optional list of provider slugs to update when --update-models is specified. "
            "Custom slugs are allowed; env vars are inferred as <SLUG>_API_KEY "
            "(or <SLUG>_ACCESS_TOKEN) and <SLUG>_BASE_URL."
        ),
    )

    args = parser.parse_args(argv)
    if args.system_prompt_b64:
        decoded_prompt = decode_system_prompt_b64(args.system_prompt_b64)
        args.system_prompt = decoded_prompt or ""
    else:
        args.system_prompt = decode_cli_system_prompt(args.system_prompt)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.update_models:
        return update_model_catalog(args.models_providers, args.models_output)

    if not args.input:
        parser.error("--input is required unless --update-models is specified.")

    if not args.model:
        parser.error("--model is required unless --update-models is specified.")
    if args.cache_pad_target_tokens < 0:
        parser.error("--cache_pad_target_tokens must be >= 0.")
    if args.create_gemini_cache and args.gemini_cached_content:
        parser.error(
            "--create_gemini_cache and --gemini_cached_content are mutually exclusive. "
            "Use --create_gemini_cache to auto-create a cache, or --gemini_cached_content "
            "to supply an existing cache resource name."
        )

    overall_start = time.perf_counter()
    env_map = parse_env_file(".env")

    provider = (args.provider or "openai").lower()
    if args.requesty_auto_cache is not None and provider != "requesty":
        logging.warning(
            "--requesty_auto_cache is set but provider is %s; this control is ignored unless --provider requesty.",
            provider,
        )
        args.requesty_auto_cache = None
    if args.vertex_auto_adc_login is not None and provider != "vertex":
        logging.warning(
            "--vertex_auto_adc_login is set but provider is %s; this control is ignored unless --provider vertex.",
            provider,
        )
        args.vertex_auto_adc_login = None
    if args.vertex_access_token_refresh_seconds is not None:
        if args.vertex_access_token_refresh_seconds <= 0:
            parser.error("--vertex_access_token_refresh_seconds must be > 0.")
        if provider != "vertex":
            logging.warning(
                "--vertex_access_token_refresh_seconds is set but provider is %s; this control is ignored unless --provider vertex.",
                provider,
            )
            args.vertex_access_token_refresh_seconds = None
    discovered_provider_defaults = discover_provider_defaults(env_map)
    provider_defaults = discovered_provider_defaults.get(provider) or infer_provider_defaults(provider)
    if args.api_key_var is None:
        args.api_key_var = provider_defaults["api_key_var"]
    if args.api_base_var is None:
        args.api_base_var = provider_defaults["api_base_var"]
    include_explanation = not args.no_explanation

    requested_control_flags = {
        "reasoning_effort": args.reasoning_effort,
        "thinking_level": args.thinking_level,
        "effort": args.effort,
        "verbosity": args.verbosity,
        "prompt_cache_key": args.prompt_cache_key,
        "gemini_cached_content": args.gemini_cached_content,
        "requesty_auto_cache": args.requesty_auto_cache,
    }
    active_requested_controls = {
        key: value for key, value in requested_control_flags.items() if value is not None
    }
    if active_requested_controls:
        logging.info("Requested controls: %s", active_requested_controls)
        if args.strict_control_acceptance:
            logging.info(
                "Strict control acceptance is enabled; runs will fail if requested controls are rejected."
            )
    if args.reasoning_effort and args.thinking_level and (
        provider == "google" or "gemini" in str(args.model).lower()
    ):
        logging.error(
            "Both --reasoning_effort and --thinking_level were requested for a Gemini target. "
            "Use only one because these controls overlap for Gemini APIs."
        )
        return 1
    if provider == "openai" and args.reasoning_effort:
        model_lower = str(args.model or "").strip().lower()
        allowed_reasoning_for_model: Optional[set[str]] = None
        if model_lower.startswith("gpt-5.2-pro"):
            allowed_reasoning_for_model = {"medium", "high", "xhigh"}
        elif model_lower.startswith("gpt-5-pro"):
            allowed_reasoning_for_model = {"high"}
        if (
            allowed_reasoning_for_model is not None
            and args.reasoning_effort not in allowed_reasoning_for_model
        ):
            logging.error(
                "Model %s does not support --reasoning_effort=%s. Allowed values are: %s.",
                args.model,
                args.reasoning_effort,
                ", ".join(sorted(allowed_reasoning_for_model)),
            )
            return 1

    access_token_provider: Optional[AccessTokenProvider] = None
    api_key = resolve_env_value(args.api_key_var, env_map)
    if provider == "vertex":
        bootstrap_token = resolve_vertex_bootstrap_token(args.api_key_var, env_map)
        access_token_provider = build_vertex_access_token_provider(
            env_map,
            bootstrap_token,
            auto_adc_login_override=args.vertex_auto_adc_login,
            refresh_interval_seconds_override=args.vertex_access_token_refresh_seconds,
        )
        try:
            api_key = access_token_provider.get_token(force_refresh=True)
            logging.info("Vertex provider auth initialized with auto-refreshing access token.")
        except RuntimeError as exc:
            if is_placeholder_value(bootstrap_token):
                logging.error("Unable to obtain Vertex access token: %s", exc)
                return 1
            api_key = bootstrap_token
            logging.warning(
                "Vertex token refresh is unavailable; continuing with static token from env. Details: %s",
                exc,
            )
    if is_placeholder_value(api_key):
        logging.error(
            "API credential not found. Ensure %s is defined in .env or the environment.",
            args.api_key_var,
        )
        return 1

    raw_api_base = resolve_env_value(args.api_base_var, env_map) if args.api_base_var else None
    api_base_url = normalize_api_base(provider, raw_api_base)
    if api_base_url:
        logging.info("Using API base URL from %s: %s", args.api_base_var, api_base_url)

    if provider != "openai":
        logging.info(
            "Provider '%s' selected. Ensure the endpoint at %s is OpenAI-compatible.",
            provider,
            api_base_url or "default base URL",
        )

    input_paths = [os.path.expanduser(path) for path in args.input]
    multiple_inputs = len(input_paths) > 1
    timestamp_tag = datetime.now().strftime("%Y-%m-%d-%H-%M")

    label_map: Optional[Dict[str, str]] = None
    if args.labels:
        logging.info("Loading labels from %s", args.labels)
        label_map = read_label_file(args.labels)
    else:
        logging.info("Using truth column embedded in the input files.")

    calibration_enabled = args.calibration
    if calibration_enabled and not ensure_calibration_dependencies():
        logging.warning(
            "Calibration dependencies unavailable. Calibration plots will be skipped."
        )
        calibration_enabled = False
    if not include_explanation:
        logging.info("Explanations disabled; model will only return labels and confidences.")
    if not args.logprobs:
        logging.info("Logprobs disabled; token-level probability estimates will be unavailable.")
    if args.prompt_layout != "standard":
        logging.info(
            "Prompt layout set to '%s' (cache-optimized payload mode).",
            args.prompt_layout,
        )
    if args.cache_pad_target_tokens > 0:
        logging.info(
            "Cache padding target configured: %d shared-prefix tokens (0 disables padding).",
            args.cache_pad_target_tokens,
        )
        if args.cache_pad_target_tokens < 1024:
            logging.warning(
                "Cache padding target %d is below 1024 shared-prefix tokens; OpenAI prompt caching usually requires at least 1024 shared-prefix tokens.",
                args.cache_pad_target_tokens,
            )
    if args.prompt_cache_key:
        logging.info("Prompt cache key configured: %s", args.prompt_cache_key)
    if args.gemini_cached_content:
        logging.info("Gemini cached content configured: %s", args.gemini_cached_content)
        if not (provider == "google" or "gemini" in str(args.model).strip().lower()):
            logging.warning(
                "--gemini_cached_content is set but target does not look like Gemini; this control may be ignored."
            )
    if args.requesty_auto_cache is not None:
        logging.info(
            "Requesty auto cache configured: %s",
            "enabled" if args.requesty_auto_cache else "disabled",
        )

    created_cache_name: Optional[str] = None
    if args.create_gemini_cache:
        is_gemini_target = provider == "google" or "gemini" in str(args.model).strip().lower()
        if not is_gemini_target:
            logging.error(
                "--create_gemini_cache is set but the target does not look like a Gemini model "
                "(provider=%s, model=%s). Aborting.",
                provider,
                args.model,
            )
            return 1
        system_for_cache = (args.system_prompt or DEFAULT_SYSTEM_PROMPT).rstrip()
        if MANDATORY_SYSTEM_APPEND:
            system_for_cache = system_for_cache.rstrip() + "\n\n" + MANDATORY_SYSTEM_APPEND
        # Gemini requires at least 1024 tokens in the cached content.  Pre-compute padding so
        # the cache content meets that minimum (and the user's --cache_pad_target_tokens target
        # if specified).  The same padding is then applied to every request so the system message
        # in each request matches what was cached.
        _GEMINI_CACHE_MIN_TOKENS = 1024
        cache_system_tokens = estimate_token_count_from_chars(len(system_for_cache))
        effective_cache_target = max(args.cache_pad_target_tokens, _GEMINI_CACHE_MIN_TOKENS)
        preset_padding_units = estimate_required_cache_padding_tokens(
            cache_system_tokens, effective_cache_target
        )
        gemini_cache_preset_padding: Optional[str] = None
        if preset_padding_units > 0:
            gemini_cache_preset_padding = build_cache_padding_text(preset_padding_units)
            system_for_cache = system_for_cache.rstrip() + "\n\n" + gemini_cache_preset_padding
            logging.info(
                "Cache padding pre-computed: system prompt ~%d tokens, target %d tokens, +%d padding units.",
                cache_system_tokens,
                effective_cache_target,
                preset_padding_units,
            )
        logging.info(
            "Creating Gemini CachedContent from system prompt (TTL %ds)...",
            args.gemini_cache_ttl,
        )
        try:
            created_cache_name = create_gemini_cached_content(
                api_key, api_base_url, args.model, system_for_cache, args.gemini_cache_ttl
            )
        except RuntimeError as exc:
            logging.error("Failed to create Gemini CachedContent: %s", exc)
            return 1
        args.gemini_cached_content = created_cache_name
        logging.info("Gemini CachedContent created: %s", created_cache_name)
        # Store the pre-computed padding so process_dataset applies the same padding to every
        # request (bypassing runtime calibration, which would measure a different prefix length).
        # Also disable the cache_pad_target_tokens calibration since we already have fixed padding.
        if gemini_cache_preset_padding:
            args._gemini_cache_preset_padding = gemini_cache_preset_padding
            args.cache_pad_target_tokens = 0

    connector = OpenAIConnector(
        api_key=api_key,
        base_url=api_base_url,
        provider=provider,
        request_interval_ms=max(0, args.request_interval_ms),
        access_token_provider=access_token_provider,
    )

    validator_client: Optional[ValidatorClient] = None
    if args.validator_cmd:
        try:
            validator_command = build_validator_command(args.validator_cmd, args.validator_args)
        except ValueError as exc:
            logging.error("Unable to build validator command: %s", exc)
            return 1
        validator_client = ValidatorClient(
            command=validator_command,
            timeout=args.validator_timeout,
            debug=args.validator_debug,
        )
        try:
            validator_client.start(
                ValidatorRunInfo(
                    provider=provider,
                    model=args.model,
                    include_explanation=include_explanation,
                    enable_cot=args.enable_cot,
                    prompt_layout=args.prompt_layout,
                    reasoning_effort=args.reasoning_effort,
                    thinking_level=args.thinking_level,
                    effort=args.effort,
                    verbosity=args.verbosity,
                    cache_pad_target_tokens=args.cache_pad_target_tokens,
                    prompt_cache_key=args.prompt_cache_key,
                    gemini_cached_content=args.gemini_cached_content,
                    requesty_auto_cache=args.requesty_auto_cache,
                    max_retries=args.max_retries,
                )
            )
            logging.info("Validator enabled: %s", " ".join(validator_client.command))
        except Exception as exc:  # noqa: BLE001
            logging.error("Failed to initialize validator: %s", exc)
            return 1

    aggregate_prompt_tokens = 0
    aggregate_completion_tokens = 0
    aggregate_reported_tokens = 0
    stopped_by_quota = False

    try:
        for resolved_input in input_paths:
            output_path = resolve_output_path(
                resolved_input,
                provider,
                args.model,
                args.output,
                timestamp_tag,
                multiple_inputs,
            )
            prompt_tokens, completion_tokens, reported_tokens, halted_by_quota = process_dataset(
                connector=connector,
                input_path=resolved_input,
                output_path=output_path,
                args=args,
                include_explanation=include_explanation,
                calibration_enabled=calibration_enabled,
                label_map=label_map,
                resolved_api_base_url=api_base_url,
                validator_client=validator_client,
            )
            aggregate_prompt_tokens += prompt_tokens
            aggregate_completion_tokens += completion_tokens
            aggregate_reported_tokens += reported_tokens
            if halted_by_quota:
                stopped_by_quota = True
                logging.error(
                    "Stopping remaining input datasets because the provider reported a retry-exhausted failure."
                )
                break
    except RequestedControlRejectedError as exc:
        logging.error("%s", exc)
        logging.error(
            "Run stopped because requested controls were rejected by the model endpoint."
        )
        return 2
    finally:
        if validator_client is not None:
            validator_client.close()
        if created_cache_name is not None:
            if args.keep_gemini_cache:
                logging.info(
                    "Gemini cache kept for reuse: --gemini_cached_content %s",
                    created_cache_name,
                )
            else:
                logging.info("Deleting auto-created Gemini cache: %s", created_cache_name)
                delete_gemini_cached_content(api_key, api_base_url, created_cache_name)

    if aggregate_prompt_tokens or aggregate_completion_tokens or aggregate_reported_tokens:
        total_token_usage = (
            aggregate_reported_tokens
            if aggregate_reported_tokens
            else aggregate_prompt_tokens + aggregate_completion_tokens
        )
        logging.info(
            "Total token usage across all inputs -> prompt: %s, completion: %s, total: %s",
            aggregate_prompt_tokens or "N/A",
            aggregate_completion_tokens or "N/A",
            total_token_usage,
        )

    elapsed_seconds = time.perf_counter() - overall_start
    logging.info("Total runtime: %.2f seconds", elapsed_seconds)

    if stopped_by_quota:
        logging.error(
            "Run ended early due to a provider-side retry-exhausted failure. "
            "Re-run later with the same --output path to resume."
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())

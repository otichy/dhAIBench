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
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from validator_client import ValidatorClient, ValidatorError, ValidatorRunInfo


NODE_MARKER_LEFT = "⟦"
NODE_MARKER_RIGHT = "⟧"
SPAN_SOURCE_NODE = "node"
SPAN_SOURCE_MARKED_SUBSPAN = "marked_subspan"
MANDATORY_SYSTEM_APPEND = (
    "Classify ONLY the text that is explicitly wrapped inside ⟦ ⟧ (the 'node' or its marked sub-span). "
    "Use the surrounding context as supporting evidence, but never change the focus away from the highlighted text. "
    'If you cannot determine the class/label for the node, return "unclassified".'
)


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


def build_default_output_filename(input_path: str, model: str, timestamp_tag: str) -> str:
    """Construct the default output filename for an input dataset."""
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    model_slug = sanitize_model_identifier(model)
    return f"{base_name}_out_{model_slug}_{timestamp_tag}.csv"


def resolve_output_path(
    input_path: str,
    model: str,
    output_argument: Optional[str],
    timestamp_tag: str,
    multiple_inputs: bool,
) -> str:
    """Determine the output path for an input file."""
    filename = build_default_output_filename(input_path, model, timestamp_tag)
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


PROVIDER_DEFAULTS: Dict[str, Dict[str, str]] = {
    "openai": {"api_key_var": "OPENAI_API_KEY", "api_base_var": "OPENAI_BASE_URL"},
    "anthropic": {"api_key_var": "ANTHROPIC_API_KEY", "api_base_var": "ANTHROPIC_BASE_URL"},
    "cohere": {"api_key_var": "COHERE_API_KEY", "api_base_var": "COHERE_BASE_URL"},
    "google": {"api_key_var": "GOOGLE_API_KEY", "api_base_var": "GOOGLE_BASE_URL"},
    "huggingface": {"api_key_var": "HF_API_KEY", "api_base_var": "HF_BASE_URL"},
    "e-infra": {"api_key_var": "E-INFRA_API_KEY", "api_base_var": "E-INFRA_BASE_URL"},
}

PROVIDER_BASE_FALLBACKS: Dict[str, str] = {
    "openai": "https://api.openai.com/v1",
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
    """Discover provider env-var mappings from known defaults plus *_API_KEY keys."""
    discovered: Dict[str, Dict[str, str]] = {
        slug: dict(defaults) for slug, defaults in PROVIDER_DEFAULTS.items()
    }

    key_pattern = re.compile(r"^([A-Z0-9][A-Z0-9_-]*)_API_KEY$")
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


def normalize_api_base(provider: str, api_base: Optional[str]) -> Optional[str]:
    """Ensure the API base ends with a version segment."""
    candidate = (api_base or PROVIDER_BASE_FALLBACKS.get(provider, "")).strip()
    if not candidate:
        return None
    trimmed = candidate.rstrip("/")
    if provider == "google":
        if re.search(r"/openai$", trimmed, re.IGNORECASE):
            return trimmed
        if re.search(r"/v\d+(?:beta\d*)?$", trimmed, re.IGNORECASE):
            return f"{trimmed}/openai"
        return f"{trimmed}/v1beta/openai"
    if not re.search(r"/v\d+(?:beta\d*)?$", trimmed, re.IGNORECASE):
        trimmed = f"{trimmed}/v1"
    return trimmed


def _parse_model_payload(payload: Dict[str, Any], provider: str, endpoint: str) -> Tuple[List[str], Optional[str]]:
    """Normalize provider model payloads into a list of model IDs."""
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


def _fetch_models_with_curl(endpoint: str, api_key: str, provider: str) -> Tuple[List[str], Optional[str]]:
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


def fetch_provider_models(provider: str, api_key: str, api_base: str) -> Tuple[List[str], Optional[str]]:
    """Fetch available models for a provider using raw HTTP."""
    endpoint = f"{api_base}/models"
    request = urllib.request.Request(
        endpoint,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
        message = f"HTTP {exc.code} {exc.reason or ''} {detail}".strip()
        logging.error("Failed to fetch models for provider %s: %s", provider, message)
        return [], message
    except urllib.error.URLError as exc:
        message = str(exc)
        logging.error("Connection error while fetching models for provider %s: %s", provider, message)
        # If Python lacks SSL support, urllib cannot handle HTTPS. Fall back to curl if available.
        if "unknown url type: https" in message.lower():
            logging.warning(
                "Python SSL support appears to be missing; trying curl fallback for provider %s.", provider
            )
            return _fetch_models_with_curl(endpoint, api_key, provider)
        return [], message
    except json.JSONDecodeError as exc:
        logging.error("Malformed JSON response from provider %s (%s): %s", provider, endpoint, exc)
        return [], "Invalid JSON response"

    return _parse_model_payload(payload, provider, endpoint)


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
            api_key = resolve_env_value(defaults["api_key_var"], env_map)
            if is_placeholder_value(api_key):
                continue
            api_base = normalize_api_base(provider_slug, resolve_env_value(defaults["api_base_var"], env_map))
            if not api_base:
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
        api_key = resolve_env_value(api_key_var, env_map)
        if is_placeholder_value(api_key):
            logging.warning("Skipping provider %s; missing API key in %s (.env first, env fallback).", provider_slug, api_key_var)
            continue
        api_base = normalize_api_base(provider_slug, resolve_env_value(api_base_var, env_map))
        if not api_base:
            logging.warning("Skipping provider %s; missing API base URL in %s (.env first, env fallback).", provider_slug, api_base_var)
            continue
        models, error = fetch_provider_models(provider_slug, api_key, api_base)
        catalog[provider_slug] = {
            "models": models,
            "api_base": api_base,
            "api_key_var": api_key_var,
            "api_base_var": api_base_var,
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
        request_interval_ms: int = 0,
    ) -> None:
        self.client_type: str
        self._chat_incompatible_models: set[str] = set()
        self._chat_unsupported_params: Dict[str, set[str]] = {}
        self._responses_unsupported_params: Dict[str, set[str]] = {}
        self._min_request_interval_seconds = max(0, request_interval_ms) / 1000.0
        self._last_request_started_at: Optional[float] = None
        try:
            from openai import OpenAI

            # Newer SDK (>= 1.0)
            kwargs: Dict[str, Any] = {"api_key": api_key}
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

                openai.api_key = api_key
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
        service_tier: Optional[str],
    ) -> CompletionResult:
        """Dispatch a chat completion request and return the message content."""
        # Top-k is not currently supported in OpenAI Chat API; we log and ignore.
        if top_k is not None:
            logging.debug("top_k is not supported by OpenAI Chat API; ignoring value %s.", top_k)
        model_key = model.strip().lower()

        def normalize_unsupported_parameter(name: Optional[str]) -> Optional[str]:
            if not isinstance(name, str):
                return None
            normalized = name.strip().lower().replace("-", "_")
            alias_map = {
                "servicetier": "service_tier",
                "service_tier": "service_tier",
                "topp": "top_p",
                "top_p": "top_p",
                "temperature": "temperature",
                "toplogprobs": "top_logprobs",
                "top_logprobs": "top_logprobs",
                "logprobs": "logprobs",
            }
            return alias_map.get(normalized, normalized or None)

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
                r"unsupported parameter:\s*['`\"]?([a-zA-Z0-9_]+)['`\"]?",
                r"unknown name\s*['`\"]([a-zA-Z0-9_]+)['`\"]\s*:\s*cannot find field",
                r"unrecognized request argument supplied:\s*['`\"]?([a-zA-Z0-9_]+)['`\"]?",
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
            if "top_logprobs" in text or "top logprobs" in text:
                if any(marker in text for marker in ("unsupported", "unknown", "unrecognized", "invalid")):
                    return "top_logprobs"
            if "logprobs" in text:
                if any(marker in text for marker in ("unsupported", "unknown", "unrecognized", "invalid", "not allowed")):
                    return "logprobs"
            return None

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

        def complete_with_responses_api() -> CompletionResult:
            request_args = {
                "model": model,
                "input": messages,
                "logprobs": True,
            }
            if temperature is not None:
                request_args["temperature"] = temperature
            if top_p is not None:
                request_args["top_p"] = top_p
            if service_tier and service_tier != "standard":
                request_args["service_tier"] = service_tier
            for unsupported in self._responses_unsupported_params.get(model_key, set()):
                request_args.pop(unsupported, None)

            while True:
                try:
                    self._throttle_request_if_needed()
                    response = self._client.responses.create(**request_args)
                    break
                except Exception as exc:  # noqa: BLE001
                    unsupported_param = extract_unsupported_parameter(exc) or infer_known_unsupported_parameter(exc)
                    if unsupported_param in {"model", "input", "messages"}:
                        raise
                    if unsupported_param and unsupported_param in request_args:
                        self._responses_unsupported_params.setdefault(model_key, set()).add(
                            unsupported_param
                        )
                        logging.info(
                            "Responses API rejected parameter '%s' for model %s; retrying without it.",
                            unsupported_param,
                            model,
                        )
                        request_args.pop(unsupported_param, None)
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

            return CompletionResult(
                text="".join(texts),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                token_logprobs=token_logprobs or None,
            )

        if self.client_type == "chat_v1":
            if hasattr(self._client, "responses") and model_key in self._chat_incompatible_models:
                return complete_with_responses_api()
            request_args = {
                "model": model,
                "messages": messages,
                "logprobs": True,
                "top_logprobs": 1,
            }
            if temperature is not None:
                request_args["temperature"] = temperature
            if top_p is not None:
                request_args["top_p"] = top_p
            if service_tier and service_tier != "standard":
                request_args["service_tier"] = service_tier
            for unsupported in self._chat_unsupported_params.get(model_key, set()):
                request_args.pop(unsupported, None)

            while True:
                try:
                    self._throttle_request_if_needed()
                    response = self._client.chat.completions.create(**request_args)
                    break
                except Exception as exc:  # noqa: BLE001
                    if hasattr(self._client, "responses") and should_retry_with_responses(exc):
                        self._chat_incompatible_models.add(model_key)
                        logging.info(
                            "Model %s is not chat-completions compatible; retrying with Responses API.",
                            model,
                        )
                        return complete_with_responses_api()
                    unsupported_param = extract_unsupported_parameter(exc) or infer_known_unsupported_parameter(exc)
                    if unsupported_param in {"model", "input", "messages"}:
                        raise
                    if unsupported_param and unsupported_param in request_args:
                        unsupported_params = self._chat_unsupported_params.setdefault(model_key, set())
                        unsupported_params.add(unsupported_param)
                        if unsupported_param == "logprobs":
                            unsupported_params.add("top_logprobs")
                        logging.info(
                            "Chat Completions rejected parameter '%s' for model %s; retrying without it.",
                            unsupported_param,
                            model,
                        )
                        request_args.pop(unsupported_param, None)
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
            return CompletionResult(
                text=message,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                token_logprobs=token_logprobs,
            )
        if self.client_type == "responses_v1":
            return complete_with_responses_api()
        # Legacy SDK path
        try:
            request_args = {
                "model": model,
                "messages": messages,
                "logprobs": True,
                "top_logprobs": 1,
            }
            if temperature is not None:
                request_args["temperature"] = temperature
            if top_p is not None:
                request_args["top_p"] = top_p
            if service_tier and service_tier != "standard":
                request_args["service_tier"] = service_tier
            self._throttle_request_if_needed()
            response = self._client.ChatCompletion.create(**request_args)
        except Exception as exc:  # noqa: BLE001
            warn_logprob_retry(exc)
            request_args.pop("logprobs", None)
            request_args.pop("top_logprobs", None)
            self._throttle_request_if_needed()
            response = self._client.ChatCompletion.create(**request_args)
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
        return CompletionResult(
            text=content or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            token_logprobs=token_logprobs,
        )


# --------------------------- Prompt Builder -------------------------------- #


def build_messages(
    example: Example,
    system_prompt: Optional[str],
    enable_cot: bool,
    include_explanation: bool,
    few_shot_context: Optional[List[Example]] = None,
) -> List[Dict[str, str]]:
    """Construct chat messages for the classification prompt."""
    if system_prompt:
        system_msg = system_prompt.strip()
    else:
        system_msg = (
            "You are a meticulous linguistic classifier. "
            "Classify the highlighted node word according to the task instructions."
        )
    system_msg = f"{system_msg.rstrip()}\n\n{MANDATORY_SYSTEM_APPEND}"

    user_instructions = [
        "You will receive a text excerpt with separate left/right context fields and a marked example where the node is wrapped as ⟦node⟧.",
        "When the node itself contains inner ⟦...⟧ spans, those marked passages are the classification target; the rest of the node and the contexts remain useful evidence only.",
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
        'If any inner ⟦...⟧ spans exist, set span_source to "marked_subspan" and set node_echo to exactly the marked text '
        "(join multiple marked spans with a single space, in order)."
    )
    user_instructions.append(
        "An additional field named 'info' may provide guidance or metadata relevant to the label; factor it into your decision."
    )
    user_instructions.append(
        "Contract: if node_echo or span_source fail to meet these requirements, the response will be rejected."
    )
    user_instructions.append("Do not include any text outside the JSON object.")

    user_content = "\n".join(user_instructions)

    if few_shot_context:
        samples = []
        for sample in few_shot_context:
            sample_target_text, sample_span_focus = resolve_span_contract(sample.node)
            samples.append(
                {
                    "left_context": sample.left_context,
                    "node": sample.node,
                    "right_context": sample.right_context,
                    "info": sample.info,
                    "label": sample.truth,
                    "marked_example": mark_node_in_context(
                        sample.left_context, sample.node, sample.right_context
                    ),
                    "classification_target": {
                        "focus": sample_span_focus,
                        "text": sample_target_text,
                        "note": (
                            "Classify only the marked sub-span; use the remaining text as context."
                            if sample_span_focus == SPAN_SOURCE_MARKED_SUBSPAN
                            else "Classify the entire node with support from the provided context."
                        ),
                    },
                }
            )
        user_content += (
            f"\n\nHere are {len(samples)} labeled example(s) you should mimic when classifying:\n"
            + json.dumps(samples, ensure_ascii=False, indent=2)
        )

    target_span_text, target_span_focus = resolve_span_contract(example.node)
    classification_note = (
        "Classify only the marked sub-span; use the rest of the node plus contexts as supporting evidence."
        if target_span_focus == SPAN_SOURCE_MARKED_SUBSPAN
        else "Classify the entire node; left/right contexts simply provide supporting evidence."
    )

    target_payload = {
        "left_context": example.left_context,
        "node": example.node,
        "right_context": example.right_context,
        "info": example.info,
        "marked_example": mark_node_in_context(
            example.left_context, example.node, example.right_context
        ),
        "classification_target": {
            "focus": target_span_focus,
            "text": target_span_text,
            "note": classification_note,
        },
    }

    user_content += "\n\nNow classify this example:\n"
    user_content += json.dumps(target_payload, ensure_ascii=False, indent=2)

    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_content}]
    return messages


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


class ProviderQuotaExceededError(RuntimeError):
    """Raised when provider quota/rate limit is exhausted after retries."""


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


def extract_label_logprob(
    response_text: str,
    label: str,
    token_logprobs: Optional[List[Dict[str, Any]]],
) -> Tuple[Optional[float], Optional[float]]:
    """Estimate the log probability and probability for the label token sequence."""
    if not token_logprobs or not label:
        return None, None

    target = f'"label": "{label}"'
    if response_text.find(target) == -1:
        # Fallback to matching the label string alone (including surrounding quotes).
        target = f'"{label}"'
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
        # Unable to map back accurately; fall back to tokens containing the label.
        matching_tokens = [
            entry for entry in token_logprobs if label in str(entry.get("token", ""))
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


# --------------------------- Main Benchmarking ----------------------------- #


def classify_example(
    connector: OpenAIConnector,
    example: Example,
    model: str,
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    service_tier: Optional[str],
    system_prompt: Optional[str],
    enable_cot: bool,
    include_explanation: bool,
    few_shot_context: Optional[List[Example]],
    max_retries: int,
    retry_delay: float,
    validator_client: Optional[ValidatorClient] = None,
    validator_prompt_max_candidates: int = 50,
    validator_prompt_max_chars: int = 8000,
    validator_exhausted_policy: str = "accept_blank_confidence",
) -> Tuple[Prediction, List[Dict[str, Any]]]:
    """Query the model and parse the prediction, returning attempt logs."""
    base_messages = build_messages(
        example,
        system_prompt,
        enable_cot,
        include_explanation,
        few_shot_context=few_shot_context,
    )
    validator_patch_message: Optional[Dict[str, str]] = None
    validator_status: Optional[str] = None
    validator_reason: Optional[str] = None
    last_error: Optional[Exception] = None
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
        }
        try:
            result = connector.complete(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                service_tier=service_tier,
            )
            raw = result.text
            log_entry["response"] = {
                "text": raw,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
                "token_logprobs": result.token_logprobs,
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
            ), interaction_logs
        except Exception as exc:  # noqa: BLE001 - surface API errors to user
            last_error = exc
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
            if attempt < max_retries:
                time.sleep(retry_delay)

    assert last_error is not None
    if is_quota_or_rate_limit_error(last_error):
        detail = str(last_error).strip() or last_error.__class__.__name__
        raise ProviderQuotaExceededError(
            f"Provider quota/rate limit exhausted for example {example.example_id}: {detail}"
        ) from last_error
    raise RuntimeError(f"Failed to classify example {example.example_id}") from last_error


def process_dataset(
    connector: OpenAIConnector,
    input_path: str,
    output_path: str,
    args: argparse.Namespace,
    include_explanation: bool,
    calibration_enabled: bool,
    label_map: Optional[Dict[str, str]],
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
                try:
                    prediction, attempt_logs = classify_example(
                        connector=connector,
                        example=example,
                        model=args.model,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        service_tier=args.service_tier,
                        system_prompt=args.system_prompt,
                        enable_cot=args.enable_cot,
                        include_explanation=include_explanation,
                        few_shot_context=few_shot_context,
                        max_retries=args.max_retries,
                        retry_delay=args.retry_delay,
                        validator_client=validator_client,
                        validator_prompt_max_candidates=args.validator_prompt_max_candidates,
                        validator_prompt_max_chars=args.validator_prompt_max_chars,
                        validator_exhausted_policy=args.validator_exhausted_policy,
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
                predictions[example.example_id] = prediction
                if prediction.prompt_tokens is not None:
                    total_prompt_tokens += prediction.prompt_tokens
                if prediction.completion_tokens is not None:
                    total_completion_tokens += prediction.completion_tokens
                if prediction.total_tokens is not None:
                    total_reported_tokens += prediction.total_tokens

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

    metrics: Dict[str, Any] = {}
    if evaluated_truths:
        metrics = compute_metrics(evaluated_truths, evaluated_preds)
        with open(metrics_output, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, ensure_ascii=False)
        logging.info("Saved metrics to %s", metrics_output)
        confusion = metrics.get("confusion_matrix")
        labels = metrics.get("labels", [])
        if confusion:
            heatmap_path = os.path.splitext(output_path)[0] + "_confusion_heatmap.png"
            generate_confusion_heatmap(confusion, labels, heatmap_path)
    else:
        logging.warning("No ground-truth labels available; skipping metric computation.")

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
            "<input>_out_<model>_<timestamp>.csv alongside each input file. "
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
        "--provider",
        default="openai",
        help=(
            "Model provider identifier used to look up default credentials. "
            "Known providers are preconfigured; custom providers are inferred from "
            "<PROVIDER>_API_KEY and <PROVIDER>_BASE_URL."
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
        "--calibration",
        action="store_true",
        help="Generate a calibration plot using the model's confidences.",
    )
    parser.add_argument(
        "--api_key_var",
        default=None,
        help="Environment variable name that stores the API key.",
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
            "Custom slugs are allowed; env vars are inferred as <SLUG>_API_KEY and <SLUG>_BASE_URL."
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

    overall_start = time.perf_counter()
    env_map = parse_env_file(".env")

    provider = (args.provider or "openai").lower()
    discovered_provider_defaults = discover_provider_defaults(env_map)
    provider_defaults = discovered_provider_defaults.get(provider) or infer_provider_defaults(provider)
    if args.api_key_var is None:
        args.api_key_var = provider_defaults["api_key_var"]
    if args.api_base_var is None:
        args.api_base_var = provider_defaults["api_base_var"]
    include_explanation = not args.no_explanation

    api_key = resolve_env_value(args.api_key_var, env_map)
    if is_placeholder_value(api_key):
        logging.error(
            "API key not found. Ensure %s is defined in .env or the environment.",
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

    connector = OpenAIConnector(
        api_key=api_key,
        base_url=api_base_url,
        request_interval_ms=max(0, args.request_interval_ms),
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
                validator_client=validator_client,
            )
            aggregate_prompt_tokens += prompt_tokens
            aggregate_completion_tokens += completion_tokens
            aggregate_reported_tokens += reported_tokens
            if halted_by_quota:
                stopped_by_quota = True
                logging.error(
                    "Stopping remaining input datasets because provider quota/rate limit was exhausted."
                )
                break
    finally:
        if validator_client is not None:
            validator_client.close()

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
            "Run ended early due to provider quota/rate-limit exhaustion. "
            "Re-run later with the same --output path to resume."
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())

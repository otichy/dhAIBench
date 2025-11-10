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
import sys
import time
import subprocess
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple


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


def load_env_file(path: str) -> None:
    """Load KEY=VALUE entries from a .env-style file into the environment."""
    if not path:
        return

    if not os.path.exists(path):
        logging.debug("Env file %s does not exist; skipping.", path)
        return

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
                os.environ.setdefault(key, value)


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


def ensure_directory(path: str) -> None:
    """Create the directory for path if it does not yet exist."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


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


def normalize_api_base(provider: str, api_base: Optional[str]) -> Optional[str]:
    """Ensure the API base ends with a version segment."""
    candidate = (api_base or PROVIDER_BASE_FALLBACKS.get(provider, "")).strip()
    if not candidate:
        return None
    trimmed = candidate.rstrip("/")
    if not re.search(r"/v\d+$", trimmed, re.IGNORECASE):
        trimmed = f"{trimmed}/v1"
    return trimmed


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
        logging.error("Connection error while fetching models for provider %s: %s", provider, exc)
        return [], str(exc)
    except json.JSONDecodeError as exc:
        logging.error("Malformed JSON response from provider %s (%s): %s", provider, endpoint, exc)
        return [], "Invalid JSON response"

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
    load_env_file(".env")
    selected = providers or sorted(PROVIDER_DEFAULTS.keys())
    catalog: Dict[str, Dict[str, Any]] = {}
    errors = 0

    for provider in selected:
        provider_slug = provider.lower()
        defaults = PROVIDER_DEFAULTS.get(provider_slug)
        if not defaults:
            logging.warning("Skipping unknown provider %s.", provider)
            continue
        api_key_var = defaults["api_key_var"]
        api_base_var = defaults["api_base_var"]
        api_key = os.environ.get(api_key_var)
        if not api_key:
            logging.warning("Skipping provider %s; missing API key in %s.", provider_slug, api_key_var)
            continue
        api_base = normalize_api_base(provider_slug, os.environ.get(api_base_var))
        if not api_base:
            logging.warning("Skipping provider %s; missing API base URL in %s.", provider_slug, api_base_var)
            continue
        models, error = fetch_provider_models(provider_slug, api_key, api_base)
        catalog[provider_slug] = {
            "models": models,
            "api_base": api_base,
            "api_key_var": api_key_var,
            "api_base_var": api_base_var,
            "error": error,
            "timestamp": datetime.utcnow().isoformat() + "Z",
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

    def __init__(self, api_key: str, base_url: Optional[str] = None) -> None:
        self.client_type: str
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

    def complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        top_p: float,
        top_k: Optional[int],
        service_tier: Optional[str],
    ) -> CompletionResult:
        """Dispatch a chat completion request and return the message content."""
        # Top-k is not currently supported in OpenAI Chat API; we log and ignore.
        if top_k is not None:
            logging.debug("top_k is not supported by OpenAI Chat API; ignoring value %s.", top_k)

        def warn_logprob_retry(exc: Exception) -> None:
            status_code = getattr(exc, "status_code", None) or getattr(exc, "http_status", None) or getattr(exc, "status", None)
            error_message = getattr(exc, "message", None)
            if error_message is None and hasattr(exc, "error"):
                error_message = getattr(exc.error, "message", None)  # type: ignore[attr-defined]
            text = str(error_message or exc)
            if isinstance(status_code, str) and status_code.isdigit():
                status_code = int(status_code)
            if status_code == 403 or "403" in text:
                logging.warning(
                    "The API rejected the logprobs request with HTTP 403. This model or service tier likely does not support token log probabilities; retrying without logprobs. Details: %s",
                    text,
                )
            else:
                logging.debug("Logprobs unavailable for this client (%s); retrying without logprobs.", exc)

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

        if self.client_type == "chat_v1":
            try:
                request_args: Dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": top_p,
                    "logprobs": True,
                    "top_logprobs": 1,
                }
                if service_tier and service_tier != "standard":
                    request_args["service_tier"] = service_tier
                response = self._client.chat.completions.create(**request_args)
            except Exception as exc:  # noqa: BLE001
                warn_logprob_retry(exc)
                request_args.pop("logprobs", None)
                request_args.pop("top_logprobs", None)
                response = self._client.chat.completions.create(**request_args)
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
            try:
                request_args = {
                    "model": model,
                    "input": messages,
                    "temperature": temperature,
                    "top_p": top_p,
                    "logprobs": True,
                }
                if service_tier and service_tier != "standard":
                    request_args["service_tier"] = service_tier
                response = self._client.responses.create(**request_args)
            except Exception as exc:  # noqa: BLE001
                warn_logprob_retry(exc)
                request_args.pop("logprobs", None)
                response = self._client.responses.create(**request_args)
            usage = getattr(response, "usage", None)
            # Responses API returns a list of content blocks
            for item in response.output:
                for segment in getattr(item, "content", []):
                    if getattr(segment, "type", "") == "output_text":
                        text = segment.text or ""
                        segment_logprobs = collect_logprobs(getattr(segment, "logprobs", None))
                        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
                        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
                        total_tokens = getattr(usage, "total_tokens", None) if usage else None
                        return CompletionResult(
                            text=text,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=total_tokens,
                            token_logprobs=segment_logprobs,
                        )
            return CompletionResult(text="")
        # Legacy SDK path
        try:
            request_args = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "logprobs": True,
                "top_logprobs": 1,
            }
            if service_tier and service_tier != "standard":
                request_args["service_tier"] = service_tier
            response = self._client.ChatCompletion.create(**request_args)
        except Exception as exc:  # noqa: BLE001
            warn_logprob_retry(exc)
            request_args.pop("logprobs", None)
            request_args.pop("top_logprobs", None)
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


def ensure_calibration_dependencies() -> bool:
    """Verify plotting dependencies are installed, installing if user agrees."""
    required_packages = ["matplotlib"]
    missing: List[str] = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if not missing:
        return True

    message = (
        f"The following packages are required for calibration plots but missing: {', '.join(missing)}.\n"
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
            return False

    logging.info("Successfully installed calibration plot dependencies.")
    return True


def generate_calibration_plot(
    confidences: List[float],
    correctness: List[bool],
    output_path: str,
    bin_count: int = 10,
) -> None:
    """Generate a reliability diagram showing calibration performance."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib not installed; skipping calibration plot.")
        return

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
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib not installed; skipping confusion heatmap.")
        return

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
    temperature: float,
    top_p: float,
    top_k: Optional[int],
    service_tier: Optional[str],
    system_prompt: Optional[str],
    enable_cot: bool,
    include_explanation: bool,
    few_shot_context: Optional[List[Example]],
    max_retries: int,
    retry_delay: float,
) -> Tuple[Prediction, List[Dict[str, Any]]]:
    """Query the model and parse the prediction, returning attempt logs."""
    messages = build_messages(
        example,
        system_prompt,
        enable_cot,
        include_explanation,
        few_shot_context=few_shot_context,
    )
    prompt_snapshot = json.dumps(messages, ensure_ascii=False, indent=2)
    logging.debug("Prompt for example %s:\n%s", example.example_id, prompt_snapshot)
    last_error: Optional[Exception] = None
    validation_failures = 0
    interaction_logs: List[Dict[str, Any]] = []

    for attempt in range(1, max_retries + 1):
        log_entry: Dict[str, Any] = {
            "attempt": attempt,
            "timestamp": datetime.utcnow().isoformat() + "Z",
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
    raise RuntimeError(f"Failed to classify example {example.example_id}") from last_error


def process_dataset(
    connector: OpenAIConnector,
    input_path: str,
    output_path: str,
    args: argparse.Namespace,
    include_explanation: bool,
    calibration_enabled: bool,
    label_map: Optional[Dict[str, str]],
) -> Tuple[int, int, int]:
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
    confidences: List[float] = []
    correctness: List[bool] = []
    few_shot_count = max(0, args.few_shot_examples)
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_reported_tokens = 0

    ensure_directory(output_path)
    log_path = os.path.splitext(output_path)[0] + ".log"
    ensure_directory(log_path)
    logging.info("Writing predictions to %s", output_path)
    logging.info("Saving prompt/response log to %s", log_path)

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
        ]
    )

    log_records: List[Dict[str, Any]] = []

    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()

        for idx, example in enumerate(examples, start=1):
            logging.info("Classifying example %s (%d/%d)", example.example_id, idx, len(examples))
            few_shot_context = select_few_shot_examples(examples, example.example_id, few_shot_count)
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
            )
            predictions[example.example_id] = prediction
            if prediction.prompt_tokens is not None:
                total_prompt_tokens += prediction.prompt_tokens
            if prediction.completion_tokens is not None:
                total_completion_tokens += prediction.completion_tokens
            if prediction.total_tokens is not None:
                total_reported_tokens += prediction.total_tokens
            if example.truth:
                is_correct = prediction.label == example.truth
                if prediction.confidence is not None:
                    correctness.append(is_correct)
                    confidences.append(prediction.confidence)

            log_records.append(
                {
                    "example_id": example.example_id,
                    "attempts": attempt_logs,
                    "final_prediction": {
                        "label": prediction.label,
                        "confidence": prediction.confidence,
                        "explanation": prediction.explanation,
                        "truth": example.truth,
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
            }
            for field in extra_field_order:
                row[field] = example.extras.get(field, "")
            writer.writerow(row)
            handle.flush()

    with open(log_path, "w", encoding="utf-8") as log_handle:
        json.dump(log_records, log_handle, ensure_ascii=False, indent=2)
    logging.info("Saved prompt log to %s", log_path)

    metrics_output = os.path.splitext(output_path)[0] + "_metrics.json"

    evaluated_truths = [ex.truth for ex in examples if ex.truth is not None]
    evaluated_preds = [
        predictions[ex.example_id].label for ex in examples if ex.truth is not None
    ]

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

    return total_prompt_tokens, total_completion_tokens, total_reported_tokens


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
            "<input>_out_<model>_<timestamp>.csv alongside each input file."
        ),
    )
    parser.add_argument("--model", help="Model name (e.g., gpt-4-turbo).")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling parameter.")
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
        choices=sorted(PROVIDER_DEFAULTS.keys()),
        default="openai",
        help="Model provider identifier used to look up default credentials.",
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
        help="Optional list of provider slugs to update when --update-models is specified.",
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

    provider = (args.provider or "openai").lower()
    provider_defaults = PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS["openai"])
    if args.api_key_var is None:
        args.api_key_var = provider_defaults["api_key_var"]
    if args.api_base_var is None:
        args.api_base_var = provider_defaults["api_base_var"]
    include_explanation = not args.no_explanation

    load_env_file(".env")
    api_key = os.environ.get(args.api_key_var)
    if not api_key:
        logging.error(
            "API key not found. Ensure %s is defined in the environment or .env.",
            args.api_key_var,
        )
        return 1

    api_base_url = os.environ.get(args.api_base_var) if args.api_base_var else None
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

    connector = OpenAIConnector(api_key=api_key, base_url=api_base_url)

    aggregate_prompt_tokens = 0
    aggregate_completion_tokens = 0
    aggregate_reported_tokens = 0

    for resolved_input in input_paths:
        output_path = resolve_output_path(
            resolved_input,
            args.model,
            args.output,
            timestamp_tag,
            multiple_inputs,
        )
        prompt_tokens, completion_tokens, reported_tokens = process_dataset(
            connector=connector,
            input_path=resolved_input,
            output_path=output_path,
            args=args,
            include_explanation=include_explanation,
            calibration_enabled=calibration_enabled,
            label_map=label_map,
        )
        aggregate_prompt_tokens += prompt_tokens
        aggregate_completion_tokens += completion_tokens
        aggregate_reported_tokens += reported_tokens

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

    return 0


if __name__ == "__main__":
    sys.exit(main())

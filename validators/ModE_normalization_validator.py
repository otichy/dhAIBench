#!/usr/bin/env python3
"""NDJSON validator for Modern English text normalization."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


FINAL_E_COST = 0.2
DOUBLED_L_COST = 0.25
SPELLING_VARIANT_COST = 0.35
DEFAULT_MAX_DISTANCE = 1.25

SUFFIX_VARIANTS: Tuple[Tuple[str, str], ...] = (
    ("or", "our"),
    ("er", "re"),
    ("ize", "ise"),
    ("ization", "isation"),
    ("yze", "yse"),
    ("og", "ogue"),
    ("ense", "ence"),
)
SEGMENT_VARIANTS: Tuple[Tuple[str, str], ...] = (
    ("ae", "e"),
    ("oe", "e"),
)


def configure_stdio() -> None:
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr, flush=True)


def write_message(message: dict) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def normalize_text(text: str) -> str:
    return (text or "").strip().lower()


def resolve_attempt_index(message: dict) -> int:
    attempt = message.get("attempt") or {}
    try:
        attempt_index = int(attempt.get("index", 1))
    except (TypeError, ValueError):
        return 1
    return max(1, attempt_index)


def resolve_effective_max_distance(
    message: dict,
    max_distance: float,
    max_distance_per_retry: float,
) -> float:
    if float(max_distance) <= 0.0:
        return 0.0
    attempt_index = resolve_attempt_index(message)
    retry_count = max(0, attempt_index - 2)
    return max(0.0, float(max_distance) + max(0.0, float(max_distance_per_retry)) * retry_count)


def iter_segments(value: str, index: int, lengths: Sequence[int]) -> Iterable[Tuple[str, int]]:
    for length in lengths:
        end = index + length
        if end <= len(value):
            yield value[index:end], length


def is_pair(left: str, right: str, pairs: Sequence[Tuple[str, str]]) -> bool:
    return any((left == a and right == b) or (left == b and right == a) for a, b in pairs)


def suffix_variant_cost(left: str, right: str, i: int, j: int) -> Optional[Tuple[float, int, int]]:
    for a, b in SUFFIX_VARIANTS:
        if left[i:] == a and right[j:] == b:
            return SPELLING_VARIANT_COST, len(a), len(b)
        if left[i:] == b and right[j:] == a:
            return SPELLING_VARIANT_COST, len(b), len(a)
    return None


def segment_substitution_cost(left: str, right: str) -> Optional[float]:
    if left == right:
        return 0.0
    if is_pair(left, right, SEGMENT_VARIANTS):
        return SPELLING_VARIANT_COST
    return None


@dataclass(frozen=True)
class Suggestion:
    distance: float
    headword: str


def load_lexicon(path: str) -> Dict[str, str]:
    headwords: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        has_header = "lemma" in sample.splitlines()[0].lower() or "headword" in sample.splitlines()[0].lower()
        if has_header:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return headwords
            field_lookup = {field.strip().lower(): field for field in reader.fieldnames if field}
            field_name = field_lookup.get("headword") or field_lookup.get("lemma") or reader.fieldnames[0]
            for row in reader:
                headword = normalize_text(row.get(field_name, ""))
                if headword:
                    headwords.setdefault(headword, headword)
            return headwords

        for raw_line in handle:
            headword = normalize_text(raw_line)
            if not headword:
                continue
            headwords.setdefault(headword, headword)
    return headwords


def weighted_distance(left: str, right: str) -> float:
    left = normalize_text(left)
    right = normalize_text(right)

    if left == right:
        return 0.0

    @lru_cache(maxsize=None)
    def dp(i: int, j: int) -> float:
        if i == len(left) and j == len(right):
            return 0.0
        if i == len(left):
            if right[j:] == "e":
                return FINAL_E_COST
            return float(len(right) - j)
        if j == len(right):
            if left[i:] == "e":
                return FINAL_E_COST
            return float(len(left) - i)

        best = math.inf

        suffix_cost = suffix_variant_cost(left, right, i, j)
        if suffix_cost is not None:
            cost, left_len, right_len = suffix_cost
            best = min(best, cost + dp(i + left_len, j + right_len))

        sub_cost = segment_substitution_cost(left[i], right[j])
        if sub_cost is not None:
            best = min(best, sub_cost + dp(i + 1, j + 1))

        for left_segment, left_len in iter_segments(left, i, (1, 2)):
            for right_segment, right_len in iter_segments(right, j, (1, 2)):
                if left_len == right_len == 1:
                    continue
                multi_cost = segment_substitution_cost(left_segment, right_segment)
                if multi_cost is not None:
                    best = min(best, multi_cost + dp(i + left_len, j + right_len))

        delete_cost = 1.0
        if i == len(left) - 1 and left[i] == "e":
            delete_cost = FINAL_E_COST
        best = min(best, delete_cost + dp(i + 1, j))

        insert_cost = 1.0
        if j == len(right) - 1 and right[j] == "e":
            insert_cost = FINAL_E_COST
        best = min(best, insert_cost + dp(i, j + 1))

        if i + 1 < len(left) and left[i : i + 2] == "ll" and right[j] == "l":
            best = min(best, DOUBLED_L_COST + dp(i + 1, j))
        if j + 1 < len(right) and right[j : j + 2] == "ll" and left[i] == "l":
            best = min(best, DOUBLED_L_COST + dp(i, j + 1))

        return best

    return dp(0, 0)


def collect_candidates(
    prediction: str,
    candidates: Dict[str, str],
    max_distance: float,
    max_suggestions: int,
) -> List[Suggestion]:
    matches: List[Suggestion] = []
    for headword_norm, headword in candidates.items():
        distance = weighted_distance(prediction, headword_norm)
        if max_distance <= 0.0 or distance <= max_distance:
            matches.append(Suggestion(distance=distance, headword=headword))
    matches.sort(key=lambda item: (item.distance, item.headword))
    return matches[:max_suggestions]


def build_retry_message(predicted_label: str, suggestions: Sequence[Suggestion]) -> str:
    parts = [f'The previous normalized form "{predicted_label}" was rejected by the Modern English validator.']
    if suggestions:
        parts.append(
            "Choose the best normalized OED headword from allowed_labels. Prefer the lexicon form when the "
            "difference is only a supported Modern English spelling variant."
        )
    else:
        parts.append("No close OED headword matches were found within the configured weighted edit-distance threshold.")
    return "\n".join(parts)


def handle_validate(
    message: dict,
    lexicon: Dict[str, str],
    max_distance: float,
    max_suggestions: int,
    max_distance_per_retry: float = 0.0,
) -> dict:
    request_id = str(message.get("request_id", "")).strip()
    prediction = message.get("prediction") or {}
    label = normalize_text(str(prediction.get("label", "") or ""))

    if not label:
        return {
            "type": "result",
            "schema_version": 1,
            "request_id": request_id,
            "action": "retry",
            "reason": "empty_label",
            "retry": {
                "allowed_labels": [],
                "instruction": "Return a non-empty Modern English normalized form.",
            },
        }

    effective_max_distance = resolve_effective_max_distance(
        message=message,
        max_distance=max_distance,
        max_distance_per_retry=max_distance_per_retry,
    )

    if label in lexicon:
        return {
            "type": "result",
            "schema_version": 1,
            "request_id": request_id,
            "action": "accept",
            "reason": "in_lexicon",
            "normalized": {"label": lexicon[label]},
        }

    suggestions = collect_candidates(
        prediction=label,
        candidates=lexicon,
        max_distance=effective_max_distance,
        max_suggestions=max_suggestions,
    )
    if suggestions and suggestions[0].distance == 0.0:
        return {
            "type": "result",
            "schema_version": 1,
            "request_id": request_id,
            "action": "accept",
            "reason": "normalized_spelling_variant",
            "normalized": {"label": suggestions[0].headword},
        }

    return {
        "type": "result",
        "schema_version": 1,
        "request_id": request_id,
        "action": "retry",
        "reason": "not_in_lexicon",
        "retry": {
            "allowed_labels": [item.headword for item in suggestions],
            "message": build_retry_message(label, suggestions),
            "instruction": "Choose the correct normalized form from allowed_labels.",
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    configure_stdio()
    parser = argparse.ArgumentParser(description="NDJSON validator for Modern English normalization.")
    parser.add_argument(
        "--lexicon",
        default="validators/oed_headwords.csv",
        help="Path to the Modern English OED headword lexicon.",
    )
    parser.add_argument(
        "--max_distance",
        type=float,
        default=DEFAULT_MAX_DISTANCE,
        help=(
            "Maximum weighted edit distance for suggestions. Set to 0 to disable the distance threshold; "
            "returned candidates are then limited by the lexicon and --max_suggestions."
        ),
    )
    parser.add_argument(
        "--max_distance_per_retry",
        type=float,
        default=0.0,
        help="Increase max_distance by this amount starting with the second retry (the third overall attempt).",
    )
    parser.add_argument(
        "--max_suggestions",
        type=int,
        default=30,
        help=(
            "Maximum number of candidate headwords returned by this validator in retry.allowed_labels. "
            "This limits what the validator sends back; the benchmark's --validator_prompt_max_candidates "
            "can still render fewer of those in the actual retry prompt."
        ),
    )
    args = parser.parse_args(argv)

    eprint(f"Loading Modern English lexicon from {args.lexicon} ...")
    lexicon = load_lexicon(args.lexicon)
    eprint(f"Loaded {len(lexicon)} unique OED headword(s).")

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except Exception as exc:  # noqa: BLE001
            eprint(f"Non-JSON input: {exc}: {line!r}")
            continue

        msg_type = str(message.get("type", "")).strip()
        schema_version = message.get("schema_version")
        if schema_version != 1:
            eprint(f"Unsupported schema_version={schema_version!r}; ignoring.")
            continue

        if msg_type == "init":
            write_message(
                {
                    "type": "init_ok",
                    "schema_version": 1,
                    "capabilities": {"supports_allowed_labels": True, "supports_normalize": True},
                }
            )
            continue

        if msg_type == "validate":
            request_id = str(message.get("request_id", "")).strip()
            if not request_id:
                eprint("validate message missing request_id; ignoring.")
                continue
            try:
                result = handle_validate(
                    message=message,
                    lexicon=lexicon,
                    max_distance=max(0.0, float(args.max_distance)),
                    max_suggestions=max(0, int(args.max_suggestions)),
                    max_distance_per_retry=max(0.0, float(args.max_distance_per_retry)),
                )
            except Exception as exc:  # noqa: BLE001
                eprint(f"Error handling request_id={request_id}: {exc}")
                result = {
                    "type": "result",
                    "schema_version": 1,
                    "request_id": request_id,
                    "action": "abort",
                    "reason": "validator_exception",
                }
            write_message(result)
            continue

        eprint(f"Unknown message type={msg_type!r}; ignoring.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

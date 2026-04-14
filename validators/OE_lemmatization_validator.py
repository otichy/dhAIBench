#!/usr/bin/env python3
"""NDJSON validator for Old English lemmatization on YCOE-style data."""

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
DOUBLED_LETTER_COST = 0.25
GE_PREFIX_COST = 0.3
VARIANT_COST = 0.35
VOWEL_SUBSTITUTION_COST = 1.0
X_CS_SUBSTITUTION_COST = 0.35
DENTAL_VARIANT_COST = 0.5
DEFAULT_MAX_DISTANCE = 1.25


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


def normalize_pos(value: str) -> str:
    text = (value or "").strip()
    for separator in ("^", "$"):
        if separator in text:
            text = text.split(separator, 1)[0]
    return text.strip()


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


def strip_ne_prefix(value: str) -> str:
    if value.startswith("ne_"):
        return value[3:]
    return value


def thorn_eth_fold(value: str) -> str:
    return value.replace("þ", "ð")


def is_ne_prefix_only_difference(a: str, b: str) -> bool:
    a_core = strip_ne_prefix(a)
    b_core = strip_ne_prefix(b)
    if a_core == a and b_core == b:
        return False
    return thorn_eth_fold(a_core) == thorn_eth_fold(b_core)


def matches_missing_ne_prefixed_lexicon_form(label: str, candidates: Dict[str, str]) -> bool:
    if not label.startswith("ne_"):
        return False
    return any(is_ne_prefix_only_difference(label, candidate) for candidate in candidates)


def is_ge_prefix_only_difference(a: str, b: str) -> bool:
    if a.startswith("ge") and a[2:] == b:
        return True
    if b.startswith("ge") and b[2:] == a:
        return True
    return False


def iter_segments(value: str, index: int, lengths: Sequence[int]) -> Iterable[Tuple[str, int]]:
    for length in lengths:
        end = index + length
        if end <= len(value):
            yield value[index:end], length


def build_variant_pairs() -> set[Tuple[str, str]]:
    groups = [
        ("i", "y", "e"),
        ("a", "o", "u"),
        ("i", "y", "e", "eo", "io", "ie"),
        ("æ", "ae", "ea"),
        ("ð", "þ", "th"),
    ]
    pairs: set[Tuple[str, str]] = set()
    for group in groups:
        for left in group:
            for right in group:
                if left != right:
                    pairs.add((left, right))
    return pairs


def build_dental_pairs() -> set[Tuple[str, str]]:
    segments = ("d", "t", "ð", "þ", "th")
    pairs: set[Tuple[str, str]] = set()
    for left in segments:
        for right in segments:
            if left != right:
                pairs.add((left, right))
    return pairs


LOW_COST_VARIANT_PAIRS = build_variant_pairs()
DENTAL_VARIANT_PAIRS = build_dental_pairs()
VOWEL_SEGMENTS = {"a", "e", "i", "o", "u", "y", "æ", "ae", "ea", "eo", "io", "ie"}


def segment_substitution_cost(left: str, right: str) -> Optional[float]:
    if left == right:
        return 0.0
    if {left, right} == {"ð", "þ"}:
        return 0.0
    if {left, right} == {"x", "cs"}:
        return X_CS_SUBSTITUTION_COST
    if (left, right) in LOW_COST_VARIANT_PAIRS:
        return VARIANT_COST
    if (left, right) in DENTAL_VARIANT_PAIRS:
        return DENTAL_VARIANT_COST
    if left in VOWEL_SEGMENTS and right in VOWEL_SEGMENTS:
        return VOWEL_SUBSTITUTION_COST
    return None


@dataclass(frozen=True)
class Suggestion:
    distance: float
    lemma: str


@dataclass
class Lexicon:
    all_lemmas: Dict[str, str]
    lemmas_by_pos: Dict[str, Dict[str, str]]


def load_lexicon(path: str) -> Lexicon:
    all_lemmas: Dict[str, str] = {}
    lemmas_by_pos: Dict[str, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        if not reader.fieldnames or "lemma" not in reader.fieldnames or "pos" not in reader.fieldnames:
            raise ValueError("Lexicon must contain 'lemma' and 'pos' columns.")
        for row in reader:
            lemma = normalize_text(row.get("lemma", ""))
            pos = normalize_pos(row.get("pos", ""))
            if not lemma:
                continue
            all_lemmas.setdefault(lemma, lemma)
            if pos:
                lemmas_by_pos.setdefault(pos, {}).setdefault(lemma, lemma)
    return Lexicon(all_lemmas=all_lemmas, lemmas_by_pos=lemmas_by_pos)


def weighted_distance(left: str, right: str) -> float:
    left = normalize_text(left)
    right = normalize_text(right)

    if left == right:
        return 0.0
    if is_ne_prefix_only_difference(left, right):
        return 0.0
    if is_ge_prefix_only_difference(left, right):
        return GE_PREFIX_COST

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

        if i + 1 < len(left) and left[i] == left[i + 1] and left[i] == right[j]:
            best = min(best, DOUBLED_LETTER_COST + dp(i + 1, j))
        if j + 1 < len(right) and right[j] == right[j + 1] and left[i] == right[j]:
            best = min(best, DOUBLED_LETTER_COST + dp(i, j + 1))

        return best

    return dp(0, 0)


def collect_candidates(
    prediction: str,
    candidates: Dict[str, str],
    max_distance: float,
    max_suggestions: int,
) -> List[Suggestion]:
    matches: List[Suggestion] = []
    for lemma_norm, lemma in candidates.items():
        distance = weighted_distance(prediction, lemma_norm)
        if max_distance <= 0.0 or distance <= max_distance:
            matches.append(Suggestion(distance=distance, lemma=lemma))
    matches.sort(key=lambda item: (item.distance, item.lemma))
    return matches[:max_suggestions]


def parse_example_pos(example: dict) -> str:
    return normalize_pos(str((example or {}).get("info", "") or ""))


def build_retry_message(predicted_label: str, pos: str, suggestions: Sequence[Suggestion]) -> str:
    parts = [f'The previous lemma "{predicted_label}" was rejected by the Old English validator.']
    if pos:
        parts.append(f'Expected POS bucket: "{pos}".')
    if suggestions:
        parts.append(
            "Choose the best normalized lexicon lemma from allowed_labels. Prefer the lexicon form when the "
            "difference is only an Old English spelling variant."
        )
    else:
        parts.append("No close lexicon matches were found within the configured weighted edit-distance threshold.")
    return "\n".join(parts)


def handle_validate(
    message: dict,
    lexicon: Lexicon,
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
                "instruction": "Return a non-empty Old English lemma.",
            },
        }

    example = message.get("example") or {}
    pos = parse_example_pos(example)
    if pos == "FW":
        return {
            "type": "result",
            "schema_version": 1,
            "request_id": request_id,
            "action": "accept",
            "reason": "foreign_word_pos",
            "normalized": {"label": "foreign_word"},
        }

    pos_candidates = lexicon.lemmas_by_pos.get(pos)
    candidate_pool = pos_candidates if pos_candidates else lexicon.all_lemmas
    effective_max_distance = resolve_effective_max_distance(
        message=message,
        max_distance=max_distance,
        max_distance_per_retry=max_distance_per_retry,
    )

    if label in candidate_pool:
        return {
            "type": "result",
            "schema_version": 1,
            "request_id": request_id,
            "action": "accept",
            "reason": "in_lexicon_pos" if pos_candidates else "in_lexicon",
            "normalized": {"label": candidate_pool[label]},
        }

    if matches_missing_ne_prefixed_lexicon_form(label, candidate_pool):
        return {
            "type": "result",
            "schema_version": 1,
            "request_id": request_id,
            "action": "accept",
            "reason": "accepted_negated_form",
            "normalized": {"label": label},
        }

    suggestions = collect_candidates(
        prediction=label,
        candidates=candidate_pool,
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
            "normalized": {"label": suggestions[0].lemma},
        }

    return {
        "type": "result",
        "schema_version": 1,
        "request_id": request_id,
        "action": "retry",
        "reason": "not_in_lexicon_pos" if pos_candidates else "not_in_lexicon",
        "retry": {
            "allowed_labels": [item.lemma for item in suggestions],
            "message": build_retry_message(label, pos, suggestions),
            "instruction": "Choose the correct lemma from allowed_labels.",
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    configure_stdio()
    parser = argparse.ArgumentParser(description="NDJSON validator for Old English lemmatization.")
    parser.add_argument(
        "--lexicon",
        default="validators/OE_lemmatization_validator_lexicon.csv",
        help="Path to the Old English lemma lexicon CSV.",
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
            "Maximum number of candidate lemmas returned by this validator in retry.allowed_labels. "
            "This limits what the validator sends back; the benchmark's --validator_prompt_max_candidates "
            "can still render fewer of those in the actual retry prompt."
        ),
    )
    args = parser.parse_args(argv)

    eprint(f"Loading Old English lexicon from {args.lexicon} ...")
    lexicon = load_lexicon(args.lexicon)
    eprint(
        "Loaded "
        f"{len(lexicon.all_lemmas)} unique lemma(s) across {len(lexicon.lemmas_by_pos)} POS bucket(s)."
    )

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

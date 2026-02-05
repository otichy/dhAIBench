#!/usr/bin/env python3
"""Reference NDJSON validator for lemmatization tasks.

Reads protocol messages on stdin (NDJSON) and writes protocol messages to stdout.
Any logs MUST go to stderr.

Lexicon format: one lemma per line (UTF-8).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr, flush=True)


def normalize_text(text: str, lowercase: bool, strip_punct: bool) -> str:
    value = text.strip()
    if lowercase:
        value = value.lower()
    if strip_punct:
        value = re.sub(r"[\W_]+", "", value, flags=re.UNICODE)
    return value


def parse_pos_from_info(info: str, regex: re.Pattern[str], whitelist: Sequence[str]) -> Optional[str]:
    text = (info or "").strip()
    if not text:
        return None
    match = regex.search(text)
    if match:
        value = (match.group(1) or "").strip()
        return value or None
    if whitelist:
        token = text.strip()
        if token in set(whitelist):
            return token
    return None


def split_lexicon_fields(line: str, sep: str) -> List[str]:
    raw = line.strip()
    if not raw:
        return []
    if sep == "tab":
        return [part.strip() for part in raw.split("\t")]
    if sep == "semicolon":
        return [part.strip() for part in raw.split(";")]
    if sep == "space":
        return [part.strip() for part in raw.split()]
    # auto
    if "\t" in raw:
        return [part.strip() for part in raw.split("\t")]
    if ";" in raw:
        return [part.strip() for part in raw.split(";")]
    return [part.strip() for part in raw.split()]


def levenshtein_bounded(a: str, b: str, max_distance: int) -> int:
    """Compute Levenshtein distance with early exit when > max_distance."""
    if a == b:
        return 0
    if max_distance < 0:
        return max_distance + 1

    len_a = len(a)
    len_b = len(b)
    if abs(len_a - len_b) > max_distance:
        return max_distance + 1

    if len_a == 0:
        return len_b if len_b <= max_distance else max_distance + 1
    if len_b == 0:
        return len_a if len_a <= max_distance else max_distance + 1

    if len_a > len_b:
        a, b = b, a
        len_a, len_b = len_b, len_a

    previous = list(range(len_a + 1))
    for i in range(1, len_b + 1):
        current = [i] + [0] * len_a
        b_char = b[i - 1]
        row_min = current[0]
        for j in range(1, len_a + 1):
            cost = 0 if a[j - 1] == b_char else 1
            current[j] = min(
                previous[j] + 1,  # deletion
                current[j - 1] + 1,  # insertion
                previous[j - 1] + cost,  # substitution
            )
            if current[j] < row_min:
                row_min = current[j]
        if row_min > max_distance:
            return max_distance + 1
        previous = current
    distance = previous[len_a]
    return distance if distance <= max_distance else max_distance + 1


@dataclass
class BKNode:
    term: str
    children: Dict[int, "BKNode"] = field(default_factory=dict)


class BKTree:
    def __init__(self) -> None:
        self._root: Optional[BKNode] = None

    def add(self, term: str, max_distance_hint: int = 2) -> None:
        if self._root is None:
            self._root = BKNode(term)
            return
        node = self._root
        while True:
            dist = levenshtein_bounded(term, node.term, max_distance_hint + 8)
            child = node.children.get(dist)
            if child is None:
                node.children[dist] = BKNode(term)
                return
            node = child

    def search(self, query: str, max_distance: int) -> List[Tuple[int, str]]:
        if self._root is None:
            return []
        results: List[Tuple[int, str]] = []
        to_visit: List[BKNode] = [self._root]
        while to_visit:
            node = to_visit.pop()
            dist = levenshtein_bounded(query, node.term, max_distance)
            if dist <= max_distance:
                results.append((dist, node.term))
            lower = dist - max_distance
            upper = dist + max_distance
            for edge, child in node.children.items():
                if lower <= edge <= upper:
                    to_visit.append(child)
        return results


def load_lexicon(
    path: str,
    lowercase: bool,
    strip_punct: bool,
    field_sep: str,
) -> Tuple[set[str], Dict[str, str], Dict[str, set[str]]]:
    """Return normalized set, mapping normalized->original, and optional normalized->POS sets.

    Lexicon line formats supported (depending on --lexicon_field_sep):
    - lemma
    - lemma<TAB>POS
    - lemma;POS
    - lemma POS
    Additional columns are ignored.
    """
    normalized_set: set[str] = set()
    canonical: Dict[str, str] = {}
    by_pos: Dict[str, set[str]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            fields = split_lexicon_fields(line, field_sep)
            if not fields:
                continue
            lemma = fields[0]
            pos = fields[1] if len(fields) > 1 else ""
            norm = normalize_text(lemma, lowercase=lowercase, strip_punct=strip_punct)
            if not norm:
                continue
            normalized_set.add(norm)
            canonical.setdefault(norm, lemma.strip())
            pos_norm = pos.strip()
            if pos_norm:
                by_pos.setdefault(pos_norm, set()).add(norm)
    return normalized_set, canonical, by_pos


def write_message(message: dict) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def handle_validate(
    message: dict,
    lexicon_norm: set[str],
    canonical: Dict[str, str],
    by_pos: Dict[str, set[str]],
    tree: BKTree,
    lowercase: bool,
    strip_punct: bool,
    use_pos: bool,
    pos_regex: re.Pattern[str],
    pos_whitelist: Sequence[str],
    pos_fallback: bool,
    pos_unknown_policy: str,
    max_distance: int,
    max_suggestions: int,
) -> dict:
    request_id = str(message.get("request_id", "")).strip()
    prediction = message.get("prediction") or {}
    label = str(prediction.get("label", "")).strip()
    norm = normalize_text(label, lowercase=lowercase, strip_punct=strip_punct)

    example = message.get("example") or {}
    info = str(example.get("info", "") or "")
    pos: Optional[str] = None
    pos_set: Optional[set[str]] = None
    if use_pos and by_pos:
        pos = parse_pos_from_info(info, regex=pos_regex, whitelist=pos_whitelist)
        if pos:
            pos_set = by_pos.get(pos)
            if pos_set is None and pos_unknown_policy == "abort":
                return {
                    "type": "result",
                    "schema_version": 1,
                    "request_id": request_id,
                    "action": "abort",
                    "reason": f"pos_not_in_lexicon:{pos}",
                }

    if norm:
        if pos_set is not None:
            if norm in pos_set:
                return {
                    "type": "result",
                    "schema_version": 1,
                    "request_id": request_id,
                    "action": "accept",
                    "reason": "in_lexicon_pos",
                    "normalized": {"label": canonical.get(norm, label)},
                }
        elif norm in lexicon_norm:
            return {
                "type": "result",
                "schema_version": 1,
                "request_id": request_id,
                "action": "accept",
                "reason": "in_lexicon",
                "normalized": {"label": canonical.get(norm, label)},
            }

    if not norm:
        return {
            "type": "result",
            "schema_version": 1,
            "request_id": request_id,
            "action": "retry",
            "reason": "empty_label",
            "retry": {
                "allowed_labels": [],
                "instruction": "Return a non-empty lemma; if unsure, return \"unclassified\".",
            },
        }

    matches = tree.search(norm, max_distance=max_distance)
    matches.sort(key=lambda pair: (pair[0], pair[1]))
    suggestions: List[str] = []
    effective_terms: Optional[set[str]] = pos_set if pos_set is not None else None
    for _, term in matches:
        if effective_terms is not None and term not in effective_terms:
            continue
        suggestions.append(canonical.get(term, term))
        if len(suggestions) >= max_suggestions:
            break
    if not suggestions and effective_terms is not None and pos_fallback:
        for _, term in matches:
            suggestions.append(canonical.get(term, term))
            if len(suggestions) >= max_suggestions:
                break

    return {
        "type": "result",
        "schema_version": 1,
        "request_id": request_id,
        "action": "retry",
        "reason": "not_in_lexicon_pos" if effective_terms is not None else "not_in_lexicon",
        "retry": {
            "allowed_labels": suggestions,
            "instruction": "Choose the correct lemma from allowed_labels.",
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="NDJSON lemmatization validator.")
    parser.add_argument("--lexicon", required=True, help="Path to lemma lexicon (one lemma per line).")
    parser.add_argument("--max_distance", type=int, default=2, help="Max edit distance for suggestions.")
    parser.add_argument("--max_suggestions", type=int, default=30, help="Max suggestions to return.")
    parser.add_argument("--lowercase", action="store_true", help="Lowercase labels before lookup.")
    parser.add_argument("--strip_punct", action="store_true", help="Strip punctuation before lookup.")
    parser.add_argument(
        "--lexicon_field_sep",
        choices=["auto", "tab", "semicolon", "space"],
        default="auto",
        help="How to split lexicon lines into fields (lemma [POS ...]).",
    )
    parser.add_argument(
        "--use_pos",
        action="store_true",
        help=(
            "If set and the lexicon provides POS in a second column, restrict membership checks and "
            "edit-distance candidates to the POS extracted from the dataset info column."
        ),
    )
    parser.add_argument(
        "--info_pos_regex",
        default=r"(?i)\b(?:pos|part[- ]of[- ]speech)\s*[:=]\s*([A-Za-z][A-Za-z0-9_-]*)\b",
        help=(
            "Regex with one capture group for extracting POS from the example's info field. "
            "Default matches e.g. 'pos=NOUN' or 'part-of-speech:VERB'."
        ),
    )
    parser.add_argument(
        "--info_pos_whitelist",
        default="",
        help=(
            "Optional comma-separated list of POS tags. If non-empty, and info contains exactly one of these tags, "
            "that value is treated as POS even without a key like pos=."
        ),
    )
    parser.add_argument(
        "--pos_fallback",
        action="store_true",
        help=(
            "When --use_pos is enabled and POS-restricted suggestions are empty, fall back to non-restricted "
            "suggestions instead of returning an empty allowed_labels."
        ),
    )
    parser.add_argument(
        "--pos_unknown_policy",
        choices=["ignore", "abort"],
        default="ignore",
        help=(
            "What to do when --use_pos is enabled, POS can be extracted from info, but the lexicon has no bucket "
            "for that POS. ignore disables POS restriction for that example; abort stops the run for that example."
        ),
    )
    args = parser.parse_args(argv)

    eprint(f"Loading lexicon from {args.lexicon} ...")
    lexicon_norm, canonical, by_pos = load_lexicon(
        args.lexicon,
        lowercase=args.lowercase,
        strip_punct=args.strip_punct,
        field_sep=args.lexicon_field_sep,
    )
    if by_pos:
        eprint(f"Loaded {len(lexicon_norm)} unique normalized lemma(s) with {len(by_pos)} POS bucket(s).")
    else:
        eprint(f"Loaded {len(lexicon_norm)} unique normalized lemma(s).")
    eprint("Building BK-tree ...")

    tree = BKTree()
    for term in lexicon_norm:
        tree.add(term, max_distance_hint=max(2, args.max_distance))
    eprint("BK-tree ready.")

    pos_regex = re.compile(str(args.info_pos_regex))
    whitelist = [part.strip() for part in str(args.info_pos_whitelist).split(",") if part.strip()]

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
                    lexicon_norm=lexicon_norm,
                    canonical=canonical,
                    by_pos=by_pos,
                    tree=tree,
                    lowercase=args.lowercase,
                    strip_punct=args.strip_punct,
                    use_pos=bool(args.use_pos),
                    pos_regex=pos_regex,
                    pos_whitelist=whitelist,
                    pos_fallback=bool(args.pos_fallback),
                    pos_unknown_policy=str(args.pos_unknown_policy),
                    max_distance=max(0, args.max_distance),
                    max_suggestions=max(0, args.max_suggestions),
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

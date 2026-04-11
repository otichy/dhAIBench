import pathlib
import re
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from validators import lemmatization_validator as lv


class LemmatizationValidatorTests(unittest.TestCase):
    def test_handle_validate_increases_max_distance_per_retry(self) -> None:
        lexicon_norm = {"cat"}
        canonical = {"cat": "cat"}
        by_pos = {}
        tree = lv.BKTree()
        tree.add("cat", max_distance_hint=2)
        base_message = {
            "request_id": "1",
            "prediction": {"label": "cut"},
            "example": {"info": ""},
        }
        first_attempt = lv.handle_validate(
            message={**base_message, "attempt": {"index": 1}},
            lexicon_norm=lexicon_norm,
            canonical=canonical,
            by_pos=by_pos,
            tree=tree,
            lowercase=False,
            strip_punct=False,
            use_pos=False,
            pos_regex=re.compile(r".*"),
            pos_whitelist=[],
            pos_fallback=False,
            pos_unknown_policy="ignore",
            max_distance=0,
            max_suggestions=30,
            max_distance_per_retry=1.0,
        )
        second_attempt = lv.handle_validate(
            message={**base_message, "attempt": {"index": 2}},
            lexicon_norm=lexicon_norm,
            canonical=canonical,
            by_pos=by_pos,
            tree=tree,
            lowercase=False,
            strip_punct=False,
            use_pos=False,
            pos_regex=re.compile(r".*"),
            pos_whitelist=[],
            pos_fallback=False,
            pos_unknown_policy="ignore",
            max_distance=0,
            max_suggestions=30,
            max_distance_per_retry=1.0,
        )
        third_attempt = lv.handle_validate(
            message={**base_message, "attempt": {"index": 3}},
            lexicon_norm=lexicon_norm,
            canonical=canonical,
            by_pos=by_pos,
            tree=tree,
            lowercase=False,
            strip_punct=False,
            use_pos=False,
            pos_regex=re.compile(r".*"),
            pos_whitelist=[],
            pos_fallback=False,
            pos_unknown_policy="ignore",
            max_distance=0,
            max_suggestions=30,
            max_distance_per_retry=1.0,
        )
        self.assertEqual(first_attempt["action"], "retry")
        self.assertEqual(first_attempt["retry"]["allowed_labels"], [])
        self.assertEqual(second_attempt["action"], "retry")
        self.assertEqual(second_attempt["retry"]["allowed_labels"], [])
        self.assertEqual(third_attempt["action"], "retry")
        self.assertEqual(third_attempt["retry"]["allowed_labels"], ["cat"])


if __name__ == "__main__":
    unittest.main()

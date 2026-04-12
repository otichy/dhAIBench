import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from validators import OE_lemmatization_validator as oe


class OELemmatizationValidatorTests(unittest.TestCase):
    def test_normalize_pos_strips_suffix_markers(self) -> None:
        self.assertEqual(oe.normalize_pos("PRO^N"), "PRO")
        self.assertEqual(oe.normalize_pos("PRO$"), "PRO")
        self.assertEqual(oe.normalize_pos("D^D$foo"), "D")

    def test_weighted_distance_treats_thorn_and_eth_as_zero_cost(self) -> None:
        self.assertEqual(oe.weighted_distance("þæt", "ðæt"), 0.0)

    def test_weighted_distance_treats_ne_prefix_as_zero_cost(self) -> None:
        self.assertEqual(oe.weighted_distance("habban", "ne_habban"), 0.0)

    def test_weighted_distance_reduces_final_e_cost(self) -> None:
        self.assertLess(oe.weighted_distance("sune", "sun"), 1.0)

    def test_weighted_distance_reduces_doubled_letter_cost(self) -> None:
        self.assertLess(oe.weighted_distance("writtan", "writan"), 1.0)
        self.assertLess(oe.weighted_distance("sunu", "sunnu"), 1.0)

    def test_weighted_distance_reduces_ge_prefix_cost(self) -> None:
        self.assertLess(oe.weighted_distance("geweaxan", "weaxan"), 1.0)

    def test_weighted_distance_keeps_legacy_low_cost_vowel_variants(self) -> None:
        self.assertEqual(oe.weighted_distance("sylf", "self"), oe.VARIANT_COST)
        self.assertEqual(oe.weighted_distance("mann", "monn"), oe.VARIANT_COST)

    def test_weighted_distance_assigns_unit_cost_to_uncovered_vowel_substitution(self) -> None:
        self.assertEqual(oe.weighted_distance("sawol", "sawel"), 1.0)
        self.assertEqual(oe.weighted_distance("dæg", "deg"), 1.0)

    def test_weighted_distance_reduces_selected_spelling_variants(self) -> None:
        self.assertEqual(oe.weighted_distance("þurh", "thurh"), oe.VARIANT_COST)
        self.assertEqual(oe.weighted_distance("wexan", "wecsan"), oe.X_CS_SUBSTITUTION_COST)

    def test_weighted_distance_assigns_dental_variant_cost(self) -> None:
        self.assertEqual(oe.weighted_distance("god", "got"), oe.DENTAL_VARIANT_COST)
        self.assertEqual(oe.weighted_distance("god", "goð"), oe.DENTAL_VARIANT_COST)
        self.assertEqual(oe.weighted_distance("smið", "smiþ"), 0.0)

    def test_handle_validate_forces_foreign_word_label(self) -> None:
        lexicon = oe.Lexicon(all_lemmas={"foreign_word": "foreign_word"}, lemmas_by_pos={"FW": {"foreign_word": "foreign_word"}})
        result = oe.handle_validate(
            message={
                "request_id": "1",
                "prediction": {"label": "anything"},
                "example": {"info": "FW"},
            },
            lexicon=lexicon,
            max_distance=oe.DEFAULT_MAX_DISTANCE,
            max_suggestions=30,
        )
        self.assertEqual(result["action"], "accept")
        self.assertEqual(result["normalized"]["label"], "foreign_word")

    def test_handle_validate_accepts_zero_cost_variant_as_normalized(self) -> None:
        lexicon = oe.Lexicon(
            all_lemmas={"ne_habban": "ne_habban"},
            lemmas_by_pos={"VB": {"ne_habban": "ne_habban"}},
        )
        result = oe.handle_validate(
            message={
                "request_id": "1",
                "prediction": {"label": "habban"},
                "example": {"info": "VB^X"},
            },
            lexicon=lexicon,
            max_distance=oe.DEFAULT_MAX_DISTANCE,
            max_suggestions=30,
        )
        self.assertEqual(result["action"], "accept")
        self.assertEqual(result["normalized"]["label"], "ne_habban")

    def test_handle_validate_accepts_ne_prefixed_prediction_not_in_lexicon(self) -> None:
        lexicon = oe.Lexicon(
            all_lemmas={"habban": "habban"},
            lemmas_by_pos={"VB": {"habban": "habban"}},
        )
        result = oe.handle_validate(
            message={
                "request_id": "1",
                "prediction": {"label": "ne_habban"},
                "example": {"info": "VB^X"},
            },
            lexicon=lexicon,
            max_distance=oe.DEFAULT_MAX_DISTANCE,
            max_suggestions=30,
        )
        self.assertEqual(result["action"], "accept")
        self.assertEqual(result["reason"], "accepted_negated_form")
        self.assertEqual(result["normalized"]["label"], "ne_habban")

    def test_handle_validate_retries_with_pos_restricted_suggestions(self) -> None:
        lexicon = oe.Lexicon(
            all_lemmas={"habban": "habban", "hatan": "hatan"},
            lemmas_by_pos={"VB": {"habban": "habban"}, "N": {"hatan": "hatan"}},
        )
        result = oe.handle_validate(
            message={
                "request_id": "1",
                "prediction": {"label": "habbane"},
                "example": {"info": "VBPS"},
            },
            lexicon=lexicon,
            max_distance=oe.DEFAULT_MAX_DISTANCE,
            max_suggestions=30,
        )
        self.assertEqual(result["action"], "retry")
        self.assertEqual(result["retry"]["allowed_labels"], ["habban"])
        self.assertIn('The previous lemma "habbane" was rejected', result["retry"]["message"])

    def test_handle_validate_increases_max_distance_per_retry(self) -> None:
        lexicon = oe.Lexicon(
            all_lemmas={"sawel": "sawel"},
            lemmas_by_pos={"N": {"sawel": "sawel"}},
        )
        base_message = {
            "request_id": "1",
            "prediction": {"label": "sawol"},
            "example": {"info": "N"},
        }
        first_attempt = oe.handle_validate(
            message={**base_message, "attempt": {"index": 1}},
            lexicon=lexicon,
            max_distance=0.75,
            max_suggestions=30,
            max_distance_per_retry=0.5,
        )
        second_attempt = oe.handle_validate(
            message={**base_message, "attempt": {"index": 2}},
            lexicon=lexicon,
            max_distance=0.75,
            max_suggestions=30,
            max_distance_per_retry=0.5,
        )
        third_attempt = oe.handle_validate(
            message={**base_message, "attempt": {"index": 3}},
            lexicon=lexicon,
            max_distance=0.75,
            max_suggestions=30,
            max_distance_per_retry=0.5,
        )
        self.assertEqual(first_attempt["action"], "retry")
        self.assertEqual(first_attempt["retry"]["allowed_labels"], [])
        self.assertEqual(second_attempt["action"], "retry")
        self.assertEqual(second_attempt["retry"]["allowed_labels"], [])
        self.assertEqual(third_attempt["action"], "retry")
        self.assertEqual(third_attempt["retry"]["allowed_labels"], ["sawel"])


if __name__ == "__main__":
    unittest.main()

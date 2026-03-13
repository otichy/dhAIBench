import csv
import os
import tempfile
import unittest

import benchmark_agent as ba


class InputSchemaTests(unittest.TestCase):
    def test_read_examples_accepts_missing_context_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "input.csv")
            with open(path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter=";")
                writer.writerow(["ID", "node"])
                writer.writerow(["1", "testnode"])

            examples, extra_fields = ba.read_examples(path)

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].example_id, "1")
        self.assertEqual(examples[0].node, "testnode")
        self.assertEqual(examples[0].left_context, "")
        self.assertEqual(examples[0].right_context, "")
        self.assertEqual(extra_fields, [])

    def test_read_examples_is_case_insensitive_for_required_and_optional_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "input.csv")
            with open(path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter=";")
                writer.writerow(
                    ["id", "LEFTCONTEXT", "NoDe", "RIGHTcontext", "INFO", "TRUTH", "customMeta"]
                )
                writer.writerow(["7", "left", "target", "right", "hint", "NOUN", "x"])

            examples, extra_fields = ba.read_examples(path)

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].example_id, "7")
        self.assertEqual(examples[0].left_context, "left")
        self.assertEqual(examples[0].node, "target")
        self.assertEqual(examples[0].right_context, "right")
        self.assertEqual(examples[0].info, "hint")
        self.assertEqual(examples[0].truth, "NOUN")
        self.assertEqual(examples[0].extras, {"customMeta": "x"})
        self.assertEqual(extra_fields, ["customMeta"])

    def test_read_examples_requires_id_and_node(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "input.csv")
            with open(path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter=";")
                writer.writerow(["ID", "leftContext", "rightContext"])
                writer.writerow(["1", "left", "right"])

            with self.assertRaises(ValueError) as ctx:
                ba.read_examples(path)

        self.assertIn("Missing required columns", str(ctx.exception))
        self.assertIn("node", str(ctx.exception))

    def test_read_label_file_is_case_insensitive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "labels.csv")
            with open(path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter=";")
                writer.writerow(["id", "TRUTH"])
                writer.writerow(["1", "NOUN"])
                writer.writerow(["2", "VERB"])

            labels = ba.read_label_file(path)

        self.assertEqual(labels, {"1": "NOUN", "2": "VERB"})

    def test_read_truth_labels_from_output_is_case_insensitive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.csv")
            with open(path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter=";")
                writer.writerow(["id", "PREDICTION", "TrUtH"])
                writer.writerow(["1", "NOUN", "NOUN"])
                writer.writerow(["2", "VERB", "NOUN"])

            labels, has_truth = ba.read_truth_labels_from_output(path)

        self.assertTrue(has_truth)
        self.assertEqual(labels, {"1": "NOUN", "2": "NOUN"})

    def test_load_existing_output_predictions_is_case_insensitive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.csv")
            with open(path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter=";")
                writer.writerow(["id", "PREDICTION", "CONFIDENCE", "PROMPTTOKENS", "TOTALTOKENS"])
                writer.writerow(["1", "NOUN", "0.8", "10", "12"])

            (
                fieldnames,
                predictions,
                total_prompt_tokens,
                total_completion_tokens,
                total_reported_tokens,
            ) = ba.load_existing_output_predictions(path)

        self.assertEqual(fieldnames, ["id", "PREDICTION", "CONFIDENCE", "PROMPTTOKENS", "TOTALTOKENS"])
        self.assertEqual(set(predictions.keys()), {"1"})
        self.assertEqual(predictions["1"].label, "NOUN")
        self.assertAlmostEqual(predictions["1"].confidence or 0.0, 0.8)
        self.assertEqual(total_prompt_tokens, 10)
        self.assertEqual(total_completion_tokens, 0)
        self.assertEqual(total_reported_tokens, 12)


if __name__ == "__main__":
    unittest.main()

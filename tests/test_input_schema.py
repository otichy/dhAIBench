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


if __name__ == "__main__":
    unittest.main()

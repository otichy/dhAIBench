import csv
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import benchmark_agent as ba


def _write_output_csv(path: str, rows: list[dict[str, str]]) -> None:
    fieldnames = ["ID", "prediction", "truth", "confidence"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_labels_csv(path: str, rows: list[dict[str, str]]) -> None:
    fieldnames = ["ID", "truth"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class MetricsOnlyTests(unittest.TestCase):
    def test_metrics_only_writes_run_metrics_without_truth_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_csv = os.path.join(tmpdir, "existing_output.csv")
            metrics_json = ba.build_artifact_path(output_csv, "_metrics.json", ba.DEFAULT_METRICS_DIR)
            with open(output_csv, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["ID", "prediction", "confidence"],
                    delimiter=";",
                )
                writer.writeheader()
                writer.writerow({"ID": "1", "prediction": "NOUN", "confidence": "0.9"})
                writer.writerow({"ID": "2", "prediction": "VERB", "confidence": "0.6"})

            with patch.object(ba, "generate_confusion_heatmap", return_value=None):
                exit_code = ba.main(
                    [
                        "--metrics_only",
                        "--input",
                        output_csv,
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.exists(metrics_json))
            with open(metrics_json, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload.get("mode"), "metrics_only")
            self.assertEqual(payload.get("truth_source"), "none")
            self.assertFalse(payload.get("label_metrics_available", True))
            self.assertNotIn("accuracy", payload)
            self.assertEqual(payload.get("prediction_count"), 2)
            self.assertEqual(payload.get("truth_label_count"), 0)
            self.assertEqual(payload.get("task_name"), "existing_output")
            self.assertEqual(payload.get("task_description"), "")
            self.assertEqual(payload.get("tags"), "")
            model_details = payload.get("model_details")
            self.assertIsInstance(model_details, dict)
            self.assertIn("provider", model_details)
            self.assertIn("model_requested", model_details)
            self.assertIn("model_for_requests", model_details)
            self.assertIn("api_base_url", model_details)
            self.assertIn("chat_completions_endpoint", model_details)

    def test_metrics_only_uses_truth_column_from_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_csv = os.path.join(tmpdir, "existing_output.csv")
            metrics_json = ba.build_artifact_path(output_csv, "_metrics.json", ba.DEFAULT_METRICS_DIR)
            _write_output_csv(
                output_csv,
                [
                    {"ID": "1", "prediction": "NOUN", "truth": "NOUN", "confidence": "0.9"},
                    {"ID": "2", "prediction": "NOUN", "truth": "VERB", "confidence": "0.6"},
                ],
            )

            with (
                patch.object(ba, "parse_env_file", side_effect=AssertionError("parse_env_file should not run")),
                patch.object(ba, "generate_confusion_heatmap", return_value=None),
            ):
                exit_code = ba.main(
                    [
                        "--metrics_only",
                        "--input",
                        output_csv,
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.exists(metrics_json))
            with open(metrics_json, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertAlmostEqual(payload.get("accuracy", 0.0), 0.5)
            self.assertEqual(payload.get("mode"), "metrics_only")
            self.assertEqual(payload.get("truth_source"), "output_csv_truth_column")

    def test_metrics_only_overrides_truths_from_labels_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_csv = os.path.join(tmpdir, "existing_output.csv")
            labels_csv = os.path.join(tmpdir, "new_labels.csv")
            metrics_json = ba.build_artifact_path(output_csv, "_metrics.json", ba.DEFAULT_METRICS_DIR)
            _write_output_csv(
                output_csv,
                [
                    {"ID": "1", "prediction": "NOUN", "truth": "NOUN", "confidence": "0.9"},
                    {"ID": "2", "prediction": "NOUN", "truth": "NOUN", "confidence": "0.6"},
                ],
            )
            _write_labels_csv(
                labels_csv,
                [
                    {"ID": "1", "truth": "NOUN"},
                    {"ID": "2", "truth": "VERB"},
                    {"ID": "3", "truth": "ADJ"},
                ],
            )

            with patch.object(ba, "generate_confusion_heatmap", return_value=None):
                exit_code = ba.main(
                    [
                        "--metrics_only",
                        "--input",
                        output_csv,
                        "--labels",
                        labels_csv,
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.exists(metrics_json))
            with open(metrics_json, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertAlmostEqual(payload.get("accuracy", 0.0), 0.5)
            self.assertEqual(payload.get("truth_source"), "labels_csv_override_with_output_fallback")


if __name__ == "__main__":
    unittest.main()

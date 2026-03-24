import csv
import json
import logging
import os
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import benchmark_agent as ba


def _reset_root_logging() -> None:
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


@contextmanager
def _isolated_data_dirs(tmpdir: str):
    data_root = os.path.join(tmpdir, "bench_data")
    input_dir = os.path.join(data_root, "input")
    output_dir = os.path.join(data_root, "output")
    metrics_dir = os.path.join(data_root, "metrics")
    logs_dir = os.path.join(data_root, "logs")
    for path in (input_dir, output_dir, metrics_dir, logs_dir):
        os.makedirs(path, exist_ok=True)
    with (
        patch.object(ba, "DATA_ROOT_DIR", data_root),
        patch.object(ba, "DEFAULT_INPUT_DIR", input_dir),
        patch.object(ba, "DEFAULT_OUTPUT_DIR", output_dir),
        patch.object(ba, "DEFAULT_METRICS_DIR", metrics_dir),
        patch.object(ba, "DEFAULT_LOGS_DIR", logs_dir),
    ):
        try:
            yield
        finally:
            logging.shutdown()
            _reset_root_logging()


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


def _assert_metrics_metadata(
    testcase: unittest.TestCase,
    payload: dict[str, object],
    *,
    task_name: str,
    task_description: str,
    tags: str,
    prompt_layout: str | None = None,
) -> None:
    run_config = payload.get("run_config")
    testcase.assertIsInstance(run_config, dict)
    testcase.assertEqual(run_config.get("task_name"), task_name)
    testcase.assertEqual(run_config.get("task_description"), task_description)
    testcase.assertEqual(run_config.get("tags"), tags)
    testcase.assertEqual(run_config.get("prompt_layout"), prompt_layout)
    testcase.assertNotIn("task_name", payload)
    testcase.assertNotIn("task_description", payload)
    testcase.assertNotIn("tags", payload)
    testcase.assertNotIn("prompt_layout", payload)


class MetricsOnlyTests(unittest.TestCase):
    def test_compute_metrics_includes_cohen_kappa(self) -> None:
        payload = ba.compute_metrics(
            ["NOUN", "VERB"],
            ["NOUN", "NOUN"],
        )

        self.assertAlmostEqual(payload.get("accuracy", 0.0), 0.5)
        self.assertAlmostEqual(payload.get("cohen_kappa", 999.0), 0.0, places=8)

    def test_compute_metrics_sets_cohen_kappa_to_one_for_single_label_perfect_agreement(self) -> None:
        payload = ba.compute_metrics(
            ["NOUN", "NOUN", "NOUN"],
            ["NOUN", "NOUN", "NOUN"],
        )

        self.assertAlmostEqual(payload.get("accuracy", 0.0), 1.0)
        self.assertAlmostEqual(payload.get("cohen_kappa", 0.0), 1.0, places=8)

    def test_metrics_only_writes_session_logs_under_sessions_subdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with _isolated_data_dirs(tmpdir):
                output_csv = os.path.join(tmpdir, "existing_output.csv")
                _write_output_csv(
                    output_csv,
                    [
                        {"ID": "1", "prediction": "NOUN", "truth": "NOUN", "confidence": "0.9"},
                    ],
                )

                exit_code = ba.main(
                    [
                        "--metrics_only",
                        "--input",
                        output_csv,
                    ]
                )

                self.assertEqual(exit_code, 0)
                session_logs_dir = os.path.join(ba.DEFAULT_LOGS_DIR, "sessions")
                self.assertTrue(os.path.isdir(session_logs_dir))
                self.assertTrue(
                    any(
                        name.startswith("benchmark_agent_") and name.endswith(".log")
                        for name in os.listdir(session_logs_dir)
                    )
                )
                self.assertFalse(
                    any(
                        name.startswith("benchmark_agent_") and name.endswith(".log")
                        for name in os.listdir(ba.DEFAULT_LOGS_DIR)
                        if os.path.isfile(os.path.join(ba.DEFAULT_LOGS_DIR, name))
                    )
                )

    def test_metrics_only_honors_no_confusion_heatmap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with _isolated_data_dirs(tmpdir):
                output_csv = os.path.join(tmpdir, "existing_output.csv")
                _write_output_csv(
                    output_csv,
                    [
                        {"ID": "1", "prediction": "NOUN", "truth": "NOUN", "confidence": "0.9"},
                        {"ID": "2", "prediction": "VERB", "truth": "VERB", "confidence": "0.6"},
                    ],
                )

                with patch.object(ba, "generate_confusion_heatmap", return_value=None) as heatmap_mock:
                    exit_code = ba.main(
                        [
                            "--metrics_only",
                            "--input",
                            output_csv,
                            "--no-confusion_heatmap",
                        ]
                    )

                self.assertEqual(exit_code, 0)
                heatmap_mock.assert_not_called()

    def test_metrics_only_writes_run_metrics_without_truth_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with _isolated_data_dirs(tmpdir):
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
                _assert_metrics_metadata(
                    self,
                    payload,
                    task_name="existing_output",
                    prompt_layout=None,
                    task_description="",
                    tags="",
                )
                calibration = payload.get("calibration_metrics")
                self.assertIsInstance(calibration, dict)
                self.assertFalse(calibration.get("available"))
                self.assertEqual(calibration.get("sample_count"), 0)
                self.assertIsNone(calibration.get("ece"))
                model_details = payload.get("model_details")
                self.assertIsInstance(model_details, dict)
                self.assertIn("provider", model_details)
                self.assertIn("model_requested", model_details)
                self.assertIn("model_for_requests", model_details)
                self.assertIn("api_base_url", model_details)
                self.assertIn("chat_completions_endpoint", model_details)

    def test_metrics_only_uses_truth_column_from_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with _isolated_data_dirs(tmpdir):
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
                self.assertAlmostEqual(payload.get("cohen_kappa", 999.0), 0.0, places=8)
                self.assertEqual(payload.get("mode"), "metrics_only")
                self.assertEqual(payload.get("truth_source"), "output_csv_truth_column")
                calibration = payload.get("calibration_metrics")
                self.assertIsInstance(calibration, dict)
                self.assertTrue(calibration.get("available"))
                self.assertEqual(calibration.get("sample_count"), 2)
                self.assertAlmostEqual(calibration.get("ece", 0.0), 0.35, places=8)
                self.assertAlmostEqual(calibration.get("mce", 0.0), 0.6, places=8)
                self.assertAlmostEqual(calibration.get("brier_score", 0.0), 0.185, places=8)

    def test_metrics_only_overrides_truths_from_labels_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with _isolated_data_dirs(tmpdir):
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
                self.assertAlmostEqual(payload.get("cohen_kappa", 999.0), 0.0, places=8)
                self.assertEqual(payload.get("truth_source"), "labels_csv_override_with_output_fallback")

    def test_metrics_only_applies_task_metadata_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with _isolated_data_dirs(tmpdir):
                output_csv = os.path.join(tmpdir, "existing_output.csv")
                metrics_json = ba.build_artifact_path(output_csv, "_metrics.json", ba.DEFAULT_METRICS_DIR)
                _write_output_csv(
                    output_csv,
                    [
                        {"ID": "1", "prediction": "NOUN", "truth": "NOUN", "confidence": "0.9"},
                    ],
                )

                with patch.object(ba, "generate_confusion_heatmap", return_value=None):
                    exit_code = ba.main(
                        [
                            "--metrics_only",
                            "--input",
                            output_csv,
                            "--task_name",
                            "Custom Task",
                            "--task_description",
                            "Custom task description",
                            "--tags",
                            "alpha;beta",
                        ]
                    )

                self.assertEqual(exit_code, 0)
                with open(metrics_json, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                _assert_metrics_metadata(
                    self,
                    payload,
                    task_name="Custom Task",
                    prompt_layout=None,
                    task_description="Custom task description",
                    tags="alpha;beta",
                )

    def test_metrics_only_preserves_existing_metadata_while_backfilling_token_totals(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with _isolated_data_dirs(tmpdir):
                output_csv = os.path.join(tmpdir, "existing_output.csv")
                metrics_json = ba.build_artifact_path(output_csv, "_metrics.json", ba.DEFAULT_METRICS_DIR)
                log_path = ba.resolve_prompt_log_path_for_output(output_csv)
                _write_output_csv(
                    output_csv,
                    [
                        {"ID": "1", "prediction": "NOUN", "truth": "NOUN", "confidence": "0.9"},
                    ],
                )

                with open(metrics_json, "w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "task_name": "Existing Task",
                            "prompt_layout": "standard",
                            "task_description": "Existing description",
                            "tags": "alpha;beta",
                            "cache_padding": {
                                "enabled": True,
                                "target_shared_prefix_tokens": 1200,
                                "calibration_shared_prefix_tokens": 1100,
                                "target_prompt_tokens": 1200,
                                "calibration_prompt_tokens": 1100,
                                "calibration_example_id": "ex-1",
                                "applied_padding_tokens_estimate": 100,
                                "examples_with_padding_applied": 3,
                            },
                            "model_details": {
                                "provider": "openai",
                                "model_requested": "gpt-5-mini",
                                "model_for_requests": "gpt-5-mini",
                                "api_base_url": "",
                                "chat_completions_endpoint": "",
                            },
                        },
                        handle,
                        indent=2,
                    )

                with open(log_path, "w", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "record_type": "example_result",
                                "example_id": "1",
                                "attempts": [
                                    {
                                        "attempt": 1,
                                        "status": "success",
                                        "response": {
                                            "usage_metadata": {
                                                "usage": {
                                                    "prompt_tokens": 10,
                                                    "completion_tokens": 6,
                                                    "total_tokens": 16,
                                                    "prompt_tokens_details": {"cached_tokens": 2},
                                                    "completion_tokens_details": {"reasoning_tokens": 4},
                                                }
                                            }
                                        },
                                    }
                                ],
                            }
                        )
                    )
                    handle.write("\n")

                with patch.object(ba, "generate_confusion_heatmap", return_value=None):
                    exit_code = ba.main(
                        [
                            "--metrics_only",
                            "--input",
                            output_csv,
                        ]
                    )

                self.assertEqual(exit_code, 0)
                with open(metrics_json, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                _assert_metrics_metadata(
                    self,
                    payload,
                    task_name="Existing Task",
                    prompt_layout="standard",
                    task_description="Existing description",
                    tags="alpha;beta",
                )
                self.assertEqual(payload.get("cache_padding", {}).get("target_shared_prefix_tokens"), 1200)
                self.assertEqual(payload.get("model_details", {}).get("model_requested"), "gpt-5-mini")
                self.assertEqual(
                    payload.get("token_usage_totals"),
                    {
                        "attempts_total": 1,
                        "attempts_with_token_usage": 1,
                        "attempts_with_output_tokens": 1,
                        "attempts_with_cached_input_tokens": 1,
                        "attempts_with_thinking_tokens": 1,
                        "input_tokens_total": 10,
                        "cached_input_tokens_total": 2,
                        "non_cached_input_tokens_total": 8,
                        "output_tokens_total": 6,
                        "thinking_tokens_total": 4,
                        "output_tokens_definition": "total_tokens - prompt_tokens (or completion_tokens + thinking_tokens fallback)",
                    },
                )

    def test_metrics_only_backfills_run_config_from_longer_legacy_top_level_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with _isolated_data_dirs(tmpdir):
                output_csv = os.path.join(tmpdir, "existing_output.csv")
                metrics_json = ba.build_artifact_path(output_csv, "_metrics.json", ba.DEFAULT_METRICS_DIR)
                _write_output_csv(
                    output_csv,
                    [
                        {"ID": "1", "prediction": "NOUN", "truth": "NOUN", "confidence": "0.9"},
                    ],
                )

                with open(metrics_json, "w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "run_config": {
                                "task_name": "Task",
                                "prompt_layout": "compact",
                                "task_description": "Short desc",
                                "tags": "alpha",
                            },
                            "task_name": "Existing Output Task",
                            "prompt_layout": "standard",
                            "task_description": "Existing description carried over from legacy metadata.",
                            "tags": "alpha;beta",
                        },
                        handle,
                        indent=2,
                    )

                with patch.object(ba, "generate_confusion_heatmap", return_value=None):
                    exit_code = ba.main(
                        [
                            "--metrics_only",
                            "--input",
                            output_csv,
                        ]
                    )

                self.assertEqual(exit_code, 0)
                with open(metrics_json, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                _assert_metrics_metadata(
                    self,
                    payload,
                    task_name="Existing Output Task",
                    prompt_layout="standard",
                    task_description="Existing description carried over from legacy metadata.",
                    tags="alpha;beta",
                )


if __name__ == "__main__":
    unittest.main()

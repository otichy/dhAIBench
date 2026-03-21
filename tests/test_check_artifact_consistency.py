import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_artifact_consistency.py"
SPEC = importlib.util.spec_from_file_location("check_artifact_consistency", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
cac = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cac
SPEC.loader.exec_module(cac)


class ArtifactConsistencyTests(unittest.TestCase):
    def test_update_metrics_token_totals_from_csv_backfills_prompt_and_output_totals(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            metrics_dir = repo_root / "data" / "metrics"
            output_dir = repo_root / "data" / "output"
            logs_dir = repo_root / "data" / "logs"
            metrics_dir.mkdir(parents=True)
            output_dir.mkdir(parents=True)
            logs_dir.mkdir(parents=True)

            metrics_path = metrics_dir / "sample__metrics.json"
            output_path = output_dir / "sample.csv"
            log_path = logs_dir / "sample.log"

            with output_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["ID", "prediction", "promptTokens", "completionTokens", "totalTokens"],
                    delimiter=";",
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "ID": "1",
                        "prediction": "A",
                        "promptTokens": "10",
                        "completionTokens": "3",
                        "totalTokens": "22",
                    }
                )
                writer.writerow(
                    {
                        "ID": "2",
                        "prediction": "B",
                        "promptTokens": "7",
                        "completionTokens": "4",
                        "totalTokens": "16",
                    }
                )

            log_path.write_text(
                json.dumps(
                    {
                        "record_type": "run_metadata",
                        "model_details": {
                            "provider": "openai",
                            "model_requested": "gpt-test",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            metrics_path.write_text(
                json.dumps(
                    {
                        "model_details": {"provider": "openai", "model_requested": "gpt-test"},
                        "prediction_count": 2,
                        "token_usage_totals": {
                            "attempts_total": 3,
                            "attempts_with_token_usage": 1,
                            "attempts_with_output_tokens": 1,
                            "attempts_with_cached_input_tokens": 0,
                            "attempts_with_thinking_tokens": 0,
                            "input_tokens_total": 1,
                            "cached_input_tokens_total": 0,
                            "non_cached_input_tokens_total": 1,
                            "output_tokens_total": 2,
                            "thinking_tokens_total": 0,
                            "output_tokens_definition": cac.OUTPUT_TOKENS_DEFINITION,
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            old_metrics_dir = cac.DEFAULT_METRICS_DIR
            old_output_dir = cac.DEFAULT_OUTPUT_DIR
            old_logs_dir = cac.DEFAULT_LOGS_DIR
            try:
                cac.DEFAULT_METRICS_DIR = metrics_dir
                cac.DEFAULT_OUTPUT_DIR = output_dir
                cac.DEFAULT_LOGS_DIR = logs_dir

                changed, resolved_output_path, csv_summary = cac.update_metrics_token_totals_from_csv(
                    metrics_path
                )
                self.assertTrue(changed)
                self.assertEqual(resolved_output_path, output_path)
                self.assertIsNotNone(csv_summary)

                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                token_usage_totals = payload["token_usage_totals"]
                self.assertEqual(token_usage_totals["attempts_total"], 3)
                self.assertEqual(token_usage_totals["attempts_with_token_usage"], 2)
                self.assertEqual(token_usage_totals["attempts_with_output_tokens"], 2)
                self.assertEqual(token_usage_totals["input_tokens_total"], 17)
                self.assertEqual(token_usage_totals["non_cached_input_tokens_total"], 17)
                self.assertEqual(token_usage_totals["output_tokens_total"], 21)
            finally:
                cac.DEFAULT_METRICS_DIR = old_metrics_dir
                cac.DEFAULT_OUTPUT_DIR = old_output_dir
                cac.DEFAULT_LOGS_DIR = old_logs_dir

    def test_analyze_metrics_artifact_flags_model_and_token_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            metrics_dir = repo_root / "data" / "metrics"
            output_dir = repo_root / "data" / "output"
            logs_dir = repo_root / "data" / "logs"
            metrics_dir.mkdir(parents=True)
            output_dir.mkdir(parents=True)
            logs_dir.mkdir(parents=True)

            metrics_path = metrics_dir / "sample__metrics.json"
            output_path = output_dir / "sample.csv"
            log_path = logs_dir / "sample.log"

            with output_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["ID", "prediction", "promptTokens", "completionTokens", "totalTokens"],
                    delimiter=";",
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "ID": "1",
                        "prediction": "A",
                        "promptTokens": "10",
                        "completionTokens": "3",
                        "totalTokens": "22",
                    }
                )

            log_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "record_type": "run_metadata",
                                "model_details": {
                                    "provider": "vertex",
                                    "model_requested": "gemini-test",
                                },
                            }
                        ),
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
                                                    "completion_tokens": 3,
                                                    "total_tokens": 22,
                                                }
                                            }
                                        },
                                    }
                                ],
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            metrics_path.write_text(
                json.dumps(
                    {
                        "model_details": {"provider": "openai", "model_requested": "gpt-test"},
                        "prediction_count": 1,
                        "request_control_summary": {"attempts_total": 2},
                        "usage_metadata_summary": {"attempts_with_usage_metadata": 0},
                        "token_usage_totals": {
                            "attempts_total": 2,
                            "attempts_with_token_usage": 0,
                            "attempts_with_output_tokens": 0,
                            "attempts_with_cached_input_tokens": 0,
                            "attempts_with_thinking_tokens": 0,
                            "input_tokens_total": 0,
                            "cached_input_tokens_total": 0,
                            "non_cached_input_tokens_total": 0,
                            "output_tokens_total": 0,
                            "thinking_tokens_total": 0,
                            "output_tokens_definition": cac.OUTPUT_TOKENS_DEFINITION,
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            old_metrics_dir = cac.DEFAULT_METRICS_DIR
            old_output_dir = cac.DEFAULT_OUTPUT_DIR
            old_logs_dir = cac.DEFAULT_LOGS_DIR
            try:
                cac.DEFAULT_METRICS_DIR = metrics_dir
                cac.DEFAULT_OUTPUT_DIR = output_dir
                cac.DEFAULT_LOGS_DIR = logs_dir

                report = cac.analyze_metrics_artifact(metrics_path)
                issue_codes = {issue.code for issue in report.issues}
                self.assertIn("attempts_total_mismatch_log", issue_codes)
                self.assertIn("usage_attempts_mismatch_log", issue_codes)
                self.assertIn("input_tokens_total_mismatch_csv", issue_codes)
                self.assertIn("output_tokens_total_mismatch_csv", issue_codes)
                self.assertIn("provider_mismatch_log", issue_codes)
                self.assertIn("model_requested_mismatch_log", issue_codes)
            finally:
                cac.DEFAULT_METRICS_DIR = old_metrics_dir
                cac.DEFAULT_OUTPUT_DIR = old_output_dir
                cac.DEFAULT_LOGS_DIR = old_logs_dir

    def test_repair_metrics_artifact_backs_up_original_and_only_lowers_from_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            metrics_dir = repo_root / "data" / "metrics"
            output_dir = repo_root / "data" / "output"
            logs_dir = repo_root / "data" / "logs"
            backup_dir = repo_root / "data" / "backup" / "artifact_consistency_test"
            metrics_dir.mkdir(parents=True)
            output_dir.mkdir(parents=True)
            logs_dir.mkdir(parents=True)

            metrics_path = metrics_dir / "sample__metrics.json"
            output_path = output_dir / "sample.csv"
            log_path = logs_dir / "sample.log"

            with output_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["ID", "prediction", "promptTokens", "completionTokens", "totalTokens"],
                    delimiter=";",
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "ID": "1",
                        "prediction": "A",
                        "promptTokens": "10",
                        "completionTokens": "2",
                        "totalTokens": "18",
                    }
                )

            log_path.write_text(
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
                                            "completion_tokens": 2,
                                            "total_tokens": 18,
                                        }
                                    }
                                },
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            original_payload = {
                "request_control_summary": {"attempts_total": 4},
                "usage_metadata_summary": {
                    "attempts_total": 4,
                    "attempts_with_usage_metadata": 1,
                },
                "token_usage_totals": {
                    "attempts_total": 4,
                    "attempts_with_token_usage": 3,
                    "attempts_with_output_tokens": 3,
                    "attempts_with_cached_input_tokens": 0,
                    "attempts_with_thinking_tokens": 0,
                    "input_tokens_total": 50,
                    "cached_input_tokens_total": 0,
                    "non_cached_input_tokens_total": 50,
                    "output_tokens_total": 40,
                    "thinking_tokens_total": 0,
                    "output_tokens_definition": cac.OUTPUT_TOKENS_DEFINITION,
                },
            }
            metrics_path.write_text(json.dumps(original_payload, indent=2), encoding="utf-8")

            old_metrics_dir = cac.DEFAULT_METRICS_DIR
            old_output_dir = cac.DEFAULT_OUTPUT_DIR
            old_logs_dir = cac.DEFAULT_LOGS_DIR
            try:
                cac.DEFAULT_METRICS_DIR = metrics_dir
                cac.DEFAULT_OUTPUT_DIR = output_dir
                cac.DEFAULT_LOGS_DIR = logs_dir

                outcome = cac.repair_metrics_artifact(
                    metrics_path,
                    backup_dir,
                    repair_lower_token_totals_from_csv=True,
                    repair_lower_attempt_totals_from_log=True,
                )

                self.assertTrue(outcome.token_totals_repaired)
                self.assertTrue(outcome.attempt_totals_repaired)
                self.assertIsNotNone(outcome.backup_path)
                self.assertTrue(outcome.backup_path.exists())

                backup_payload = json.loads(outcome.backup_path.read_text(encoding="utf-8"))
                self.assertEqual(backup_payload, original_payload)

                repaired_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                self.assertEqual(repaired_payload["request_control_summary"]["attempts_total"], 1)
                self.assertEqual(repaired_payload["usage_metadata_summary"]["attempts_total"], 1)
                self.assertEqual(repaired_payload["token_usage_totals"]["attempts_total"], 1)
                self.assertEqual(repaired_payload["token_usage_totals"]["attempts_with_token_usage"], 1)
                self.assertEqual(repaired_payload["token_usage_totals"]["input_tokens_total"], 10)
                self.assertEqual(repaired_payload["token_usage_totals"]["output_tokens_total"], 8)
            finally:
                cac.DEFAULT_METRICS_DIR = old_metrics_dir
                cac.DEFAULT_OUTPUT_DIR = old_output_dir
                cac.DEFAULT_LOGS_DIR = old_logs_dir


if __name__ == "__main__":
    unittest.main()

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import benchmark_agent as ba


def request(duration=None, status="success", timestamp="2026-11-20T10:00:00Z"):
    return {"status": status, "duration_seconds": duration, "timestamp": timestamp}


class RequestDurationTests(unittest.TestCase):
    def test_failed_attempts_and_resume_gaps_do_not_change_mean(self):
        records = [
            {"record_type": "run_metadata", "run_config": {"threads": 5}},
            {"attempts": [request(500, "error"), request(3)]},
            {"record_type": "run_metadata", "run_config": {"threads": 20}},
            {"attempts": [request(900, "validator_retry"), request(7, timestamp="2026-11-25T10:00:00Z")]},
        ]
        result = ba.compute_mean_request_duration(iter(records))
        self.assertEqual(result["mean_request_duration_seconds"], 5)
        self.assertEqual(result["request_duration_sample_count"], 2)
        self.assertEqual(result["request_duration_excluded_attempt_count"], 2)
        self.assertEqual(result["mean_request_duration_source"], "logs")

    def test_batch_is_one_request_and_individual_fallback_is_separate(self):
        records = [
            {"record_type": "batch_result", "example_ids": ["a", "b"], "attempts": [request(10, "partial_success")]},
            {"record_type": "example_result", "example_id": "a", "attempts": []},
            {"record_type": "example_result", "example_id": "b", "attempts": [request(2)]},
        ]
        self.assertEqual(ba.compute_mean_request_duration(records)["mean_request_duration_seconds"], 6)

    def test_partial_coverage_uses_measurements_instead_of_elapsed_estimate(self):
        result = ba.compute_mean_request_duration(
            [{"attempts": [request(None), request(4), request(float("nan")), request(-2)]}],
            {"overall_time_seconds": 1000, "prediction_count": 4, "run_config": {"threads": 10}},
        )
        self.assertEqual(result["mean_request_duration_seconds"], 4)
        self.assertEqual(result["request_duration_missing_count"], 3)
        self.assertEqual(result["mean_request_duration_source"], "logs_partial")

    def test_fallback_recomputes_session_windows_and_recovers_threads(self):
        records = [
            {"record_type": "run_metadata", "run_config": {"threads": 3}},
            {"attempts": [request(timestamp="2026-11-20T10:00:00Z")]},
            {"attempts": [request(timestamp="2026-11-20T10:01:00Z")]},
            {"record_type": "run_metadata"},
            {"attempts": [request(timestamp="2026-11-25T10:00:00Z")]},
            {"attempts": [request(timestamp="2026-11-25T10:01:00Z")]},
        ]
        result = ba.compute_mean_request_duration(records, {"prediction_count": 4, "overall_time_seconds": 999999})
        self.assertEqual(result["mean_request_duration_seconds"], 90)
        self.assertEqual(result["mean_request_duration_source"], "estimated_elapsed_threads")
        self.assertEqual(result["request_duration_estimate_threads"], 3)

    def test_fallback_from_saved_metrics_and_command_threads(self):
        result = ba.compute_mean_request_duration(
            [{"record_type": "run_command", "argv": ["python", "benchmark_agent.py", "--threads=4"]}],
            {"prediction_count": 10, "overall_time_seconds": 100},
        )
        self.assertEqual(result["mean_request_duration_seconds"], 40)
        self.assertFalse(result["request_duration_estimate_threads_assumed"])

    def test_missing_threads_are_explicitly_assumed_sequential(self):
        result = ba.compute_mean_request_duration([], {"prediction_count": 10, "overall_time_seconds": 100})
        self.assertEqual(result["mean_request_duration_seconds"], 10)
        self.assertTrue(result["request_duration_estimate_threads_assumed"])

    def test_zero_duration_is_valid_but_zero_elapsed_window_is_unknown(self):
        self.assertEqual(ba.compute_mean_request_duration([{"attempts": [request(0)]}])["mean_request_duration_seconds"], 0)
        result = ba.compute_mean_request_duration([{"attempts": [request()]}], {"prediction_count": 1})
        self.assertIsNone(result["mean_request_duration_seconds"])
        self.assertEqual(result["mean_request_duration_source"], "unavailable")

    def test_compact_logs_preserve_request_duration(self):
        attempt = request(1.234)
        self.assertEqual(ba.compact_prompt_attempt_log(attempt)["duration_seconds"], 1.234)

    def test_metrics_only_writes_request_duration_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "fixture__openai__model__2026-11-25-12-00.csv"
            output.write_text("ID;prediction;truth\na;N;N\n", encoding="utf-8")
            log = output.with_suffix(".log")
            log.write_text(json.dumps({"record_type": "example_result", "example_id": "a",
                                       "attempts": [request(999, "error"), request(2.5)]}) + "\n", encoding="utf-8")
            with (patch.object(ba, "DEFAULT_METRICS_DIR", str(root)),
                  patch.object(ba, "DEFAULT_LOGS_DIR", str(root)),
                  patch.object(ba, "ensure_data_layout"),
                  patch.object(ba, "refresh_agreement_summary_best_effort")):
                ba.process_metrics_only_output(str(output), argparse.Namespace(confusion_heatmap=False), False, None)
            payload = json.loads(next(root.glob("*metrics.json")).read_text(encoding="utf-8"))
            self.assertEqual(payload["mean_request_duration_seconds"], 2.5)
            self.assertEqual(payload["mean_request_duration_source"], "logs")
            self.assertEqual(payload["request_duration_sample_count"], 1)

    def test_metrics_only_can_estimate_from_saved_runtime_without_a_log(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "fixture__openai__model__2026-11-25-12-00.csv"
            output.write_text("ID;prediction;truth\na;N;N\n", encoding="utf-8")
            metrics_path = root / (output.stem + "__metrics.json")
            metrics_path.write_text(json.dumps({"overall_time_seconds": 10, "run_config": {"threads": 4}}), encoding="utf-8")
            with (patch.object(ba, "DEFAULT_METRICS_DIR", str(root)),
                  patch.object(ba, "DEFAULT_LOGS_DIR", str(root)),
                  patch.object(ba, "ensure_data_layout"),
                  patch.object(ba, "refresh_agreement_summary_best_effort")):
                ba.process_metrics_only_output(str(output), argparse.Namespace(confusion_heatmap=False), False, None)
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["mean_request_duration_seconds"], 40)
            self.assertEqual(payload["mean_request_duration_source"], "estimated_elapsed_threads")
            self.assertEqual(payload["overall_time_seconds"], 10)


if __name__ == "__main__":
    unittest.main()

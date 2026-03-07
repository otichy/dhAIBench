import argparse
import csv
import json
import os
import tempfile
import unittest
from typing import Any, Dict, List
from unittest.mock import patch

import benchmark_agent as ba


def _build_args(
    *,
    threads: int = 1,
    prompt_log_detail: str = "full",
    flush_rows: int = 2,
    flush_seconds: float = 60.0,
) -> argparse.Namespace:
    return argparse.Namespace(
        few_shot_examples=0,
        cache_pad_target_tokens=0,
        provider="openai",
        model="gpt-4o-mini",
        api_key_var="OPENAI_API_KEY",
        api_base_var="OPENAI_BASE_URL",
        gemini_cached_content=None,
        prompt_layout="compact",
        temperature=0.0,
        top_p=1.0,
        top_k=None,
        verbosity=None,
        service_tier="standard",
        logprobs=False,
        reasoning_effort=None,
        thinking_level=None,
        effort=None,
        system_prompt="test prompt",
        enable_cot=False,
        max_retries=1,
        retry_delay=0.0,
        validator_prompt_max_candidates=50,
        validator_prompt_max_chars=8000,
        validator_exhausted_policy="accept_blank_confidence",
        strict_control_acceptance=False,
        prompt_cache_key=None,
        requesty_auto_cache=None,
        prompt_log_detail=prompt_log_detail,
        flush_rows=flush_rows,
        flush_seconds=flush_seconds,
        threads=threads,
    )


def _write_input(path: str, ids: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["ID", "leftContext", "node", "rightContext", "info"])
        for example_id in ids:
            writer.writerow([example_id, "L", f"N_{example_id}", "R", ""])


def _prediction_for(example: ba.Example) -> ba.Prediction:
    return ba.Prediction(
        label="X",
        explanation="ok",
        confidence=0.5,
        raw_response='{"label":"X","confidence":0.5}',
        prompt_tokens=10,
        completion_tokens=2,
        total_tokens=12,
        node_echo=example.node,
        span_source=ba.SPAN_SOURCE_NODE,
        shared_prefix_tokens_estimate=100,
        variable_prompt_tokens_estimate=20,
    )


class PromptLogNdjsonTests(unittest.TestCase):
    def test_main_defaults_logprobs_off_and_new_flush_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_csv = os.path.join(tmpdir, "existing_output.csv")
            with open(output_csv, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["ID", "prediction"], delimiter=";")
                writer.writeheader()
                writer.writerow({"ID": "1", "prediction": "NOUN"})

            captured: Dict[str, Any] = {}

            def fake_metrics(
                output_path: str,
                args: argparse.Namespace,
                calibration_enabled: bool,
                label_map: Any,
            ):
                captured["args"] = args
                self.assertEqual(output_path, output_csv)
                self.assertFalse(calibration_enabled)
                self.assertIsNone(label_map)
                return (0, 0, 0)

            with patch.object(ba, "process_metrics_only_output", side_effect=fake_metrics):
                exit_code = ba.main(["--metrics_only", "--input", output_csv])

            self.assertEqual(exit_code, 0)
            parsed_args = captured["args"]
            self.assertFalse(parsed_args.logprobs)
            self.assertEqual(parsed_args.prompt_log_detail, "full")
            self.assertEqual(parsed_args.flush_rows, 100)
            self.assertEqual(parsed_args.flush_seconds, 2.0)

    def test_prompt_log_writer_emits_ndjson(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "run.log")
            writer = ba.PromptLogWriter(log_path, flush_rows=2, flush_seconds=60.0)
            writer.write_record({"record_type": "run_metadata", "timestamp": ba.utc_timestamp()})
            writer.write_record({"record_type": "run_command", "command": "python benchmark_agent.py"})
            writer.write_record({"record_type": "example_result", "example_id": "x1", "attempts": []})
            writer.close()

            with open(log_path, "r", encoding="utf-8") as handle:
                lines = [line.strip() for line in handle if line.strip()]
            self.assertEqual(len(lines), 3)
            for line in lines:
                self.assertIsInstance(json.loads(line), dict)

    def test_compact_attempt_logs_drop_heavy_text_fields(self) -> None:
        attempt_logs = [
            {
                "attempt": 1,
                "timestamp": ba.utc_timestamp(),
                "request": [{"role": "user", "content": "very large request"}],
                "response": {"text": "very large response", "prompt_tokens": 10},
                "status": "success",
                "parsed_payload": {"label": "X", "confidence": 0.5},
                "validator_result": {
                    "action": "retry",
                    "reason": "bad label",
                    "retry": {"allowed_labels": ["X", "Y"], "instruction": "try again"},
                },
            }
        ]
        compact = ba.prepare_attempt_logs_for_storage(attempt_logs, prompt_log_detail="compact")
        self.assertEqual(len(compact), 1)
        self.assertNotIn("request", compact[0])
        self.assertIn("response", compact[0])
        self.assertNotIn("text", compact[0]["response"])
        self.assertEqual(compact[0]["validator_result"]["retry_allowed_labels_count"], 2)

    def test_resume_migrates_legacy_prompt_log_to_ndjson(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.csv")
            output_path = os.path.join(tmpdir, "out.csv")
            log_path = ba.build_artifact_path(output_path, ".log", ba.DEFAULT_LOGS_DIR)
            _write_input(input_path, ["id1", "id2"])

            with open(output_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter=";")
                writer.writerow(["ID", "prediction"])
                writer.writerow(["id1", "existing"])

            with open(log_path, "w", encoding="utf-8") as handle:
                json.dump(
                    [
                        {
                            "record_type": "run_command",
                            "timestamp": ba.utc_timestamp(),
                            "resume_mode": False,
                            "reason": "initial_run",
                            "command": "python benchmark_agent.py --threads 1 --model old",
                            "argv": ["benchmark_agent.py", "--threads", "1", "--model", "old"],
                        }
                    ],
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )

            def fake_classify_example(*_args: Any, **kwargs: Any):
                example = kwargs["example"]
                prediction = _prediction_for(example)
                attempt_logs = [
                    {
                        "attempt": 1,
                        "timestamp": ba.utc_timestamp(),
                        "request": [{"role": "user", "content": "payload"}],
                        "response": {"text": "x", "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                        "status": "success",
                        "parsed_payload": {"label": "X", "confidence": 0.5},
                    }
                ]
                return prediction, attempt_logs

            args = _build_args(prompt_log_detail="full", flush_rows=1, flush_seconds=0.0)
            with patch.object(ba, "classify_example", side_effect=fake_classify_example):
                ba.process_dataset(
                    connector=object(),
                    input_path=input_path,
                    output_path=output_path,
                    args=args,
                    include_explanation=True,
                    calibration_enabled=False,
                    label_map=None,
                    resolved_api_base_url=None,
                    validator_client=None,
                    before_example_hook=None,
                    run_command="python benchmark_agent.py --threads 2 --model gpt-4o-mini",
                    run_command_argv=[
                        "benchmark_agent.py",
                        "--threads",
                        "2",
                        "--model",
                        "gpt-4o-mini",
                    ],
                )

            self.assertTrue(os.path.exists(log_path + ".legacy.json"))
            self.assertEqual(ba.detect_prompt_log_format(log_path), "ndjson")
            records = list(ba.iter_prompt_log_records(log_path))
            self.assertTrue(any(record.get("record_type") == "example_result" for record in records))
            run_commands = [
                record
                for record in records
                if isinstance(record, dict) and record.get("record_type") == "run_command"
            ]
            self.assertEqual(len(run_commands), 2)

    def test_migrate_corrupted_legacy_json_array_recovers_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "corrupted.log")
            with open(log_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "[\n"
                    '{"record_type":"run_command","command":"python benchmark_agent.py --model old"},\n'
                    '{"record_type":"example_result","example_id":"id1","attempts":[{"attempt":1,"status":"success"}]},\n'
                    '{"record_type":"example_result","example_id":'  # intentionally truncated tail
                )

            migrated = ba.migrate_legacy_prompt_log_to_ndjson(log_path)
            self.assertTrue(migrated)
            self.assertEqual(ba.detect_prompt_log_format(log_path), "ndjson")
            self.assertTrue(os.path.exists(log_path + ".legacy.json"))

            records = list(ba.iter_prompt_log_records(log_path))
            self.assertGreaterEqual(len(records), 2)
            self.assertEqual(records[0].get("record_type"), "run_command")


if __name__ == "__main__":
    unittest.main()

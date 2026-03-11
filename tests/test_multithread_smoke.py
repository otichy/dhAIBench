import argparse
import csv
import logging
import os
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from typing import Any, Dict, List
from unittest.mock import patch

import benchmark_agent as ba


def _write_input_csv(path: str, ids: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["ID", "leftContext", "node", "rightContext", "info"])
        for example_id in ids:
            writer.writerow([example_id, f"L_{example_id}", f"N_{example_id}", f"R_{example_id}", ""])


def _read_output_ids(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        return [str(row.get("ID", "")) for row in reader]


def _read_json_log(path: str) -> List[Dict[str, Any]]:
    return list(ba.iter_prompt_log_records(path))


def _build_args(threads: int) -> argparse.Namespace:
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
        prompt_cache_key="shared-cache-key",
        requesty_auto_cache=True,
        threads=threads,
        reprompt_unclassified=False,
    )


class _DummyConnector:
    pass


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


class MultithreadSmokeTests(unittest.TestCase):
    def _make_stub(self, ids: List[str]) -> Any:
        index_by_id = {example_id: index for index, example_id in enumerate(ids)}
        delays = {
            example_id: float(len(ids) - index) * 0.01 for example_id, index in index_by_id.items()
        }
        lock = threading.Lock()
        call_records: List[Dict[str, Any]] = []

        def fake_classify_example(*_args: Any, **kwargs: Any):
            example = kwargs["example"]
            time.sleep(delays.get(example.example_id, 0.0))
            with lock:
                call_records.append(
                    {
                        "id": example.example_id,
                        "thread_id": threading.get_ident(),
                        "prompt_cache_key": kwargs.get("prompt_cache_key"),
                        "gemini_cached_content": kwargs.get("gemini_cached_content"),
                        "requesty_auto_cache": kwargs.get("requesty_auto_cache"),
                    }
                )
            prediction = ba.Prediction(
                label=f"label_{example.example_id}",
                explanation=f"explanation_{example.example_id}",
                confidence=0.5,
                raw_response="{}",
                prompt_tokens=10,
                completion_tokens=2,
                total_tokens=12,
                node_echo=example.node,
                span_source=ba.SPAN_SOURCE_NODE,
                shared_prefix_tokens_estimate=1200,
                variable_prompt_tokens_estimate=40,
            )
            attempt_logs = [
                {
                    "attempt": 1,
                    "timestamp": ba.utc_timestamp(),
                    "status": "success",
                    "response": {"usage_metadata": {}},
                }
            ]
            return prediction, attempt_logs

        return fake_classify_example, call_records

    def test_multithreaded_output_preserves_input_order(self) -> None:
        ids = ["id1", "id2", "id3", "id4", "id5", "id6"]
        args = _build_args(threads=4)
        stub, call_records = self._make_stub(ids)

        with tempfile.TemporaryDirectory() as tmpdir:
            with _isolated_data_dirs(tmpdir):
                input_path = os.path.join(tmpdir, "input.csv")
                output_path = os.path.join(tmpdir, "output.csv")
                log_path = ba.build_artifact_path(output_path, ".log", ba.DEFAULT_LOGS_DIR)
                metrics_path = ba.build_artifact_path(output_path, "_metrics.json", ba.DEFAULT_METRICS_DIR)
                _write_input_csv(input_path, ids)

                with patch.object(ba, "classify_example", side_effect=stub):
                    _, _, _, halted = ba.process_dataset(
                        connector=_DummyConnector(),
                        input_path=input_path,
                        output_path=output_path,
                        args=args,
                        include_explanation=True,
                        calibration_enabled=False,
                        label_map=None,
                        resolved_api_base_url=None,
                        validator_client=None,
                        before_example_hook=None,
                        run_command="python benchmark_agent.py --threads 4 --input input.csv --model gpt-4o-mini",
                        run_command_argv=[
                            "benchmark_agent.py",
                            "--threads",
                            "4",
                            "--input",
                            "input.csv",
                            "--model",
                            "gpt-4o-mini",
                        ],
                    )

                output_ids = _read_output_ids(output_path)
                log_records = _read_json_log(log_path)
                run_command_records = [
                    record
                    for record in log_records
                    if isinstance(record, dict) and record.get("record_type") == "run_command"
                ]
                self.assertFalse(halted)
                self.assertEqual(output_ids, ids)
                self.assertEqual(len(call_records), len(ids))
                self.assertGreaterEqual(len({record["thread_id"] for record in call_records}), 2)
                self.assertTrue(
                    all(record["prompt_cache_key"] == "shared-cache-key" for record in call_records)
                )
                self.assertTrue(all(record["gemini_cached_content"] is None for record in call_records))
                self.assertTrue(all(record["requesty_auto_cache"] is True for record in call_records))
                self.assertEqual(len(run_command_records), 1)
                self.assertTrue(os.path.exists(metrics_path))
                with open(metrics_path, "r", encoding="utf-8") as handle:
                    metrics_payload = ba.json.load(handle)
                self.assertFalse(metrics_payload.get("label_metrics_available", True))
                self.assertNotIn("accuracy", metrics_payload)
                self.assertEqual(metrics_payload.get("prediction_count"), len(ids))
                self.assertEqual(metrics_payload.get("truth_label_count"), 0)
                self.assertEqual(metrics_payload.get("task_name"), "output")
                self.assertEqual(metrics_payload.get("task_description"), "")
                self.assertEqual(metrics_payload.get("tags"), "")
                model_details = metrics_payload.get("model_details")
                self.assertIsInstance(model_details, dict)
                self.assertIn("provider", model_details)
                self.assertIn("model_requested", model_details)
                self.assertIn("model_for_requests", model_details)
                self.assertIn("api_base_url", model_details)
                self.assertIn("chat_completions_endpoint", model_details)
                self.assertEqual(
                    run_command_records[0].get("reason"),
                    "initial_run",
                )
                self.assertEqual(
                    run_command_records[0].get("command"),
                    "python benchmark_agent.py --threads 4 --input input.csv --model gpt-4o-mini",
                )

    def test_resume_keeps_order_and_processes_only_missing_ids(self) -> None:
        ids = ["id1", "id2", "id3", "id4", "id5"]
        args = _build_args(threads=3)
        stub, call_records = self._make_stub(ids)

        with tempfile.TemporaryDirectory() as tmpdir:
            with _isolated_data_dirs(tmpdir):
                input_path = os.path.join(tmpdir, "input.csv")
                output_path = os.path.join(tmpdir, "output.csv")
                log_path = ba.build_artifact_path(output_path, ".log", ba.DEFAULT_LOGS_DIR)
                _write_input_csv(input_path, ids)

                with open(output_path, "w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle, delimiter=";")
                    writer.writerow(["ID", "prediction"])
                    writer.writerow(["id1", "existing_label_1"])
                    writer.writerow(["id2", "existing_label_2"])
                with open(log_path, "w", encoding="utf-8") as handle:
                    ba.json.dump(
                        [
                            {
                                "record_type": "run_command",
                                "timestamp": ba.utc_timestamp(),
                                "resume_mode": False,
                                "reason": "initial_run",
                                "command": "python benchmark_agent.py --threads 1 --input input.csv --model old-model",
                                "argv": ["benchmark_agent.py", "--threads", "1", "--input", "input.csv", "--model", "old-model"],
                            }
                        ],
                        handle,
                        ensure_ascii=False,
                        indent=2,
                    )

                with patch.object(ba, "classify_example", side_effect=stub):
                    _, _, _, halted = ba.process_dataset(
                        connector=_DummyConnector(),
                        input_path=input_path,
                        output_path=output_path,
                        args=args,
                        include_explanation=True,
                        calibration_enabled=False,
                        label_map=None,
                        resolved_api_base_url=None,
                        validator_client=None,
                        before_example_hook=None,
                        run_command="python benchmark_agent.py --threads 3 --input input.csv --model gpt-4o-mini",
                        run_command_argv=[
                            "benchmark_agent.py",
                            "--threads",
                            "3",
                            "--input",
                            "input.csv",
                            "--model",
                            "gpt-4o-mini",
                        ],
                    )

                output_ids = _read_output_ids(output_path)
                log_records = _read_json_log(log_path)
                run_command_records = [
                    record
                    for record in log_records
                    if isinstance(record, dict) and record.get("record_type") == "run_command"
                ]
                processed_ids = {record["id"] for record in call_records}
                self.assertFalse(halted)
                self.assertEqual(output_ids, ids)
                self.assertEqual(processed_ids, {"id3", "id4", "id5"})
                self.assertEqual(len(call_records), 3)
                self.assertEqual(len(run_command_records), 2)
                self.assertEqual(
                    run_command_records[-1].get("reason"),
                    "resume_command_changed",
                )
                self.assertEqual(
                    run_command_records[-1].get("command"),
                    "python benchmark_agent.py --threads 3 --input input.csv --model gpt-4o-mini",
                )

    def test_resume_reprompts_only_unclassified_rows(self) -> None:
        ids = ["id1", "id2", "id3"]
        args = _build_args(threads=1)
        args.reprompt_unclassified = True
        call_ids: List[str] = []

        def stub_unclassified_only(*_args: Any, **kwargs: Any):
            example = kwargs["example"]
            call_ids.append(example.example_id)
            prediction = ba.Prediction(
                label="fixed_label",
                explanation="fixed explanation",
                confidence=0.75,
                raw_response="{}",
                prompt_tokens=9,
                completion_tokens=3,
                total_tokens=12,
                node_echo=example.node,
                span_source=ba.SPAN_SOURCE_NODE,
                shared_prefix_tokens_estimate=1200,
                variable_prompt_tokens_estimate=40,
            )
            attempt_logs = [
                {
                    "attempt": 1,
                    "timestamp": ba.utc_timestamp(),
                    "status": "success",
                    "response": {"usage_metadata": {}},
                }
            ]
            return prediction, attempt_logs

        with tempfile.TemporaryDirectory() as tmpdir:
            with _isolated_data_dirs(tmpdir):
                input_path = os.path.join(tmpdir, "input.csv")
                output_path = os.path.join(tmpdir, "output.csv")
                _write_input_csv(input_path, ids)

                with open(output_path, "w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle, delimiter=";")
                    writer.writerow(["ID", "leftContext", "node", "rightContext", "info", "truth", "prediction"])
                    writer.writerow(["id1", "L_id1", "N_id1", "R_id1", "", "", "singular"])
                    writer.writerow(["id2", "L_id2", "N_id2", "R_id2", "", "", "unclassified"])
                    writer.writerow(["id3", "L_id3", "N_id3", "R_id3", "", "", "plural"])

                with patch.object(ba, "classify_example", side_effect=stub_unclassified_only):
                    _, _, _, halted = ba.process_dataset(
                        connector=_DummyConnector(),
                        input_path=input_path,
                        output_path=output_path,
                        args=args,
                        include_explanation=True,
                        calibration_enabled=False,
                        label_map=None,
                        resolved_api_base_url=None,
                        validator_client=None,
                        before_example_hook=None,
                        run_command="python benchmark_agent.py --unclassified",
                        run_command_argv=["benchmark_agent.py", "--unclassified"],
                    )

                self.assertFalse(halted)
                self.assertEqual(call_ids, ["id2"])

                with open(output_path, "r", encoding="utf-8-sig", newline="") as handle:
                    rows = list(csv.DictReader(handle, delimiter=";"))
                self.assertEqual(len(rows), 3)
                id2_rows = [row for row in rows if row.get("ID") == "id2"]
                self.assertEqual(len(id2_rows), 1)
                self.assertEqual(id2_rows[0].get("prediction"), "fixed_label")


if __name__ == "__main__":
    unittest.main()

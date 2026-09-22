import argparse
import base64
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import benchmark_agent as ba


CRITERIA = {"NOUN": "A noun", "VERB": "A verb"}

def encoded(value):
    return base64.b64encode(json.dumps(value, ensure_ascii=False).encode()).decode()


class JevTests(unittest.TestCase):
    def test_criteria_validation(self):
        self.assertEqual(ba.decode_decision_criteria(encoded(CRITERIA)), CRITERIA)
        for invalid in ({}, {"NOUN": "noun"}, {"": "noun", "V": "verb"},
                        {"N": "noun", "V": 2}, ["N", "V"]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                ba.decode_decision_criteria(encoded(invalid))
        duplicate = base64.b64encode(b'{"N": "noun", " N ": "other"}').decode()
        with self.assertRaises(ValueError):
            ba.decode_decision_criteria(duplicate)

    def test_prediction_and_wire_request(self):
        answer = {"classification": {"type": "choice", "choice": "NOUN",
                  "confidence": 0.99, "probabilities": {"NOUN": 0.7, "VERB": 0.3}}}
        connector = ba.OpenAIConnector.__new__(ba.OpenAIConnector)
        connector.decision_criteria = CRITERIA
        connector._provider = "requesty"
        connector.client_type = "chat_v1"
        connector._request_timeout_seconds = 20
        connector._refresh_access_token_if_needed = Mock()
        connector._throttle_request_if_needed = Mock()
        usage = SimpleNamespace(prompt_tokens=30, completion_tokens=10, total_tokens=40,
                                model_dump=lambda: {"prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40})
        create = Mock(return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(answer)))], usage=usage))
        connector._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        prediction, logs = ba.classify_example(
            connector=connector, example=ba.Example("1", "the", "cat", "sleeps", truth="SECRET_TRUTH"),
            model="typesafe/jev-1.13.0", temperature=None, top_p=None, top_k=None,
            verbosity=None, service_tier="standard", include_logprobs=False,
            reasoning_effort=None, thinking_level=None, effort=None,
            system_prompt="Which part of speech is the marked node?", enable_cot=False,
            include_explanation=False, prompt_layout="standard", few_shot_context=None,
            max_retries=2, retry_delay=0, prompt_log_detail="full")
        self.assertEqual(prediction.label, "NOUN")
        self.assertEqual(prediction.confidence, 0.7)
        self.assertEqual(prediction.explanation, "")
        self.assertIsNone(prediction.node_echo)
        self.assertIsNone(prediction.span_source)
        self.assertEqual(prediction.total_tokens, 40)
        self.assertEqual(json.loads(prediction.raw_response), answer)
        self.assertEqual(create.call_count, 1)
        request = create.call_args.kwargs
        self.assertEqual(set(request), {"model", "messages", "response_format", "timeout"})
        self.assertEqual([m["role"] for m in request["messages"]], ["user"])
        self.assertNotIn("SECRET_TRUTH", json.dumps(request))
        self.assertEqual(request["response_format"]["questions"]["classification"]["criteria"], CRITERIA)
        self.assertEqual(logs[0]["span_verification"], "unavailable")
        self.assertEqual(logs[0]["parsed_payload"]["decision"]["confidence"], 0.99)

    def test_bad_probability_is_rejected(self):
        for probs in ({"NOUN": 2, "VERB": -1}, {"NOUN": 0.4},
                      {"NOUN": 0.1, "VERB": 0.2}, {"NOUN": True, "VERB": 0},
                      {"NOUN": 0.7, "VERB": 0.289}, {"NOUN": 0.7, "VERB": 0.311},
                      {"NOUN": float("nan"), "VERB": 0.3},
                      {"NOUN": float("inf"), "VERB": 0.3}):
            with self.subTest(probs=probs), self.assertRaises(ValueError):
                ba.parse_jev_answer(json.dumps({"classification": {
                    "type": "choice", "choice": "NOUN", "probabilities": probs}}), CRITERIA)

    def test_probability_rounding_boundaries(self):
        for other_probability in (0.29, 0.3, 0.31):
            with self.subTest(other_probability=other_probability):
                answer = {"type": "choice", "choice": "NOUN",
                          "probabilities": {"NOUN": 0.7, "VERB": other_probability}}
                parsed = ba.parse_jev_answer(json.dumps({"classification": answer}), CRITERIA)
                self.assertEqual(parsed["confidence"], 0.7)
                self.assertEqual(parsed["decision"], answer)

    def test_invalid_answer_retries_then_falls_back_or_recovers(self):
        valid = {"type": "choice", "choice": "NOUN",
                 "probabilities": {"NOUN": 0.7, "VERB": 0.3}}
        invalid_answers = [
            {**valid, "probabilities": {"NOUN": 0.1, "VERB": 0.2}},
            {**valid, "choice": "UNKNOWN"},
            {"type": "choice", "choice": "NOUN"},
            {"type": "unknown"},
        ]
        for invalid in invalid_answers:
            for recover in (False, True):
                with self.subTest(invalid=invalid, recover=recover):
                    responses = [json.dumps({"classification": answer})
                                 for answer in (invalid, valid if recover else invalid)]
                    connector = SimpleNamespace(
                        decision_criteria=CRITERIA,
                        request_timeout_seconds=30.0,
                        complete=Mock(side_effect=[ba.CompletionResult(
                            text=raw, prompt_tokens=30, completion_tokens=10, total_tokens=40)
                            for raw in responses]))
                    prediction, logs = ba.classify_example(
                        connector=connector, example=ba.Example("671", "the", "cat", "sleeps"),
                        model="typesafe/jev-1.13.0", temperature=None, top_p=None, top_k=None,
                        verbosity=None, service_tier="standard", include_logprobs=False,
                        reasoning_effort=None, thinking_level=None, effort=None,
                        system_prompt="Which class?", enable_cot=False,
                        include_explanation=False, prompt_layout="standard", few_shot_context=None,
                        max_retries=2, retry_delay=0, prompt_log_detail="full")
                    self.assertEqual(connector.complete.call_count, 2)
                    self.assertEqual(len(logs), 2)
                    self.assertEqual(logs[0]["error_type"], "JevAnswerValidationError")
                    self.assertEqual(prediction.raw_response, responses[-1])
                    self.assertEqual(prediction.total_tokens, 40)
                    if recover:
                        self.assertEqual(prediction.label, "NOUN")
                        self.assertEqual(prediction.confidence, 0.7)
                    else:
                        self.assertEqual(prediction.label, "unclassified")
                        self.assertIsNone(prediction.confidence)
                        self.assertEqual(prediction.validator_status, "accepted_after_parse_error")
                        self.assertTrue(prediction.validator_reason)

    def test_load_params_restores_jev_settings_before_execution(self):
        config = {"provider": "requesty", "model": "typesafe/jev-1.13.0",
                  "system_prompt": "Which class?", "decision_criteria_b64": encoded(CRITERIA),
                  "input": ["example_input.csv"]}
        class StopBeforeExecution(Exception):
            pass
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "metrics.json"
            source.write_text(json.dumps({"run_config": config}), encoding="utf-8")
            with patch.object(ba, "ensure_data_layout", side_effect=StopBeforeExecution), \
                 patch.object(ba, "build_run_config_snapshot", wraps=ba.build_run_config_snapshot) as snapshot:
                with self.assertRaises(StopBeforeExecution):
                    ba.main(["--load_params", str(source)])
                args = snapshot.call_args.args[0]
                self.assertTrue(args.no_explanation)
                self.assertEqual(ba.decode_decision_criteria(args.decision_criteria_b64), CRITERIA)
                self.assertEqual(args.system_prompt, "Which class?")

    def test_config_recovers_criteria_and_confidence_source(self):
        args = argparse.Namespace(model="typesafe/jev-1.13.0", system_prompt="Which class?",
                                  decision_criteria_b64=encoded(CRITERIA))
        snapshot = ba.build_run_config_snapshot(args)
        self.assertEqual(snapshot["confidence_source"], "choice_probability")
        recovered = ba.normalize_resume_run_config(snapshot)
        self.assertEqual(ba.decode_decision_criteria(recovered["decision_criteria_b64"]), CRITERIA)


if __name__ == "__main__":
    unittest.main()

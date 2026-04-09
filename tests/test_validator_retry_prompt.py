import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import benchmark_agent as ba


class _RecordingConnector:
    def __init__(self) -> None:
        self.calls = []
        self._responses = [
            ba.CompletionResult(
                text=json.dumps(
                    {
                        "label": "bad",
                        "confidence": 0.8,
                        "node_echo": "node",
                        "span_source": "node",
                    }
                )
            ),
            ba.CompletionResult(
                text=json.dumps(
                    {
                        "label": "good",
                        "confidence": 0.9,
                        "node_echo": "node",
                        "span_source": "node",
                    }
                )
            ),
        ]

    def complete(self, **kwargs):
        self.calls.append(kwargs["messages"])
        return self._responses[len(self.calls) - 1]


class _RetryThenAcceptValidator:
    def __init__(self) -> None:
        self.requests = []

    def validate(self, payload):
        self.requests.append(payload)
        if len(self.requests) == 1:
            return {
                "type": "result",
                "schema_version": 1,
                "request_id": payload["request_id"],
                "action": "retry",
                "reason": "not_in_lexicon",
                "retry": {
                    "allowed_labels": ["good"],
                    "instruction": "ignored fallback instruction",
                    "message": 'The previous label "bad" is not accepted.\nChoose one of the validator suggestions.',
                },
            }
        return {
            "type": "result",
            "schema_version": 1,
            "request_id": payload["request_id"],
            "action": "accept",
            "reason": "ok",
        }


class ValidatorRetryPromptTests(unittest.TestCase):
    def test_render_validator_retry_message_uses_custom_message_when_present(self) -> None:
        text = ba.render_validator_retry_message(
            allowed_labels=["good", "better"],
            instruction="fallback instruction",
            retry_message="Custom retry context.",
            max_candidates=10,
            max_chars=5000,
        )
        self.assertTrue(text.startswith("Custom retry context."))
        self.assertNotIn("External validator rejected the previous label.", text)
        self.assertIn('"good"', text)
        self.assertIn('You MUST set "label" to exactly one item in allowed_labels', text)

    def test_render_validator_retry_message_falls_back_to_instruction(self) -> None:
        text = ba.render_validator_retry_message(
            allowed_labels=["good"],
            instruction="Choose the corrected lemma.",
            max_candidates=10,
            max_chars=5000,
        )
        self.assertTrue(text.startswith("External validator rejected the previous label."))
        self.assertIn("Choose the corrected lemma.", text)

    def test_classify_example_appends_custom_validator_retry_message(self) -> None:
        connector = _RecordingConnector()
        validator = _RetryThenAcceptValidator()
        example = ba.Example(
            example_id="e1",
            left_context="left",
            node="node",
            right_context="right",
            info="N",
        )

        prediction, attempt_logs = ba.classify_example(
            connector=connector,  # type: ignore[arg-type]
            example=example,
            model="gpt-4o-mini",
            temperature=0.0,
            top_p=1.0,
            top_k=None,
            verbosity=None,
            service_tier=None,
            include_logprobs=False,
            reasoning_effort=None,
            thinking_level=None,
            effort=None,
            system_prompt="test prompt",
            enable_cot=False,
            include_explanation=False,
            prompt_layout="compact",
            few_shot_context=None,
            max_retries=2,
            retry_delay=0.0,
            validator_client=validator,  # type: ignore[arg-type]
            prompt_log_detail="full",
        )

        self.assertEqual(prediction.label, "good")
        self.assertEqual(len(connector.calls), 2)
        expected_retry_message = ba.render_validator_retry_message(
            allowed_labels=["good"],
            instruction="ignored fallback instruction",
            retry_message='The previous label "bad" is not accepted.\nChoose one of the validator suggestions.',
            max_candidates=50,
            max_chars=8000,
        )
        self.assertEqual(connector.calls[1][-1]["content"], expected_retry_message)
        self.assertEqual(attempt_logs[0]["status"], "validator_retry")


if __name__ == "__main__":
    unittest.main()

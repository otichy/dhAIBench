import unittest

import benchmark_agent as ba


class _ProviderError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class _AlwaysCancelledConnector:
    request_timeout_seconds = 30.0

    def complete(self, **_kwargs):
        raise _ProviderError(
            "Error code: 499 - [{'error': {'code': 499, 'message': 'The operation was cancelled.', 'status': 'CANCELLED'}}]",
            status_code=499,
        )


class ErrorHandlingTests(unittest.TestCase):
    def test_detects_retryable_provider_cancellation_error(self) -> None:
        exc = _ProviderError("The operation was cancelled.", status_code=499)
        self.assertTrue(ba.is_retryable_provider_cancellation_error(exc))

    def test_classify_example_falls_back_on_provider_cancellation(self) -> None:
        connector = _AlwaysCancelledConnector()
        example = ba.Example(
            example_id="e1",
            left_context="left",
            node=f"{ba.NODE_MARKER_LEFT}node{ba.NODE_MARKER_RIGHT}",
            right_context="right",
            info="",
        )

        prediction, attempt_logs = ba.classify_example(
            connector=connector,  # type: ignore[arg-type]
            example=example,
            model="gemini-2.5-flash",
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
            include_explanation=True,
            prompt_layout="compact",
            few_shot_context=None,
            max_retries=2,
            retry_delay=0.0,
        )

        self.assertEqual(prediction.label, "unclassified")
        self.assertIsNone(prediction.confidence)
        self.assertEqual(
            prediction.validator_status,
            "accepted_after_provider_cancellation",
        )
        self.assertEqual(len(attempt_logs), 2)
        self.assertTrue(all(log.get("error_category") == "provider_cancellation" for log in attempt_logs))


if __name__ == "__main__":
    unittest.main()

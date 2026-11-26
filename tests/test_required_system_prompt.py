import argparse
import base64
from contextlib import redirect_stderr
import io
import json
import unittest
from unittest.mock import mock_open, patch

import benchmark_agent as ba


PROMPT = "Classify grammatical number. Use singular or plural."
RUN = ["--input", "example_input.csv", "--model", "test-model"]


class StopBeforeExecution(Exception):
    pass


class RequiredSystemPromptTests(unittest.TestCase):
    def assert_rejected(self, argv, message="task-specific system prompt"):
        with (
            patch.object(ba, "ensure_data_layout") as layout,
            patch.object(ba, "parse_env_file") as credentials,
            patch.object(ba, "OpenAIConnector") as connector,
            redirect_stderr(io.StringIO()) as stderr,
        ):
            with self.assertRaises(SystemExit) as raised:
                ba.main(argv)
            self.assertEqual(raised.exception.code, 2)
            self.assertIn(message, stderr.getvalue())
            layout.assert_not_called()
            credentials.assert_not_called()
            connector.assert_not_called()

    def assert_accepted(self, argv, expected_prompt=None):
        with (
            patch.object(ba, "ensure_data_layout", side_effect=StopBeforeExecution),
            patch.object(ba, "build_run_config_snapshot", wraps=ba.build_run_config_snapshot) as snapshot,
        ):
            with self.assertRaises(StopBeforeExecution):
                ba.main(argv)
            args = snapshot.call_args.args[0]
            self.assertEqual(args.system_prompt, expected_prompt)
            self.assertEqual(args._run_config_snapshot["system_prompt"], expected_prompt or "")

    def test_missing_blank_and_legacy_prompts_fail_before_execution(self):
        for value in (None, "", " \t\n", r"\n\t", ba.LEGACY_DEFAULT_SYSTEM_PROMPT):
            with self.subTest(value=value):
                flags = [] if value is None else ["--system_prompt", value]
                self.assert_rejected(RUN + flags)

    def test_missing_prompt_is_rejected_for_timeout_probe(self):
        self.assert_rejected(RUN + ["--timeout_probe"])

    def test_invalid_or_blank_encoded_prompt_never_falls_back(self):
        for encoded in ("", "not-base64!", "/w==", "IAkK",
                        base64.b64encode(ba.LEGACY_DEFAULT_SYSTEM_PROMPT.encode()).decode()):
            with self.subTest(encoded=encoded):
                self.assert_rejected(RUN + ["--system_prompt_b64", encoded], "prompt")

    def test_plain_and_encoded_prompts_are_accepted(self):
        for prompt in (PROMPT, PROMPT + "\nUse the context as evidence."):
            with self.subTest(prompt=prompt):
                self.assert_accepted(RUN + ["--system_prompt", prompt], prompt)
                encoded = base64.b64encode(prompt.encode()).decode()
                self.assert_accepted(RUN + ["--system_prompt_b64", encoded], prompt)

    def test_non_model_modes_do_not_require_a_prompt(self):
        for flags in (["--metrics_only"], ["--update-models"],
                      ["--summarize-log-errors", "run.log"]):
            with self.subTest(flags=flags):
                self.assert_accepted(flags)

    def test_loaded_prompts_and_explicit_overrides(self):
        encoded = base64.b64encode(PROMPT.encode()).decode()
        cases = [
            ({"system_prompt": PROMPT}, [], PROMPT),
            ({"system_prompt_b64": encoded}, [], PROMPT),
            ({"system_prompt_b64": encoded}, ["--system_prompt", "Override task."], "Override task."),
            ({"system_prompt": PROMPT}, ["--system_prompt_b64", encoded], PROMPT),
            ({}, [], None),
            ({"system_prompt": ba.LEGACY_DEFAULT_SYSTEM_PROMPT}, [], None),
            ({"system_prompt": PROMPT}, ["--system_prompt", ""], None),
            ({"system_prompt": PROMPT}, ["--system_prompt_b64", ""], None),
        ]
        for saved, overrides, expected in cases:
            with self.subTest(saved=saved, overrides=overrides):
                payload = json.dumps({"run_config": saved})
                with patch.object(ba.os.path, "isfile", return_value=True), \
                     patch("builtins.open", mock_open(read_data=payload)):
                    argv = RUN + ["--load_params", "saved.json"] + overrides
                    if expected is None:
                        self.assert_rejected(argv, "prompt")
                    else:
                        self.assert_accepted(argv, expected)

    def test_resume_restores_prompt_before_validation(self):
        for saved_prompt in (None, "", ba.LEGACY_DEFAULT_SYSTEM_PROMPT, PROMPT):
            with self.subTest(saved_prompt=saved_prompt):
                saved = {"input": ["example_input.csv"], "model": "test-model"}
                if saved_prompt is not None:
                    saved["system_prompt"] = saved_prompt
                with patch.object(ba.os.path, "isfile", return_value=True), \
                     patch.object(ba, "recover_resume_defaults_from_artifacts", return_value=saved):
                    argv = ["--resume", "--output", "existing.csv"]
                    if saved_prompt == PROMPT:
                        self.assert_accepted(argv, PROMPT)
                    else:
                        self.assert_rejected(argv)
                    self.assert_accepted(argv + ["--system_prompt", PROMPT], PROMPT)

    def test_snapshot_does_not_invent_a_prompt(self):
        self.assertEqual(ba.build_run_config_snapshot(argparse.Namespace())["system_prompt"], "")


if __name__ == "__main__":
    unittest.main()

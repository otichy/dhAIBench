import unittest
from pathlib import Path

import benchmark_agent as ba


def _load_file(name: str) -> str:
    return (Path(__file__).resolve().parents[1] / name).read_text(encoding="utf-8")


def _to_gui_instruction_template(text: str) -> str:
    marker_pair = f"{ba.NODE_MARKER_LEFT}...{ba.NODE_MARKER_RIGHT}"
    rendered = text.replace(
        f"{ba.NODE_MARKER_LEFT}node{ba.NODE_MARKER_RIGHT}",
        "${NODE_MARKER_LEFT}node${NODE_MARKER_RIGHT}",
    )
    rendered = rendered.replace(marker_pair, "${markerPair}")
    return rendered


class PromptGuiSyncTests(unittest.TestCase):
    def test_canonical_markers_are_used(self) -> None:
        self.assertEqual(ba.NODE_MARKER_LEFT, "\u27E6")
        self.assertEqual(ba.NODE_MARKER_RIGHT, "\u27E7")

        sample = ba.Example(
            example_id="1",
            left_context="left",
            node="node",
            right_context="right",
            info="",
        )
        artifacts = ba.build_prompt_artifacts(
            example=sample,
            system_prompt=None,
            enable_cot=False,
            include_explanation=True,
            prompt_layout="standard",
        )
        system_text = artifacts.messages[0]["content"]
        self.assertIn(f"{ba.NODE_MARKER_LEFT} {ba.NODE_MARKER_RIGHT}", system_text)
        self.assertNotIn(ba.LEGACY_NODE_MARKER_LEFT, system_text)

    def test_gui_includes_mandatory_system_append_text(self) -> None:
        shared_js = _load_file("config_gui_shared.js")
        self.assertIn(
            "Classify ONLY the text that is explicitly wrapped inside ${NODE_MARKER_LEFT} ${NODE_MARKER_RIGHT}",
            shared_js,
        )
        self.assertIn("(the 'node' or its marked sub-span). ", shared_js)
        self.assertIn(
            "Use the surrounding context as supporting evidence, but never change the focus away from the highlighted text. ",
            shared_js,
        )
        self.assertIn(
            'If you cannot determine the class/label for the node, return "unclassified".',
            shared_js,
        )

    def test_gui_instruction_lines_stay_in_sync_with_script(self) -> None:
        shared_js = _load_file("config_gui_shared.js")
        combinations = [
            ("standard", False, True),
            ("standard", False, False),
            ("standard", True, True),
            ("compact", False, True),
            ("compact", False, False),
        ]
        for layout, enable_cot, include_explanation in combinations:
            lines = ba.build_user_instruction_lines(
                layout=layout,
                enable_cot=enable_cot,
                include_explanation=include_explanation,
            )
            for line in lines:
                expected_fragment = _to_gui_instruction_template(line)
                self.assertIn(
                    expected_fragment,
                    shared_js,
                    msg=(
                        "Shared GUI prompt source is out of sync with benchmark_agent.py for "
                        f"layout={layout}, enable_cot={enable_cot}, include_explanation={include_explanation}. "
                        f"Missing: {expected_fragment!r}"
                    ),
                )

    def test_main_gui_references_shared_script(self) -> None:
        html = _load_file("config_gui.html")
        self.assertIn('<script src="config_gui_shared.js"></script>', html)
        self.assertIn('data-gui-variant="mode-first"', html)
        self.assertNotIn("Mode-First Preview", html)

    def test_cli_flag_reference_includes_current_mode_and_provider_flags(self) -> None:
        shared_js = _load_file("config_gui_shared.js")
        self.assertIn('const cliFlagReferenceSections = [', shared_js)
        self.assertIn('"--system_prompt_b64"', shared_js)
        self.assertIn('"--resume"', shared_js)
        self.assertIn('"--metrics_only"', shared_js)
        self.assertIn('"--no-confusion_heatmap"', shared_js)
        self.assertIn('"--create_gemini_cache"', shared_js)
        self.assertIn('"--no-gemini_cache_ttl_autoupdate"', shared_js)
        self.assertIn('"--max_retries"', shared_js)
        self.assertIn('"--retry_delay"', shared_js)
        self.assertIn('"--validator_exhausted_policy"', shared_js)
        self.assertIn("renderCliFlagReference(ctx);", shared_js)

    def test_gui_builds_validator_args_from_dedicated_fields(self) -> None:
        shared_js = _load_file("config_gui_shared.js")
        self.assertIn("function buildValidatorArgsValue(data)", shared_js)
        self.assertIn('data.get("validator_lexicon")', shared_js)
        self.assertIn('data.get("validator_max_distance")', shared_js)
        self.assertIn('data.get("validator_max_suggestions")', shared_js)

    def test_metrics_mode_disables_logprobs_control(self) -> None:
        shared_js = _load_file("config_gui_shared.js")
        self.assertIn('logprobsInput: document.getElementById("logprobs")', shared_js)
        self.assertIn('ctx.logprobsInput.disabled = !logprobsEnabled;', shared_js)
        self.assertIn('ctx.logprobsInput.setAttribute("aria-disabled", logprobsEnabled ? "false" : "true");', shared_js)

    def test_classic_builder_omits_logprobs_in_metrics_only_mode(self) -> None:
        shared_js = _load_file("config_gui_shared.js")
        self.assertIn("if (!metricsOnly && data.get(\"logprobs\")) {", shared_js)


if __name__ == "__main__":
    unittest.main()

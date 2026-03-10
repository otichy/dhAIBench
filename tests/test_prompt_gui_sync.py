import unittest
from pathlib import Path

import benchmark_agent as ba


def _load_gui_html() -> str:
    gui_path = Path(__file__).resolve().parents[1] / "config_gui.html"
    return gui_path.read_text(encoding="utf-8")


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
        html = _load_gui_html()
        self.assertIn(
            "Classify ONLY the text that is explicitly wrapped inside ${NODE_MARKER_LEFT} ${NODE_MARKER_RIGHT}",
            html,
        )
        self.assertIn("(the 'node' or its marked sub-span). ", html)
        self.assertIn(
            "Use the surrounding context as supporting evidence, but never change the focus away from the highlighted text. ",
            html,
        )
        self.assertIn(
            'If you cannot determine the class/label for the node, return "unclassified".',
            html,
        )

    def test_gui_instruction_lines_stay_in_sync_with_script(self) -> None:
        html = _load_gui_html()
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
                    html,
                    msg=(
                        "GUI prompt preview is out of sync with benchmark_agent.py for "
                        f"layout={layout}, enable_cot={enable_cot}, include_explanation={include_explanation}. "
                        f"Missing: {expected_fragment!r}"
                    ),
                )


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path


def _load_main_gui_html() -> str:
    return (Path(__file__).resolve().parents[1] / "config_gui.html").read_text(encoding="utf-8")


class ConfigGuiModeFirstTests(unittest.TestCase):
    def test_mode_labels_exist(self) -> None:
        html = _load_main_gui_html()
        self.assertIn('data-mode-choice="run"', html)
        self.assertIn("Run &amp; Validate", html)
        self.assertIn('data-mode-choice="resume"', html)
        self.assertIn("Metrics only", html)
        self.assertIn('data-mode-choice="metrics"', html)
        self.assertIn('data-mode-choice="validator"', html)

    def test_sidebar_has_collapsed_markup(self) -> None:
        html = _load_main_gui_html()
        self.assertIn('class="command-panel is-collapsed"', html)
        self.assertIn('id="sidebar-toggle-button"', html)
        self.assertIn('id="sidebar-mode-label"', html)
        self.assertIn('id="sidebar-flag-count"', html)

    def test_secondary_sections_are_collapsible(self) -> None:
        html = _load_main_gui_html()
        self.assertIn("<summary>Execution</summary>", html)
        self.assertIn("<summary>Provider controls</summary>", html)
        self.assertIn("<summary>Evaluation &amp; Metadata</summary>", html)
        self.assertIn("<summary>Logging</summary>", html)
        self.assertIn("<summary>Inspect Prompt &amp; Cache</summary>", html)
        self.assertIn("<summary>CLI Flag Reference</summary>", html)
        self.assertNotIn('class="section-card" open', html)

    def test_cli_flag_reference_footer_is_present(self) -> None:
        html = _load_main_gui_html()
        self.assertIn('class="reference-wrap"', html)
        self.assertIn('id="cli-flag-reference"', html)
        self.assertIn(
            "All CLI flags this GUI can currently emit. Flags not listed here are still available only through the terminal.",
            html,
        )

    def test_main_gui_omits_legacy_mode_checkboxes(self) -> None:
        html = _load_main_gui_html()
        self.assertNotIn('id="metrics_only"', html)
        self.assertNotIn('name="metrics_only"', html)
        self.assertNotIn('id="validator_enable"', html)
        self.assertNotIn('name="validator_enable"', html)
        self.assertNotIn('id="api_key_var"', html)
        self.assertNotIn('name="api_key_var"', html)
        self.assertNotIn('id="api_base_var"', html)
        self.assertNotIn('name="api_base_var"', html)

    def test_validator_setup_uses_dedicated_validator_arg_fields(self) -> None:
        html = _load_main_gui_html()
        self.assertIn("Validator path", html)
        self.assertNotIn("Validator command", html)
        self.assertIn('id="validator_lexicon"', html)
        self.assertIn('id="validator_max_distance"', html)
        self.assertIn('id="validator_max_distance_per_retry"', html)
        self.assertIn('id="validator_max_suggestions"', html)
        self.assertNotIn('id="validator_args"', html)

    def test_provider_specific_controls_are_badged(self) -> None:
        html = _load_main_gui_html()
        self.assertIn('class="field-tag">OpenAI<', html)
        self.assertIn('class="field-tag">Gemini<', html)
        self.assertIn('class="field-tag">Claude<', html)
        self.assertIn('class="field-tag">Vertex<', html)
        self.assertIn('class="field-tag">Requesty<', html)

    def test_requested_default_toggles_start_checked(self) -> None:
        html = _load_main_gui_html()
        self.assertIn('id="include_explanations" name="include_explanations" checked', html)
        self.assertIn('id="enable_cot" name="enable_cot" checked', html)
        self.assertIn(
            'id="strict_control_acceptance" name="strict_control_acceptance" checked',
            html,
        )
        self.assertIn('id="calibration" name="calibration" checked', html)
        self.assertIn('id="confusion_heatmap" name="confusion_heatmap" checked', html)

    def test_resume_reprompt_switch_is_scoped_to_resume_markup(self) -> None:
        html = _load_main_gui_html()
        self.assertEqual(
            html.count("Resume mode: re-prompt only <code>unclassified</code> rows"),
            1,
        )
        self.assertIn('<div data-mode-visible="resume">', html)

    def test_repeat_unclassified_switch_is_available_in_run_and_resume(self) -> None:
        html = _load_main_gui_html()
        self.assertEqual(
            html.count(
                "Auto-repeat remaining <code>unclassified</code> rows until resolved or stable"
            ),
            1,
        )
        self.assertIn('<div data-mode-visible="run resume">', html)

    def test_execution_section_includes_max_retries(self) -> None:
        html = _load_main_gui_html()
        self.assertIn('id="max_retries"', html)
        self.assertIn('id="retry_delay"', html)

    def test_model_provider_layout_matches_provider_first_refresh_right(self) -> None:
        html = _load_main_gui_html()
        self.assertIn('class="model-provider-layout"', html)
        provider_pos = html.find('>Provider <span class="field-tags"><span class="field-tag">Required</span></span></span>')
        model_pos = html.find('>Model Name <span class="field-tags"><span class="field-tag">Required</span></span></span>')
        refresh_pos = html.find('id="refresh-models-button"')
        self.assertGreaterEqual(provider_pos, 0)
        self.assertGreaterEqual(model_pos, 0)
        self.assertGreaterEqual(refresh_pos, 0)
        self.assertLess(provider_pos, model_pos)
        self.assertLess(model_pos, refresh_pos)

    def test_setup_fields_mark_required_and_optional_states(self) -> None:
        html = _load_main_gui_html()
        self.assertIn('<span class="field-tag">Required</span>', html)
        self.assertIn('id="output-path-mode-badge"', html)
        self.assertIn(">Optional<", html)
        self.assertIn('id="output-path-mode-hint"', html)
        self.assertEqual(html.count("<span>Output CSV Path</span>"), 1)

    def test_field_help_styles_are_present(self) -> None:
        html = _load_main_gui_html()
        self.assertIn(".field-help-button {", html)
        self.assertIn(".field-help-popover {", html)

    def test_hidden_attribute_has_explicit_css_support(self) -> None:
        html = _load_main_gui_html()
        self.assertIn("[hidden] {", html)
        self.assertIn("display: none !important;", html)

    def test_logprobs_control_is_scoped_away_from_metrics_mode(self) -> None:
        html = _load_main_gui_html()
        self.assertIn('data-mode-hidden="metrics"><input type="checkbox" id="logprobs"', html)


if __name__ == "__main__":
    unittest.main()

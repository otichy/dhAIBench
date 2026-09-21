"""Optional real-browser test; use DHAIBENCH_BROWSER_CHANNEL=msedge on Windows."""
import base64
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import unittest

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

ROOT = Path(__file__).resolve().parents[1]

class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

@unittest.skipIf(sync_playwright is None, "Playwright is optional")
class JevGuiTests(unittest.TestCase):
    def test_editor_commands_persistence_and_import(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(ROOT)))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with sync_playwright() as p:
                options = {"headless": True}
                if os.environ.get("DHAIBENCH_BROWSER_CHANNEL"):
                    options["channel"] = os.environ["DHAIBENCH_BROWSER_CHANNEL"]
                browser = p.chromium.launch(**options)
                page = browser.new_page()
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(f"http://127.0.0.1:{server.server_port}/config_gui.html")
                page.locator("#provider").select_option("requesty")
                page.locator("#model").fill("typesafe/jev-1.13.0")
                self.assertTrue(page.locator("#decision-editor").is_visible())
                self.assertEqual(page.locator("#system-prompt-label").inner_text(), "Choice question")
                self.assertTrue(page.locator("#enable_cot").is_disabled())
                page.locator("#system_prompt").fill("Which class applies to the marked node?")
                labels = page.get_by_role("textbox", name="Choice label", exact=True)
                descriptions = page.get_by_role("textbox", name="Choice description", exact=True)
                labels.nth(0).fill("NOUN")
                descriptions.nth(0).fill("A noun")
                labels.nth(1).fill("VERB")
                descriptions.nth(1).fill("A verb")
                command = page.locator("#command-output").inner_text()
                self.assertIn("--decision_criteria_b64", command)
                self.assertNotIn("--enable_cot", command)
                self.assertIn("--no_explanation", command)
                flag = command.split("--decision_criteria_b64")[1].split()[0]
                self.assertEqual(json.loads(base64.b64decode(flag)), {"NOUN": "A noun", "VERB": "A verb"})
                page.get_by_role("button", name="+ Add choice", exact=True).click()
                self.assertEqual(labels.count(), 3)
                labels.nth(2).fill("NOUN")
                descriptions.nth(2).fill("Duplicate")
                self.assertIn("Duplicate", page.locator("#decision-error").inner_text())
                page.get_by_role("button", name="Remove choice", exact=True).nth(2).click()
                page.reload()
                self.assertEqual(labels.count(), 2)
                self.assertEqual(labels.nth(0).input_value(), "NOUN")
                self.assertEqual(page.locator("#system_prompt").input_value(), "Which class applies to the marked node?")
                page.locator("#model").fill("openai/gpt-4o-mini")
                self.assertFalse(page.locator("#decision-editor").is_visible())
                self.assertFalse(page.locator("#enable_cot").is_disabled())
                self.assertNotIn("--decision_criteria_b64", page.locator("#command-output").inner_text())
                criteria = {"ADJ": "An adjective", "OTHER": "Something else"}
                metrics = {"accuracy": 0.8, "run_config": {
                    "provider": "requesty", "model": "typesafe/jev-1.13.0",
                    "system_prompt": "Which category?", "input": ["example_input.csv"],
                    "decision_criteria_b64": base64.b64encode(json.dumps(criteria).encode()).decode()}}
                page.locator("#load-metrics-file").set_input_files({
                    "name": "run__metrics.json", "mimeType": "application/json",
                    "buffer": json.dumps(metrics).encode()})
                page.wait_for_function("document.getElementById('system_prompt').value === 'Which category?'")
                self.assertEqual(labels.nth(0).input_value(), "ADJ")
                self.assertEqual(descriptions.nth(1).input_value(), "Something else")
                self.assertIn("--decision_criteria_b64", page.locator("#command-output").inner_text())
                self.assertFalse(errors, errors)
                browser.close()
        finally:
            server.shutdown()
            server.server_close()

if __name__ == "__main__":
    unittest.main()

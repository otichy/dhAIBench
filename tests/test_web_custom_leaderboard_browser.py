"""Optional browser smoke test: install Playwright and Chromium (or Edge)."""
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
class CustomLeaderboardBrowserTests(unittest.TestCase):
    def test_weights_filters_sharing_export_and_mobile(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(ROOT)))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        fixtures = {}
        for index, (model, task, score) in enumerate([
            ("complete", "A", .9), ("complete", "B", .7),
            ("incomplete", "A", 1), ("unpriced", "A", .8), ("unpriced", "B", .8),
        ]):
            name = f"fixture{index}__fixture__{model}__2026-09-09-12-00__metrics.json"
            fixtures[name] = {
                "model_details": {"provider": "fixture", "model_requested": model},
                "run_config": {"task_name": task}, "accuracy": score, "macro_f1": score - .1,
                "prediction_count": 100, "evaluated_example_count": 100, "truth_label_count": 100,
                "token_usage_totals": {"input_tokens_total": 100000, "output_tokens_total": 0},
            }
        catalog = {"updated_at": "2026-09-09", "providers": {"fixture": {"models": {
            model: {"service_tiers": {"standard": {"input_usd_per_mtokens": 1, "output_usd_per_mtokens": 1}}}
            for model in ["complete", "incomplete"]
        }}}}
        try:
            with sync_playwright() as p:
                options = {"headless": True}
                if os.environ.get("DHAIBENCH_BROWSER_CHANNEL"):
                    options["channel"] = os.environ["DHAIBENCH_BROWSER_CHANNEL"]
                browser = p.chromium.launch(**options)
                page = browser.new_page(accept_downloads=True)
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.route("**/metrics-manifest.json", lambda route: route.fulfill(json={"metrics_files": ["../data/metrics/" + name for name in fixtures]}))
                page.route("**/fixture*__metrics.json", lambda route: route.fulfill(json=fixtures[route.request.url.rsplit("/", 1)[-1]]))
                page.route("**/config_prices.js", lambda route: route.fulfill(content_type="text/javascript", body="window.MODEL_PRICING_CATALOG=" + json.dumps(catalog)))
                page.goto(f"http://127.0.0.1:{server.server_port}/web/#tab=custom&v=1")
                page.locator(".custom-ranking").wait_for()
                self.assertIn("2 ranked configurations", page.locator(".custom-summary").inner_text())
                self.assertEqual(page.locator(".custom-point").count(), 1)
                self.assertEqual(page.locator(".custom-legend-item").count(), 3)
                def marker_styles():
                    return page.locator(".custom-ranking tbody tr").evaluate_all("rows => Object.fromEntries(rows.map(row => { const marker = row.querySelector('.custom-series-marker'); return [row.dataset.customSeries, [marker.dataset.shape, marker.dataset.color]]; }))")
                initial_styles = marker_styles()
                legend_item = page.locator(".custom-legend-item").first
                legend_item.hover()
                self.assertEqual(page.locator(".custom-point.is-highlighted").count(), 1)
                self.assertEqual(page.locator(".custom-ranking tr.is-highlighted").count(), 1)
                self.assertEqual(page.locator(".custom-point").get_attribute("data-color"), legend_item.locator(".custom-series-marker").get_attribute("data-color"))
                legend_item.focus()
                page.keyboard.press("Enter")
                self.assertEqual(page.locator(".custom-ranking details[open]").count(), 1)
                field = page.get_by_role("spinbutton", name="Weight for A", exact=True)
                field.fill("3")
                self.assertTrue(field.evaluate("node => node === document.activeElement"))
                self.assertIn("85%", page.locator(".custom-ranking tbody tr").first.inner_text())
                self.assertIn("75%", page.locator(".custom-weights").inner_text())
                self.assertEqual(marker_styles(), initial_styles)
                page.locator(".custom-point").focus()
                page.keyboard.press("Enter")
                self.assertEqual(page.locator(".custom-ranking details[open]").count(), 1)
                page.locator("#customMetric").select_option("macro_f1")
                self.assertIn("75%", page.locator(".custom-ranking tbody tr").first.inner_text())
                shared = page.url
                page.reload()
                page.locator(".custom-ranking").wait_for()
                self.assertEqual(field.input_value(), "3")
                self.assertEqual(page.locator("#customMetric").input_value(), "macro_f1")
                with page.expect_download() as info:
                    page.get_by_role("button", name="Export CSV", exact=True).click()
                exported = Path(info.value.path()).read_text(encoding="utf-8-sig")
                self.assertIn("USD / 1000 predictions", exported)
                self.assertIn("fixture0", exported)
                page.get_by_role("button", name="Clear", exact=True).click()
                self.assertEqual(page.locator(".custom-ranking").count(), 0)
                page.get_by_role("button", name="Select all", exact=True).click()
                field.fill("-1")
                self.assertEqual(page.locator(".custom-ranking").count(), 0)
                field.fill("3")
                page.locator("#customTask1").uncheck()
                self.assertIn("3 ranked configurations", page.locator(".custom-summary").inner_text())
                self.assertEqual(marker_styles(), initial_styles)
                self.assertEqual(len(set(page.locator(".custom-point").evaluate_all("nodes => nodes.map(n => n.dataset.shape)"))), 2)
                self.assertEqual(len(set(page.locator(".custom-point").evaluate_all("nodes => nodes.map(n => n.dataset.color)"))), 2)
                # Ranking changes after selecting one task; styles do not.
                self.assertEqual(page.locator(".custom-legend-item").first.locator("strong").inner_text(), "incomplete")
                page.locator("#customTask1").check()
                # A model filter must not renormalize away its missing B task.
                page.locator("#modelSelect").select_option("incomplete")
                self.assertIn("0 ranked configurations", page.locator(".custom-summary").inner_text())
                page.evaluate("localStorage.clear()")
                page.goto(shared)
                page.reload()
                page.locator(".custom-ranking").wait_for()
                self.assertEqual(field.input_value(), "3")
                page.set_viewport_size({"width": 390, "height": 844})
                self.assertLessEqual(page.evaluate("document.documentElement.scrollWidth"), 390,
                    page.evaluate("Array.from(document.querySelectorAll('body *')).filter(n => n.getBoundingClientRect().right > 400 && n.getBoundingClientRect().width > 0).slice(0, 18).map(n => [n.tagName, n.className, n.getBoundingClientRect().width])"))
                page.evaluate("document.documentElement.dataset.theme = 'light'")
                self.assertTrue(page.locator(".custom-ranking").is_visible())
                screenshot_dir = os.environ.get("DHAIBENCH_SCREENSHOT_DIR")
                if screenshot_dir:
                    page.screenshot(path=str(Path(screenshot_dir) / "custom-leaderboard-mobile.png"), full_page=True)
                # Existing views remain usable.
                page.set_viewport_size({"width": 1280, "height": 900})
                for tab in ["Chart", "Scatter", "Table", "Radar", "Agreement", "Custom Leaderboard"]:
                    page.get_by_role("button", name=tab, exact=True).click()
                if screenshot_dir:
                    page.screenshot(path=str(Path(screenshot_dir) / "custom-leaderboard-desktop.png"), full_page=True)
                # Saved B stays required even if all its files disappear.
                for name in list(fixtures):
                    if fixtures[name]["run_config"]["task_name"] == "B":
                        del fixtures[name]
                page.reload()
                page.locator(".custom-ranking").wait_for()
                self.assertIn("0 ranked configurations", page.locator(".custom-summary").inner_text())
                self.assertEqual(page.get_by_role("spinbutton", name="Weight for B", exact=True).count(), 1)
                self.assertEqual(errors, [])
                browser.close()
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()

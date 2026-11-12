"""Exercise tag availability and native disabled buttons in headless Edge."""
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import unittest
import uuid


ROOT = Path(__file__).resolve().parents[1]


class TagFilterBrowserTests(unittest.TestCase):
    def test_tag_counts_ignore_tag_selection_and_follow_other_filters(self):
        browser = Path(os.environ.get(
            "DHAIBENCH_EDGE_PATH",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        ))
        if not browser.is_file():
            self.skipTest("Set DHAIBENCH_EDGE_PATH to a Chromium/Edge executable")
        markup = (ROOT / "web/index.html").read_text(encoding="utf-8")

        def inline(match):
            name = match.group(1).split("?")[0].removeprefix("./")
            if name == "analytics.local.js":
                return ""
            code = (ROOT / "web" / name).read_text(encoding="utf-8")
            if name == "app.js":
                code = code.rsplit("init();", 1)[0]
            return "<script>" + code.replace("</script", "<\\/script") + "</script>"

        markup = re.sub(r'<script src="([^"]+)"></script>', inline, markup)
        css = (ROOT / "web/styles.css").read_text(encoding="utf-8")
        markup = markup.replace("</head>", "<style>" + css + "</style></head>")
        assertions = r"""
const assert = (condition, message) => { if (!condition) throw new Error(message); };
try {
  const fixture = (stem, model, task, tags) => normalizeRun(
    `${stem}__fixture__${model}__2026-11-03-00-00__metrics.json`, {
      model_details: {provider: 'fixture', model_requested: model},
      run_config: {task_name: task, tags}, accuracy: .8, macro_f1: .8,
      prediction_count: 10, evaluated_example_count: 10, truth_label_count: 10,
    });
  state.runs = [fixture('one', 'M1', 'A', 'red'), fixture('two', 'M1', 'A', 'blue'),
    fixture('three', 'M1', 'B', 'green'), fixture('four', 'M2', 'B', 'red;green')];
  state.models = ['M1', 'M2']; state.tasks = ['A', 'B']; state.tags = ['red', 'blue', 'green'];
  state.selectedModels = []; state.selectedTasks = []; state.selectedTags = [];
  state.timeRanges = []; state.hideNoAccuracy = false;
  const chips = () => Array.from(els.tagChips.children);
  const labels = () => chips().map(button => button.textContent).join('|');
  // Keep real filter setters, click handlers and tag rendering; omit unrelated charts.
  render = () => { state.filtered = getFilteredRuns(); renderTagControls(); };
  render();
  assert(labels() === 'All Tags (4)|red (2)|blue (1)|green (2)', 'Initial counts');
  setSelectedModels(['M1']);
  assert(labels() === 'All Tags (3)|red (1)|blue (1)|green (1)', 'Model counts');
  setSelectedTasks(['A']);
  assert(labels() === 'All Tags (2)|red (1)|blue (1)|green (0)', 'Combined counts');
  assert(chips()[3].disabled, 'Zero tag must be disabled');
  assert(getComputedStyle(chips()[3]).opacity < getComputedStyle(chips()[1]).opacity,
    'Disabled tag must be visibly dimmed');
  chips()[3].click();
  assert(state.selectedTags.length === 0, 'Disabled tag click changed selection');
  chips()[1].click();
  assert(state.filtered.length === 1, 'Tag click did not filter runs');
  assert(labels() === 'All Tags (2)|red (1)|blue (1)|green (0)', 'Tag selection changed counts');
  chips()[2].click();
  assert(state.filtered.length === 2, 'Other tags must remain selectable');
  setSelectedModels(['M2']);
  assert(chips().length === 4 && chips().slice(1).every(button => button.disabled),
    'Empty model/task intersection must preserve all tags and disable them');
  assert(!chips()[0].disabled, 'All Tags must allow clearing unavailable selections');
  chips()[0].click();
  assert(state.selectedTags.length === 0, 'All Tags did not clear selection');
  setSelectedTasks([]);
  assert(labels() === 'All Tags (1)|red (1)|blue (0)|green (1)', 'Model-only counts');
  setSelectedModels([]); setSelectedTasks(['B']);
  assert(labels() === 'All Tags (2)|red (1)|blue (0)|green (2)', 'Task-only counts');
  // Explicit temperatures remain inspectable even when comparison views omit them.
  const zero = fixture('zero', 'M1', 'A', 'red;temp_0.0');
  zero.runConfig.temperature = 0;
  const one = fixture('oneTemp', 'M2', 'B', 'green;temp_1.0');
  one.runConfig.temperature = 1;
  state.runs.push(zero, one);
  state.tags.push('temp_0.0', 'temp_1.0');
  setSelectedTasks([]);
  assert(labels() === 'All Tags (6)|red (3)|blue (1)|green (3)|temp_0.0 (1)|temp_1.0 (1)',
    'Temperature runs missing from counts with all filters cleared');
  assert(!chips()[4].disabled && !chips()[5].disabled, 'Temperature chips must be selectable');
  assert(state.filtered.length === 4, 'Temperature overrides entered comparisons');
  chips()[4].click();
  assert(state.selectedTags.includes('temp_0.0'), 'Temperature tag click did not select');
  assert(getFilteredRuns({includeTemperatureOverrides: true}).length === 1,
    'Temperature tag must reveal its run in the runs table');
  assert(state.filtered.length === 1 && state.filtered[0] === zero,
    'Selected temperature run must enter leaderboard and comparison inputs');
  assert(chips()[4].classList.contains('active') && chips()[4].getAttribute('aria-pressed') === 'true',
    'Clicked temperature tag must visibly indicate selection');
  // Simulate hover with a class of identical specificity in the real CSS rules.
  // This exercises the cascade deterministically without pointer coordinates.
  for (const sheet of document.styleSheets) {
    if (sheet.ownerNode.tagName !== 'STYLE') continue;
    for (const rule of sheet.cssRules) {
      if (rule.selectorText?.includes('.chip:hover')) {
        rule.selectorText = rule.selectorText.replaceAll(':hover', '.test-hover');
      }
    }
  }
  chips()[4].classList.add('test-hover');
  assert(getComputedStyle(chips()[4]).backgroundImage.includes('linear-gradient'),
    'Hover must preserve the selected tag gradient');
  chips()[5].click();
  assert(state.filtered.length === 2, 'Multiple temperature tags must match with OR');
  chips()[0].click();
  assert(state.filtered.length === 4, 'Clearing temperature selections must restore baseline comparisons');
  chips()[4].click();
  assert(chips()[5].textContent === 'temp_1.0 (1)' && !chips()[5].disabled,
    'Tag selection must not hide other temperature tags');
  setSelectedModels(['M1']);
  assert(chips()[4].textContent === 'temp_0.0 (1)' && !chips()[4].disabled,
    'Matching model must retain temperature chip');
  assert(chips()[5].disabled, 'Model filter must disable unavailable temperature');
  setSelectedTasks(['B']);
  assert(chips()[4].disabled, 'Task filter must disable unavailable temperature');
  document.getElementById('tagTestResult').textContent = JSON.stringify({passed: true});
} catch (error) {
  document.getElementById('tagTestResult').textContent = JSON.stringify({passed: false, error: error.stack});
}
"""
        markup = markup.replace("</body>", '<pre id="tagTestResult">pending</pre><script>' + assertions + "</script></body>")
        base = (ROOT / "verification_tmp").resolve()
        temp = base / ("tag-filters-" + uuid.uuid4().hex)
        temp.mkdir(parents=True)
        try:
            page = temp / "tag_filters.html"
            page.write_text(markup, encoding="utf-8")
            options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
            result = subprocess.run([
                str(browser), "--headless", "--disable-gpu", "--disable-extensions",
                "--no-first-run", "--no-default-browser-check", "--disable-background-networking",
                "--user-data-dir=" + str(temp / "profile"), "--dump-dom", page.as_uri(),
            ], capture_output=True, encoding="utf-8", errors="replace", timeout=60, **options)
            report = re.search(r'<pre id="tagTestResult">(.*?)</pre>', result.stdout, re.S)
            self.assertIsNotNone(report, (result.stderr + result.stdout)[-2500:])
            data = json.loads(html.unescape(report.group(1)))
            self.assertTrue(data["passed"], data)
        finally:
            self.assertEqual(temp.resolve().parent, base)
            shutil.rmtree(temp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

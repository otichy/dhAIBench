(function initCustomLeaderboard(global) {
  "use strict";

  const finite = (value) => typeof value === "number" && Number.isFinite(value);
  const mean = (values) => values.length ? values.reduce((sum, value) => sum + value / values.length, 0) : null;

  function normalizeSettings(value) {
    const source = value && typeof value === "object" ? value : {};
    const weights = Object.create(null);
    Object.entries(source.weights || {}).forEach(([task, setting]) => {
      if (setting && finite(setting.weight) && setting.weight >= 0) {
        weights[task] = { weight: setting.weight, enabled: setting.enabled !== false };
      }
    });
    return { metric: source.metric === "macro_f1" ? "macro_f1" : "accuracy", weights };
  }

  function taskSetting(settings, task) {
    return settings.weights[task] || { weight: 1, enabled: true };
  }

  function modelIdentity(run) {
    const config = run.runConfig || {};
    const controls = run.controlSummary?.configured || {};
    const keys = ["reasoning_effort", "thinking_level", "effort", "temperature", "top_p", "top_k", "verbosity", "prompt_layout", "enable_cot", "no_explanation", "few_shot_examples", "prompt_batch_size"];
    const values = keys.map((key) => [key, controls[key] ?? config[key] ?? null]);
    const provider = run.provider === "einfra" ? "e-infra" : run.provider || "unknown provider";
    const tier = run.serviceTier || "standard";
    const details = values.filter(([, value]) => value !== null && value !== "").map(([key, value]) => `${key}=${value}`);
    return {
      // Match the Chart tab's Model grouping, including across providers/settings.
      key: run.model,
      label: run.model,
      provider,
      runLabel: `${run.model} · ${provider} · ${tier}${details.length ? " · " + details.join(", ") : " · unspecified settings"}`,
    };
  }

  // Records are already filtered by task/model/tag/date. Never derive required
  // tasks from those records: a filter must not silently improve task coverage.
  function calculate(records, taskNames, rawSettings) {
    const settings = normalizeSettings(rawSettings);
    const tasks = [...new Set(taskNames)].map((task) => ({ task, ...taskSetting(settings, task) }))
      .filter((item) => item.enabled && item.weight > 0);
    const maxWeight = Math.max(0, ...tasks.map((task) => task.weight));
    const scaledTotal = tasks.reduce((sum, task) => sum + task.weight / maxWeight, 0);
    tasks.forEach((task) => { task.share = (task.weight / maxWeight) / scaledTotal; });
    const required = new Set(tasks.map((task) => task.task));
    const groups = new Map();
    const variants = new Map();
    records.forEach((record) => {
      if (!required.has(record.task)) return;
      if (!groups.has(record.key)) groups.set(record.key, { key: record.key, label: record.label, records: [] });
      groups.get(record.key).records.push(record);
      // Missing historical metadata is not evidence of a distinct dataset.
      if (record.dataset) {
        if (!variants.has(record.task)) variants.set(record.task, new Set());
        variants.get(record.task).add(record.dataset);
      }
    });
    const rows = [...groups.values()].map((group) => {
      const breakdown = tasks.map((task) => {
        const all = group.records.filter((record) => record.task === task.task);
        const eligible = all.filter((record) => finite(record.metric) && !record.partial);
        const differentPrompts = new Set(eligible.map((record) => record.protocol).filter(Boolean)).size > 1;
        const ambiguous = (variants.get(task.task)?.size || 0) > 1 || differentPrompts;
        const score = ambiguous ? null : mean(eligible.map((record) => record.metric));
        const prices = eligible.map((record) => finite(record.cost) && record.cost >= 0 && finite(record.predictions) && record.predictions > 0
          ? record.cost / record.predictions * 1000 : null);
        const cost = score !== null && prices.length && prices.every(finite) ? mean(prices) : null;
        return { ...task, score, cost, contribution: score === null ? null : score * task.share,
          runs: eligible, excludedRuns: all.filter((record) => !eligible.includes(record)),
          reason: ambiguous ? "Multiple datasets or prompts; narrow the filters" : score === null ? "No eligible result" : "" };
      });
      const covered = breakdown.filter((task) => task.score !== null);
      const complete = tasks.length > 0 && covered.length === tasks.length;
      return { key: group.key, label: group.label, breakdown, complete,
        providers: [...new Set(group.records.map((record) => record.provider).filter(Boolean))].sort(),
        coverage: covered.reduce((sum, task) => sum + task.share, 0),
        coveredTasks: covered.length,
        score: complete ? breakdown.reduce((sum, task) => sum + task.contribution, 0) : null,
        cost: complete && breakdown.every((task) => finite(task.cost))
          ? breakdown.reduce((sum, task) => sum + task.cost * task.share, 0) : null };
    });
    rows.sort((a, b) => Number(b.complete) - Number(a.complete) || (b.score ?? -Infinity) - (a.score ?? -Infinity) || a.label.localeCompare(b.label));
    let lastScore = null;
    let rank = 0;
    rows.forEach((row, index) => {
      if (!row.complete) { row.rank = null; return; }
      if (row.score !== lastScore) rank = index + 1;
      row.rank = rank;
      lastScore = row.score;
    });
    return { tasks, rows, metric: settings.metric };
  }

  function csv(result, pricingDate, shareUrl) {
    const cells = (values) => values.map((value) => {
      let text = value == null ? "" : String(value);
      if (/^[=+@\-\t\r]/.test(text)) text = "'" + text;
      return '"' + text.replace(/"/g, '""') + '"';
    }).join(",");
    const lines = [cells(["Rank", "Model", "Metric", "Weighted score (%)", "USD / 1000 predictions", "Coverage (%)", "Task", "Weight", "Share (%)", "Task score (%)", "Contribution (percentage points)", "Task USD / 1000 predictions", "Eligible runs", "Evaluated examples by run", "Run references", "Excluded run references", "Status", "Pricing updated", "Share URL", "Providers", "Run configurations"])];
    result.rows.forEach((row) => row.breakdown.forEach((task) => {
      lines.push(cells([row.rank, row.label, result.metric, row.score, row.cost, row.coverage * 100, task.task, task.weight, task.share * 100, task.score, task.contribution, task.cost, task.runs.length,
        task.runs.map((run) => run.samples ?? "unknown").join("; "), task.runs.map((run) => run.path).join("; "),
        task.excludedRuns.map((run) => run.path).join("; "), task.reason, pricingDate, shareUrl,
        row.providers.join("; "), task.runs.map((run) => run.runLabel || run.label).join("; ")]));
    }));
    return lines.join("\r\n");
  }

  global.DHAIBenchCustomLeaderboard = { normalizeSettings, taskSetting, modelIdentity, calculate, csv };
})(typeof globalThis !== "undefined" ? globalThis : this);

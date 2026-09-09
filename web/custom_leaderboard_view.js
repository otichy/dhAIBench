/* Dashboard integration. Calculations live in custom_leaderboard.js. */
function renderCustomLeaderboard(container, runs) {
  const api = window.DHAIBenchCustomLeaderboard;
  const settings = state.customLeaderboard;
  const taskNames = state.selectedTasks.length ? state.selectedTasks
    : uniqueNonEmptyStrings([...state.tasks, ...Object.keys(settings.weights)]);
  let addedTask = false;
  taskNames.forEach((task) => {
    if (!Object.hasOwn(settings.weights, task)) {
      settings.weights[task] = { weight: 1, enabled: true };
      addedTask = true;
    }
  });
  if (addedTask) persistUiState();
  const element = (tag, text, className) => {
    const node = document.createElement(tag);
    if (text != null) node.textContent = text;
    if (className) node.className = className;
    return node;
  };
  const button = (label, action) => {
    const node = element("button", label, "btn");
    node.type = "button";
    node.addEventListener("click", action);
    return node;
  };
  container.classList.add("custom-leaderboard");
  container.append(element("h3", "Rank models using your task priorities"));
  container.append(element("p", "Select tasks and assign relative weights. Sidebar filters apply to this comparison.", "muted"));
  const controls = element("div", null, "custom-controls");
  const metricLabel = element("label", "Scoring metric ", "field");
  const metricSelect = element("select");
  metricSelect.id = "customMetric";
  [["accuracy", "Accuracy"], ["macro_f1", "Macro F1"]].forEach(([value, label]) => {
    const option = element("option", label);
    option.value = value;
    metricSelect.append(option);
  });
  metricSelect.value = settings.metric;
  metricLabel.append(metricSelect);
  controls.append(metricLabel);
  const setAll = (enabled, equal = false) => {
    taskNames.forEach((task) => {
      const current = api.taskSetting(settings, task);
      settings.weights[task] = { weight: equal ? 1 : current.weight, enabled: enabled ?? current.enabled };
    });
    persistUiState();
    renderLeaderboard(state.filtered);
  };
  controls.append(button("Select all", () => setAll(true)), button("Clear", () => setAll(false)), button("Equal weights", () => setAll(null, true)));
  container.append(controls);

  const records = runs.map((run) => {
    const raw = run.rawMetrics || {};
    const source = raw.source_input_csv || (Array.isArray(run.runConfig?.input) && run.runConfig.input.length === 1 ? run.runConfig.input[0] : "");
    return { ...api.modelIdentity(run), task: run.task,
      metric: settings.metric === "macro_f1" ? run.macroF1 : run.accuracy,
      cost: Number.isFinite(run.inputTokensTotal) && Number.isFinite(run.outputTokensTotal)
        && raw.token_usage_totals?.attempts_with_token_usage !== 0 ? run.estimatedCostUsd : null,
      // Never substitute labelled/evaluated examples for an unknown prediction count.
      predictions: run.predictionCount,
      samples: raw.evaluated_example_count ?? run.totalExamples,
      partial: Boolean(raw.stop_reason) || (Number.isFinite(raw.truth_label_count) && Number.isFinite(raw.evaluated_example_count) && raw.evaluated_example_count < raw.truth_label_count),
      dataset: source ? normalizeSlashes(source).split("/").pop() : "",
      protocol: run.runConfig?.system_prompt || "",
      path: run.filePath, run };
  });
  const weights = element("div", null, "custom-weights");
  weights.setAttribute("role", "group");
  weights.setAttribute("aria-label", "Task weights");
  const shares = new Map();
  const fields = [];
  taskNames.forEach((task, index) => {
    const setting = api.taskSetting(settings, task);
    const row = element("div", null, "custom-weight-row");
    const label = element("label", null, "custom-task-label");
    const checkbox = element("input");
    checkbox.type = "checkbox";
    checkbox.checked = setting.enabled;
    checkbox.id = `customTask${index}`;
    label.append(checkbox, element("span", task));
    const input = element("input");
    input.type = "number";
    input.min = "0";
    input.step = "any";
    input.value = setting.weight;
    input.disabled = !setting.enabled;
    input.setAttribute("aria-label", `Weight for ${task}`);
    input.id = `customWeight${index}`;
    fields.push(input);
    const share = element("span", "0%", "custom-task-share");
    shares.set(task, share);
    row.append(label, input, share);
    weights.append(row);
    checkbox.addEventListener("change", () => {
      settings.weights[task] = { ...api.taskSetting(settings, task), enabled: checkbox.checked };
      input.disabled = !checkbox.checked;
      if (checkbox.checked) input.value = settings.weights[task].weight;
      refresh();
      persistUiState();
    });
    input.addEventListener("input", () => {
      const weight = input.valueAsNumber;
      const valid = Number.isFinite(weight) && weight >= 0;
      input.setAttribute("aria-invalid", String(!valid));
      if (valid) {
        settings.weights[task] = { weight, enabled: checkbox.checked };
        persistUiState();
      }
      refresh();
    });
  });
  container.append(weights);
  const results = element("div", null, "custom-results");
  container.append(results);
  metricSelect.addEventListener("change", () => {
    settings.metric = metricSelect.value;
    persistUiState();
    renderLeaderboard(state.filtered);
    document.getElementById("customMetric")?.focus();
  });

  function refresh() {
    results.replaceChildren();
    if (fields.some((field) => !field.disabled && (!Number.isFinite(field.valueAsNumber) || field.valueAsNumber < 0))) {
      const error = element("p", "Enter a non-negative number for each enabled task weight.", "warn");
      error.setAttribute("role", "status");
      results.append(error);
      shares.forEach((share) => { share.textContent = "—"; });
      return;
    }
    const result = api.calculate(records, taskNames, settings);
    shares.forEach((share, task) => { share.textContent = `${formatNum((result.tasks.find((item) => item.task === task)?.share || 0) * 100, 1)}%`; });
    if (!result.tasks.length) {
      results.append(element("p", "Select at least one task with a positive weight to create a leaderboard.", "muted"));
      return;
    }
    const complete = result.rows.filter((row) => row.complete);
    const pricingDate = window.MODEL_PRICING_CATALOG?.updated_at || "Unknown";
    const summary = element("p", `${complete.length} ranked configurations · ${result.rows.length - complete.length} incomplete · ${result.tasks.length} tasks`, "custom-summary");
    summary.setAttribute("role", "status");
    summary.setAttribute("aria-live", "polite");
    results.append(summary);
    const methodology = element("details", null, "custom-methodology");
    methodology.append(element("summary", "How scores and costs are calculated"));
    methodology.append(element("p", "Scores are averaged across eligible repeats within each task, then combined using the task weights. Cost is the mean of run costs per 1,000 predictions within each task, then the weighted mean across tasks. Unknown prices or prediction counts are never treated as zero.", "muted"));
    methodology.append(element("p", `Estimates use catalogue pricing updated ${pricingDate}. Shared links require the same metrics source; results can change when data or prices change.`, "muted"));
    methodology.append(element("p", "Only configurations with all selected task scores are ranked. Known partial/stopped runs are excluded. Distinct known input filenames or system prompts under one task require narrower filters. Older artifacts may lack dataset, prompt or completion metadata; task names alone cannot establish full comparability.", "muted"));
    results.append(methodology);
    const exportButton = button("Export CSV", () => {
      const blob = new Blob(["\uFEFF", api.csv(result, pricingDate, buildShareUrl())], { type: "text/csv;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const anchor = element("a");
      anchor.href = url;
      anchor.download = "custom-leaderboard.csv";
      document.body.append(anchor);
      anchor.click();
      anchor.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
    exportButton.disabled = !result.rows.length;
    results.append(exportButton);
    if (!complete.length) results.append(element("p", "No configuration has eligible results for every selected task. Review the missing-task details below, select fewer tasks, or broaden the sidebar filters.", "muted"));
    const detailTargets = new Map();
    const chart = element("div", null, "custom-chart");
    results.append(chart);
    const tableWrap = element("div", null, "custom-table-wrap");
    const table = element("table", null, "custom-ranking");
    table.append(element("caption", `Custom leaderboard by ${METRIC_LABELS[settings.metric]}`));
    const head = element("thead");
    const headings = element("tr");
    ["Rank", "Model / configuration and task breakdown", "Weighted score", "USD / 1,000 predictions", "Task coverage"].forEach((text) => {
      const th = element("th", text); th.scope = "col"; headings.append(th);
    });
    head.append(headings); table.append(head);
    const body = element("tbody");
    result.rows.forEach((row) => {
      const tr = element("tr", null, row.complete ? "" : "custom-incomplete");
      tr.append(element("td", row.rank ?? "Unranked"));
      const modelCell = element("td");
      const details = element("details");
      const title = element("summary", row.label);
      details.append(title);
      detailTargets.set(row.key, { details, title });
      const breakdown = element("div", null, "custom-breakdown");
      row.breakdown.forEach((task) => {
        const section = element("div", null, "custom-task-detail");
        section.append(element("strong", `${task.task} · ${formatNum(task.share * 100, 1)}% weight`));
        section.append(element("p", task.score === null ? task.reason : `Score ${formatNum(task.score, 2)}% · contribution ${formatNum(task.contribution, 2)} percentage points · ${task.cost === null ? "unknown cost" : formatUsd(task.cost) + " / 1,000 predictions"}`));
        task.runs.forEach((record) => {
          const link = button(`${record.run.timestamp || record.path} · ${record.samples ?? "unknown"} evaluated examples`, () => openRunModal(record.run));
          link.title = record.path;
          section.append(link);
        });
        task.excludedRuns.forEach((record) => section.append(button(`Excluded: ${record.partial ? "partial/stopped run" : "missing metric"} · ${record.path}`, () => openRunModal(record.run))));
        breakdown.append(section);
      });
      details.append(breakdown); modelCell.append(details); tr.append(modelCell);
      tr.append(element("td", row.score === null ? "—" : `${formatNum(row.score, 2)}%`), element("td", row.cost === null ? "Unknown" : formatUsd(row.cost)), element("td", `${row.coveredTasks}/${result.tasks.length} tasks · ${formatNum(row.coverage * 100, 1)}% weight`));
      body.append(tr);
    });
    table.append(body); tableWrap.append(table); results.append(tableWrap);
    renderCustomLeaderboardScatter(chart, complete, settings.metric, (row) => {
      const target = detailTargets.get(row.key);
      target.details.open = true;
      target.title.focus();
      target.details.scrollIntoView({ block: "nearest", behavior: "smooth" });
    });
  }
  refresh();
}

function renderCustomLeaderboardScatter(container, rows, metric, onSelect) {
  const numeric = rows.filter((row) => row.cost !== null);
  const note = document.createElement("p");
  note.className = "muted";
  const unknownCount = rows.length - numeric.length;
  note.textContent = `${unknownCount ? `${unknownCount} with unknown cost shown only in the table. ` : ""}Select a point for task details. Higher and further left is better.`;
  container.append(note);
  if (!numeric.length) return;
  const width = 900, height = 410;
  const margin = { left: 78, right: 32, top: 30, bottom: 65 };
  const svg = createSvgNode("svg", { viewBox: `0 0 ${width} ${height}`, role: "group", "aria-label": `Weighted ${METRIC_LABELS[metric]} versus estimated USD per 1,000 predictions`, class: "custom-scatter" });
  const maxCost = Math.max(...numeric.map((row) => row.cost)) * 1.05 || 1;
  const x = (cost) => margin.left + cost / maxCost * (width - margin.left - margin.right);
  const y = (score) => height - margin.bottom - score / 100 * (height - margin.top - margin.bottom);
  const text = (label, attrs) => {
    const node = createSvgNode("text", attrs); node.textContent = label; svg.append(node); return node;
  };
  for (let i = 0; i <= 5; i += 1) {
    const score = i * 20;
    svg.append(createSvgNode("line", { x1: margin.left, x2: width - margin.right, y1: y(score), y2: y(score), class: "custom-grid-line" }));
    text(`${score}%`, { x: margin.left - 12, y: y(score) + 4, "text-anchor": "end" });
    const cost = maxCost * i / 5;
    text(formatUsd(cost), { x: x(cost), y: height - margin.bottom + 25, "text-anchor": "middle" });
  }
  text("Estimated USD per 1,000 predictions", { x: width / 2, y: height - 12, "text-anchor": "middle" });
  text(`Weighted ${METRIC_LABELS[metric]} (%)`, { transform: `translate(18 ${height / 2}) rotate(-90)`, "text-anchor": "middle" });
  numeric.forEach((row) => {
    const label = `${row.label}: ${formatNum(row.score, 2)}%, ${formatUsd(row.cost)} per 1,000 predictions, rank ${row.rank}`;
    const point = createSvgNode("g", { role: "button", tabindex: "0", "aria-label": label, class: "custom-point" });
    const title = createSvgNode("title"); title.textContent = label; point.append(title);
    point.append(createSvgNode("circle", { cx: x(row.cost), cy: y(row.score), r: 8 }));
    const rank = createSvgNode("text", { x: x(row.cost) + 10, y: y(row.score) - 10 });
    rank.textContent = `#${row.rank}`; point.append(rank);
    point.addEventListener("click", () => onSelect(row));
    point.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); onSelect(row); }
    });
    svg.append(point);
  });
  container.append(svg);
}

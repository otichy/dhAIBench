/* Dashboard integration. Calculations live in custom_leaderboard.js. */
function renderCustomLeaderboard(container, runs) {
  const api = window.DHAIBenchCustomLeaderboard;
  const settings = state.customLeaderboard;
  // Use the full loaded catalogue, not rank or the filtered rows, for stable styles.
  const modelNames = new Map(state.runs.map((run) => [api.modelIdentity(run).key, run.model]));
  const seriesStyles = buildModelSeriesStyleMap([...modelNames.keys()].sort());
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
    const plotted = complete.filter((row) => row.cost !== null);
    const pricingDate = window.MODEL_PRICING_CATALOG?.updated_at || "Unknown";
    const summary = element("p", `${complete.length} ranked models · ${result.rows.length - complete.length} incomplete · ${result.tasks.length} tasks`, "custom-summary");
    summary.setAttribute("role", "status");
    summary.setAttribute("aria-live", "polite");
    results.append(summary);
    const methodology = element("details", null, "custom-methodology");
    methodology.append(element("summary", "How scores and costs are calculated"));
    methodology.append(element("p", "Runs with the same model name are grouped across providers and settings, as in the Chart tab. Scores are averaged across eligible runs within each task, then combined using the task weights. Cost is the mean of run costs per 1,000 predictions within each task, then the weighted mean across tasks. Unknown prices or prediction counts are never treated as zero.", "muted"));
    methodology.append(element("p", `Estimates use catalogue pricing updated ${pricingDate}. Shared links require the same metrics source; results can change when data or prices change.`, "muted"));
    methodology.append(element("p", "Only models with all selected task scores are ranked. Known partial/stopped runs are excluded. Distinct known input filenames or system prompts within one model's eligible task runs require narrower filters. Dataset differences in other models do not remove coverage. Older artifacts may lack dataset, prompt or completion metadata; task names alone cannot establish full comparability.", "muted"));
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
    if (!complete.length) results.append(element("p", "No model has eligible results for every selected task. Review the missing-task details below, select fewer tasks, or broaden the sidebar filters.", "muted"));
    else if (!plotted.length) results.append(element("p", "No ranked model has a known cost to plot. Scores are available in the table below.", "muted"));
    const detailTargets = new Map();
    const linkedNodes = new Map();
    const highlight = (key) => {
      linkedNodes.forEach((nodes, seriesKey) => nodes.forEach((node) => {
        node.classList.toggle("is-highlighted", seriesKey === key);
        node.classList.toggle("is-muted", Boolean(key) && seriesKey !== key && node.classList.contains("custom-point"));
      }));
    };
    const linkSeries = (node, row) => {
      node.dataset.customSeries = row.key;
      node.style.setProperty("--series-color", seriesStyles.get(row.key).color);
      if (!linkedNodes.has(row.key)) linkedNodes.set(row.key, []);
      linkedNodes.get(row.key).push(node);
      node.addEventListener("mouseenter", () => highlight(row.key));
      node.addEventListener("mouseleave", () => highlight(document.activeElement?.closest("[data-custom-series]")?.dataset.customSeries));
      node.addEventListener("focusin", () => highlight(row.key));
      node.addEventListener("focusout", (event) => highlight(event.relatedTarget?.closest("[data-custom-series]")?.dataset.customSeries));
    };
    const selectRow = (row) => {
      const target = detailTargets.get(row.key);
      target.details.open = true;
      target.title.focus();
      highlight(row.key);
      target.details.scrollIntoView({ block: "nearest", behavior: "smooth" });
    };
    const chart = element("div", null, "custom-chart");
    if (plotted.length) results.append(chart);
    const legend = element("div", null, "custom-legend");
    legend.setAttribute("role", "group");
    legend.setAttribute("aria-label", "Model colors and shapes");
    plotted.forEach((row) => {
      const item = button("", () => selectRow(row));
      item.className = "custom-legend-item";
      item.title = row.label;
      item.setAttribute("aria-label", `Show task breakdown for ${row.label}`);
      item.append(createCustomSeriesMarker(seriesStyles.get(row.key)), createCustomSeriesLabel(row, modelNames.get(row.key)));
      const status = `#${row.rank} · ${formatNum(row.score, 2)}%`;
      item.append(element("span", status, "custom-legend-status"));
      linkSeries(item, row);
      legend.append(item);
    });
    if (plotted.length) {
      results.append(element("p", "Model legend · Hover or focus to highlight; select to open task details.", "muted custom-legend-heading"), legend);
    }
    const tableWrap = element("div", null, "custom-table-wrap");
    const table = element("table", null, "custom-ranking");
    table.append(element("caption", `Custom leaderboard by ${METRIC_LABELS[settings.metric]}`));
    const head = element("thead");
    const headings = element("tr");
    ["Rank", "Model and task breakdown", "Weighted score", "USD / 1,000 predictions", "Task coverage"].forEach((text) => {
      const th = element("th", text); th.scope = "col"; headings.append(th);
    });
    head.append(headings); table.append(head);
    const body = element("tbody");
    result.rows.forEach((row) => {
      const tr = element("tr", null, row.complete ? "" : "custom-incomplete");
      linkSeries(tr, row);
      const rankCell = element("td");
      rankCell.append(element("span", row.rank === null ? "Unranked" : `#${row.rank}`, "custom-rank-badge"));
      tr.append(rankCell);
      const modelCell = element("td");
      const details = element("details");
      const title = element("summary", null, "custom-model-summary");
      title.title = row.label;
      title.append(createCustomSeriesMarker(seriesStyles.get(row.key)), createCustomSeriesLabel(row, modelNames.get(row.key)));
      details.append(title);
      detailTargets.set(row.key, { details, title });
      const breakdown = element("div", null, "custom-breakdown");
      breakdown.append(element("p", row.label, "muted"));
      row.breakdown.forEach((task) => {
        const section = element("div", null, "custom-task-detail");
        section.append(element("strong", `${task.task} · ${formatNum(task.share * 100, 1)}% weight`));
        section.append(element("p", task.score === null ? task.reason : `Score ${formatNum(task.score, 2)}% · contribution ${formatNum(task.contribution, 2)} percentage points · ${task.cost === null ? "unknown cost" : formatUsd(task.cost) + " / 1,000 predictions"}`));
        task.runs.forEach((record) => {
          const link = button(`${record.provider} · ${record.run.timestamp || record.path} · ${record.samples ?? "unknown"} evaluated examples`, () => openRunModal(record.run));
          link.title = `${record.runLabel} · ${record.path}`;
          section.append(link);
        });
        task.excludedRuns.forEach((record) => section.append(button(`Excluded: ${record.partial ? "partial/stopped run" : "missing metric"} · ${record.path}`, () => openRunModal(record.run))));
        breakdown.append(section);
      });
      details.append(breakdown); modelCell.append(details); tr.append(modelCell);
      const scoreCell = element("td", null, "custom-score-cell");
      scoreCell.append(element("strong", row.score === null ? "—" : `${formatNum(row.score, 2)}%`));
      if (row.score !== null) {
        const track = element("div", null, "custom-score-track");
        track.setAttribute("aria-hidden", "true");
        const fill = element("span");
        fill.style.width = `${Math.max(0, Math.min(100, row.score))}%`;
        track.append(fill); scoreCell.append(track);
      }
      const coverageCell = element("td", `${row.coveredTasks}/${result.tasks.length} tasks`);
      coverageCell.append(element("small", `${formatNum(row.coverage * 100, 1)}% weight`, "custom-cell-secondary"));
      tr.append(scoreCell, element("td", row.cost === null ? "Unknown" : formatUsd(row.cost)), coverageCell);
      body.append(tr);
    });
    table.append(body); tableWrap.append(table); results.append(tableWrap);
    if (plotted.length) renderCustomLeaderboardScatter(chart, complete, settings.metric, selectRow, seriesStyles, linkSeries);
  }
  refresh();
}

function createCustomSeriesMarker(style) {
  const svg = createSvgNode("svg", { viewBox: "0 0 24 24", class: "custom-series-marker", "aria-hidden": "true", "data-shape": style.shape, "data-color": style.color });
  svg.append(buildTimeSeriesShape(style.shape, 12, 12, 16, style.color, "var(--ink)", 1));
  return svg;
}

function createCustomSeriesLabel(row, modelName = row.label) {
  const copy = document.createElement("span");
  copy.className = "custom-series-copy";
  const name = document.createElement("strong");
  name.textContent = modelName;
  const configuration = document.createElement("small");
  configuration.textContent = (row.providers || []).join(" · ");
  copy.append(name, configuration);
  return copy;
}

function renderCustomLeaderboardScatter(container, rows, metric, onSelect, seriesStyles, linkSeries) {
  const numeric = rows.filter((row) => row.cost !== null);
  if (!numeric.length) return;
  const labels = [];
  const toolbar = document.createElement("div");
  toolbar.className = "custom-chart-toolbar";
  const toolbarLabel = document.createElement("span");
  toolbarLabel.className = "custom-chart-toolbar-label";
  toolbarLabel.textContent = "Point labels";
  const namesToggle = createTimeSeriesToggleControl("Model names", state.customLeaderboard.showModelNames, () => {
    state.customLeaderboard.showModelNames = !state.customLeaderboard.showModelNames;
    const enabled = state.customLeaderboard.showModelNames;
    namesToggle.classList.toggle("active", enabled);
    namesToggle.setAttribute("aria-checked", String(enabled));
    namesToggle.setAttribute("aria-label", `Model names: ${enabled ? "on" : "off"}`);
    namesToggle.querySelector(".time-series-toggle-status").textContent = enabled ? "On" : "Off";
    updateLabels();
    persistUiState();
  });
  toolbar.append(toolbarLabel, namesToggle);
  container.append(toolbar);
  const note = document.createElement("p");
  note.className = "muted custom-chart-note";
  const unknownCount = rows.length - numeric.length;
  note.textContent = `${unknownCount ? `${unknownCount} with unknown cost shown only in the table. ` : ""}Select a point for task details. Higher and further left is better.`;
  container.append(note);
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
    const style = seriesStyles.get(row.key);
    point.dataset.shape = style.shape;
    point.dataset.color = style.color;
    const marker = buildTimeSeriesShape(style.shape, x(row.cost), y(row.score), 18, style.color, "var(--ink)", 1.3);
    marker.classList.add("custom-point-mark");
    point.append(marker);
    linkSeries(point, row);
    const rank = createSvgNode("text", { x: x(row.cost) + 10, y: y(row.score) - 10, class: "custom-point-label" });
    labels.push({ node: rank, row });
    point.append(rank);
    point.addEventListener("click", () => onSelect(row));
    point.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); onSelect(row); }
    });
    svg.append(point);
  });
  const viewport = document.createElement("div");
  viewport.className = "custom-chart-viewport";
  viewport.append(svg);
  container.append(viewport);
  function updateLabels() {
    labels.forEach(({ node, row }) => {
      node.textContent = `#${row.rank}${state.customLeaderboard.showModelNames ? " " + row.label : ""}`;
      node.removeAttribute("textLength");
      node.removeAttribute("lengthAdjust");
      const measured = node.getComputedTextLength();
      const available = width - margin.left - margin.right - 20;
      const labelWidth = Math.min(measured, available);
      if (measured > available) {
        node.setAttribute("textLength", available);
        node.setAttribute("lengthAdjust", "spacingAndGlyphs");
      }
      // Keep names at the expensive end of the chart inside the SVG bounds.
      const placeLeft = x(row.cost) + 10 + labelWidth > width - 8;
      node.setAttribute("text-anchor", placeLeft ? "end" : "start");
      node.setAttribute("x", placeLeft ? Math.max(margin.left + labelWidth, x(row.cost) - 10) : x(row.cost) + 10);
    });
  }
  updateLabels();
}

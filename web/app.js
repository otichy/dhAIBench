const STORAGE_KEY = "dhAIBench.metricsDashboard.state.v1";
const METRICS_MANIFEST_PATH = "./metrics-manifest.json";
const METRICS_SERVER_DIR = "../data/metrics";

const state = {
  runs: [],
  filtered: [],
  tasks: [],
  selectedTask: "ALL",
  selectedRunPath: null,
  modelQuery: "",
  sortBy: "accuracy",
  hideNoAccuracy: false,
  theme: "light",
  sourceMode: "none",
  sourceFileCount: 0,
  warnings: [],
  activeDirectoryHandle: null,
  activeFiles: [],
};

const els = {
  taskSelect: document.querySelector("#taskSelect"),
  modelSearch: document.querySelector("#modelSearch"),
  sortSelect: document.querySelector("#sortSelect"),
  hideNoAccuracy: document.querySelector("#hideNoAccuracy"),
  themeToggle: document.querySelector("#themeToggle"),
  taskChips: document.querySelector("#taskChips"),
  heroTitle: document.querySelector("#heroTitle"),
  heroSubtitle: document.querySelector("#heroSubtitle"),
  btnAutoServer: document.querySelector("#btnAutoServer"),
  btnOpenFolder: document.querySelector("#btnOpenFolder"),
  btnOpenFiles: document.querySelector("#btnOpenFiles"),
  reloadBtn: document.querySelector("#reloadBtn"),
  metricsFileInput: document.querySelector("#metricsFileInput"),
  sourceStatus: document.querySelector("#sourceStatus"),
  sourceHint: document.querySelector("#sourceHint"),
  sourceWarnings: document.querySelector("#sourceWarnings"),
  kpiRuns: document.querySelector("#kpiRuns"),
  kpiTasks: document.querySelector("#kpiTasks"),
  kpiBestAccuracy: document.querySelector("#kpiBestAccuracy"),
  kpiRequests: document.querySelector("#kpiRequests"),
  leaderboardChart: document.querySelector("#leaderboardChart"),
  taskBestChart: document.querySelector("#taskBestChart"),
  tokenChart: document.querySelector("#tokenChart"),
  runsTableBody: document.querySelector("#runsTableBody"),
  tableMeta: document.querySelector("#tableMeta"),
  runModal: document.querySelector("#runModal"),
  runModalTitle: document.querySelector("#runModalTitle"),
  runModalMeta: document.querySelector("#runModalMeta"),
  runModalContent: document.querySelector("#runModalContent"),
  runModalClose: document.querySelector("#runModalClose"),
  barRowTemplate: document.querySelector("#barRowTemplate"),
};

function supportsDirectoryPicker() {
  return typeof window.showDirectoryPicker === "function";
}

function isFileProtocol() {
  return window.location.protocol === "file:";
}

function getFileNameFromPath(filePath) {
  const normalized = String(filePath || "").replace(/\\/g, "/");
  const parts = normalized.split("/");
  return parts[parts.length - 1] || normalized;
}

function parseRunName(fileName) {
  const raw = fileName.replace(/_metrics\.json$/i, "");
  const match = raw.match(/^(.*)_out_(.*)_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})$/i);
  if (!match) {
    return {
      task: raw,
      model: "unknown",
      timestamp: null,
    };
  }
  return {
    task: match[1],
    model: match[2],
    timestamp: match[3],
  };
}

function toPct(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return null;
  }
  return value <= 1 ? value * 100 : value;
}

function safeNum(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function normalizeRun(filePath, payload) {
  const fileName = getFileNameFromPath(filePath);
  const nameParts = parseRunName(fileName);
  const modelDetails = payload.model_details || {};
  const usage = payload.usage_metadata_summary || {};
  const tokenTotals = payload.token_usage_totals || {};
  const controls = payload.request_control_summary || {};
  const ts = payload.first_prompt_timestamp || nameParts.timestamp;

  const accuracy = toPct(safeNum(payload.accuracy));
  const macroF1 = toPct(safeNum(payload.macro_f1));

  return {
    filePath,
    fileName,
    task: nameParts.task,
    model:
      modelDetails.model_requested ||
      modelDetails.model_for_requests ||
      nameParts.model,
    timestamp: ts,
    accuracy,
    macroF1,
    totalExamples:
      safeNum(payload.total_examples) ??
      safeNum(payload.evaluated_example_count) ??
      safeNum(payload.prediction_count),
    predictionCount: safeNum(payload.prediction_count),
    requestsTotal:
      safeNum(controls.attempts_total) ??
      safeNum(usage.attempts_total),
    attemptsWithUsage: safeNum(usage.attempts_with_usage_metadata),
    cachedTokens:
      safeNum(usage.cached_tokens_total_estimate) ??
      safeNum(usage.cache_read_tokens_total),
    inputTokensTotal: safeNum(tokenTotals.input_tokens_total),
    cachedInputTokensTotal:
      safeNum(tokenTotals.cached_input_tokens_total) ??
      safeNum(usage.cached_tokens_total_estimate) ??
      safeNum(usage.cache_read_tokens_total),
    outputTokensTotal: safeNum(tokenTotals.output_tokens_total),
    thinkingTokensTotal: safeNum(tokenTotals.thinking_tokens_total),
    nonCachedInputTokensTotal: safeNum(tokenTotals.non_cached_input_tokens_total),
    overallTimeSeconds: safeNum(payload.overall_time_seconds),
    overallTimeHuman: payload.overall_time_human || null,
    firstPromptTimestamp: payload.first_prompt_timestamp || null,
    lastPromptTimestamp: payload.last_prompt_timestamp || null,
    truthLabelCount: safeNum(payload.truth_label_count),
    labelMetricsAvailable:
      payload.label_metrics_available !== false &&
      (safeNum(payload.accuracy) !== null || safeNum(payload.macro_f1) !== null),
    labelMetricsReason: payload.label_metrics_reason || null,
    modelDetails,
    usageSummary: usage,
    tokenUsageTotals: tokenTotals,
    controlSummary: controls,
    rawMetrics: payload,
  };
}

function parseMetricText(filePath, rawText, warnings) {
  let payload;
  try {
    payload = JSON.parse(rawText);
  } catch (error) {
    warnings.push({
      file: filePath,
      message: `Invalid JSON (${error.message})`,
    });
    return null;
  }

  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    warnings.push({
      file: filePath,
      message: "Unexpected JSON structure (expected object).",
    });
    return null;
  }

  return normalizeRun(filePath, payload);
}

function dedupeRuns(runs) {
  const byFile = new Map();
  runs.forEach((run) => {
    byFile.set(run.filePath, run);
  });
  return Array.from(byFile.values());
}

async function discoverMetricFilesFromServer() {
  try {
    const manifestRes = await fetch(METRICS_MANIFEST_PATH, { cache: "no-store" });
    if (manifestRes.ok) {
      const manifest = await manifestRes.json();
      if (Array.isArray(manifest.metrics_files) && manifest.metrics_files.length) {
        return manifest.metrics_files;
      }
    }
  } catch (_) {
    // Fallback below.
  }

  const dirRes = await fetch(`${METRICS_SERVER_DIR}/`, { cache: "no-store" });
  if (!dirRes.ok) {
    throw new Error(`Unable to load metrics manifest or metrics directory listing from ${METRICS_SERVER_DIR}/.`);
  }
  const html = await dirRes.text();
  const matches = [...html.matchAll(/href=\"([^\"]+_metrics\.json)\"/gi)];
  const files = matches.map((match) => `${METRICS_SERVER_DIR}/${decodeURIComponent(match[1])}`);
  if (!files.length) {
    throw new Error(`No *_metrics.json files discovered on server path ${METRICS_SERVER_DIR}/.`);
  }
  return files.sort();
}

async function loadFromServer() {
  const files = await discoverMetricFilesFromServer();
  const warnings = [];
  const runs = [];

  for (const path of files) {
    try {
      const res = await fetch(path, { cache: "no-store" });
      if (!res.ok) {
        warnings.push({
          file: path,
          message: `HTTP ${res.status} while loading file.`,
        });
        continue;
      }
      const text = await res.text();
      const run = parseMetricText(path, text, warnings);
      if (run) {
        runs.push(run);
      }
    } catch (error) {
      warnings.push({
        file: path,
        message: `Failed to fetch file (${error.message}).`,
      });
    }
  }

  return {
    mode: "server",
    fileCount: files.length,
    runs: dedupeRuns(runs),
    warnings,
  };
}

async function collectMetricFilesFromDirectoryHandle(dirHandle, prefix = "") {
  const files = [];
  for await (const [name, entry] of dirHandle.entries()) {
    if (entry.kind === "file" && name.toLowerCase().endsWith("_metrics.json")) {
      files.push({
        path: `${prefix}${name}`,
        handle: entry,
      });
      continue;
    }
    if (entry.kind === "directory") {
      const nested = await collectMetricFilesFromDirectoryHandle(entry, `${prefix}${name}/`);
      files.push(...nested);
    }
  }
  return files;
}

async function loadFromDirectoryHandle(dirHandle) {
  const metricFiles = await collectMetricFilesFromDirectoryHandle(dirHandle);
  if (!metricFiles.length) {
    throw new Error("No *_metrics.json files found in selected folder.");
  }

  const warnings = [];
  const runs = [];

  for (const item of metricFiles) {
    try {
      const fileObj = await item.handle.getFile();
      const text = await fileObj.text();
      const run = parseMetricText(item.path, text, warnings);
      if (run) {
        runs.push(run);
      }
    } catch (error) {
      warnings.push({
        file: item.path,
        message: `Cannot read file (${error.message}).`,
      });
    }
  }

  return {
    mode: "folder",
    fileCount: metricFiles.length,
    runs: dedupeRuns(runs),
    warnings,
  };
}

async function loadFromFiles(fileList) {
  const files = Array.from(fileList || []).filter((file) =>
    String(file.name || "").toLowerCase().endsWith("_metrics.json")
  );

  if (!files.length) {
    throw new Error("No *_metrics.json files selected.");
  }

  const warnings = [];
  const runs = [];

  for (const file of files) {
    try {
      const text = await file.text();
      const filePath = file.webkitRelativePath || file.name;
      const run = parseMetricText(filePath, text, warnings);
      if (run) {
        runs.push(run);
      }
    } catch (error) {
      warnings.push({
        file: file.name,
        message: `Cannot read file (${error.message}).`,
      });
    }
  }

  return {
    mode: "files",
    fileCount: files.length,
    runs: dedupeRuns(runs),
    warnings,
  };
}

function warningSummary(warnings, maxItems = 3) {
  if (!warnings.length) {
    return "";
  }
  const items = warnings.slice(0, maxItems).map((warning) => `${warning.file}: ${warning.message}`);
  if (warnings.length > maxItems) {
    items.push(`...and ${warnings.length - maxItems} more`);
  }
  return items.join(" | ");
}

function updateSourceStatus(customHint = "") {
  els.sourceStatus.textContent = `Mode: ${state.sourceMode} | Files: ${state.sourceFileCount} | Warnings: ${state.warnings.length}`;

  let hint = customHint;
  if (!hint) {
    if (isFileProtocol() && state.sourceMode === "none") {
      hint = "file:// mode: browsers block automatic folder scanning. Use Open Metrics Folder or Open Metrics Files.";
    } else if (state.sourceMode === "server") {
      hint = `Server mode: loading from ${METRICS_MANIFEST_PATH} or ${METRICS_SERVER_DIR}/ listing.`;
    } else if (state.sourceMode === "folder") {
      hint = "Folder mode: loaded from browser-granted local folder access.";
    } else if (state.sourceMode === "files") {
      hint = "Files mode: loaded from selected local files.";
    } else {
      hint = "Choose a data source to load metrics.";
    }
  }
  els.sourceHint.textContent = hint;

  const summary = warningSummary(state.warnings);
  els.sourceWarnings.textContent = summary ? `Warnings: ${summary}` : "";
}

function persistUiState() {
  const payload = {
    selectedTask: state.selectedTask,
    modelQuery: state.modelQuery,
    sortBy: state.sortBy,
    hideNoAccuracy: state.hideNoAccuracy,
    theme: state.theme,
  };
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  } catch (_) {
    // Ignore storage failures.
  }
}

function restoreUiState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return;
    }
    const payload = JSON.parse(raw);
    if (payload && typeof payload === "object") {
      if (typeof payload.selectedTask === "string") {
        state.selectedTask = payload.selectedTask;
      }
      if (typeof payload.modelQuery === "string") {
        state.modelQuery = payload.modelQuery;
      }
      if (typeof payload.sortBy === "string") {
        state.sortBy = payload.sortBy;
      }
      if (typeof payload.hideNoAccuracy === "boolean") {
        state.hideNoAccuracy = payload.hideNoAccuracy;
      }
      if (payload.theme === "dark" || payload.theme === "light") {
        state.theme = payload.theme;
      }
    }
  } catch (_) {
    // Ignore corrupted storage.
  }
}

function applyUiStateToControls() {
  els.modelSearch.value = state.modelQuery;
  els.sortSelect.value = state.sortBy;
  els.hideNoAccuracy.checked = state.hideNoAccuracy;
  els.themeToggle.checked = state.theme === "dark";
}

function applyTheme() {
  document.documentElement.setAttribute("data-theme", state.theme);
}

function setSelectedTask(task) {
  state.selectedTask = task;
  if (els.taskSelect.value !== task) {
    els.taskSelect.value = task;
  }
  persistUiState();
  render();
}

function setupFilters() {
  els.taskSelect.addEventListener("change", (event) => {
    setSelectedTask(event.target.value);
  });

  els.modelSearch.addEventListener("input", (event) => {
    state.modelQuery = event.target.value.trim().toLowerCase();
    persistUiState();
    render();
  });

  els.sortSelect.addEventListener("change", (event) => {
    state.sortBy = event.target.value;
    persistUiState();
    render();
  });

  els.hideNoAccuracy.addEventListener("change", (event) => {
    state.hideNoAccuracy = event.target.checked;
    persistUiState();
    render();
  });

  els.themeToggle.addEventListener("change", (event) => {
    state.theme = event.target.checked ? "dark" : "light";
    applyTheme();
    persistUiState();
  });
}

function setupSourceControls() {
  els.btnAutoServer.addEventListener("click", () => {
    activateServerSource();
  });

  els.btnOpenFolder.addEventListener("click", async () => {
    if (!supportsDirectoryPicker()) {
      renderError(
        "Open Metrics Folder is not supported by this browser. Use Open Metrics Files instead.",
        true
      );
      return;
    }
    try {
      const handle = await window.showDirectoryPicker({ mode: "read" });
      state.activeDirectoryHandle = handle;
      state.activeFiles = [];
      await activateFolderSource(handle);
    } catch (error) {
      if (error && error.name === "AbortError") {
        return;
      }
      renderError(`Folder selection failed: ${error.message}`, true);
    }
  });

  els.btnOpenFiles.addEventListener("click", () => {
    els.metricsFileInput.value = "";
    els.metricsFileInput.click();
  });

  els.metricsFileInput.addEventListener("change", async () => {
    const files = Array.from(els.metricsFileInput.files || []);
    if (!files.length) {
      return;
    }
    state.activeFiles = files;
    state.activeDirectoryHandle = null;
    await activateFilesSource(files);
  });

  els.reloadBtn.addEventListener("click", async () => {
    if (state.sourceMode === "folder" && state.activeDirectoryHandle) {
      await activateFolderSource(state.activeDirectoryHandle);
      return;
    }
    if (state.sourceMode === "files" && state.activeFiles.length) {
      await activateFilesSource(state.activeFiles);
      return;
    }
    if (state.sourceMode === "server" || !isFileProtocol()) {
      await activateServerSource();
      return;
    }
    renderError(
      "No active local source to reload. Choose Open Metrics Folder or Open Metrics Files.",
      true
    );
  });
}

function scoreForSort(run, key) {
  if (key === "accuracy") return run.accuracy ?? -Infinity;
  if (key === "macro_f1") return run.macroF1 ?? -Infinity;
  if (key === "cached_tokens") return run.cachedTokens ?? -Infinity;
  if (key === "attempts_total") return run.requestsTotal ?? -Infinity;
  if (key === "timestamp") {
    const ts = Date.parse(run.timestamp || "");
    return Number.isFinite(ts) ? ts : -Infinity;
  }
  return -Infinity;
}

function getFilteredRuns() {
  const query = state.modelQuery;
  const selectedTask = state.selectedTask;
  let runs = state.runs.filter((run) => {
    if (selectedTask !== "ALL" && run.task !== selectedTask) {
      return false;
    }
    if (query && !run.model.toLowerCase().includes(query) && !run.fileName.toLowerCase().includes(query)) {
      return false;
    }
    if (state.hideNoAccuracy && run.accuracy === null) {
      return false;
    }
    return true;
  });

  runs.sort((a, b) => {
    const valueA = scoreForSort(a, state.sortBy);
    const valueB = scoreForSort(b, state.sortBy);
    if (valueB !== valueA) return valueB - valueA;
    return (b.macroF1 ?? -Infinity) - (a.macroF1 ?? -Infinity);
  });

  return runs;
}

function formatNum(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "N/A";
  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: 0,
  });
}

function formatTs(ts) {
  const dt = new Date(ts || "");
  if (Number.isNaN(dt.getTime())) return "N/A";
  return dt.toLocaleString();
}

function renderTaskControls() {
  const tasks = ["ALL", ...state.tasks];
  els.taskSelect.innerHTML = tasks
    .map((task) => `<option value="${task}">${task === "ALL" ? "All Tasks" : task}</option>`)
    .join("");

  if (!tasks.includes(state.selectedTask)) {
    state.selectedTask = "ALL";
  }
  els.taskSelect.value = state.selectedTask;

  const counts = state.runs.reduce((acc, run) => {
    acc[run.task] = (acc[run.task] || 0) + 1;
    return acc;
  }, {});

  els.taskChips.innerHTML = "";
  tasks.forEach((task) => {
    const button = document.createElement("button");
    button.className = `chip${state.selectedTask === task ? " active" : ""}`;
    button.textContent = task === "ALL" ? `All (${state.runs.length})` : `${task} (${counts[task] || 0})`;
    button.type = "button";
    button.addEventListener("click", () => setSelectedTask(task));
    els.taskChips.appendChild(button);
  });
}

function renderKpis(runs) {
  const uniqueTasks = new Set(runs.map((run) => run.task)).size;
  const bestAccuracy = runs.reduce(
    (max, run) => (run.accuracy !== null ? Math.max(max, run.accuracy) : max),
    -Infinity
  );
  const totalRequests = runs.reduce((sum, run) => sum + (run.requestsTotal || 0), 0);

  els.kpiRuns.textContent = formatNum(runs.length, 0);
  els.kpiTasks.textContent = formatNum(uniqueTasks, 0);
  els.kpiBestAccuracy.textContent = Number.isFinite(bestAccuracy)
    ? `${formatNum(bestAccuracy, 2)}%`
    : "N/A";
  els.kpiRequests.textContent = formatNum(totalRequests, 0);

  els.heroTitle.textContent = state.selectedTask === "ALL" ? "All Tasks" : state.selectedTask;
  els.heroSubtitle.textContent = `${formatNum(runs.length, 0)} runs loaded.`;
}

function createBarRow(label, value, max, formatter, colorClass, onClick) {
  const node = els.barRowTemplate.content.firstElementChild.cloneNode(true);
  const labelEl = node.querySelector(".bar-label");
  const fillEl = node.querySelector(".bar-fill");
  const valueEl = node.querySelector(".bar-value");
  labelEl.textContent = label;

  const ratio = max > 0 ? Math.max(0, Math.min(1, value / max)) : 0;
  fillEl.style.width = `${ratio * 100}%`;

  if (colorClass === "warm") {
    fillEl.style.background = "linear-gradient(90deg, #f59e0b, #ea580c)";
  } else if (colorClass === "blue") {
    fillEl.style.background = "linear-gradient(90deg, #2563eb, #1e3a8a)";
  }

  if (typeof onClick === "function") {
    node.classList.add("chart-clickable");
    node.tabIndex = 0;
    node.setAttribute("role", "button");
    node.addEventListener("click", onClick);
    node.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        onClick();
      }
    });
  }

  valueEl.textContent = formatter(value);
  return node;
}

function renderLeaderboard(runs) {
  els.leaderboardChart.innerHTML = "";
  const source = runs.filter((run) => run.accuracy !== null).slice(0, 12);

  if (!source.length) {
    els.leaderboardChart.innerHTML = '<p class="muted">No runs with accuracy in current filter.</p>';
    return;
  }

  const max = Math.max(...source.map((run) => run.accuracy));
  source.forEach((run) => {
    els.leaderboardChart.appendChild(
      createBarRow(
        run.model,
        run.accuracy,
        max,
        (value) => `${formatNum(value, 2)}%`,
        null,
        () => openRunModal(run)
      )
    );
  });
}

function renderBestByTask() {
  els.taskBestChart.innerHTML = "";
  const bestByTask = {};

  state.runs.forEach((run) => {
    if (run.accuracy === null) return;
    if (!bestByTask[run.task] || run.accuracy > bestByTask[run.task].accuracy) {
      bestByTask[run.task] = run;
    }
  });

  const items = Object.values(bestByTask)
    .sort((a, b) => b.accuracy - a.accuracy)
    .slice(0, 16);

  if (!items.length) {
    els.taskBestChart.innerHTML = '<p class="muted">No task-level accuracy data found.</p>';
    return;
  }

  const max = Math.max(...items.map((run) => run.accuracy));
  items.forEach((run) => {
    els.taskBestChart.appendChild(
      createBarRow(
        `${run.task} / ${run.model}`,
        run.accuracy,
        max,
        (value) => `${formatNum(value, 2)}%`,
        "warm",
        () => openRunModal(run)
      )
    );
  });
}

function renderTokenSignals(runs) {
  els.tokenChart.innerHTML = "";
  const tokenRuns = runs
    .filter((run) => (run.cachedTokens || 0) > 0 || (run.requestsTotal || 0) > 0)
    .slice(0, 14);

  if (!tokenRuns.length) {
    els.tokenChart.innerHTML = '<p class="muted">No token/request metadata for current filter.</p>';
    return;
  }

  const maxCached = Math.max(...tokenRuns.map((run) => run.cachedTokens || 0), 1);
  tokenRuns.forEach((run) => {
    const row = createBarRow(
      run.model,
      run.cachedTokens || 0,
      maxCached,
      (value) => `${formatNum(value, 0)} cached`,
      "blue",
      () => openRunModal(run)
    );
    const value = row.querySelector(".bar-value");
    value.innerHTML = `<span class="mono">${formatNum(run.cachedTokens || 0, 0)}</span><br><span class="muted">${formatNum(run.requestsTotal || 0, 0)} req</span>`;
    els.tokenChart.appendChild(row);
  });
}

function renderTable(runs) {
  els.runsTableBody.innerHTML = "";
  els.tableMeta.textContent = `${formatNum(runs.length, 0)} rows`;

  runs.forEach((run) => {
    const tr = document.createElement("tr");
    tr.className = "clickable-row";
    if (run.filePath === state.selectedRunPath) {
      tr.classList.add("selected-row");
    }
    tr.innerHTML = `
      <td>${run.task}</td>
      <td>${run.model}</td>
      <td>${formatTs(run.timestamp)}</td>
      <td>${run.accuracy !== null ? `${formatNum(run.accuracy, 2)}%` : '<span class="muted">N/A</span>'}</td>
      <td>${run.macroF1 !== null ? `${formatNum(run.macroF1, 2)}%` : '<span class="muted">N/A</span>'}</td>
      <td class="mono">${formatNum(run.requestsTotal, 0)}</td>
      <td class="mono">${formatNum(run.cachedTokens, 0)}</td>
      <td class="mono">${run.fileName}</td>
    `;
    tr.addEventListener("click", () => {
      state.selectedRunPath = run.filePath;
      openRunModal(run);
      render();
    });
    els.runsTableBody.appendChild(tr);
  });
}

function findRunByPath(path) {
  if (!path) {
    return null;
  }
  return state.runs.find((run) => run.filePath === path) || null;
}

function createDetailItem(label, value) {
  const wrapper = document.createElement("div");
  wrapper.className = "detail-item";
  const labelEl = document.createElement("span");
  labelEl.className = "label";
  labelEl.textContent = label;
  const valueEl = document.createElement("div");
  valueEl.className = "value mono";
  valueEl.textContent = value == null || value === "" ? "N/A" : String(value);
  wrapper.appendChild(labelEl);
  wrapper.appendChild(valueEl);
  return wrapper;
}

function fillRunDetailsContent(run) {
  if (!run) {
    els.runModalTitle.textContent = "Run Details";
    els.runModalMeta.textContent = "No run selected.";
    els.runModalContent.innerHTML = "";
    return;
  }
  const detailPairs = [
    ["Task", run.task],
    ["Model", run.model],
    ["Timestamp", formatTs(run.timestamp)],
    ["Accuracy", run.accuracy == null ? "N/A" : `${formatNum(run.accuracy, 2)}%`],
    ["Macro F1", run.macroF1 == null ? "N/A" : `${formatNum(run.macroF1, 2)}%`],
    ["Predictions", formatNum(run.predictionCount, 0)],
    ["Evaluated Examples", formatNum(run.totalExamples, 0)],
    ["Truth Label Count", formatNum(run.truthLabelCount, 0)],
    ["Requests (attempts_total)", formatNum(run.requestsTotal, 0)],
    ["Attempts With Usage", formatNum(run.attemptsWithUsage, 0)],
    ["Input Tokens (total)", formatNum(run.inputTokensTotal, 0)],
    ["Cached Input Tokens (total)", formatNum(run.cachedInputTokensTotal, 0)],
    ["Non-Cached Input Tokens", formatNum(run.nonCachedInputTokensTotal, 0)],
    ["Output Tokens (total)", formatNum(run.outputTokensTotal, 0)],
    ["Thinking Tokens (total)", formatNum(run.thinkingTokensTotal, 0)],
    ["Runtime (seconds)", formatNum(run.overallTimeSeconds, 2)],
    ["Runtime (human)", run.overallTimeHuman],
    ["First Prompt", formatTs(run.firstPromptTimestamp)],
    ["Last Prompt", formatTs(run.lastPromptTimestamp)],
    ["Label Metrics Available", run.labelMetricsAvailable ? "true" : "false"],
    ["Label Metrics Reason", run.labelMetricsReason],
    ["Provider", run.modelDetails.provider || "N/A"],
    ["Model Requested", run.modelDetails.model_requested || "N/A"],
    ["Model For Requests", run.modelDetails.model_for_requests || "N/A"],
  ];

  els.runModalTitle.textContent = `${run.task} / ${run.model}`;
  els.runModalMeta.textContent = run.fileName;
  els.runModalContent.innerHTML = "";

  detailPairs.forEach(([label, value]) => {
    els.runModalContent.appendChild(createDetailItem(label, value));
  });

  const rawWrap = document.createElement("div");
  rawWrap.className = "detail-item detail-raw";
  const detailsEl = document.createElement("details");
  const summaryEl = document.createElement("summary");
  summaryEl.className = "label";
  summaryEl.textContent = "Raw Metrics JSON";
  const pre = document.createElement("pre");
  pre.textContent = JSON.stringify(run.rawMetrics, null, 2);
  detailsEl.appendChild(summaryEl);
  detailsEl.appendChild(pre);
  rawWrap.appendChild(detailsEl);
  els.runModalContent.appendChild(rawWrap);
}

function openRunModal(run) {
  if (!run) {
    return;
  }
  state.selectedRunPath = run.filePath;
  fillRunDetailsContent(run);
  els.runModal.classList.remove("hidden");
}

function closeRunModal() {
  els.runModal.classList.add("hidden");
}

function setupModalControls() {
  els.runModalClose.addEventListener("click", closeRunModal);
  els.runModal.addEventListener("click", (event) => {
    if (event.target === els.runModal) {
      closeRunModal();
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !els.runModal.classList.contains("hidden")) {
      closeRunModal();
    }
  });
}

function render() {
  state.filtered = getFilteredRuns();
  renderTaskControls();
  renderKpis(state.filtered);
  renderLeaderboard(state.filtered);
  renderBestByTask();
  renderTokenSignals(state.filtered);
  renderTable(state.filtered);
}

function renderError(message, preserveExisting = false) {
  els.heroSubtitle.innerHTML = `<span class="warn">${message}</span>`;
  if (!preserveExisting) {
    els.leaderboardChart.innerHTML = "";
    els.taskBestChart.innerHTML = "";
    els.tokenChart.innerHTML = "";
    els.runsTableBody.innerHTML = "";
    closeRunModal();
    fillRunDetailsContent(null);
  }
}

function validateLoadedResult(result) {
  if (!result || !Array.isArray(result.runs) || !Array.isArray(result.warnings)) {
    throw new Error("Unexpected loader result.");
  }
  if (!result.runs.length) {
    const details = warningSummary(result.warnings, 2);
    const suffix = details ? ` ${details}` : "";
    throw new Error(`No valid metrics JSON files loaded.${suffix}`);
  }
}

function applyLoadedResult(result) {
  validateLoadedResult(result);

  state.runs = result.runs;
  state.tasks = [...new Set(result.runs.map((run) => run.task))].sort((a, b) => a.localeCompare(b));
  state.sourceMode = result.mode;
  state.sourceFileCount = result.fileCount;
  state.warnings = result.warnings;

  if (state.selectedTask !== "ALL" && !state.tasks.includes(state.selectedTask)) {
    state.selectedTask = "ALL";
  }
  if (state.selectedRunPath && !findRunByPath(state.selectedRunPath)) {
    state.selectedRunPath = null;
  }

  render();
  updateSourceStatus();
}

async function activateServerSource() {
  els.heroSubtitle.textContent = "Loading metrics from server source...";
  try {
    const result = await loadFromServer();
    state.activeDirectoryHandle = null;
    state.activeFiles = [];
    applyLoadedResult(result);
    const warningInfo = result.warnings.length ? ` (${result.warnings.length} warning(s))` : "";
    els.heroSubtitle.textContent = `Loaded ${result.runs.length.toLocaleString()} runs from server source${warningInfo}.`;
  } catch (error) {
    const base = `Server source failed: ${error.message}`;
    const suffix = isFileProtocol()
      ? " In file:// mode, use Open Metrics Folder or Open Metrics Files."
      : "";
    renderError(base + suffix, true);
    updateSourceStatus();
  }
}

async function activateFolderSource(dirHandle) {
  els.heroSubtitle.textContent = "Loading metrics from selected folder...";
  try {
    const result = await loadFromDirectoryHandle(dirHandle);
    applyLoadedResult(result);
    const warningInfo = result.warnings.length ? ` (${result.warnings.length} warning(s))` : "";
    els.heroSubtitle.textContent = `Loaded ${result.runs.length.toLocaleString()} runs from local folder${warningInfo}.`;
  } catch (error) {
    renderError(`Folder source failed: ${error.message}`, true);
    updateSourceStatus();
  }
}

async function activateFilesSource(files) {
  els.heroSubtitle.textContent = "Loading metrics from selected files...";
  try {
    const result = await loadFromFiles(files);
    applyLoadedResult(result);
    const warningInfo = result.warnings.length ? ` (${result.warnings.length} warning(s))` : "";
    els.heroSubtitle.textContent = `Loaded ${result.runs.length.toLocaleString()} runs from selected files${warningInfo}.`;
  } catch (error) {
    renderError(`Files source failed: ${error.message}`, true);
    updateSourceStatus();
  }
}

async function init() {
  restoreUiState();
  applyUiStateToControls();
  applyTheme();
  setupFilters();
  setupSourceControls();
  setupModalControls();

  if (!supportsDirectoryPicker()) {
    els.btnOpenFolder.disabled = true;
    els.btnOpenFolder.title = "Directory picker is not supported in this browser.";
  }

  if (isFileProtocol()) {
    state.sourceMode = "none";
    state.sourceFileCount = 0;
    state.warnings = [];
    updateSourceStatus(
      "file:// mode detected. Browsers block auto-scanning local folders, so choose Open Metrics Folder or Open Metrics Files."
    );
    els.heroSubtitle.textContent = "Choose a local data source to load metrics.";
    return;
  }

  await activateServerSource();
}

init();

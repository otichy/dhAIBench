const STORAGE_KEY = "dhAIBench.metricsDashboard.state.v1";
const METRICS_MANIFEST_PATH = "./metrics-manifest.json";
const METRICS_SERVER_DIR = "../data/metrics";
const METRICS_SERVER_DIR_CANDIDATES = [METRICS_SERVER_DIR, "./metrics", "./data/metrics"];

const state = {
  runs: [],
  filtered: [],
  tasks: [],
  models: [],
  selectedTask: "ALL",
  selectedModel: "ALL",
  selectedRunPath: null,
  modelQuery: "",
  sortBy: "accuracy",
  tokenSortBy: "runtime_seconds",
  tokenSortDir: "desc",
  hideNoAccuracy: false,
  theme: "dark",
  sourceMode: "none",
  sourceFileCount: 0,
  warnings: [],
  activeDirectoryHandle: null,
  activeFiles: [],
  expandedLeaderboardModels: new Set(),
};

const els = {
  taskSelect: document.querySelector("#taskSelect"),
  modelSearch: document.querySelector("#modelSearch"),
  sortSelect: document.querySelector("#sortSelect"),
  hideNoAccuracy: document.querySelector("#hideNoAccuracy"),
  themeToggle: document.querySelector("#themeToggle"),
  taskChips: document.querySelector("#taskChips"),
  modelChips: document.querySelector("#modelChips"),
  heroTitle: document.querySelector("#heroTitle"),
  heroSubtitle: document.querySelector("#heroSubtitle"),
  btnAutoServer: document.querySelector("#btnAutoServer"),
  btnOpenFolder: document.querySelector("#btnOpenFolder"),
  reloadBtn: document.querySelector("#reloadBtn"),
  sourceStatus: document.querySelector("#sourceStatus"),
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

function isExplicitUrlOrAbsolutePath(path) {
  return /^(https?:)?\/\//i.test(path) || path.startsWith("/") || /^[A-Za-z]:\//.test(path);
}

function normalizeSlashes(path) {
  return String(path || "").replace(/\\/g, "/");
}

function trimTrailingSlash(path) {
  return normalizeSlashes(path).replace(/\/+$/, "");
}

function uniqueNonEmptyStrings(values) {
  const seen = new Set();
  const out = [];
  (values || []).forEach((value) => {
    const normalized = String(value || "").trim();
    if (!normalized || seen.has(normalized)) {
      return;
    }
    seen.add(normalized);
    out.push(normalized);
  });
  return out;
}

function joinPath(baseDir, name) {
  const base = trimTrailingSlash(baseDir);
  const fileName = normalizeSlashes(name).replace(/^\/+/, "");
  return base ? `${base}/${fileName}` : fileName;
}

function getLocationAwareMetricsBaseDirs() {
  const dirs = ["/data/metrics", "/metrics"];
  const pathname = normalizeSlashes(window.location.pathname || "/");
  const currentDirPath = trimTrailingSlash(pathname.replace(/\/[^/]*$/, "")) || "/";
  const segments = currentDirPath.split("/").filter(Boolean);

  for (let i = segments.length; i >= 1; i -= 1) {
    const prefix = `/${segments.slice(0, i).join("/")}`;
    dirs.push(`${prefix}/data/metrics`, `${prefix}/metrics`);
  }

  return uniqueNonEmptyStrings(dirs).map((dir) => trimTrailingSlash(dir));
}

function getFileNameFromPath(filePath) {
  const normalized = normalizeSlashes(filePath);
  const parts = normalized.split("/");
  return parts[parts.length - 1] || normalized;
}

function normalizeMetricsPath(filePath) {
  const normalized = normalizeSlashes(filePath).trim();
  if (!normalized) {
    return normalized;
  }

  // Keep explicit URLs and absolute paths unchanged.
  if (isExplicitUrlOrAbsolutePath(normalized)) {
    return normalized;
  }

  // Already points into metrics via relative traversal.
  if (normalized.includes("/data/metrics/") || normalized.startsWith("../data/metrics/")) {
    return normalized;
  }

  // Relative bare paths from local file/folder loaders are resolved under data/metrics.
  return `${METRICS_SERVER_DIR}/${normalized.replace(/^\.?\//, "")}`;
}

function getDirectoryFromPath(filePath) {
  const normalized = normalizeSlashes(filePath);
  const idx = normalized.lastIndexOf("/");
  return idx >= 0 ? normalized.slice(0, idx) : "";
}

function replaceFileNameInPath(filePath, newFileName) {
  const dir = getDirectoryFromPath(filePath);
  if (!dir) {
    return newFileName;
  }
  return `${dir}/${newFileName}`;
}

function metricFileToRunStem(fileName) {
  return String(fileName || "")
    .replace(/__(metrics|calibration|heatmap)\.(json|png)$/i, "")
    .replace(/_(metrics|calibration|confusion_heatmap)\.(json|png)$/i, "");
}

function mapMetricsPathToSiblingDir(filePath, siblingDir) {
  const normalized = normalizeSlashes(filePath);
  if (normalized.includes("/metrics/")) {
    return normalized.replace("/metrics/", `/${siblingDir}/`);
  }
  if (normalized.startsWith("metrics/")) {
    return normalized.replace(/^metrics\//, `${siblingDir}/`);
  }
  return `../data/${siblingDir}/${getFileNameFromPath(normalized)}`;
}

function parseRunName(fileName) {
  const raw = metricFileToRunStem(fileName);

  const canonical = raw.match(
    /^(.*)__(.*?)__(.*?)__(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})(?:__(.+))?$/i
  );
  if (canonical) {
    return {
      task: canonical[1],
      provider: canonical[2] || "",
      model: canonical[3] || "unknown",
      timestamp: canonical[4],
      extra: canonical[5] || "",
    };
  }

  const legacy = raw.match(
    /^(.*)_out_(.+)_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})(?:__(.+))?$/i
  );
  if (legacy) {
    const task = legacy[1];
    const tail = legacy[2];
    const tokens = tail.split("_").filter(Boolean);
    const knownProviders = new Set([
      "openai",
      "anthropic",
      "cohere",
      "google",
      "huggingface",
      "einfra",
      "requesty",
      "vertex",
    ]);

    if (tokens.length === 1) {
      return {
        task,
        provider: "",
        model: tokens[0],
        timestamp: legacy[3],
        extra: legacy[4] || "",
      };
    }

    const firstToken = (tokens[0] || "").toLowerCase();
    if (knownProviders.has(firstToken)) {
      return {
        task,
        provider: tokens[0],
        model: tokens.slice(1).join("_") || "unknown",
        timestamp: legacy[3],
        extra: legacy[4] || "",
      };
    }

    return {
      task,
      provider: "",
      model: tail || "unknown",
      timestamp: legacy[3],
      extra: legacy[4] || "",
    };
  }

  return {
    task: raw,
    provider: "",
    model: "unknown",
    timestamp: null,
    extra: "",
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

function asTrimmedString(value) {
  return typeof value === "string" ? value.trim() : "";
}

function parseSemicolonTags(value) {
  if (Array.isArray(value)) {
    return value
      .map((tag) => asTrimmedString(tag))
      .filter(Boolean);
  }
  if (typeof value !== "string") {
    return [];
  }
  return value
    .split(";")
    .map((tag) => tag.trim())
    .filter(Boolean);
}

function parseRunTimestampMs(run) {
  const ts = Date.parse(run.timestamp || "");
  return Number.isFinite(ts) ? ts : -Infinity;
}

function getConfiguredControl(run, key) {
  const configured =
    run && run.controlSummary && typeof run.controlSummary === "object"
      ? run.controlSummary.configured
      : null;
  if (!configured || typeof configured !== "object") {
    return "";
  }
  return asTrimmedString(configured[key]);
}

function getRunEffortSuffix(run) {
  const thinkingLevel = getConfiguredControl(run, "thinking_level");
  const reasoningEffort =
    getConfiguredControl(run, "reasoning_effort") || getConfiguredControl(run, "effort");
  const parts = [];
  if (thinkingLevel) {
    parts.push(`thinking:${thinkingLevel}`);
  }
  if (reasoningEffort) {
    parts.push(`effort:${reasoningEffort}`);
  }
  return parts.length ? `[${parts.join(", ")}]` : "";
}

function getRunModelDisplayName(run) {
  const suffix = getRunEffortSuffix(run);
  return suffix ? `${run.model} ${suffix}` : run.model;
}

function getSharedRunEffortSuffix(runs) {
  const suffixes = runs.map((run) => getRunEffortSuffix(run));
  if (!suffixes.length) {
    return "";
  }
  const first = suffixes[0];
  if (!first) {
    return "";
  }
  return suffixes.every((suffix) => suffix === first) ? first : "";
}

function normalizeRun(filePath, payload) {
  const normalizedFilePath = normalizeMetricsPath(filePath);
  const fileName = getFileNameFromPath(normalizedFilePath);
  const runStem = metricFileToRunStem(fileName);
  const nameParts = parseRunName(fileName);
  const modelDetails = payload.model_details || {};
  const usage = payload.usage_metadata_summary || {};
  const tokenTotals = payload.token_usage_totals || {};
  const controls = payload.request_control_summary || {};
  const ts = payload.first_prompt_timestamp || nameParts.timestamp;
  const taskNameFromMetrics = asTrimmedString(payload.task_name);
  const providerFromMetrics =
    asTrimmedString(modelDetails.provider) || asTrimmedString(payload.provider);
  const modelFromMetrics =
    asTrimmedString(modelDetails.model_requested) ||
    asTrimmedString(modelDetails.model_for_requests) ||
    asTrimmedString(payload.model_requested) ||
    asTrimmedString(payload.model);
  const taskDescription = asTrimmedString(payload.task_description);
  const tags = parseSemicolonTags(payload.tags);

  const accuracy = toPct(safeNum(payload.accuracy));
  const macroF1 = toPct(safeNum(payload.macro_f1));

  return {
    filePath: normalizedFilePath,
    fileName,
    runStem,
    task: taskNameFromMetrics || nameParts.task,
    taskDescription,
    tags,
    tagsDisplay: tags.join("; "),
    provider: providerFromMetrics || nameParts.provider || "",
    model: modelFromMetrics || nameParts.model,
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

function parseManifestBaseDirs(manifest) {
  if (!manifest || typeof manifest !== "object") {
    return [];
  }

  const fromArray = Array.isArray(manifest.metrics_base_dirs) ? manifest.metrics_base_dirs : [];
  const fromSingle = typeof manifest.metrics_base_dir === "string" ? [manifest.metrics_base_dir] : [];
  const fromLegacy = typeof manifest.metrics_base_url === "string" ? [manifest.metrics_base_url] : [];

  return uniqueNonEmptyStrings([...fromArray, ...fromSingle, ...fromLegacy]).map((dir) =>
    trimTrailingSlash(dir)
  );
}

function buildServerMetricPathCandidates(rawPath, manifestBaseDirs = []) {
  const normalized = normalizeSlashes(rawPath).trim();
  if (!normalized) {
    return [];
  }

  if (isExplicitUrlOrAbsolutePath(normalized)) {
    return [normalized];
  }

  const fileName = getFileNameFromPath(normalized);
  const baseDirs = uniqueNonEmptyStrings([...manifestBaseDirs, ...METRICS_SERVER_DIR_CANDIDATES]).map((dir) =>
    trimTrailingSlash(dir)
  );

  const directCandidates = [normalized];
  if (normalized.startsWith("./")) {
    directCandidates.push(normalized.replace(/^\.\/+/, ""));
  } else if (!normalized.startsWith("../")) {
    directCandidates.push(`./${normalized}`);
  }

  const byFileName = fileName ? baseDirs.map((dir) => joinPath(dir, fileName)) : [];
  if (fileName) {
    byFileName.push(fileName, `./${fileName}`);
  }

  return uniqueNonEmptyStrings([...directCandidates, ...byFileName]);
}

function getDirectoryListingCandidates() {
  return uniqueNonEmptyStrings([
    ...METRICS_SERVER_DIR_CANDIDATES,
    ...getLocationAwareMetricsBaseDirs(),
  ]).map((dir) => trimTrailingSlash(dir));
}

async function discoverMetricFilesFromDirectoryListings(candidateDirs = getDirectoryListingCandidates()) {
  const listingDirs = uniqueNonEmptyStrings(candidateDirs).map((dir) => trimTrailingSlash(dir));
  const listingErrors = [];

  for (const dir of listingDirs) {
    try {
      const dirRes = await fetch(`${dir}/`, { cache: "no-store" });
      if (!dirRes.ok) {
        listingErrors.push(`${dir}/ -> HTTP ${dirRes.status}`);
        continue;
      }

      const html = await dirRes.text();
      const matches = [...html.matchAll(/href=\"([^\"]+_metrics\.json)\"/gi)];
      const files = uniqueNonEmptyStrings(
        matches.map((match) => {
          const href = decodeURIComponent(match[1]);
          if (isExplicitUrlOrAbsolutePath(href) || href.startsWith("./") || href.startsWith("../")) {
            return href;
          }
          return joinPath(dir, href);
        })
      );

      if (files.length) {
        return {
          files: files.sort(),
          source: "directory-listing",
          manifestBaseDirs: [],
        };
      }

      listingErrors.push(`${dir}/ -> no *_metrics.json links`);
    } catch (error) {
      listingErrors.push(`${dir}/ -> ${error.message}`);
    }
  }

  const detailText = listingErrors.length
    ? ` Tried: ${listingErrors.slice(0, 3).join(" | ")}${listingErrors.length > 3 ? ` | ...and ${listingErrors.length - 3} more` : ""}`
    : "";
  throw new Error(`Unable to load metrics manifest or discover metrics directory listings.${detailText}`);
}

async function discoverMetricFilesFromServer() {
  try {
    const manifestRes = await fetch(METRICS_MANIFEST_PATH, { cache: "no-store" });
    if (manifestRes.ok) {
      const manifest = await manifestRes.json();
      if (Array.isArray(manifest.metrics_files) && manifest.metrics_files.length) {
        return {
          files: manifest.metrics_files,
          source: "manifest",
          manifestBaseDirs: parseManifestBaseDirs(manifest),
        };
      }
    }
  } catch (_) {
    // Fallback below.
  }

  return discoverMetricFilesFromDirectoryListings();
}

async function loadRunsFromServerFileList(files, manifestBaseDirs, warnings) {
  const runs = [];
  for (const path of files) {
    const run = await loadRunFromServerCandidates(path, manifestBaseDirs, warnings);
    if (run) {
      runs.push(run);
    }
  }
  return runs;
}

async function loadRunFromServerCandidates(rawPath, manifestBaseDirs, warnings) {
  const candidates = buildServerMetricPathCandidates(rawPath, manifestBaseDirs);
  if (!candidates.length) {
    warnings.push({
      file: String(rawPath || ""),
      message: "Empty metrics path entry.",
    });
    return null;
  }

  const candidateErrors = [];
  for (const candidatePath of candidates) {
    try {
      const res = await fetch(candidatePath, { cache: "no-store" });
      if (!res.ok) {
        candidateErrors.push(`${candidatePath}: HTTP ${res.status}`);
        continue;
      }

      const text = await res.text();
      const parseWarnings = [];
      const run = parseMetricText(candidatePath, text, parseWarnings);
      if (run) {
        return run;
      }

      parseWarnings.forEach((warning) => {
        candidateErrors.push(`${candidatePath}: ${warning.message}`);
      });
    } catch (error) {
      candidateErrors.push(`${candidatePath}: ${error.message}`);
    }
  }

  const details = candidateErrors.slice(0, 2).join(" | ");
  const overflow =
    candidateErrors.length > 2 ? ` | ...and ${candidateErrors.length - 2} more attempts` : "";
  warnings.push({
    file: String(rawPath),
    message: details ? `Failed all path candidates. ${details}${overflow}` : "Failed all path candidates.",
  });
  return null;
}

async function loadFromServer() {
  const discovery = await discoverMetricFilesFromServer();
  const files = discovery.files;
  const manifestBaseDirs = discovery.manifestBaseDirs || [];
  const warnings = [];
  let runs = await loadRunsFromServerFileList(files, manifestBaseDirs, warnings);
  let fileCount = files.length;

  if (!runs.length && discovery.source === "manifest") {
    try {
      const listingDiscovery = await discoverMetricFilesFromDirectoryListings(manifestBaseDirs);
      const listingRuns = await loadRunsFromServerFileList(listingDiscovery.files, [], warnings);
      if (listingRuns.length) {
        runs = listingRuns;
        fileCount = listingDiscovery.files.length;
        warnings.push({
          file: METRICS_MANIFEST_PATH,
          message: "Manifest paths failed; loaded metrics via directory listing fallback.",
        });
      }
    } catch (_) {
      // Keep original manifest warnings.
    }
  }

  return {
    mode: "server",
    fileCount,
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

function updateSourceStatus() {
  els.sourceStatus.textContent = `Mode: ${state.sourceMode} | Files: ${state.sourceFileCount} | Warnings: ${state.warnings.length}`;

  const summary = warningSummary(state.warnings);
  els.sourceWarnings.textContent = summary ? `Warnings: ${summary}` : "";
}

function persistUiState() {
  const payload = {
    selectedTask: state.selectedTask,
    selectedModel: state.selectedModel,
    modelQuery: state.modelQuery,
    sortBy: state.sortBy,
    tokenSortBy: state.tokenSortBy,
    tokenSortDir: state.tokenSortDir,
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
      if (typeof payload.selectedModel === "string") {
        state.selectedModel = payload.selectedModel;
      }
      if (typeof payload.modelQuery === "string") {
        state.modelQuery = payload.modelQuery;
      }
      if (typeof payload.sortBy === "string") {
        state.sortBy = payload.sortBy;
      }
      if (typeof payload.tokenSortBy === "string") {
        state.tokenSortBy = payload.tokenSortBy;
      }
      if (payload.tokenSortDir === "asc" || payload.tokenSortDir === "desc") {
        state.tokenSortDir = payload.tokenSortDir;
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

function setSelectedModel(model) {
  state.selectedModel = model;
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
      renderError("Open Metrics Folder is not supported by this browser.", true);
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
    renderError("No active local source to reload. Choose Open Metrics Folder.", true);
  });
}

function scoreForSort(run, key) {
  if (key === "accuracy") return run.accuracy ?? -Infinity;
  if (key === "macro_f1") return run.macroF1 ?? -Infinity;
  if (key === "cached_tokens") return run.cachedTokens ?? -Infinity;
  if (key === "attempts_total") return run.requestsTotal ?? -Infinity;
  if (key === "timestamp") return parseRunTimestampMs(run);
  return -Infinity;
}

function getFilteredRuns() {
  const query = state.modelQuery;
  const selectedTask = state.selectedTask;
  const selectedModel = state.selectedModel;
  let runs = state.runs.filter((run) => {
    if (selectedTask !== "ALL" && run.task !== selectedTask) {
      return false;
    }
    if (selectedModel !== "ALL" && run.model !== selectedModel) {
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

function renderModelControls() {
  const models = ["ALL", ...state.models];
  const counts = state.runs.reduce((acc, run) => {
    acc[run.model] = (acc[run.model] || 0) + 1;
    return acc;
  }, {});

  if (!models.includes(state.selectedModel)) {
    state.selectedModel = "ALL";
  }

  els.modelChips.innerHTML = "";
  models.forEach((model) => {
    const button = document.createElement("button");
    button.className = `chip${state.selectedModel === model ? " active" : ""}`;
    button.textContent =
      model === "ALL"
        ? `All Models (${state.runs.length})`
        : `${model} (${counts[model] || 0})`;
    button.type = "button";
    button.addEventListener("click", () => setSelectedModel(model));
    els.modelChips.appendChild(button);
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
  const source = runs.filter((run) => run.accuracy !== null);

  if (!source.length) {
    els.leaderboardChart.innerHTML = '<p class="muted">No runs with accuracy in current filter.</p>';
    return;
  }

  const groups = new Map();
  source.forEach((run) => {
    const key = run.model;
    if (!groups.has(key)) {
      groups.set(key, []);
    }
    groups.get(key).push(run);
  });

  const entries = Array.from(groups.entries()).map(([model, modelRuns]) => {
    const runsSorted = [...modelRuns].sort((a, b) => {
      if ((b.accuracy || 0) !== (a.accuracy || 0)) {
        return (b.accuracy || 0) - (a.accuracy || 0);
      }
      return parseRunTimestampMs(b) - parseRunTimestampMs(a);
    });
    if (runsSorted.length === 1) {
      return {
        type: "run",
        key: runsSorted[0].filePath,
        label: getRunModelDisplayName(runsSorted[0]),
        score: runsSorted[0].accuracy || 0,
        run: runsSorted[0],
      };
    }

    const avgAccuracy =
      runsSorted.reduce((sum, run) => sum + (run.accuracy || 0), 0) / runsSorted.length;
    const sharedSuffix = getSharedRunEffortSuffix(runsSorted);
    const groupLabel = sharedSuffix ? `${model} ${sharedSuffix}` : model;
    return {
      type: "group",
      key: model,
      label: groupLabel,
      score: avgAccuracy,
      avgAccuracy,
      runs: runsSorted,
    };
  });

  entries.sort((a, b) => {
    if (b.score !== a.score) {
      return b.score - a.score;
    }
    return a.label.localeCompare(b.label);
  });

  const maxScore = Math.max(...entries.map((entry) => entry.score), 1);
  entries.forEach((entry) => {
    if (entry.type === "run") {
      els.leaderboardChart.appendChild(
        createBarRow(
          entry.label,
          entry.score,
          maxScore,
          (value) => `${formatNum(value, 2)}%`,
          null,
          () => openRunModal(entry.run)
        )
      );
      return;
    }

    const details = document.createElement("details");
    details.className = "leaderboard-group";
    details.dataset.modelKey = entry.key;
    details.open = state.expandedLeaderboardModels.has(entry.key);
    details.addEventListener("toggle", () => {
      if (details.open) {
        state.expandedLeaderboardModels.add(entry.key);
      } else {
        state.expandedLeaderboardModels.delete(entry.key);
      }
    });

    const summary = document.createElement("summary");
    summary.className = "leaderboard-summary";
    summary.appendChild(
      createBarRow(
        `${entry.label} (${entry.runs.length})`,
        entry.avgAccuracy || 0,
        maxScore,
        (value) => `${formatNum(value, 2)}% avg`,
        null
      )
    );
    details.appendChild(summary);

    const runsWrap = document.createElement("div");
    runsWrap.className = "leaderboard-group-runs";
    entry.runs.forEach((run) => {
      runsWrap.appendChild(
        createBarRow(
          getRunModelDisplayName(run),
          run.accuracy || 0,
          maxScore,
          (value) => `${formatNum(value, 2)}%`,
          null,
          () => openRunModal(run)
        )
      );
    });
    details.appendChild(runsWrap);
    els.leaderboardChart.appendChild(details);
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
        `${run.task} / ${getRunModelDisplayName(run)}`,
        run.accuracy,
        max,
        (value) => `${formatNum(value, 2)}%`,
        "warm",
        () => openRunModal(run)
      )
    );
  });
}

function getPromptCountForRun(run) {
  return run.requestsTotal ?? run.attemptsWithUsage ?? run.predictionCount ?? run.totalExamples ?? 0;
}

function formatTotalAndAvg(total, prompts) {
  if (total === null || total === undefined || Number.isNaN(total)) {
    return "N/A";
  }
  const avg = prompts > 0 ? total / prompts : null;
  if (avg === null || Number.isNaN(avg)) {
    return `${formatNum(total, 0)} / N/A`;
  }
  return `${formatNum(total, 0)} / ${formatNum(avg, 2)}`;
}

function formatRuntimeCell(seconds, humanText = "") {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) {
    return "N/A";
  }
  const human = asTrimmedString(humanText);
  return human ? `${human} (${formatNum(seconds, 2)}s)` : `${formatNum(seconds, 2)}s`;
}

function computeTokenSignalRows(runs) {
  return runs.map((run) => {
    const prompts = getPromptCountForRun(run);
    const input = run.inputTokensTotal;
    const cached = run.cachedInputTokensTotal ?? run.cachedTokens;
    const output = run.outputTokensTotal;
    const thinking = run.thinkingTokensTotal;
    const runtime = run.overallTimeSeconds;
    return {
      run,
      modelLabel: getRunModelDisplayName(run),
      prompts,
      input,
      cached,
      output,
      thinking,
      runtime,
      runtimeHuman: run.overallTimeHuman,
    };
  });
}

function getTokenSortValue(row, key) {
  if (key === "model") return row.modelLabel.toLowerCase();
  if (key === "prompts") return row.prompts ?? -Infinity;
  if (key === "input") return row.input ?? -Infinity;
  if (key === "cached") return row.cached ?? -Infinity;
  if (key === "output") return row.output ?? -Infinity;
  if (key === "thinking") return row.thinking ?? -Infinity;
  if (key === "runtime_seconds") return row.runtime ?? -Infinity;
  return -Infinity;
}

function sortTokenRows(rows) {
  const key = state.tokenSortBy;
  const dir = state.tokenSortDir === "asc" ? 1 : -1;
  rows.sort((a, b) => {
    const va = getTokenSortValue(a, key);
    const vb = getTokenSortValue(b, key);
    if (typeof va === "string" || typeof vb === "string") {
      const cmp = String(va).localeCompare(String(vb));
      if (cmp !== 0) return cmp * dir;
    } else if (vb !== va) {
      return (va - vb) * dir;
    }
    const tsCmp = parseRunTimestampMs(b.run) - parseRunTimestampMs(a.run);
    if (tsCmp !== 0) return tsCmp;
    return b.modelLabel.localeCompare(a.modelLabel);
  });
}

function toggleTokenSort(key) {
  if (state.tokenSortBy === key) {
    state.tokenSortDir = state.tokenSortDir === "desc" ? "asc" : "desc";
  } else {
    state.tokenSortBy = key;
    state.tokenSortDir = key === "model" ? "asc" : "desc";
  }
  persistUiState();
  renderTokenSignals(state.filtered);
}

function createTokenHeader(label, key) {
  const th = document.createElement("th");
  const button = document.createElement("button");
  button.type = "button";
  button.className = "sort-head";
  const active = state.tokenSortBy === key;
  const arrow = !active ? "" : state.tokenSortDir === "desc" ? " v" : " ^";
  button.textContent = `${label}${arrow}`;
  button.addEventListener("click", () => toggleTokenSort(key));
  th.appendChild(button);
  return th;
}

function renderTokenSignals(runs) {
  els.tokenChart.innerHTML = "";
  const rows = computeTokenSignalRows(runs).filter(
    (row) =>
      row.prompts > 0 ||
      row.input !== null ||
      row.cached !== null ||
      row.output !== null ||
      row.thinking !== null ||
      row.runtime !== null
  );

  if (!rows.length) {
    els.tokenChart.innerHTML = '<p class="muted">No token/request metadata for current filter.</p>';
    return;
  }

  sortTokenRows(rows);

  const wrap = document.createElement("div");
  wrap.className = "token-table-wrap";
  const table = document.createElement("table");
  table.className = "token-table";
  const thead = document.createElement("thead");
  const trHead = document.createElement("tr");
  trHead.appendChild(createTokenHeader("Model", "model"));
  trHead.appendChild(createTokenHeader("Prompts", "prompts"));
  trHead.appendChild(createTokenHeader("Input Tokens (total / avg)", "input"));
  trHead.appendChild(createTokenHeader("Cached Tokens (total / avg)", "cached"));
  trHead.appendChild(createTokenHeader("Output Tokens (total / avg)", "output"));
  trHead.appendChild(createTokenHeader("Thinking Tokens (total / avg)", "thinking"));
  trHead.appendChild(createTokenHeader("Total Runtime", "runtime_seconds"));
  thead.appendChild(trHead);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.className = "clickable-row";
    tr.innerHTML = `
      <td>${row.modelLabel}</td>
      <td class="mono">${formatNum(row.prompts, 0)}</td>
      <td class="mono">${formatTotalAndAvg(row.input, row.prompts)}</td>
      <td class="mono">${formatTotalAndAvg(row.cached, row.prompts)}</td>
      <td class="mono">${formatTotalAndAvg(row.output, row.prompts)}</td>
      <td class="mono">${formatTotalAndAvg(row.thinking, row.prompts)}</td>
      <td class="mono">${formatRuntimeCell(row.runtime, row.runtimeHuman)}</td>
    `;
    tr.addEventListener("click", () => openRunModal(row.run));
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  wrap.appendChild(table);
  els.tokenChart.appendChild(wrap);
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

function createLinkDetailItem(label, href, hint = "") {
  const wrapper = document.createElement("div");
  wrapper.className = "detail-item";
  const labelEl = document.createElement("span");
  labelEl.className = "label";
  labelEl.textContent = label;

  const valueEl = document.createElement("div");
  valueEl.className = "value mono";

  const link = document.createElement("a");
  link.href = href;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.className = "detail-link";
  link.textContent = "Open";
  valueEl.appendChild(link);

  const hintText = hint || href;
  if (hintText) {
    const hintEl = document.createElement("div");
    hintEl.className = "muted";
    hintEl.textContent = hintText;
    valueEl.appendChild(hintEl);
  }

  wrapper.appendChild(labelEl);
  wrapper.appendChild(valueEl);
  return wrapper;
}

function createChartsDetailItem(charts) {
  const wrapper = document.createElement("div");
  wrapper.className = "detail-item detail-raw";
  const labelEl = document.createElement("span");
  labelEl.className = "label";
  labelEl.textContent = "Metric Charts";
  wrapper.appendChild(labelEl);

  const grid = document.createElement("div");
  grid.className = "chart-preview-grid";

  charts.forEach((chart) => {
    const link = document.createElement("a");
    link.href = chart.href;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.className = "chart-preview";

    const img = document.createElement("img");
    img.src = chart.href;
    img.alt = chart.label;
    img.loading = "lazy";
    img.addEventListener("error", () => {
      link.classList.add("missing");
      img.remove();
    });

    const caption = document.createElement("span");
    caption.textContent = chart.label;

    link.appendChild(img);
    link.appendChild(caption);
    grid.appendChild(link);
  });

  wrapper.appendChild(grid);
  return wrapper;
}

function buildRunResourceLinks(run) {
  const metricsPath = run.filePath;
  const canonicalMetrics = String(run.fileName || "").toLowerCase().endsWith("__metrics.json");
  const heatmapFile = canonicalMetrics
    ? `${run.runStem}__heatmap.png`
    : `${run.runStem}_confusion_heatmap.png`;
  const calibrationFile = canonicalMetrics
    ? `${run.runStem}__calibration.png`
    : `${run.runStem}_calibration.png`;
  const logFile = `${run.runStem}.log`;
  const outputFile = `${run.runStem}.csv`;
  const inputFile = `${run.task}.csv`;

  return {
    metrics: metricsPath,
    heatmap: replaceFileNameInPath(metricsPath, heatmapFile),
    calibration: replaceFileNameInPath(metricsPath, calibrationFile),
    log: replaceFileNameInPath(mapMetricsPathToSiblingDir(metricsPath, "logs"), logFile),
    output: replaceFileNameInPath(mapMetricsPathToSiblingDir(metricsPath, "output"), outputFile),
    input: replaceFileNameInPath(mapMetricsPathToSiblingDir(metricsPath, "input"), inputFile),
  };
}

function fillRunDetailsContent(run) {
  if (!run) {
    els.runModalTitle.textContent = "Run Details";
    els.runModalMeta.textContent = "No run selected.";
    els.runModalContent.innerHTML = "";
    return;
  }
  const runThinkingLevel = getConfiguredControl(run, "thinking_level");
  const runReasoningEffort =
    getConfiguredControl(run, "reasoning_effort") || getConfiguredControl(run, "effort");
  const detailPairs = [
    ["Task", run.task],
    ["Task Description", run.taskDescription],
    ["Tags", run.tagsDisplay],
    ["Model", run.model],
    ["Thinking Level", runThinkingLevel],
    ["Reasoning/Effort", runReasoningEffort],
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
    ["Provider", run.provider || "N/A"],
    ["Model Requested", run.modelDetails.model_requested || "N/A"],
    ["Model For Requests", run.modelDetails.model_for_requests || "N/A"],
  ];

  els.runModalTitle.textContent = `${run.task} / ${run.model}`;
  els.runModalMeta.textContent = run.fileName;
  els.runModalContent.innerHTML = "";

  detailPairs.forEach(([label, value]) => {
    els.runModalContent.appendChild(createDetailItem(label, value));
  });

  const links = buildRunResourceLinks(run);
  els.runModalContent.appendChild(
    createLinkDetailItem("Metrics JSON", links.metrics, getFileNameFromPath(links.metrics))
  );
  els.runModalContent.appendChild(
    createLinkDetailItem("Heatmap", links.heatmap, getFileNameFromPath(links.heatmap))
  );
  els.runModalContent.appendChild(
    createLinkDetailItem("Calibration", links.calibration, getFileNameFromPath(links.calibration))
  );
  els.runModalContent.appendChild(
    createLinkDetailItem("Log File", links.log, getFileNameFromPath(links.log))
  );
  els.runModalContent.appendChild(
    createLinkDetailItem("Output CSV", links.output, getFileNameFromPath(links.output))
  );
  els.runModalContent.appendChild(
    createLinkDetailItem("Input CSV", links.input, getFileNameFromPath(links.input))
  );
  els.runModalContent.appendChild(
    createChartsDetailItem([
      { label: "Heatmap Preview", href: links.heatmap },
      { label: "Calibration Preview", href: links.calibration },
    ])
  );

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
  renderModelControls();
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
  state.models = [...new Set(result.runs.map((run) => run.model))].sort((a, b) => a.localeCompare(b));
  state.sourceMode = result.mode;
  state.sourceFileCount = result.fileCount;
  state.warnings = result.warnings;
  state.expandedLeaderboardModels = new Set();

  if (state.selectedTask !== "ALL" && !state.tasks.includes(state.selectedTask)) {
    state.selectedTask = "ALL";
  }
  if (state.selectedModel !== "ALL" && !state.models.includes(state.selectedModel)) {
    state.selectedModel = "ALL";
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
      ? " In file:// mode, use Open Metrics Folder."
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
    updateSourceStatus();
    els.heroSubtitle.textContent = "Choose a local data source to load metrics.";
    return;
  }

  await activateServerSource();
}

init();

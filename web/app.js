const STORAGE_KEY = "dhAIBench.metricsDashboard.state.v1";
const METRICS_MANIFEST_PATH = "./metrics-manifest.json";
const METRICS_SERVER_DIR = "../data/metrics";
const METRICS_SERVER_DIR_CANDIDATES = [METRICS_SERVER_DIR, "./metrics", "./data/metrics"];
const MOBILE_LAYOUT_BREAKPOINT = 900;
const mobileLayoutQuery =
  typeof window.matchMedia === "function"
    ? window.matchMedia(`(max-width: ${MOBILE_LAYOUT_BREAKPOINT}px)`)
    : null;

const state = {
  runs: [],
  filtered: [],
  tasks: [],
  models: [],
  tags: [],
  selectedTasks: [],
  selectedModels: [],
  selectedTags: [],
  selectedRunPath: null,
  sortBy: "accuracy",
  leaderboardTab: "chart",
  radarAxis: "task",
  hideNoAccuracy: false,
  theme: "dark",
  sourceMode: "none",
  sourceFileCount: 0,
  warnings: [],
  activeDirectoryHandle: null,
  activeFiles: [],
  expandedLeaderboardModels: new Set(),
  mobileSidebarOpen: false,
};

const els = {
  dashboardSidebar: document.querySelector("#dashboardSidebar"),
  mobileSidebarToggle: document.querySelector("#mobileSidebarToggle"),
  mobileFilterSummary: document.querySelector("#mobileFilterSummary"),
  sidebarCloseBtn: document.querySelector("#sidebarCloseBtn"),
  sidebarBackdrop: document.querySelector("#sidebarBackdrop"),
  taskSelect: document.querySelector("#taskSelect"),
  taskChipList: document.querySelector("#taskChipList"),
  modelSelect: document.querySelector("#modelSelect"),
  modelChipList: document.querySelector("#modelChipList"),
  sortSelect: document.querySelector("#sortSelect"),
  hideNoAccuracy: document.querySelector("#hideNoAccuracy"),
  themeToggle: document.querySelector("#themeToggle"),
  tagChips: document.querySelector("#tagChips"),
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
  leaderboardTabs: document.querySelector("#leaderboardTabs"),
  leaderboardChart: document.querySelector("#leaderboardChart"),
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

const METRIC_KEYS = new Set(["accuracy", "macro_f1", "macro_precision", "macro_recall"]);
const PERCENT_METRIC_KEYS = new Set(["accuracy", "macro_f1", "macro_precision", "macro_recall"]);
const RADAR_AXIS_KEYS = new Set(["task", "tag"]);
const LEADERBOARD_TAB_KEYS = new Set(["chart", "table", "best_by_task"]);
const LEADERBOARD_TABLE_METRICS = ["accuracy", "macro_f1", "macro_precision", "macro_recall"];

const METRIC_LABELS = {
  accuracy: "Accuracy",
  macro_f1: "Macro F1",
  macro_precision: "Macro Precision",
  macro_recall: "Macro Recall",
};

const RADAR_AXIS_LABELS = {
  task: "Task",
  tag: "Tag",
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
  const macroPrecision = toPct(safeNum(payload.macro_precision));
  const macroRecall = toPct(safeNum(payload.macro_recall));
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
    macroPrecision,
    macroRecall,
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
      (
        safeNum(payload.accuracy) !== null ||
        safeNum(payload.macro_precision) !== null ||
        safeNum(payload.macro_recall) !== null ||
        safeNum(payload.macro_f1) !== null
      ),
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
    selectedTasks: state.selectedTasks,
    selectedModels: state.selectedModels,
    selectedTags: state.selectedTags,
    sortBy: state.sortBy,
    leaderboardTab: state.leaderboardTab,
    radarAxis: state.radarAxis,
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
      if (Array.isArray(payload.selectedTasks)) {
        state.selectedTasks = uniqueNonEmptyStrings(payload.selectedTasks);
      } else if (typeof payload.selectedTask === "string" && payload.selectedTask !== "ALL") {
        state.selectedTasks = [payload.selectedTask];
      }
      if (Array.isArray(payload.selectedModels)) {
        state.selectedModels = uniqueNonEmptyStrings(payload.selectedModels);
      } else if (typeof payload.selectedModel === "string" && payload.selectedModel !== "ALL") {
        state.selectedModels = [payload.selectedModel];
      }
      if (Array.isArray(payload.selectedTags)) {
        state.selectedTags = uniqueNonEmptyStrings(payload.selectedTags);
      }
      if (typeof payload.sortBy === "string" && METRIC_KEYS.has(payload.sortBy)) {
        state.sortBy = payload.sortBy;
      }
      if (typeof payload.leaderboardTab === "string" && LEADERBOARD_TAB_KEYS.has(payload.leaderboardTab)) {
        state.leaderboardTab = payload.leaderboardTab;
      }
      if (typeof payload.radarAxis === "string" && RADAR_AXIS_KEYS.has(payload.radarAxis)) {
        state.radarAxis = payload.radarAxis;
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
  els.sortSelect.value = state.sortBy;
  els.hideNoAccuracy.checked = state.hideNoAccuracy;
  els.themeToggle.checked = state.theme === "dark";
}

function applyTheme() {
  document.documentElement.setAttribute("data-theme", state.theme);
}

function isMobileLayout() {
  return mobileLayoutQuery ? mobileLayoutQuery.matches : window.innerWidth <= MOBILE_LAYOUT_BREAKPOINT;
}

function setMobileSidebarOpen(nextOpen) {
  const shouldOpen = isMobileLayout() && Boolean(nextOpen);
  state.mobileSidebarOpen = shouldOpen;
  document.body.classList.toggle("mobile-sidebar-open", shouldOpen);
  els.mobileSidebarToggle.setAttribute("aria-expanded", shouldOpen ? "true" : "false");
  els.dashboardSidebar.setAttribute("aria-hidden", shouldOpen ? "false" : String(isMobileLayout()));

  if (shouldOpen) {
    requestAnimationFrame(() => {
      els.sidebarCloseBtn.focus();
    });
  }
}

function updateMobileFilterSummary() {
  const activeFilterCount =
    (state.selectedTasks.length ? 1 : 0) +
    (state.selectedModels.length ? 1 : 0) +
    (state.selectedTags.length ? 1 : 0) +
    (state.hideNoAccuracy ? 1 : 0);
  const summaryText = activeFilterCount ? `${activeFilterCount} active` : "All";

  els.mobileFilterSummary.textContent = summaryText;
  els.mobileSidebarToggle.title = activeFilterCount ? `${summaryText} filters` : "All filters visible";
}

function sanitizeSelections(selectedValues, allowedValues) {
  const allowedSet = new Set(allowedValues || []);
  return uniqueNonEmptyStrings(selectedValues || []).filter((value) => allowedSet.has(value));
}

function isAllSelected(values) {
  return !Array.isArray(values) || values.length === 0;
}

function setSelectedTasks(values) {
  state.selectedTasks = sanitizeSelections(values, state.tasks);
  persistUiState();
  render();
}

function setSelectedModels(values) {
  state.selectedModels = sanitizeSelections(values, state.models);
  persistUiState();
  render();
}

function setSelectedTags(values) {
  state.selectedTags = sanitizeSelections(values, state.tags);
  persistUiState();
  render();
}

function toggleTaskSelection(task) {
  if (task === "ALL") {
    setSelectedTasks([]);
    return;
  }
  if (state.selectedTasks.includes(task)) {
    setSelectedTasks(state.selectedTasks.filter((value) => value !== task));
    return;
  }
  setSelectedTasks([...state.selectedTasks, task]);
}

function toggleModelSelection(model) {
  if (model === "ALL") {
    setSelectedModels([]);
    return;
  }
  if (state.selectedModels.includes(model)) {
    setSelectedModels(state.selectedModels.filter((value) => value !== model));
    return;
  }
  setSelectedModels([...state.selectedModels, model]);
}

function toggleTagSelection(tag) {
  if (tag === "ALL") {
    setSelectedTags([]);
    return;
  }
  if (state.selectedTags.includes(tag)) {
    setSelectedTags(state.selectedTags.filter((value) => value !== tag));
    return;
  }
  setSelectedTags([...state.selectedTags, tag]);
}

function setRadarAxis(axis) {
  if (!RADAR_AXIS_KEYS.has(axis) || state.radarAxis === axis) {
    return;
  }
  state.radarAxis = axis;
  persistUiState();
  renderTokenSignals(state.filtered);
}

function setLeaderboardTab(tab) {
  if (!LEADERBOARD_TAB_KEYS.has(tab) || state.leaderboardTab === tab) {
    return;
  }
  state.leaderboardTab = tab;
  persistUiState();
  renderLeaderboard(state.filtered);
}

function syncTaskSelectValue() {
  const selected = new Set(state.selectedTasks);
  Array.from(els.taskSelect.options).forEach((option) => {
    option.selected = selected.has(option.value);
  });
}

function syncModelSelectValue() {
  const selected = new Set(state.selectedModels);
  Array.from(els.modelSelect.options).forEach((option) => {
    option.selected = selected.has(option.value);
  });
}

function getSelectValues(selectElement) {
  return Array.from(selectElement.selectedOptions)
    .map((option) => option.value)
    .filter(Boolean);
}

function clearSelectionForAll(values) {
  const normalized = uniqueNonEmptyStrings(values || []);
  if (!normalized.length) {
    return [];
  }
  if (!normalized.includes("ALL")) {
    return normalized;
  }
  if (normalized.length === 1) {
    return [];
  }
  return normalized.filter((value) => value !== "ALL");
}

function setTaskSelectionFromSelect(values) {
  setSelectedTasks(clearSelectionForAll(values));
}

function setModelSelectionFromSelect(values) {
  setSelectedModels(clearSelectionForAll(values));
}

function renderChoiceChipList(container, options, selectedValues, allLabel, onToggle) {
  container.innerHTML = "";

  const items = [{ value: "ALL", label: allLabel, active: isAllSelected(selectedValues) }].concat(
    options.map((option) => ({
      value: option,
      label: option,
      active: selectedValues.includes(option),
    }))
  );

  items.forEach((item) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `choice-chip${item.active ? " active" : ""}`;
    button.textContent = item.label;
    button.setAttribute("aria-pressed", item.active ? "true" : "false");
    button.addEventListener("click", () => onToggle(item.value));
    container.appendChild(button);
  });
}

function setupFilters() {
  els.taskSelect.addEventListener("change", () => {
    setTaskSelectionFromSelect(getSelectValues(els.taskSelect));
  });

  els.modelSelect.addEventListener("change", () => {
    setModelSelectionFromSelect(getSelectValues(els.modelSelect));
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

function setupResponsiveShell() {
  els.mobileSidebarToggle.addEventListener("click", () => {
    setMobileSidebarOpen(!state.mobileSidebarOpen);
  });
  els.sidebarCloseBtn.addEventListener("click", () => {
    setMobileSidebarOpen(false);
  });
  els.sidebarBackdrop.addEventListener("click", () => {
    setMobileSidebarOpen(false);
  });

  const handleViewportChange = () => {
    if (!isMobileLayout()) {
      setMobileSidebarOpen(false);
      return;
    }
    els.dashboardSidebar.setAttribute("aria-hidden", state.mobileSidebarOpen ? "false" : "true");
  };

  if (mobileLayoutQuery) {
    if (typeof mobileLayoutQuery.addEventListener === "function") {
      mobileLayoutQuery.addEventListener("change", handleViewportChange);
    } else if (typeof mobileLayoutQuery.addListener === "function") {
      mobileLayoutQuery.addListener(handleViewportChange);
    }
  } else {
    window.addEventListener("resize", handleViewportChange);
  }

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && state.mobileSidebarOpen) {
      setMobileSidebarOpen(false);
    }
  });

  handleViewportChange();
  updateMobileFilterSummary();
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

function getMetricValueForRun(run, key) {
  if (key === "accuracy") return run.accuracy;
  if (key === "macro_f1") return run.macroF1;
  if (key === "macro_precision") return run.macroPrecision;
  if (key === "macro_recall") return run.macroRecall;
  return null;
}

function scoreForSort(run, key) {
  return getMetricValueForRun(run, key) ?? -Infinity;
}

function isPercentMetric(metricKey) {
  return PERCENT_METRIC_KEYS.has(metricKey);
}

function resolveLeaderboardBarMax(metricKey, values) {
  if (isPercentMetric(metricKey)) {
    return 100;
  }
  const finiteValues = (values || []).filter((value) => typeof value === "number" && Number.isFinite(value));
  if (!finiteValues.length) {
    return 1;
  }
  return Math.max(...finiteValues, 1);
}

function formatMetric(metricKey, value, digits = 2, suffix = "") {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "N/A";
  }
  const unit = isPercentMetric(metricKey) ? "%" : "";
  return `${formatNum(value, digits)}${unit}${suffix}`;
}

function getRunSampleSize(run) {
  const candidates = [run.totalExamples, run.predictionCount, run.truthLabelCount];
  for (const value of candidates) {
    if (typeof value === "number" && Number.isFinite(value) && value > 1) {
      return value;
    }
  }
  return null;
}

function computeApproximateCi95(metricValue, sampleSize) {
  if (typeof metricValue !== "number" || !Number.isFinite(metricValue)) {
    return null;
  }
  if (typeof sampleSize !== "number" || !Number.isFinite(sampleSize) || sampleSize <= 1) {
    return null;
  }
  const p = Math.max(0, Math.min(1, metricValue / 100));
  const standardError = Math.sqrt((p * (1 - p)) / sampleSize);
  const margin = 1.96 * standardError * 100;
  return {
    low: Math.max(0, metricValue - margin),
    high: Math.min(100, metricValue + margin),
    sampleSize,
  };
}

function getRunMetricConfidence(run, metricKey) {
  if (!isPercentMetric(metricKey)) {
    return null;
  }
  const metricValue = getMetricValueForRun(run, metricKey);
  const sampleSize = getRunSampleSize(run);
  return computeApproximateCi95(metricValue, sampleSize);
}

function getMeanMetricConfidence(runs, metricKey, meanValue) {
  if (!isPercentMetric(metricKey)) {
    return null;
  }
  const ciRows = runs
    .map((run) => getRunMetricConfidence(run, metricKey))
    .filter((row) => row && typeof row.low === "number" && typeof row.high === "number");
  if (!ciRows.length) {
    return null;
  }
  const standardErrorsSquared = ciRows.map((ci) => {
    const margin = (ci.high - ci.low) / 2;
    const sePercent = margin / 1.96;
    const se = sePercent / 100;
    return se * se;
  });
  const meanSe = Math.sqrt(standardErrorsSquared.reduce((sum, value) => sum + value, 0)) / ciRows.length;
  const margin = 1.96 * meanSe * 100;
  return {
    low: Math.max(0, meanValue - margin),
    high: Math.min(100, meanValue + margin),
    sampleSize: ciRows.reduce((sum, ci) => sum + (ci.sampleSize || 0), 0),
  };
}

function formatCiRange(ci) {
  if (!ci) {
    return "";
  }
  return ` | CI ${formatNum(ci.low, 2)}-${formatNum(ci.high, 2)}%`;
}

function quantileSorted(sortedValues, q) {
  if (!Array.isArray(sortedValues) || !sortedValues.length) {
    return null;
  }
  const clampedQ = Math.max(0, Math.min(1, q));
  const index = (sortedValues.length - 1) * clampedQ;
  const lowIndex = Math.floor(index);
  const highIndex = Math.ceil(index);
  const lowValue = sortedValues[lowIndex];
  const highValue = sortedValues[highIndex];
  if (lowIndex === highIndex) {
    return lowValue;
  }
  const weight = index - lowIndex;
  return lowValue + (highValue - lowValue) * weight;
}

function getDistributionStats(values) {
  const numeric = (values || [])
    .filter((value) => typeof value === "number" && Number.isFinite(value))
    .sort((a, b) => a - b);
  if (!numeric.length) {
    return null;
  }
  return {
    min: numeric[0],
    q1: quantileSorted(numeric, 0.25),
    median: quantileSorted(numeric, 0.5),
    q3: quantileSorted(numeric, 0.75),
    max: numeric[numeric.length - 1],
    values: numeric,
  };
}

function getFilteredRuns() {
  const selectedTasks = state.selectedTasks;
  const selectedModels = state.selectedModels;
  const selectedTags = state.selectedTags;
  let runs = state.runs.filter((run) => {
    if (!isAllSelected(selectedTasks) && !selectedTasks.includes(run.task)) {
      return false;
    }
    if (!isAllSelected(selectedModels) && !selectedModels.includes(run.model)) {
      return false;
    }
    if (!isAllSelected(selectedTags)) {
      const runTags = Array.isArray(run.tags) ? run.tags : [];
      if (!runTags.some((tag) => selectedTags.includes(tag))) {
        return false;
      }
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
    const f1A = scoreForSort(a, "macro_f1");
    const f1B = scoreForSort(b, "macro_f1");
    if (f1B !== f1A) return f1B - f1A;
    return parseRunTimestampMs(b) - parseRunTimestampMs(a);
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

  state.selectedTasks = sanitizeSelections(state.selectedTasks, state.tasks);
  if (isAllSelected(state.selectedTasks)) {
    const first = els.taskSelect.querySelector('option[value="ALL"]');
    if (first) {
      first.selected = true;
    }
  } else {
    syncTaskSelectValue();
  }

  renderChoiceChipList(els.taskChipList, state.tasks, state.selectedTasks, "All Tasks", toggleTaskSelection);
}

function renderModelControls() {
  const models = ["ALL", ...state.models];

  els.modelSelect.innerHTML = models
    .map((model) => `<option value="${model}">${model === "ALL" ? "All Models" : model}</option>`)
    .join("");

  state.selectedModels = sanitizeSelections(state.selectedModels, state.models);
  if (isAllSelected(state.selectedModels)) {
    const first = els.modelSelect.querySelector('option[value="ALL"]');
    if (first) {
      first.selected = true;
    }
  } else {
    syncModelSelectValue();
  }

  renderChoiceChipList(els.modelChipList, state.models, state.selectedModels, "All Models", toggleModelSelection);
}

function renderTagControls() {
  const tags = ["ALL", ...state.tags];
  state.selectedTags = sanitizeSelections(state.selectedTags, state.tags);

  const counts = state.runs.reduce((acc, run) => {
    const runTags = Array.isArray(run.tags) ? run.tags : [];
    runTags.forEach((tag) => {
      acc[tag] = (acc[tag] || 0) + 1;
    });
    return acc;
  }, {});

  els.tagChips.innerHTML = "";
  tags.forEach((tag) => {
    const isActive = tag === "ALL" ? isAllSelected(state.selectedTags) : state.selectedTags.includes(tag);
    const button = document.createElement("button");
    button.className = `chip${isActive ? " active" : ""}`;
    button.type = "button";
    button.textContent = tag === "ALL" ? `All Tags (${state.runs.length})` : `${tag} (${counts[tag] || 0})`;
    button.addEventListener("click", () => toggleTagSelection(tag));
    els.tagChips.appendChild(button);
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

  const selectedTaskCount = state.selectedTasks.length;
  els.heroTitle.textContent = selectedTaskCount === 0 ? "All Tasks" : `${selectedTaskCount} Task(s) Selected`;
  const selectedModelCount = state.selectedModels.length;
  const modelScope = selectedModelCount === 0 ? "all models" : `${selectedModelCount} model(s)`;
  els.heroSubtitle.textContent = `${formatNum(runs.length, 0)} runs in view (${modelScope}).`;
}

function createBarRow(
  label,
  value,
  max,
  formatter,
  colorClass,
  onClick,
  ci = null,
  insideBarText = "",
  options = {}
) {
  const node = els.barRowTemplate.content.firstElementChild.cloneNode(true);
  const labelEl = node.querySelector(".bar-label");
  const trackEl = node.querySelector(".bar-track");
  const fillEl = node.querySelector(".bar-fill");
  const valueEl = node.querySelector(".bar-value");
  const labelText = asTrimmedString(label);
  labelEl.textContent = labelText;
  labelEl.title = label;

  const rowClass = asTrimmedString(options.rowClass);
  if (rowClass) {
    node.classList.add(rowClass);
  }

  const ratio = max > 0 ? Math.max(0, Math.min(1, value / max)) : 0;
  fillEl.style.width = `${ratio * 100}%`;

  if (colorClass === "warm") {
    fillEl.style.background = "linear-gradient(90deg, #f59e0b, #ea580c)";
  } else if (colorClass === "blue") {
    fillEl.style.background = "linear-gradient(90deg, #2563eb, #1e3a8a)";
  }

  const badges = Array.isArray(options.badges) ? options.badges.filter((badge) => asTrimmedString(badge)) : [];
  if (badges.length) {
    labelEl.textContent = "";
    const textNode = document.createElement("span");
    textNode.className = "bar-label-text";
    textNode.textContent = labelText;
    labelEl.appendChild(textNode);
    const badgesWrap = document.createElement("span");
    badgesWrap.className = "bar-label-badges";
    badges.forEach((badge) => {
      const token = document.createElement("span");
      token.className = `bar-label-badge bar-label-badge-${String(badge).toLowerCase()}`;
      token.textContent = badge;
      badgesWrap.appendChild(token);
    });
    labelEl.appendChild(badgesWrap);
  }

  const referenceBand = options.referenceBand;
  if (
    referenceBand &&
    typeof referenceBand.low === "number" &&
    Number.isFinite(referenceBand.low) &&
    typeof referenceBand.high === "number" &&
    Number.isFinite(referenceBand.high) &&
    max > 0
  ) {
    const lowClamped = Math.max(0, Math.min(max, referenceBand.low));
    const highClamped = Math.max(lowClamped, Math.min(max, referenceBand.high));
    const bandEl = document.createElement("span");
    bandEl.className = "bar-reference-band";
    bandEl.style.left = `${(lowClamped / max) * 100}%`;
    bandEl.style.width = `${Math.max(((highClamped - lowClamped) / max) * 100, 0.8)}%`;
    bandEl.title = `Grouped IQR ${formatNum(lowClamped, 2)}-${formatNum(highClamped, 2)}%`;
    trackEl.appendChild(bandEl);
  }

  const distribution = options.distribution;
  if (distribution && distribution.values && Array.isArray(distribution.values) && distribution.values.length && max > 0) {
    const distributionWrap = document.createElement("span");
    distributionWrap.className = "bar-distribution";
    const stats = getDistributionStats(distribution.values);
    if (stats) {
      const lowClamped = Math.max(0, Math.min(max, stats.min));
      const highClamped = Math.max(lowClamped, Math.min(max, stats.max));
      const whiskerEl = document.createElement("span");
      whiskerEl.className = "bar-distribution-whisker";
      whiskerEl.style.left = `${(lowClamped / max) * 100}%`;
      whiskerEl.style.width = `${Math.max(((highClamped - lowClamped) / max) * 100, 0.5)}%`;
      distributionWrap.appendChild(whiskerEl);

      const q1Clamped = Math.max(0, Math.min(max, stats.q1));
      const q3Clamped = Math.max(q1Clamped, Math.min(max, stats.q3));
      const boxEl = document.createElement("span");
      boxEl.className = "bar-distribution-box";
      boxEl.style.left = `${(q1Clamped / max) * 100}%`;
      boxEl.style.width = `${Math.max(((q3Clamped - q1Clamped) / max) * 100, 0.8)}%`;
      distributionWrap.appendChild(boxEl);

      const medianClamped = Math.max(0, Math.min(max, stats.median));
      const medianEl = document.createElement("span");
      medianEl.className = "bar-distribution-median";
      medianEl.style.left = `${(medianClamped / max) * 100}%`;
      distributionWrap.appendChild(medianEl);

      stats.values.forEach((entryValue) => {
        const clamped = Math.max(0, Math.min(max, entryValue));
        const tick = document.createElement("span");
        tick.className = "bar-distribution-tick";
        tick.style.left = `${(clamped / max) * 100}%`;
        distributionWrap.appendChild(tick);
      });
      distributionWrap.title = `Runs: min ${formatNum(stats.min, 2)}%, median ${formatNum(stats.median, 2)}%, max ${formatNum(
        stats.max,
        2
      )}%`;
      trackEl.appendChild(distributionWrap);
    }
  }

  const insideLabel = asTrimmedString(insideBarText);
  if (insideLabel) {
    const trackLabel = document.createElement("span");
    trackLabel.className = "bar-track-label";
    trackLabel.textContent = insideLabel;
    trackLabel.title = insideLabel;
    trackEl.classList.add("with-label");
    trackEl.appendChild(trackLabel);
  }

  if (ci && typeof ci.low === "number" && typeof ci.high === "number" && max > 0) {
    const lowClamped = Math.max(0, Math.min(max, ci.low));
    const highClamped = Math.max(lowClamped, Math.min(max, ci.high));
    const ciRange = document.createElement("span");
    ciRange.className = "bar-ci-range";
    ciRange.style.left = `${(lowClamped / max) * 100}%`;
    ciRange.style.width = `${Math.max(((highClamped - lowClamped) / max) * 100, 0.6)}%`;
    ciRange.title = `95% CI ${formatNum(lowClamped, 2)}-${formatNum(highClamped, 2)}%`;
    trackEl.appendChild(ciRange);
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

  valueEl.textContent = formatter(value, ci);
  return node;
}

function getConcatenatedTaskLabel(runs) {
  const tasks = [...new Set((runs || []).map((run) => asTrimmedString(run.task)).filter(Boolean))].sort((a, b) =>
    a.localeCompare(b)
  );
  return tasks.join(" + ");
}

function renderLeaderboardTabControls() {
  if (!els.leaderboardTabs) {
    return;
  }
  els.leaderboardTabs.innerHTML = "";
  const tabs = [
    { key: "chart", label: "Chart" },
    { key: "table", label: "Metrics Table" },
    { key: "best_by_task", label: "Best Run Per Task" },
  ];
  tabs.forEach((tab) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `leaderboard-tab-btn${state.leaderboardTab === tab.key ? " active" : ""}`;
    button.textContent = tab.label;
    button.setAttribute("aria-pressed", state.leaderboardTab === tab.key ? "true" : "false");
    button.addEventListener("click", () => setLeaderboardTab(tab.key));
    els.leaderboardTabs.appendChild(button);
  });
}

function renderLeaderboardChart(container, runs) {
  const metricKey = state.sortBy;
  const metricLabel = METRIC_LABELS[metricKey] || metricKey;
  const source = runs.filter((run) => getMetricValueForRun(run, metricKey) !== null);

  if (!source.length) {
    container.innerHTML = `<p class="muted">No runs with ${metricLabel.toLowerCase()} in current filter.</p>`;
    return;
  }

  const ciNote = document.createElement("p");
  ciNote.className = "leaderboard-ci-note muted";
  ciNote.textContent = "95% CI is approximated from each run's evaluated examples.";
  container.appendChild(ciNote);
  const hasMultipleTasks = new Set(runs.map((run) => asTrimmedString(run.task)).filter(Boolean)).size > 1;

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
      const scoreA = scoreForSort(a, metricKey);
      const scoreB = scoreForSort(b, metricKey);
      if (scoreB !== scoreA) {
        return scoreB - scoreA;
      }
      return parseRunTimestampMs(b) - parseRunTimestampMs(a);
    });
    if (runsSorted.length === 1) {
      const run = runsSorted[0];
      const score = scoreForSort(run, metricKey);
      return {
        type: "run",
        key: run.filePath,
        label: getRunModelDisplayName(run),
        score: Number.isFinite(score) ? score : 0,
        ci: getRunMetricConfidence(run, metricKey),
        run,
      };
    }

    const avgMetric =
      runsSorted.reduce((sum, run) => sum + (getMetricValueForRun(run, metricKey) ?? 0), 0) / runsSorted.length;
    const sharedSuffix = getSharedRunEffortSuffix(runsSorted);
    const groupLabel = sharedSuffix ? `${model} ${sharedSuffix}` : model;
    return {
      type: "group",
      key: model,
      label: groupLabel,
      score: avgMetric,
      avgMetric,
      ci: getMeanMetricConfidence(runsSorted, metricKey, avgMetric),
      runs: runsSorted,
    };
  });

  entries.sort((a, b) => {
    if (b.score !== a.score) {
      return b.score - a.score;
    }
    return a.label.localeCompare(b.label);
  });

  const groupedEntries = entries.filter((entry) => entry.type === "group");
  const rankedRuns = [...source].sort((a, b) => {
    const scoreA = scoreForSort(a, metricKey);
    const scoreB = scoreForSort(b, metricKey);
    if (scoreB !== scoreA) {
      return scoreB - scoreA;
    }
    return parseRunTimestampMs(b) - parseRunTimestampMs(a);
  });
  const topRunPath = rankedRuns.length ? rankedRuns[0].filePath : null;

  if (groupedEntries.length) {
    const groupedSummary = document.createElement("p");
    groupedSummary.className = "leaderboard-ci-note muted";
    groupedSummary.textContent =
      groupedEntries.length > 1
        ? "TOP is based on the best individual run; if hidden in a collapsed group, the marker appears on the group summary."
        : "Grouped rows include run distribution overlays with CI.";
    container.appendChild(groupedSummary);
  }

  const maxScore = resolveLeaderboardBarMax(metricKey, [
    ...entries.map((entry) => entry.score),
    ...source.map((run) => getMetricValueForRun(run, metricKey)),
  ]);
  entries.forEach((entry) => {
    if (entry.type === "run") {
      const isTopRun = topRunPath && entry.run.filePath === topRunPath;
      const badges = [];
      if (isTopRun) {
        badges.push("TOP");
      }
      let rowClass = "";
      if (isTopRun) {
        rowClass = "bar-row-top";
      }
      container.appendChild(
        createBarRow(
          entry.label,
          entry.score,
          maxScore,
          (value, ci) => `${formatMetric(metricKey, value)}${formatCiRange(ci)}`,
          null,
          () => openRunModal(entry.run),
          entry.ci,
          hasMultipleTasks ? entry.run.task : "",
          { badges, rowClass }
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
    const groupHasTopRun = Boolean(topRunPath && entry.runs.some((run) => run.filePath === topRunPath));
    const badges = [];
    if (groupHasTopRun) {
      badges.push("TOP");
    }
    let rowClass = "";
    if (groupHasTopRun) {
      rowClass = "bar-row-top";
    }
    const distributionValues = entry.runs
      .map((run) => getMetricValueForRun(run, metricKey))
      .filter((value) => typeof value === "number" && Number.isFinite(value));
    summary.appendChild(
      createBarRow(
        `${entry.label} (${entry.runs.length})`,
        entry.avgMetric || 0,
        maxScore,
        (value, ci) => `${formatMetric(metricKey, value)} avg${formatCiRange(ci)}`,
        null,
        null,
        entry.ci,
        hasMultipleTasks ? getConcatenatedTaskLabel(entry.runs) : "",
        {
          badges,
          rowClass,
          distribution: distributionValues.length ? { values: distributionValues } : null,
        }
      )
    );
    details.appendChild(summary);

    const runsWrap = document.createElement("div");
    runsWrap.className = "leaderboard-group-runs";
    entry.runs.forEach((run) => {
      const runCi = getRunMetricConfidence(run, metricKey);
      const showTopOnRun = Boolean(topRunPath && run.filePath === topRunPath);
      const runBadges = [];
      if (showTopOnRun) {
        runBadges.push("TOP");
      }
      let runRowClass = "";
      if (showTopOnRun) {
        runRowClass = "bar-row-top";
      }
      runsWrap.appendChild(
        createBarRow(
          getRunModelDisplayName(run),
          getMetricValueForRun(run, metricKey) ?? 0,
          maxScore,
          (value, ci) => `${formatMetric(metricKey, value)}${formatCiRange(ci)}`,
          null,
          () => openRunModal(run),
          runCi,
          hasMultipleTasks ? run.task : "",
          { badges: runBadges, rowClass: runRowClass }
        )
      );
    });
    details.appendChild(runsWrap);
    container.appendChild(details);
  });
}

function renderLeaderboardMetricsTable(container, runs) {
  const metricKey = state.sortBy;
  const source = runs.filter((run) =>
    LEADERBOARD_TABLE_METRICS.some((key) => getMetricValueForRun(run, key) !== null)
  );

  if (!source.length) {
    container.innerHTML = '<p class="muted">No run-level metric values in current filter.</p>';
    return;
  }

  const bestByMetric = {};
  LEADERBOARD_TABLE_METRICS.forEach((key) => {
    const values = source
      .map((run) => getMetricValueForRun(run, key))
      .filter((value) => typeof value === "number" && Number.isFinite(value));
    bestByMetric[key] = values.length ? Math.max(...values) : null;
  });

  const sorted = [...source].sort((a, b) => {
    const diff = scoreForSort(b, metricKey) - scoreForSort(a, metricKey);
    if (diff !== 0) {
      return diff;
    }
    return parseRunTimestampMs(b) - parseRunTimestampMs(a);
  });

  const summary = document.createElement("p");
  summary.className = "leaderboard-metrics-summary muted";
  summary.textContent = "Highlighted cells mark best score per metric in the current selection.";
  container.appendChild(summary);

  const wrap = document.createElement("div");
  wrap.className = "leaderboard-metrics-wrap";
  const table = document.createElement("table");
  table.className = "leaderboard-metrics-table";

  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  ["Run", "Accuracy", "Macro F1", "Macro Precision", "Macro Recall"].forEach((label) => {
    const th = document.createElement("th");
    th.textContent = label;
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  sorted.forEach((run) => {
    const tr = document.createElement("tr");
    tr.className = "clickable-row";
    if (run.filePath === state.selectedRunPath) {
      tr.classList.add("selected-row");
    }
    tr.addEventListener("click", () => {
      state.selectedRunPath = run.filePath;
      openRunModal(run);
      render();
    });

    const runCell = document.createElement("td");
    runCell.className = "leaderboard-run-cell";
    runCell.title = `${run.task} / ${getRunModelDisplayName(run)} / ${run.fileName}`;
    runCell.textContent = `${run.task} / ${getRunModelDisplayName(run)}`;
    tr.appendChild(runCell);

    LEADERBOARD_TABLE_METRICS.forEach((key) => {
      const value = getMetricValueForRun(run, key);
      const td = document.createElement("td");
      td.className = "mono";
      td.textContent = value == null ? "N/A" : `${formatNum(value, 2)}%`;
      if (value != null && bestByMetric[key] != null && Math.abs(value - bestByMetric[key]) < 1e-9) {
        td.classList.add("metric-best");
        td.title = `Best ${METRIC_LABELS[key]} in current selection`;
      }
      tr.appendChild(td);
    });

    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  wrap.appendChild(table);
  container.appendChild(wrap);
}

function renderLeaderboard(runs) {
  renderLeaderboardTabControls();
  els.leaderboardChart.innerHTML = "";

  const panel = document.createElement("div");
  panel.className = "leaderboard-panel";
  els.leaderboardChart.appendChild(panel);

  if (state.leaderboardTab === "table") {
    renderLeaderboardMetricsTable(panel, runs);
    return;
  }
  if (state.leaderboardTab === "best_by_task") {
    renderBestByTask(panel, runs);
    return;
  }
  renderLeaderboardChart(panel, runs);
}

function renderBestByTask(container, runs) {
  const metricKey = state.sortBy;
  const metricLabel = METRIC_LABELS[metricKey] || metricKey;
  const bestByTask = {};

  runs.forEach((run) => {
    const metricValue = getMetricValueForRun(run, metricKey);
    if (metricValue === null || Number.isNaN(metricValue)) return;
    const current = bestByTask[run.task];
    if (!current || metricValue > current.metricValue) {
      bestByTask[run.task] = { run, metricValue };
    }
  });

  const items = Object.values(bestByTask)
    .sort((a, b) => b.metricValue - a.metricValue)
    .slice(0, 16);

  if (!items.length) {
    container.innerHTML = `<p class="muted">No task-level ${metricLabel.toLowerCase()} data found.</p>`;
    return;
  }

  const max = resolveLeaderboardBarMax(
    metricKey,
    items.map((item) => item.metricValue)
  );
  items.forEach(({ run, metricValue }) => {
    container.appendChild(
      createBarRow(
        `${run.task} / ${getRunModelDisplayName(run)}`,
        metricValue,
        max,
        (value) => formatMetric(metricKey, value),
        "warm",
        () => openRunModal(run),
        null,
        "",
        {}
      )
    );
  });
}

function getPromptCountForRun(run) {
  return run.requestsTotal ?? run.attemptsWithUsage ?? run.predictionCount ?? run.totalExamples ?? 0;
}

const TOKEN_SEGMENTS = [
  { key: "avgInput", label: "Input", color: "#6ea8ff" },
  { key: "avgCached", label: "Cached", color: "#50e3c2" },
  { key: "avgOutput", label: "Output", color: "#ffb36b" },
  { key: "avgThinking", label: "Thinking", color: "#f472b6" },
];

const RADAR_COLORS = ["#6ea8ff", "#50e3c2", "#ffb36b", "#f472b6", "#facc15", "#22d3ee", "#fb7185", "#4ade80"];

function toNonNegativeNumber(value) {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : 0;
}

function averageOrZero(total, denominator) {
  return denominator > 0 ? total / denominator : 0;
}

function computeRunSignalRows(runs) {
  return runs
    .map((run) => {
      const prompts = toNonNegativeNumber(getPromptCountForRun(run));
      const avgInput = averageOrZero(toNonNegativeNumber(run.inputTokensTotal), prompts);
      const avgCached = averageOrZero(toNonNegativeNumber(run.cachedInputTokensTotal ?? run.cachedTokens), prompts);
      const avgOutput = averageOrZero(toNonNegativeNumber(run.outputTokensTotal), prompts);
      const avgThinking = averageOrZero(toNonNegativeNumber(run.thinkingTokensTotal), prompts);
      return {
        run,
        model: run.model || "unknown",
        modelDisplay: getRunModelDisplayName(run),
        task: run.task || "unknown",
        prompts,
        avgInput,
        avgCached,
        avgOutput,
        avgThinking,
        totalAvg: avgInput + avgCached + avgOutput + avgThinking,
      };
    })
    .sort((a, b) => {
      const modelCmp = a.model.localeCompare(b.model);
      if (modelCmp !== 0) return modelCmp;
      const displayCmp = a.modelDisplay.localeCompare(b.modelDisplay);
      if (displayCmp !== 0) return displayCmp;
      const tsCmp = parseRunTimestampMs(b.run) - parseRunTimestampMs(a.run);
      if (tsCmp !== 0) return tsCmp;
      return a.run.fileName.localeCompare(b.run.fileName);
    });
}

function appendTokenLegend(container) {
  const legend = document.createElement("div");
  legend.className = "token-legend";
  TOKEN_SEGMENTS.forEach((segment) => {
    const item = document.createElement("div");
    item.className = "token-legend-item";
    const swatch = document.createElement("span");
    swatch.className = "token-legend-swatch";
    swatch.style.background = segment.color;
    const label = document.createElement("span");
    label.textContent = segment.label;
    item.appendChild(swatch);
    item.appendChild(label);
    legend.appendChild(item);
  });
  container.appendChild(legend);
}

function truncateAxisLabel(label, maxChars = 14) {
  const text = String(label || "");
  if (text.length <= maxChars) {
    return text;
  }
  return `${text.slice(0, maxChars - 1)}...`;
}

function buildRadarAxisDataset(runs, metricKey, axisMode) {
  const axesCandidates = (
    axisMode === "tag"
      ? (
          state.selectedTags.length
            ? state.selectedTags
            : [...new Set(runs.flatMap((run) => run.tags || []))].sort((a, b) => a.localeCompare(b))
        )
      : (
          state.selectedTasks.length
            ? state.selectedTasks
            : [...new Set(runs.map((run) => run.task))].sort((a, b) => a.localeCompare(b))
        )
  ).filter((axisValue) =>
    runs.some((run) =>
      axisMode === "tag"
        ? (run.tags || []).includes(axisValue)
        : run.task === axisValue
    )
  );

  const modelCandidates = (state.selectedModels.length
    ? state.selectedModels
    : [...new Set(runs.map((run) => run.model))].sort((a, b) => a.localeCompare(b))
  ).filter((model) => runs.some((run) => run.model === model));

  const series = modelCandidates.map((model) => {
    const values = axesCandidates.map((axisValue) => {
      const matches = runs.filter((run) =>
        run.model === model &&
        (axisMode === "tag" ? (run.tags || []).includes(axisValue) : run.task === axisValue)
      );
      const numeric = matches
        .map((run) => getMetricValueForRun(run, metricKey))
        .filter((value) => typeof value === "number" && Number.isFinite(value));
      if (!numeric.length) {
        return 0;
      }
      return numeric.reduce((sum, value) => sum + value, 0) / numeric.length;
    });
    const nonZero = values.filter((value) => value > 0);
    const mean = nonZero.length
      ? nonZero.reduce((sum, value) => sum + value, 0) / nonZero.length
      : 0;
    return { model, values, mean };
  });

  const useAutoTopModels = state.selectedModels.length === 0 && series.length > 8;
  const visibleSeries = useAutoTopModels
    ? [...series]
        .sort((a, b) => {
          if (b.mean !== a.mean) return b.mean - a.mean;
          return a.model.localeCompare(b.model);
        })
        .slice(0, 8)
    : series;

  return {
    axes: axesCandidates,
    series: visibleSeries,
    hiddenModelCount: Math.max(0, series.length - visibleSeries.length),
  };
}

function buildRadarSvg(tasks, seriesRows, metricKey) {
  const ns = "http://www.w3.org/2000/svg";
  const size = 360;
  const center = size / 2;
  const radius = 126;
  const metricIsPercent = isPercentMetric(metricKey);
  const axisMaxima = tasks.map((_, index) => {
    if (metricIsPercent) {
      return 100;
    }
    return Math.max(
      ...seriesRows.map((row) => toNonNegativeNumber(row.values[index])),
      1
    );
  });

  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("class", "radar-svg");
  svg.setAttribute("viewBox", `0 0 ${size} ${size}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Radar chart comparing model proficiency across selected dimensions");

  const angleFor = (index) => -Math.PI / 2 + (2 * Math.PI * index) / tasks.length;
  const pointAt = (angle, dist) => [center + Math.cos(angle) * dist, center + Math.sin(angle) * dist];

  for (let step = 1; step <= 5; step += 1) {
    const dist = (radius * step) / 5;
    const points = tasks
      .map((_, index) => pointAt(angleFor(index), dist))
      .map((xy) => xy.join(","))
      .join(" ");
    const ring = document.createElementNS(ns, "polygon");
    ring.setAttribute("class", "radar-grid");
    ring.setAttribute("points", points);
    svg.appendChild(ring);
  }

  tasks.forEach((task, index) => {
    const [x2, y2] = pointAt(angleFor(index), radius);
    const axisLine = document.createElementNS(ns, "line");
    axisLine.setAttribute("class", "radar-axis");
    axisLine.setAttribute("x1", String(center));
    axisLine.setAttribute("y1", String(center));
    axisLine.setAttribute("x2", String(x2));
    axisLine.setAttribute("y2", String(y2));
    svg.appendChild(axisLine);

    const [lx, ly] = pointAt(angleFor(index), radius + 18);
    const text = document.createElementNS(ns, "text");
    text.setAttribute("class", "radar-label");
    text.setAttribute("x", String(lx));
    text.setAttribute("y", String(ly));
    text.setAttribute("text-anchor", lx >= center + 8 ? "start" : lx <= center - 8 ? "end" : "middle");
    text.textContent = truncateAxisLabel(task);
    svg.appendChild(text);
  });

  seriesRows.forEach((row, idx) => {
    const color = RADAR_COLORS[idx % RADAR_COLORS.length];
    const points = tasks
      .map((_, axisIndex) => {
        const raw = toNonNegativeNumber(row.values[axisIndex]);
        const axisMax = axisMaxima[axisIndex] || 1;
        const normalized = Math.max(0, Math.min(1, raw / axisMax));
        return pointAt(angleFor(axisIndex), radius * normalized);
      })
      .map((xy) => xy.join(","))
      .join(" ");
    const polygon = document.createElementNS(ns, "polygon");
    polygon.setAttribute("class", "radar-series");
    polygon.setAttribute("points", points);
    polygon.setAttribute("fill", color);
    polygon.setAttribute("stroke", color);
    svg.appendChild(polygon);
  });

  return svg;
}

function renderRadarPanel(panel, runs) {
  const metricKey = state.sortBy;
  const metricLabel = METRIC_LABELS[metricKey] || metricKey;
  const axisMode = state.radarAxis;
  const axisLabel = RADAR_AXIS_LABELS[axisMode] || "Axis";
  const header = document.createElement("h4");
  header.className = "token-radar-header";
  header.textContent = `Proficiency by ${axisLabel.toLowerCase()} (${metricLabel})`;
  panel.appendChild(header);

  const toggleWrap = document.createElement("div");
  toggleWrap.className = "radar-mode-toggle";
  ["task", "tag"].forEach((mode) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `radar-mode-btn${state.radarAxis === mode ? " active" : ""}`;
    button.textContent = mode === "task" ? "Per Task" : "Per Tag";
    button.setAttribute("aria-pressed", state.radarAxis === mode ? "true" : "false");
    button.addEventListener("click", () => setRadarAxis(mode));
    toggleWrap.appendChild(button);
  });
  panel.appendChild(toggleWrap);

  const dataset = buildRadarAxisDataset(runs, metricKey, axisMode);
  if (dataset.axes.length < 3) {
    const msg = document.createElement("p");
    msg.className = "muted";
    msg.textContent = `Select at least 3 ${axisLabel.toLowerCase()}s to render a radar profile.`;
    panel.appendChild(msg);
    return;
  }
  if (!dataset.series.length) {
    const msg = document.createElement("p");
    msg.className = "muted";
    msg.textContent = `No model data for the selected metric/${axisLabel.toLowerCase()}s.`;
    panel.appendChild(msg);
    return;
  }

  const wrap = document.createElement("div");
  wrap.className = "radar-wrap";
  wrap.appendChild(buildRadarSvg(dataset.axes, dataset.series, metricKey));

  const legend = document.createElement("div");
  legend.className = "radar-legend";
  const suffix = isPercentMetric(metricKey) ? "%" : "";
  dataset.series.forEach((row, idx) => {
    const entry = document.createElement("div");
    entry.className = "radar-legend-row";
    const line = document.createElement("span");
    line.className = "radar-legend-line";
    line.style.background = RADAR_COLORS[idx % RADAR_COLORS.length];
    const text = document.createElement("span");
    text.className = "mono";
    text.textContent = `${row.model} | avg ${formatNum(row.mean, 2)}${suffix}`;
    entry.appendChild(line);
    entry.appendChild(text);
    legend.appendChild(entry);
  });
  wrap.appendChild(legend);

  if (dataset.hiddenModelCount > 0) {
    const note = document.createElement("p");
    note.className = "muted";
    note.textContent = `Radar shows top ${dataset.series.length} models by average ${metricLabel.toLowerCase()}.`;
    wrap.appendChild(note);
  }

  panel.appendChild(wrap);
}

function renderTokenSignals(runs) {
  els.tokenChart.innerHTML = "";
  const runRows = computeRunSignalRows(runs).filter((row) => row.prompts > 0 || row.totalAvg > 0);
  if (!runRows.length) {
    els.tokenChart.innerHTML = '<p class="muted">No token/request metadata for current filter.</p>';
    return;
  }

  const layout = document.createElement("div");
  layout.className = "token-visual-grid";

  const stackPanel = document.createElement("section");
  stackPanel.className = "token-stack-panel";
  const stackHeader = document.createElement("h4");
  stackHeader.className = "token-stack-header";
  stackHeader.textContent = "Average tokens per prompt by run (sorted by model name)";
  stackPanel.appendChild(stackHeader);
  appendTokenLegend(stackPanel);

  const rowsWrap = document.createElement("div");
  rowsWrap.className = "token-rows";
  const rowLimit = 60;
  runRows.slice(0, rowLimit).forEach((row) => {
    const rowEl = document.createElement("div");
    rowEl.className = "token-row";
    rowEl.style.cursor = "pointer";

    const head = document.createElement("div");
    head.className = "token-row-head";
    const modelLabel = document.createElement("span");
    modelLabel.className = "token-model-label";
    modelLabel.textContent = row.modelDisplay;
    modelLabel.title = row.modelDisplay;
    const meta = document.createElement("span");
    meta.className = "mono";
    meta.textContent = `${row.task} | avg ${formatNum(row.totalAvg, 2)} | prompts ${formatNum(row.prompts, 0)}`;
    head.appendChild(modelLabel);
    head.appendChild(meta);
    rowEl.appendChild(head);

    const track = document.createElement("div");
    track.className = "token-stack-track";
    const total = row.totalAvg;
    TOKEN_SEGMENTS.forEach((segment) => {
      const value = row[segment.key];
      const percent = total > 0 ? (value / total) * 100 : 0;
      const seg = document.createElement("div");
      seg.className = "token-segment";
      seg.style.width = `${percent}%`;
      seg.style.background = segment.color;
      seg.title = `${segment.label}: ${formatNum(value, 2)} avg/prompt`;
      track.appendChild(seg);
    });
    rowEl.appendChild(track);

    const values = document.createElement("div");
    values.className = "token-row-values mono";
    values.textContent = `in ${formatNum(row.avgInput, 2)} | cached ${formatNum(row.avgCached, 2)} | out ${formatNum(row.avgOutput, 2)} | think ${formatNum(row.avgThinking, 2)}`;
    rowEl.appendChild(values);

    rowEl.addEventListener("click", () => openRunModal(row.run));
    rowsWrap.appendChild(rowEl);
  });
  stackPanel.appendChild(rowsWrap);

  if (runRows.length > rowLimit) {
    const note = document.createElement("p");
    note.className = "muted";
    note.textContent = `Showing ${rowLimit} of ${runRows.length} runs. Narrow filters to focus.`;
    stackPanel.appendChild(note);
  }

  const radarPanel = document.createElement("section");
  radarPanel.className = "token-radar-panel";
  renderRadarPanel(radarPanel, runs);

  layout.appendChild(stackPanel);
  layout.appendChild(radarPanel);
  els.tokenChart.appendChild(layout);
}

function renderTableCell(label, value, className = "") {
  const classAttr = className ? ` class="${className}"` : "";
  return `<td data-label="${label}"${classAttr}>${value}</td>`;
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
    tr.innerHTML = [
      renderTableCell("Task", run.task, "table-cell-primary"),
      renderTableCell("Model", run.model, "table-cell-model table-cell-full"),
      renderTableCell("Timestamp", formatTs(run.timestamp)),
      renderTableCell(
        "Accuracy",
        run.accuracy !== null ? `${formatNum(run.accuracy, 2)}%` : '<span class="muted">N/A</span>'
      ),
      renderTableCell(
        "Macro F1",
        run.macroF1 !== null ? `${formatNum(run.macroF1, 2)}%` : '<span class="muted">N/A</span>'
      ),
      renderTableCell("Requests", formatNum(run.requestsTotal, 0), "mono"),
      renderTableCell("Cached Input Tokens", formatNum(run.cachedTokens, 0), "mono"),
      renderTableCell("File", run.fileName, "mono table-cell-full"),
    ].join("");
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
    ["Macro Precision", run.macroPrecision == null ? "N/A" : `${formatNum(run.macroPrecision, 2)}%`],
    ["Macro Recall", run.macroRecall == null ? "N/A" : `${formatNum(run.macroRecall, 2)}%`],
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
  renderTagControls();
  updateMobileFilterSummary();
  renderKpis(state.filtered);
  renderLeaderboard(state.filtered);
  renderTokenSignals(state.filtered);
  renderTable(state.filtered);
}

function renderError(message, preserveExisting = false) {
  els.heroSubtitle.innerHTML = `<span class="warn">${message}</span>`;
  if (!preserveExisting) {
    els.leaderboardChart.innerHTML = "";
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
  state.tags = [...new Set(result.runs.flatMap((run) => run.tags || []))].sort((a, b) => a.localeCompare(b));
  state.sourceMode = result.mode;
  state.sourceFileCount = result.fileCount;
  state.warnings = result.warnings;
  state.expandedLeaderboardModels = new Set();

  state.selectedTasks = sanitizeSelections(state.selectedTasks, state.tasks);
  state.selectedModels = sanitizeSelections(state.selectedModels, state.models);
  state.selectedTags = sanitizeSelections(state.selectedTags, state.tags);
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
  setupResponsiveShell();
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

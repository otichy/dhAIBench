const STORAGE_KEY = "dhAIBench.metricsDashboard.state.v1";
const METRICS_MANIFEST_PATH = "./metrics-manifest.json";
const METRICS_SERVER_DIR = "../data/metrics";
const METRICS_SERVER_DIR_CANDIDATES = [METRICS_SERVER_DIR, "./metrics", "./data/metrics"];
const DESKTOP_LAYOUT_BREAKPOINT = 1100;
const MOBILE_LAYOUT_BREAKPOINT = 900;
const DESKTOP_SIDEBAR_EXPANDED_WIDTH = 420;
const DESKTOP_SIDEBAR_COLLAPSED_WIDTH = 78;
const TOKEN_SIGNAL_PAGE_SIZE = 60;
const BEST_BY_TASK_PAGE_SIZE = 16;
const RADAR_MODEL_PAGE_SIZE = 8;
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
  timeRanges: [{ from: "", to: "" }],
  selectedRunPath: null,
  sortBy: "accuracy",
  leaderboardTableSortKey: null,
  leaderboardTableSortDirection: null,
  leaderboardTab: "chart",
  leaderboardChartGroupBy: "model",
  timeSeriesShowLabels: false,
  timeSeriesViewport: null,
  radarAxis: "task",
  radarScale: "linear",
  tokenSignalsVisibleCount: TOKEN_SIGNAL_PAGE_SIZE,
  bestByTaskVisibleCount: BEST_BY_TASK_PAGE_SIZE,
  radarVisibleSeriesCount: RADAR_MODEL_PAGE_SIZE,
  hideNoAccuracy: false,
  theme: "dark",
  sourceMode: "none",
  sourceFileCount: 0,
  warnings: [],
  activeDirectoryHandle: null,
  activeFiles: [],
  expandedLeaderboardGroups: new Set(),
  mobileSidebarOpen: false,
  desktopSidebarCollapsed: false,
  activeTimeRangeEditor: null,
  isLoading: false,
  loadingMessage: "",
  loadingProgressCurrent: 0,
  loadingProgressTotal: 0,
};

const els = {
  main: document.querySelector(".main"),
  dashboardSidebar: document.querySelector("#dashboardSidebar"),
  dashboardSidebarScroll: document.querySelector("#dashboardSidebarScroll"),
  sidebarCollapseBtn: document.querySelector("#sidebarCollapseBtn"),
  sidebarScrollUpOverlay: document.querySelector("#sidebarScrollUpOverlay"),
  sidebarScrollDownOverlay: document.querySelector("#sidebarScrollDownOverlay"),
  sidebarScrollUpBtn: document.querySelector("#sidebarScrollUpBtn"),
  sidebarScrollDownBtn: document.querySelector("#sidebarScrollDownBtn"),
  resetFiltersBtn: document.querySelector("#resetFiltersBtn"),
  mobileSidebarToggle: document.querySelector("#mobileSidebarToggle"),
  mobileFilterSummary: document.querySelector("#mobileFilterSummary"),
  sidebarCloseBtn: document.querySelector("#sidebarCloseBtn"),
  sidebarBackdrop: document.querySelector("#sidebarBackdrop"),
  taskSelect: document.querySelector("#taskSelect"),
  taskChipList: document.querySelector("#taskChipList"),
  modelSelect: document.querySelector("#modelSelect"),
  modelChipList: document.querySelector("#modelChipList"),
  timeRangeList: document.querySelector("#timeRangeList"),
  addTimeRangeBtn: document.querySelector("#addTimeRangeBtn"),
  timeRangeSummary: document.querySelector("#timeRangeSummary"),
  sortSelect: document.querySelector("#sortSelect"),
  hideNoAccuracy: document.querySelector("#hideNoAccuracy"),
  themeToggle: document.querySelector("#themeToggle"),
  tagChips: document.querySelector("#tagChips"),
  heroTitle: document.querySelector("#heroTitle"),
  heroSubtitle: document.querySelector("#heroSubtitle"),
  loadingNotice: document.querySelector("#loadingNotice"),
  loadingNoticeMessage: document.querySelector("#loadingNoticeMessage"),
  loadingProgress: document.querySelector("#loadingProgress"),
  loadingProgressLabel: document.querySelector("#loadingProgressLabel"),
  loadingProgressPercent: document.querySelector("#loadingProgressPercent"),
  loadingProgressTrack: document.querySelector("#loadingProgressTrack"),
  loadingProgressFill: document.querySelector("#loadingProgressFill"),
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
  leaderboardGroupSwitch: document.querySelector("#leaderboardGroupSwitch"),
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

const METRIC_KEYS = new Set(["accuracy", "macro_f1", "macro_precision", "macro_recall", "calibration_ece"]);
const PERCENT_METRIC_KEYS = new Set([
  "accuracy",
  "macro_f1",
  "macro_precision",
  "macro_recall",
  "calibration_ece",
]);
const APPROX_CI_METRIC_KEYS = new Set(["accuracy", "macro_f1", "macro_precision", "macro_recall"]);
const LOWER_IS_BETTER_METRIC_KEYS = new Set(["calibration_ece"]);
const RADAR_AXIS_KEYS = new Set(["task", "tag"]);
const RADAR_SCALE_KEYS = new Set(["linear", "contrast"]);
const LEADERBOARD_TAB_KEYS = new Set(["chart", "time_series", "table", "best_by_task", "radar"]);
const LEADERBOARD_CHART_GROUP_BY_KEYS = new Set(["model", "task"]);
const LEADERBOARD_TABLE_METRICS = [
  "accuracy",
  "macro_f1",
  "macro_precision",
  "macro_recall",
  "calibration_ece",
];
const LEADERBOARD_TABLE_SORTABLE_KEYS = new Set(["run", "timestamp", ...LEADERBOARD_TABLE_METRICS]);
const SORT_DIRECTIONS = new Set(["asc", "desc"]);
let leaderboardMetricsScrollCleanup = null;

const METRIC_LABELS = {
  accuracy: "Accuracy",
  macro_f1: "Macro F1",
  macro_precision: "Macro Precision",
  macro_recall: "Macro Recall",
  calibration_ece: "Calibration ECE",
};

const RADAR_AXIS_LABELS = {
  task: "Task",
  tag: "Tag",
};

const RADAR_SCALE_LABELS = {
  linear: "Linear",
  contrast: "Contrast",
};

const LEADERBOARD_CHART_GROUP_BY_LABELS = {
  model: "Model",
  task: "Task",
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

function createEmptyTimeRange() {
  return { from: "", to: "" };
}

function normalizeTimeRanges(ranges) {
  const source = Array.isArray(ranges) ? ranges : [];
  const normalized = [];
  let keptEmptyRange = false;

  source.forEach((range) => {
    const nextRange = {
      from: asTrimmedString(range && range.from),
      to: asTrimmedString(range && range.to),
    };
    if (nextRange.from || nextRange.to) {
      normalized.push(nextRange);
      return;
    }
    if (!keptEmptyRange) {
      normalized.push(createEmptyTimeRange());
      keptEmptyRange = true;
    }
  });

  return normalized.length ? normalized : [createEmptyTimeRange()];
}

function isTimeRangeActive(range) {
  return Boolean(asTrimmedString(range && range.from) || asTrimmedString(range && range.to));
}

function padDatePart(value) {
  return String(value).padStart(2, "0");
}

function getPreferredLocales() {
  if (typeof navigator !== "undefined") {
    if (Array.isArray(navigator.languages) && navigator.languages.length) {
      return navigator.languages;
    }
    const single = asTrimmedString(navigator.language);
    if (single) {
      return [single];
    }
  }
  return undefined;
}

function getPrimaryLocale() {
  const locales = getPreferredLocales();
  return Array.isArray(locales) && locales.length ? locales[0] : "";
}

function parseTimestampToMs(value) {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }

  const raw = asTrimmedString(value);
  if (!raw) {
    return null;
  }

  const localDateTimeMatch = raw.match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})$/);
  if (localDateTimeMatch) {
    const [, year, month, day, hour, minute] = localDateTimeMatch;
    return new Date(
      Number(year),
      Number(month) - 1,
      Number(day),
      Number(hour),
      Number(minute),
      0,
      0
    ).getTime();
  }

  const direct = Date.parse(raw);
  if (Number.isFinite(direct)) {
    return direct;
  }

  const compactMatch = raw.match(/^(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})$/);
  if (compactMatch) {
    const [, year, month, day, hour, minute] = compactMatch;
    return new Date(
      Number(year),
      Number(month) - 1,
      Number(day),
      Number(hour),
      Number(minute),
      0,
      0
    ).getTime();
  }

  const dateOnlyMatch = raw.match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (dateOnlyMatch) {
    const [, year, month, day] = dateOnlyMatch;
    return new Date(Number(year), Number(month) - 1, Number(day), 0, 0, 0, 0).getTime();
  }

  return null;
}

function parseLocalDateTimeInputMs(value) {
  const raw = asTrimmedString(value);
  if (!raw) {
    return null;
  }

  const match = raw.match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})$/);
  if (!match) {
    return parseTimestampToMs(raw);
  }

  const [, year, month, day, hour, minute] = match;
  return new Date(
    Number(year),
    Number(month) - 1,
    Number(day),
    Number(hour),
    Number(minute),
    0,
    0
  ).getTime();
}

function formatDateTimeLocalInput(value) {
  const ms = parseTimestampToMs(value);
  if (!Number.isFinite(ms)) {
    return "";
  }
  const dt = new Date(ms);
  return `${dt.getFullYear()}-${padDatePart(dt.getMonth() + 1)}-${padDatePart(dt.getDate())}T${padDatePart(
    dt.getHours()
  )}:${padDatePart(dt.getMinutes())}`;
}

function formatTimeRangeDisplayDate(value) {
  const ms = parseTimestampToMs(value);
  if (!Number.isFinite(ms)) {
    return "Set date";
  }
  return new Date(ms).toLocaleDateString(getPreferredLocales(), {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
  });
}

function getActiveTimeRanges(ranges = state.timeRanges) {
  return normalizeTimeRanges(ranges)
    .map((range) => {
      const fromMs = parseLocalDateTimeInputMs(range.from);
      const toMs = parseLocalDateTimeInputMs(range.to);
      if (!Number.isFinite(fromMs) && !Number.isFinite(toMs)) {
        return null;
      }
      if (Number.isFinite(fromMs) && Number.isFinite(toMs) && fromMs > toMs) {
        return {
          fromMs: toMs,
          toMs: fromMs,
          from: range.to,
          to: range.from,
        };
      }
      return {
        fromMs: Number.isFinite(fromMs) ? fromMs : null,
        toMs: Number.isFinite(toMs) ? toMs : null,
        from: range.from,
        to: range.to,
      };
    })
    .filter(Boolean);
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

function isLowerBetterMetric(metricKey) {
  return LOWER_IS_BETTER_METRIC_KEYS.has(metricKey);
}

function supportsApproximateCi(metricKey) {
  return APPROX_CI_METRIC_KEYS.has(metricKey);
}

function compareMetricNumbers(valueA, valueB, metricKey) {
  const aValid = typeof valueA === "number" && Number.isFinite(valueA);
  const bValid = typeof valueB === "number" && Number.isFinite(valueB);
  if (!aValid && !bValid) {
    return 0;
  }
  if (!aValid) {
    return 1;
  }
  if (!bValid) {
    return -1;
  }
  if (valueA === valueB) {
    return 0;
  }
  return isLowerBetterMetric(metricKey) ? valueA - valueB : valueB - valueA;
}

function compareNullableNumbers(valueA, valueB) {
  const aValid = typeof valueA === "number" && Number.isFinite(valueA);
  const bValid = typeof valueB === "number" && Number.isFinite(valueB);
  if (!aValid && !bValid) {
    return 0;
  }
  if (!aValid) {
    return 1;
  }
  if (!bValid) {
    return -1;
  }
  if (valueA === valueB) {
    return 0;
  }
  return valueA < valueB ? -1 : 1;
}

function getPreferredMetricValue(values, metricKey) {
  const numeric = (values || []).filter((value) => typeof value === "number" && Number.isFinite(value));
  if (!numeric.length) {
    return null;
  }
  return isLowerBetterMetric(metricKey) ? Math.min(...numeric) : Math.max(...numeric);
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
  const ts = parseTimestampToMs(run && run.timestamp);
  return Number.isFinite(ts) ? ts : -Infinity;
}

function getDefaultSortDirectionForMetric(metricKey) {
  if (metricKey === "timestamp") {
    return "desc";
  }
  return isLowerBetterMetric(metricKey) ? "asc" : "desc";
}

function getLeaderboardMetricsTableRunLabel(run) {
  return `${run.task} / ${getRunModelDisplayName(run)}`;
}

function resolveLeaderboardMetricsTableSortSpec() {
  const isExplicitSort =
    LEADERBOARD_TABLE_SORTABLE_KEYS.has(state.leaderboardTableSortKey) &&
    SORT_DIRECTIONS.has(state.leaderboardTableSortDirection);
  if (isExplicitSort) {
    return {
      key: state.leaderboardTableSortKey,
      direction: state.leaderboardTableSortDirection,
    };
  }
  return {
    key: state.sortBy,
    direction: getDefaultSortDirectionForMetric(state.sortBy),
  };
}

function compareLeaderboardMetricsTableRows(runA, runB, sortSpec) {
  const key = sortSpec && sortSpec.key ? sortSpec.key : state.sortBy;
  const direction = sortSpec && sortSpec.direction === "asc" ? "asc" : "desc";
  let diff = 0;

  if (key === "run") {
    diff = getLeaderboardMetricsTableRunLabel(runA).localeCompare(
      getLeaderboardMetricsTableRunLabel(runB)
    );
  } else if (key === "timestamp") {
    const tsA = parseTimestampToMs(runA && runA.timestamp);
    const tsB = parseTimestampToMs(runB && runB.timestamp);
    diff = compareNullableNumbers(Number.isFinite(tsA) ? tsA : null, Number.isFinite(tsB) ? tsB : null);
  } else {
    diff = compareNullableNumbers(getMetricValueForRun(runA, key), getMetricValueForRun(runB, key));
  }

  if (direction === "desc") {
    diff *= -1;
  }
  if (diff !== 0) {
    return diff;
  }
  const tsCmp = parseRunTimestampMs(runB) - parseRunTimestampMs(runA);
  if (tsCmp !== 0) {
    return tsCmp;
  }
  return runA.fileName.localeCompare(runB.fileName);
}

function setLeaderboardMetricsTableSort(nextKey) {
  if (!LEADERBOARD_TABLE_SORTABLE_KEYS.has(nextKey)) {
    return;
  }
  const current = resolveLeaderboardMetricsTableSortSpec();
  if (current.key === nextKey) {
    state.leaderboardTableSortKey = nextKey;
    state.leaderboardTableSortDirection = current.direction === "asc" ? "desc" : "asc";
  } else {
    state.leaderboardTableSortKey = nextKey;
    state.leaderboardTableSortDirection =
      nextKey === "run" ? "asc" : getDefaultSortDirectionForMetric(nextKey);
  }
  persistUiState();
  render();
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
  const calibrationMetrics =
    payload.calibration_metrics && typeof payload.calibration_metrics === "object"
      ? payload.calibration_metrics
      : {};
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
  const calibrationEce = toPct(safeNum(calibrationMetrics.ece));
  const calibrationMce = toPct(safeNum(calibrationMetrics.mce));
  const calibrationBrierScore = safeNum(calibrationMetrics.brier_score);

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
    calibrationEce,
    calibrationMce,
    calibrationBrierScore,
    calibrationSampleCount: safeNum(calibrationMetrics.sample_count),
    calibrationAvailable: calibrationMetrics.available === true && calibrationEce !== null,
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

async function loadRunsFromServerFileList(files, manifestBaseDirs, warnings, onProgress = null) {
  const runs = [];
  const total = Array.isArray(files) ? files.length : 0;
  if (typeof onProgress === "function" && total > 0) {
    onProgress(0, total);
  }

  let processed = 0;
  for (const path of files) {
    const run = await loadRunFromServerCandidates(path, manifestBaseDirs, warnings);
    if (run) {
      runs.push(run);
    }
    processed += 1;
    if (typeof onProgress === "function" && total > 0) {
      onProgress(processed, total);
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

async function loadFromServer(onProgress = null) {
  const discovery = await discoverMetricFilesFromServer();
  const files = discovery.files;
  const manifestBaseDirs = discovery.manifestBaseDirs || [];
  const warnings = [];
  let runs = await loadRunsFromServerFileList(files, manifestBaseDirs, warnings, onProgress);
  let fileCount = files.length;

  if (!runs.length && discovery.source === "manifest") {
    try {
      const listingDiscovery = await discoverMetricFilesFromDirectoryListings(manifestBaseDirs);
      const listingRuns = await loadRunsFromServerFileList(listingDiscovery.files, [], warnings, onProgress);
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

async function loadFromDirectoryHandle(dirHandle, onProgress = null) {
  const metricFiles = await collectMetricFilesFromDirectoryHandle(dirHandle);
  if (!metricFiles.length) {
    throw new Error("No *_metrics.json files found in selected folder.");
  }

  const warnings = [];
  const runs = [];
  const total = metricFiles.length;
  if (typeof onProgress === "function") {
    onProgress(0, total);
  }
  let processed = 0;

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
    } finally {
      processed += 1;
      if (typeof onProgress === "function") {
        onProgress(processed, total);
      }
    }
  }

  return {
    mode: "folder",
    fileCount: metricFiles.length,
    runs: dedupeRuns(runs),
    warnings,
  };
}

async function loadFromFiles(fileList, onProgress = null) {
  const files = Array.from(fileList || []).filter((file) =>
    String(file.name || "").toLowerCase().endsWith("_metrics.json")
  );

  if (!files.length) {
    throw new Error("No *_metrics.json files selected.");
  }

  const warnings = [];
  const runs = [];
  const total = files.length;
  if (typeof onProgress === "function") {
    onProgress(0, total);
  }
  let processed = 0;

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
    } finally {
      processed += 1;
      if (typeof onProgress === "function") {
        onProgress(processed, total);
      }
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
    timeRanges: normalizeTimeRanges(state.timeRanges),
    desktopSidebarCollapsed: state.desktopSidebarCollapsed,
    sortBy: state.sortBy,
    leaderboardTableSortKey: state.leaderboardTableSortKey,
    leaderboardTableSortDirection: state.leaderboardTableSortDirection,
    leaderboardTab: state.leaderboardTab,
    leaderboardChartGroupBy: state.leaderboardChartGroupBy,
    timeSeriesShowLabels: state.timeSeriesShowLabels,
    timeSeriesViewport: state.timeSeriesViewport,
    radarAxis: state.radarAxis,
    radarScale: state.radarScale,
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
      if (Array.isArray(payload.timeRanges)) {
        state.timeRanges = normalizeTimeRanges(payload.timeRanges);
      }
      if (typeof payload.desktopSidebarCollapsed === "boolean") {
        state.desktopSidebarCollapsed = payload.desktopSidebarCollapsed;
      }
      if (typeof payload.sortBy === "string" && METRIC_KEYS.has(payload.sortBy)) {
        state.sortBy = payload.sortBy;
      }
      if (
        typeof payload.leaderboardTableSortKey === "string" &&
        LEADERBOARD_TABLE_SORTABLE_KEYS.has(payload.leaderboardTableSortKey)
      ) {
        state.leaderboardTableSortKey = payload.leaderboardTableSortKey;
      }
      if (
        typeof payload.leaderboardTableSortDirection === "string" &&
        SORT_DIRECTIONS.has(payload.leaderboardTableSortDirection)
      ) {
        state.leaderboardTableSortDirection = payload.leaderboardTableSortDirection;
      }
      if (typeof payload.leaderboardTab === "string" && LEADERBOARD_TAB_KEYS.has(payload.leaderboardTab)) {
        state.leaderboardTab = payload.leaderboardTab;
      }
      if (
        typeof payload.leaderboardChartGroupBy === "string" &&
        LEADERBOARD_CHART_GROUP_BY_KEYS.has(payload.leaderboardChartGroupBy)
      ) {
        state.leaderboardChartGroupBy = payload.leaderboardChartGroupBy;
      }
      if (typeof payload.timeSeriesShowLabels === "boolean") {
        state.timeSeriesShowLabels = payload.timeSeriesShowLabels;
      }
      if (payload.timeSeriesViewport && typeof payload.timeSeriesViewport === "object") {
        state.timeSeriesViewport = payload.timeSeriesViewport;
      }
      if (typeof payload.radarAxis === "string" && RADAR_AXIS_KEYS.has(payload.radarAxis)) {
        state.radarAxis = payload.radarAxis;
      }
      if (typeof payload.radarScale === "string") {
        if (RADAR_SCALE_KEYS.has(payload.radarScale)) {
          state.radarScale = payload.radarScale;
        } else if (payload.radarScale === "log") {
          state.radarScale = "contrast";
        }
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
  renderLeaderboardGroupSwitch();
}

function applyTheme() {
  document.documentElement.setAttribute("data-theme", state.theme);
}

function isMobileLayout() {
  return mobileLayoutQuery ? mobileLayoutQuery.matches : window.innerWidth <= MOBILE_LAYOUT_BREAKPOINT;
}

function supportsDesktopSidebarCollapse() {
  return window.innerWidth > DESKTOP_LAYOUT_BREAKPOINT;
}

function shouldUseDesktopSidebarScrollAffordances() {
  return supportsDesktopSidebarCollapse() && !isMobileLayout() && !state.desktopSidebarCollapsed;
}

function updateSidebarScrollAffordances() {
  const upOverlay = els.sidebarScrollUpOverlay;
  const downOverlay = els.sidebarScrollDownOverlay;
  const scrollEl = els.dashboardSidebarScroll;
  if (!upOverlay || !downOverlay || !scrollEl) {
    return;
  }

  if (!shouldUseDesktopSidebarScrollAffordances()) {
    upOverlay.hidden = true;
    downOverlay.hidden = true;
    return;
  }

  const maxScrollTop = Math.max(scrollEl.scrollHeight - scrollEl.clientHeight, 0);
  const hasOverflow = maxScrollTop > 16;
  const showUp = hasOverflow && scrollEl.scrollTop > 8;
  const showDown = hasOverflow && maxScrollTop - scrollEl.scrollTop > 8;
  upOverlay.hidden = !showUp;
  downOverlay.hidden = !showDown;
}

function scrollSidebarByPage(direction) {
  const scrollEl = els.dashboardSidebarScroll;
  if (!scrollEl) {
    return;
  }
  const delta = Math.max(scrollEl.clientHeight * 0.8, 80) * direction;
  scrollEl.scrollBy({
    top: delta,
    behavior: "smooth",
  });
}

function applySidebarLayoutState() {
  const collapsed = supportsDesktopSidebarCollapse() && state.desktopSidebarCollapsed;
  document.body.classList.toggle("desktop-sidebar-collapsed", collapsed);
  document.documentElement.style.setProperty(
    "--sidebar-width",
    `${collapsed ? DESKTOP_SIDEBAR_COLLAPSED_WIDTH : DESKTOP_SIDEBAR_EXPANDED_WIDTH}px`
  );

  if (els.sidebarCollapseBtn) {
    els.sidebarCollapseBtn.setAttribute("aria-expanded", collapsed ? "false" : "true");
    els.sidebarCollapseBtn.setAttribute("aria-label", collapsed ? "Expand filters" : "Collapse filters");
    els.sidebarCollapseBtn.title = collapsed ? "Expand filters" : "Collapse filters";
  }

  if (!isMobileLayout()) {
    els.dashboardSidebar.setAttribute("aria-hidden", "false");
  }

  requestAnimationFrame(() => {
    updateSidebarScrollAffordances();
  });
}

function setDesktopSidebarCollapsed(nextCollapsed) {
  state.desktopSidebarCollapsed = supportsDesktopSidebarCollapse() && Boolean(nextCollapsed);
  applySidebarLayoutState();
  persistUiState();
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
    return;
  }
}

function updateMobileFilterSummary() {
  const activeFilterCount = getActiveFilterCount();
  const summaryText = activeFilterCount ? `${activeFilterCount} active` : "All";

  els.mobileFilterSummary.textContent = summaryText;
  els.mobileSidebarToggle.title = activeFilterCount ? `${summaryText} filters` : "All filters visible";
}

function getActiveFilterCount() {
  return (
    (state.selectedTasks.length ? 1 : 0) +
    (state.selectedModels.length ? 1 : 0) +
    (state.selectedTags.length ? 1 : 0) +
    (getActiveTimeRanges().length ? 1 : 0) +
    (state.hideNoAccuracy ? 1 : 0)
  );
}

function updateResetFiltersButton() {
  if (!els.resetFiltersBtn) {
    return;
  }
  els.resetFiltersBtn.disabled = getActiveFilterCount() === 0;
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

function setTimeRanges(ranges) {
  state.timeRanges = normalizeTimeRanges(ranges);
  persistUiState();
  render();
}

function setActiveTimeRangeEditor(index, key) {
  if (typeof index !== "number" || !["from", "to"].includes(key)) {
    state.activeTimeRangeEditor = null;
    renderTimeRangeControls();
    return;
  }
  state.activeTimeRangeEditor = { index, key };
  renderTimeRangeControls();
}

function updateTimeRangeValue(index, key, value) {
  if (!["from", "to"].includes(key)) {
    return;
  }
  const nextRanges = normalizeTimeRanges(state.timeRanges).map((range, rangeIndex) =>
    rangeIndex === index
      ? {
          ...range,
          [key]: asTrimmedString(value),
        }
      : { ...range }
  );
  state.activeTimeRangeEditor = null;
  setTimeRanges(nextRanges);
}

function addTimeRange() {
  setTimeRanges([...normalizeTimeRanges(state.timeRanges), createEmptyTimeRange()]);
}

function removeTimeRange(index) {
  const source = normalizeTimeRanges(state.timeRanges);
  const nextRanges = source.filter((_, rangeIndex) => rangeIndex !== index);
  state.activeTimeRangeEditor = null;
  setTimeRanges(nextRanges);
}

function resetAllFilters() {
  state.selectedTasks = [];
  state.selectedModels = [];
  state.selectedTags = [];
  state.timeRanges = [createEmptyTimeRange()];
  state.activeTimeRangeEditor = null;
  state.hideNoAccuracy = false;
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
  renderLeaderboard(state.filtered);
}

function setRadarScale(scale) {
  if (!RADAR_SCALE_KEYS.has(scale) || state.radarScale === scale) {
    return;
  }
  state.radarScale = scale;
  persistUiState();
  renderLeaderboard(state.filtered);
}

function setLeaderboardTab(tab) {
  if (!LEADERBOARD_TAB_KEYS.has(tab) || state.leaderboardTab === tab) {
    return;
  }
  state.leaderboardTab = tab;
  persistUiState();
  renderLeaderboard(state.filtered);
}

function setTimeSeriesShowLabels(nextValue) {
  state.timeSeriesShowLabels = Boolean(nextValue);
  persistUiState();
  renderLeaderboard(state.filtered);
}

function normalizeTimeSeriesViewport(viewport) {
  if (!viewport || typeof viewport !== "object") {
    return null;
  }
  const xMin = Number(viewport.xMin);
  const xMax = Number(viewport.xMax);
  const yMin = Number(viewport.yMin);
  const yMax = Number(viewport.yMax);
  if (![xMin, xMax, yMin, yMax].every((value) => Number.isFinite(value))) {
    return null;
  }
  if (xMin >= xMax || yMin >= yMax) {
    return null;
  }
  return { xMin, xMax, yMin, yMax };
}

function setTimeSeriesViewport(nextViewport) {
  state.timeSeriesViewport = normalizeTimeSeriesViewport(nextViewport);
  persistUiState();
  renderLeaderboard(state.filtered);
}

function resetTimeSeriesZoom() {
  if (!state.timeSeriesViewport) {
    return;
  }
  state.timeSeriesViewport = null;
  persistUiState();
  renderLeaderboard(state.filtered);
}

function syncMultiSelectValue(selectElement, selectedValues) {
  if (!selectElement) {
    return;
  }

  const selected = new Set(Array.isArray(selectedValues) ? selectedValues : []);
  const allSelected = selected.size === 0;
  Array.from(selectElement.options).forEach((option) => {
    const isSelected = allSelected ? option.value === "ALL" : selected.has(option.value);
    option.selected = isSelected;
    option.classList.toggle("is-selected", isSelected);
  });
  selectElement.classList.toggle("has-active-selection", !allSelected);
}

function syncTaskSelectValue() {
  syncMultiSelectValue(els.taskSelect, state.selectedTasks);
}

function syncModelSelectValue() {
  syncMultiSelectValue(els.modelSelect, state.selectedModels);
}

function getSelectValues(selectElement) {
  return Array.from(selectElement.selectedOptions)
    .map((option) => option.value)
    .filter(Boolean);
}

function syncSelectOptions(selectElement, values, labelForValue) {
  const desiredValues = Array.isArray(values) ? values : [];
  const existingOptions = Array.from(selectElement.options);
  const hasSameOptions =
    existingOptions.length === desiredValues.length &&
    existingOptions.every((option, index) => {
      const desiredValue = desiredValues[index];
      return option.value === desiredValue && option.textContent === labelForValue(desiredValue);
    });

  if (hasSameOptions) {
    return;
  }

  const fragment = document.createDocumentFragment();
  desiredValues.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = labelForValue(value);
    fragment.appendChild(option);
  });

  selectElement.replaceChildren(fragment);
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

function renderTimeRangeControls() {
  if (!els.timeRangeList) {
    return;
  }

  const ranges = normalizeTimeRanges(state.timeRanges);
  const activeRanges = getActiveTimeRanges(ranges);
  els.timeRangeList.innerHTML = "";

  ranges.forEach((range, index) => {
    const row = document.createElement("div");
    row.className = "time-range-row";

    const fromField = document.createElement("div");
    fromField.className = "field time-range-field";
    const fromLabel = document.createElement("span");
    fromLabel.textContent = "From";
    fromField.appendChild(fromLabel);
    const isEditingFrom =
      state.activeTimeRangeEditor &&
      state.activeTimeRangeEditor.index === index &&
      state.activeTimeRangeEditor.key === "from";
    if (isEditingFrom) {
      const fromInput = document.createElement("input");
      fromInput.type = "datetime-local";
      fromInput.className = "time-range-input";
      const primaryLocale = getPrimaryLocale();
      if (primaryLocale) {
        fromInput.setAttribute("lang", primaryLocale);
      }
      fromInput.value = formatDateTimeLocalInput(range.from);
      fromInput.addEventListener("change", (event) => {
        updateTimeRangeValue(index, "from", event.target.value);
      });
      fromInput.addEventListener("blur", () => {
        if (
          state.activeTimeRangeEditor &&
          state.activeTimeRangeEditor.index === index &&
          state.activeTimeRangeEditor.key === "from"
        ) {
          setActiveTimeRangeEditor(null, null);
        }
      });
      fromField.appendChild(fromInput);
      requestAnimationFrame(() => {
        fromInput.focus();
        if (typeof fromInput.showPicker === "function") {
          try {
            fromInput.showPicker();
          } catch (_) {
            // Ignore browsers that block programmatic picker opening.
          }
        }
      });
    } else {
      const fromButton = document.createElement("button");
      fromButton.type = "button";
      fromButton.className = "btn time-range-display";
      fromButton.textContent = formatTimeRangeDisplayDate(range.from);
      fromButton.title = range.from ? formatTs(range.from) : "Set start date";
      fromButton.addEventListener("click", () => setActiveTimeRangeEditor(index, "from"));
      fromField.appendChild(fromButton);
    }

    const toField = document.createElement("div");
    toField.className = "field time-range-field";
    const toLabel = document.createElement("span");
    toLabel.textContent = "To";
    toField.appendChild(toLabel);
    const isEditingTo =
      state.activeTimeRangeEditor &&
      state.activeTimeRangeEditor.index === index &&
      state.activeTimeRangeEditor.key === "to";
    if (isEditingTo) {
      const toInput = document.createElement("input");
      toInput.type = "datetime-local";
      toInput.className = "time-range-input";
      const primaryLocale = getPrimaryLocale();
      if (primaryLocale) {
        toInput.setAttribute("lang", primaryLocale);
      }
      toInput.value = formatDateTimeLocalInput(range.to);
      toInput.addEventListener("change", (event) => {
        updateTimeRangeValue(index, "to", event.target.value);
      });
      toInput.addEventListener("blur", () => {
        if (
          state.activeTimeRangeEditor &&
          state.activeTimeRangeEditor.index === index &&
          state.activeTimeRangeEditor.key === "to"
        ) {
          setActiveTimeRangeEditor(null, null);
        }
      });
      toField.appendChild(toInput);
      requestAnimationFrame(() => {
        toInput.focus();
        if (typeof toInput.showPicker === "function") {
          try {
            toInput.showPicker();
          } catch (_) {
            // Ignore browsers that block programmatic picker opening.
          }
        }
      });
    } else {
      const toButton = document.createElement("button");
      toButton.type = "button";
      toButton.className = "btn time-range-display";
      toButton.textContent = formatTimeRangeDisplayDate(range.to);
      toButton.title = range.to ? formatTs(range.to) : "Set end date";
      toButton.addEventListener("click", () => setActiveTimeRangeEditor(index, "to"));
      toField.appendChild(toButton);
    }

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "btn time-range-btn time-range-remove";
    removeBtn.textContent = "Remove";
    removeBtn.disabled = ranges.length === 1 && !isTimeRangeActive(range);
    removeBtn.addEventListener("click", () => removeTimeRange(index));

    row.appendChild(fromField);
    row.appendChild(toField);
    row.appendChild(removeBtn);
    els.timeRangeList.appendChild(row);
  });

  if (els.addTimeRangeBtn) {
    els.addTimeRangeBtn.disabled = activeRanges.length === 0;
  }
  if (els.timeRangeSummary) {
    els.timeRangeSummary.textContent = "";
  }
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

  if (els.addTimeRangeBtn) {
    els.addTimeRangeBtn.addEventListener("click", () => {
      addTimeRange();
    });
  }
  if (els.resetFiltersBtn) {
    els.resetFiltersBtn.addEventListener("click", () => {
      resetAllFilters();
    });
  }
}

function setupResponsiveShell() {
  if (els.sidebarCollapseBtn) {
    els.sidebarCollapseBtn.addEventListener("click", () => {
      setDesktopSidebarCollapsed(!state.desktopSidebarCollapsed);
    });
  }
  els.mobileSidebarToggle.addEventListener("click", () => {
    setMobileSidebarOpen(!state.mobileSidebarOpen);
  });
  els.sidebarCloseBtn.addEventListener("click", () => {
    setMobileSidebarOpen(false);
  });
  els.sidebarBackdrop.addEventListener("click", () => {
    setMobileSidebarOpen(false);
  });
  if (els.sidebarScrollUpBtn) {
    els.sidebarScrollUpBtn.addEventListener("click", () => {
      scrollSidebarByPage(-1);
    });
  }
  if (els.sidebarScrollDownBtn) {
    els.sidebarScrollDownBtn.addEventListener("click", () => {
      scrollSidebarByPage(1);
    });
  }
  if (els.dashboardSidebarScroll) {
    els.dashboardSidebarScroll.addEventListener(
      "scroll",
      () => {
        updateSidebarScrollAffordances();
      },
      { passive: true }
    );
  }

  const handleViewportChange = () => {
    if (!isMobileLayout()) {
      setMobileSidebarOpen(false);
      applySidebarLayoutState();
      return;
    }
    applySidebarLayoutState();
    els.dashboardSidebar.setAttribute("aria-hidden", state.mobileSidebarOpen ? "false" : "true");
    updateSidebarScrollAffordances();
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
  window.addEventListener("resize", () => {
    updateSidebarScrollAffordances();
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && state.mobileSidebarOpen) {
      setMobileSidebarOpen(false);
    }
  });

  handleViewportChange();
  updateMobileFilterSummary();
  updateSidebarScrollAffordances();
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
  if (key === "calibration_ece") return run.calibrationEce;
  return null;
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
  if (!supportsApproximateCi(metricKey)) {
    return null;
  }
  const metricValue = getMetricValueForRun(run, metricKey);
  const sampleSize = getRunSampleSize(run);
  return computeApproximateCi95(metricValue, sampleSize);
}

function getMeanMetricConfidence(runs, metricKey, meanValue) {
  if (!supportsApproximateCi(metricKey)) {
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
  const activeTimeRanges = getActiveTimeRanges();
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
    if (activeTimeRanges.length) {
      const runTs = parseRunTimestampMs(run);
      if (!Number.isFinite(runTs)) {
        return false;
      }
      const matchesRange = activeTimeRanges.some((range) => {
        if (range.fromMs !== null && runTs < range.fromMs) {
          return false;
        }
        if (range.toMs !== null && runTs > range.toMs) {
          return false;
        }
        return true;
      });
      if (!matchesRange) {
        return false;
      }
    }
    return true;
  });

  runs.sort((a, b) => {
    const primary = compareMetricNumbers(getMetricValueForRun(a, state.sortBy), getMetricValueForRun(b, state.sortBy), state.sortBy);
    if (primary !== 0) return primary;
    const secondary = compareMetricNumbers(getMetricValueForRun(a, "macro_f1"), getMetricValueForRun(b, "macro_f1"), "macro_f1");
    if (secondary !== 0) return secondary;
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
  const ms = parseTimestampToMs(ts);
  if (!Number.isFinite(ms)) return "N/A";
  return new Date(ms).toLocaleString();
}

function formatDateOnly(ts) {
  const ms = parseTimestampToMs(ts);
  if (!Number.isFinite(ms)) return "N/A";
  return new Date(ms).toLocaleDateString();
}

function setSourceControlsDisabled(disabled) {
  [els.btnAutoServer, els.btnOpenFolder, els.reloadBtn].forEach((button) => {
    if (!button) {
      return;
    }

    if (disabled) {
      button.dataset.wasDisabledBeforeLoad = button.disabled ? "true" : "false";
      button.disabled = true;
      return;
    }

    if (button.dataset.wasDisabledBeforeLoad !== "true") {
      button.disabled = false;
    }
    delete button.dataset.wasDisabledBeforeLoad;
  });
}

function setLoadingState(isLoading, message = "") {
  state.isLoading = Boolean(isLoading);
  state.loadingMessage = state.isLoading ? String(message || "").trim() : "";
  if (!state.isLoading) {
    state.loadingProgressCurrent = 0;
    state.loadingProgressTotal = 0;
  }

  if (els.main) {
    els.main.setAttribute("aria-busy", state.isLoading ? "true" : "false");
  }

  setSourceControlsDisabled(state.isLoading);

  if (!els.loadingNotice || !els.loadingNoticeMessage) {
    return;
  }

  if (state.isLoading) {
    els.loadingNoticeMessage.textContent = state.loadingMessage || "Loading metrics data...";
    els.loadingNotice.hidden = false;
  } else {
    els.loadingNotice.hidden = true;
    els.loadingNoticeMessage.textContent = "Loading metrics data...";
  }

  renderLoadingProgress();
}

function renderLoadingProgress() {
  if (
    !els.loadingProgress ||
    !els.loadingProgressLabel ||
    !els.loadingProgressPercent ||
    !els.loadingProgressTrack ||
    !els.loadingProgressFill
  ) {
    return;
  }

  const total = Number.isFinite(state.loadingProgressTotal) ? Math.max(0, state.loadingProgressTotal) : 0;
  const current = Number.isFinite(state.loadingProgressCurrent)
    ? Math.max(0, Math.min(state.loadingProgressCurrent, total))
    : 0;

  if (!state.isLoading || total <= 0) {
    els.loadingProgress.hidden = true;
    els.loadingProgressLabel.textContent = "0 / 0 files";
    els.loadingProgressPercent.textContent = "0%";
    els.loadingProgressTrack.setAttribute("aria-valuemax", "0");
    els.loadingProgressTrack.setAttribute("aria-valuenow", "0");
    els.loadingProgressTrack.setAttribute("aria-valuetext", "0 of 0 metric files processed");
    els.loadingProgressFill.style.width = "0%";
    return;
  }

  const percent = Math.round((current / total) * 100);
  els.loadingProgress.hidden = false;
  els.loadingProgressLabel.textContent = `${formatNum(current, 0)} / ${formatNum(total, 0)} files`;
  els.loadingProgressPercent.textContent = `${percent}%`;
  els.loadingProgressTrack.setAttribute("aria-valuemax", String(total));
  els.loadingProgressTrack.setAttribute("aria-valuenow", String(current));
  els.loadingProgressTrack.setAttribute(
    "aria-valuetext",
    `${formatNum(current, 0)} of ${formatNum(total, 0)} metric files processed`
  );
  els.loadingProgressFill.style.width = `${percent}%`;
}

function setLoadingProgress(current, total) {
  state.loadingProgressCurrent = Number.isFinite(current) ? current : 0;
  state.loadingProgressTotal = Number.isFinite(total) ? total : 0;
  renderLoadingProgress();
}

function waitForNextPaint() {
  return new Promise((resolve) => {
    if (typeof window.requestAnimationFrame === "function") {
      window.requestAnimationFrame(() => resolve());
      return;
    }
    window.setTimeout(resolve, 0);
  });
}

async function runWithLoadingNotice(message, loader) {
  const normalizedMessage = String(message || "").trim() || "Loading metrics data...";
  setLoadingState(true, normalizedMessage);
  els.heroSubtitle.textContent = normalizedMessage;
  await waitForNextPaint();
  try {
    return await loader((current, total) => setLoadingProgress(current, total));
  } finally {
    setLoadingState(false);
  }
}

function renderTaskControls() {
  const tasks = ["ALL", ...state.tasks];
  syncSelectOptions(els.taskSelect, tasks, (task) => (task === "ALL" ? "All Tasks" : task));

  state.selectedTasks = sanitizeSelections(state.selectedTasks, state.tasks);
  syncTaskSelectValue();

  renderChoiceChipList(els.taskChipList, state.tasks, state.selectedTasks, "All Tasks", toggleTaskSelection);
}

function renderModelControls() {
  const models = ["ALL", ...state.models];
  syncSelectOptions(els.modelSelect, models, (model) => (model === "ALL" ? "All Models" : model));

  state.selectedModels = sanitizeSelections(state.selectedModels, state.models);
  syncModelSelectValue();

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
  const trackBadges = Array.isArray(options.trackBadges)
    ? uniqueNonEmptyStrings(options.trackBadges.map((badge) => asTrimmedString(badge)))
    : [];
  if (insideLabel || trackBadges.length) {
    const trackLabel = document.createElement("span");
    trackLabel.className = "bar-track-label";
    const trackContent = document.createElement("span");
    trackContent.className = "bar-track-label-content";

    if (insideLabel) {
      const trackText = document.createElement("span");
      trackText.className = "bar-track-label-text";
      trackText.textContent = insideLabel;
      trackContent.appendChild(trackText);
    }

    if (trackBadges.length) {
      const badgesWrap = document.createElement("span");
      badgesWrap.className = "bar-track-label-badges";
      const trackBadgeColorMap = options.trackBadgeColorMap instanceof Map ? options.trackBadgeColorMap : new Map();
      trackBadges.forEach((badge) => {
        badgesWrap.appendChild(createHtmlTagBadge(badge, trackBadgeColorMap.get(badge), "tag-badge tag-badge-compact"));
      });
      trackContent.appendChild(badgesWrap);
    }

    trackLabel.appendChild(trackContent);
    trackLabel.title = [insideLabel, ...trackBadges].filter(Boolean).join(" | ");
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

function getConcatenatedModelLabel(runs) {
  const models = [...new Set((runs || []).map((run) => getRunModelDisplayName(run)).filter(Boolean))].sort((a, b) =>
    a.localeCompare(b)
  );
  return models.join(" + ");
}

function getTopRunForMetric(runs, metricKey) {
  if (!Array.isArray(runs) || !runs.length) {
    return null;
  }
  const rankedRuns = [...runs].sort((a, b) => {
    const diff = compareMetricNumbers(getMetricValueForRun(a, metricKey), getMetricValueForRun(b, metricKey), metricKey);
    if (diff !== 0) {
      return diff;
    }
    return parseRunTimestampMs(b) - parseRunTimestampMs(a);
  });
  return rankedRuns[0] || null;
}

function getTaskGroupedTopRunLabel(runs, metricKey) {
  const topRun = getTopRunForMetric(runs, metricKey);
  if (!topRun) {
    return "";
  }
  const metricValue = getMetricValueForRun(topRun, metricKey);
  if (metricValue === null || metricValue === undefined || Number.isNaN(metricValue)) {
    return "";
  }
  const metricLabel = (METRIC_LABELS[metricKey] || metricKey).toLowerCase();
  return `top: ${getRunModelDisplayName(topRun)} (${metricLabel}: ${formatMetric(metricKey, metricValue)})`;
}

function shouldShowLeaderboardContextLabels() {
  return state.leaderboardChartGroupBy === "task" ? state.selectedModels.length !== 1 : state.selectedTasks.length !== 1;
}

function setLeaderboardChartGroupBy(groupBy) {
  if (!LEADERBOARD_CHART_GROUP_BY_KEYS.has(groupBy) || state.leaderboardChartGroupBy === groupBy) {
    return;
  }
  state.leaderboardChartGroupBy = groupBy;
  persistUiState();
  renderLeaderboard(state.filtered);
}

function renderLeaderboardGroupSwitch() {
  if (!els.leaderboardGroupSwitch) {
    return;
  }
  els.leaderboardGroupSwitch.hidden = state.leaderboardTab !== "chart";
  els.leaderboardGroupSwitch.innerHTML = "";
  if (state.leaderboardTab !== "chart") {
    return;
  }

  const label = document.createElement("span");
  label.className = "leaderboard-group-switch-label";
  label.textContent = "Group Chart By";
  els.leaderboardGroupSwitch.appendChild(label);

  const toggle = document.createElement("div");
  toggle.className = "leaderboard-group-toggle";
  toggle.setAttribute("role", "group");
  toggle.setAttribute("aria-label", "Leaderboard chart grouping");

  Object.entries(LEADERBOARD_CHART_GROUP_BY_LABELS).forEach(([key, text]) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `leaderboard-group-btn${state.leaderboardChartGroupBy === key ? " active" : ""}`;
    button.textContent = text;
    button.setAttribute("aria-pressed", state.leaderboardChartGroupBy === key ? "true" : "false");
    button.addEventListener("click", () => setLeaderboardChartGroupBy(key));
    toggle.appendChild(button);
  });

  els.leaderboardGroupSwitch.appendChild(toggle);
}

function renderLeaderboardTabControls() {
  if (!els.leaderboardTabs) {
    return;
  }
  renderLeaderboardGroupSwitch();
  els.leaderboardTabs.innerHTML = "";
  const tabs = [
    { key: "chart", label: "Chart" },
    { key: "time_series", label: "Time Series" },
    { key: "table", label: "Metrics Table" },
    { key: "best_by_task", label: "Best Run Per Task" },
    { key: "radar", label: "Radar" },
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
  const metricIsLowerBetter = isLowerBetterMetric(metricKey);
  const suppressTopMarkers = state.leaderboardChartGroupBy === "task";
  const showApproximateCi = supportsApproximateCi(metricKey);
  const showSelectedTagBadges = hasMultipleSelectedTags();
  const selectedTagColorMap = buildSelectedTagColorMap();
  const groupBy = state.leaderboardChartGroupBy;
  const showContextLabels = shouldShowLeaderboardContextLabels();
  const grouping =
    groupBy === "task"
      ? {
          key: "task",
          getGroupValue: (run) => asTrimmedString(run.task) || "Unknown task",
          getGroupLabel: (run) => asTrimmedString(run.task) || "Unknown task",
          getSingleRowLabel: (run) => asTrimmedString(run.task) || "Unknown task",
          getSingleRowContextLabel: (run) => (showContextLabels ? getRunModelDisplayName(run) : ""),
          getGroupedRunLabel: (run) => getRunModelDisplayName(run),
          getGroupedRunContextLabel: () => "",
          getGroupContextLabel: (groupRuns) => (showContextLabels ? getConcatenatedModelLabel(groupRuns) : ""),
        }
      : {
          key: "model",
          getGroupValue: (run) => run.model,
          getGroupLabel: (run, groupRuns) => {
            const sharedSuffix = getSharedRunEffortSuffix(groupRuns);
            return sharedSuffix ? `${run.model} ${sharedSuffix}` : run.model;
          },
          getSingleRowLabel: (run) => getRunModelDisplayName(run),
          getSingleRowContextLabel: (run) => (showContextLabels ? run.task : ""),
          getGroupedRunLabel: (run) => getRunModelDisplayName(run),
          getGroupedRunContextLabel: (run) => (showContextLabels ? run.task : ""),
          getGroupContextLabel: (groupRuns) => (showContextLabels ? getConcatenatedTaskLabel(groupRuns) : ""),
        };

  if (!source.length) {
    container.innerHTML = `<p class="muted">No runs with ${metricLabel.toLowerCase()} in current filter.</p>`;
    return;
  }

  if (showApproximateCi) {
    const ciNote = document.createElement("p");
    ciNote.className = "leaderboard-ci-note muted";
    ciNote.textContent = "95% CI is approximated from each run's evaluated examples.";
    container.appendChild(ciNote);
  }
  if (metricIsLowerBetter) {
    const directionNote = document.createElement("p");
    directionNote.className = "leaderboard-ci-note muted";
    directionNote.textContent = `${metricLabel} is lower-is-better.`;
    container.appendChild(directionNote);
  }
  const groups = new Map();
  source.forEach((run) => {
    const key = grouping.getGroupValue(run);
    if (!groups.has(key)) {
      groups.set(key, []);
    }
    groups.get(key).push(run);
  });

  const entries = Array.from(groups.entries()).map(([groupValue, groupRuns]) => {
    const runsSorted = [...groupRuns].sort((a, b) => {
      const diff = compareMetricNumbers(getMetricValueForRun(a, metricKey), getMetricValueForRun(b, metricKey), metricKey);
      if (diff !== 0) {
        return diff;
      }
      return parseRunTimestampMs(b) - parseRunTimestampMs(a);
    });
    if (runsSorted.length === 1) {
      const run = runsSorted[0];
      const metricValue = getMetricValueForRun(run, metricKey);
      return {
        type: "run",
        key: run.filePath,
        label: grouping.getSingleRowLabel(run),
        metricValue: typeof metricValue === "number" && Number.isFinite(metricValue) ? metricValue : 0,
        ci: showApproximateCi ? getRunMetricConfidence(run, metricKey) : null,
        run,
      };
    }

    const avgMetric =
      runsSorted.reduce((sum, run) => sum + (getMetricValueForRun(run, metricKey) ?? 0), 0) / runsSorted.length;
    return {
      type: "group",
      key: `${grouping.key}:${groupValue}`,
      label: grouping.getGroupLabel(runsSorted[0], runsSorted),
      metricValue: avgMetric,
      avgMetric,
      ci: showApproximateCi ? getMeanMetricConfidence(runsSorted, metricKey, avgMetric) : null,
      runs: runsSorted,
    };
  });

  entries.sort((a, b) => {
    const diff = compareMetricNumbers(a.metricValue, b.metricValue, metricKey);
    if (diff !== 0) {
      return diff;
    }
    return a.label.localeCompare(b.label);
  });

  const groupedEntries = entries.filter((entry) => entry.type === "group");
  const rankedRuns = [...source].sort((a, b) => {
    const diff = compareMetricNumbers(getMetricValueForRun(a, metricKey), getMetricValueForRun(b, metricKey), metricKey);
    if (diff !== 0) {
      return diff;
    }
    return parseRunTimestampMs(b) - parseRunTimestampMs(a);
  });
  const topRunPath = !suppressTopMarkers && rankedRuns.length ? rankedRuns[0].filePath : null;

  if (groupedEntries.length) {
    const groupedSummary = document.createElement("p");
    groupedSummary.className = "leaderboard-ci-note muted";
    groupedSummary.textContent =
      groupBy === "task"
        ? `Grouped task rows show average ${metricLabel.toLowerCase()} with run distribution overlays and the top model label.`
        : groupedEntries.length > 1
        ? "TOP is based on the best individual run; if hidden in a collapsed group, the marker appears on the group summary."
        : `Grouped rows include run distribution overlays${showApproximateCi ? " with CI" : ""}.`;
    container.appendChild(groupedSummary);
  }

  const maxScore = resolveLeaderboardBarMax(metricKey, [
    ...entries.map((entry) => entry.metricValue),
    ...source.map((run) => getMetricValueForRun(run, metricKey)),
  ]);
  entries.forEach((entry) => {
    if (entry.type === "run") {
      const isTopRun = topRunPath && entry.run.filePath === topRunPath;
      const badges = [];
      if (isTopRun) {
        badges.push("TOP");
      }
      const trackBadges = showSelectedTagBadges ? getSelectedTagsForRun(entry.run) : [];
      let rowClass = "";
      if (isTopRun) {
        rowClass = "bar-row-top";
      }
      container.appendChild(
        createBarRow(
          entry.label,
          entry.metricValue,
          maxScore,
          (value, ci) => `${formatMetric(metricKey, value)}${formatCiRange(ci)}`,
          null,
          () => openRunModal(entry.run),
          entry.ci,
          groupBy === "task" ? getTaskGroupedTopRunLabel([entry.run], metricKey) : grouping.getSingleRowContextLabel(entry.run),
          { badges, rowClass, trackBadges, trackBadgeColorMap: selectedTagColorMap }
        )
      );
      return;
    }

    const details = document.createElement("details");
    details.className = "leaderboard-group";
    details.dataset.groupKey = entry.key;
    details.open = state.expandedLeaderboardGroups.has(entry.key);
    details.addEventListener("toggle", () => {
      if (details.open) {
        state.expandedLeaderboardGroups.add(entry.key);
      } else {
        state.expandedLeaderboardGroups.delete(entry.key);
      }
    });

    const summary = document.createElement("summary");
    summary.className = "leaderboard-summary";
    const groupHasTopRun = Boolean(topRunPath && entry.runs.some((run) => run.filePath === topRunPath));
    const badges = [];
    if (groupHasTopRun) {
      badges.push("TOP");
    }
    const groupTrackBadges = showSelectedTagBadges ? getSelectedTagsForRuns(entry.runs) : [];
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
        groupBy === "task" ? getTaskGroupedTopRunLabel(entry.runs, metricKey) : grouping.getGroupContextLabel(entry.runs),
        {
          badges: suppressTopMarkers ? [] : badges,
          rowClass: suppressTopMarkers ? "" : rowClass,
          trackBadges: groupTrackBadges,
          trackBadgeColorMap: selectedTagColorMap,
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
      const runTrackBadges = showSelectedTagBadges ? getSelectedTagsForRun(run) : [];
      let runRowClass = "";
      if (showTopOnRun) {
        runRowClass = "bar-row-top";
      }
      runsWrap.appendChild(
        createBarRow(
          grouping.getGroupedRunLabel(run),
          getMetricValueForRun(run, metricKey) ?? 0,
          maxScore,
          (value, ci) => `${formatMetric(metricKey, value)}${formatCiRange(ci)}`,
          null,
          () => openRunModal(run),
          runCi,
          grouping.getGroupedRunContextLabel(run),
          {
            badges: suppressTopMarkers ? [] : runBadges,
            rowClass: suppressTopMarkers ? "" : runRowClass,
            trackBadges: runTrackBadges,
            trackBadgeColorMap: selectedTagColorMap,
          }
        )
      );
    });
    details.appendChild(runsWrap);
    container.appendChild(details);
  });
}

function createSvgNode(name, attributes = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attributes).forEach(([key, value]) => {
    node.setAttribute(key, String(value));
  });
  return node;
}

function polygonPointString(cx, cy, radius, sides, rotationDegrees = 0) {
  const points = [];
  const rotation = (rotationDegrees * Math.PI) / 180;
  for (let index = 0; index < sides; index += 1) {
    const angle = rotation + (Math.PI * 2 * index) / sides;
    points.push(`${cx + Math.cos(angle) * radius},${cy + Math.sin(angle) * radius}`);
  }
  return points.join(" ");
}

function buildTimeSeriesShape(shape, cx, cy, size, fill, stroke = "currentColor", strokeWidth = 1.5) {
  const radius = size / 2;
  if (shape === "square") {
    return createSvgNode("rect", {
      x: cx - radius,
      y: cy - radius,
      width: size,
      height: size,
      rx: 1.5,
      fill,
      stroke,
      "stroke-width": strokeWidth,
    });
  }
  if (shape === "diamond") {
    return createSvgNode("polygon", {
      points: `${cx},${cy - radius} ${cx + radius},${cy} ${cx},${cy + radius} ${cx - radius},${cy}`,
      fill,
      stroke,
      "stroke-width": strokeWidth,
    });
  }
  if (shape === "triangle") {
    return createSvgNode("polygon", {
      points: polygonPointString(cx, cy, radius, 3, -90),
      fill,
      stroke,
      "stroke-width": strokeWidth,
    });
  }
  if (shape === "triangle_down") {
    return createSvgNode("polygon", {
      points: polygonPointString(cx, cy, radius, 3, 90),
      fill,
      stroke,
      "stroke-width": strokeWidth,
    });
  }
  if (shape === "hexagon") {
    return createSvgNode("polygon", {
      points: polygonPointString(cx, cy, radius, 6, -90),
      fill,
      stroke,
      "stroke-width": strokeWidth,
    });
  }
  if (shape === "pentagon") {
    return createSvgNode("polygon", {
      points: polygonPointString(cx, cy, radius, 5, -90),
      fill,
      stroke,
      "stroke-width": strokeWidth,
    });
  }
  if (shape === "octagon") {
    return createSvgNode("polygon", {
      points: polygonPointString(cx, cy, radius, 8, -90),
      fill,
      stroke,
      "stroke-width": strokeWidth,
    });
  }
  return createSvgNode("circle", {
    cx,
    cy,
    r: radius,
    fill,
    stroke,
    "stroke-width": strokeWidth,
  });
}

function buildNumericTicks(minValue, maxValue, count) {
  if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) {
    return [];
  }
  if (count <= 1 || minValue === maxValue) {
    return [minValue];
  }
  return Array.from({ length: count }, (_, index) => minValue + ((maxValue - minValue) * index) / (count - 1));
}

function formatTimeSeriesTick(ms, spanMs) {
  const dt = new Date(ms);
  if (spanMs >= 1000 * 60 * 60 * 24 * 2) {
    return dt.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  }
  return dt.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

function createTimeSeriesLegend(title, rows) {
  const wrap = document.createElement("section");
  wrap.className = "time-series-legend";
  const heading = document.createElement("h4");
  heading.textContent = title;
  wrap.appendChild(heading);

  const grid = document.createElement("div");
  grid.className = "time-series-legend-grid";
  rows.forEach((row) => {
    const entry = document.createElement("div");
    entry.className = "time-series-legend-row";
    if (row.type === "task") {
      const swatch = document.createElement("span");
      swatch.className = "time-series-task-swatch";
      swatch.style.background = row.color;
      entry.appendChild(swatch);
    } else {
      const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      icon.setAttribute("viewBox", "0 0 16 16");
      icon.setAttribute("class", "time-series-shape-icon");
      icon.appendChild(buildTimeSeriesShape(row.shape, 8, 8, 9, "currentColor", "currentColor", 1.2));
      entry.appendChild(icon);
    }

    const label = document.createElement("span");
    label.textContent = row.label;
    entry.appendChild(label);
    grid.appendChild(entry);
  });
  wrap.appendChild(grid);
  return wrap;
}

function truncateTimeSeriesLabel(label, maxLength = 18) {
  const text = asTrimmedString(label);
  if (!text || text.length <= maxLength) {
    return text;
  }
  return `${text.slice(0, maxLength - 1)}...`;
}

function hasMultipleSelectedTags() {
  return Array.isArray(state.selectedTags) && state.selectedTags.length > 1;
}

function buildSelectedTagColorMap(tags = state.selectedTags) {
  const uniqueTags = uniqueNonEmptyStrings(tags);
  return new Map(
    uniqueTags.map((tag, index) => [tag, SELECTED_TAG_BADGE_COLORS[index % SELECTED_TAG_BADGE_COLORS.length]])
  );
}

function getSelectedTagsForRun(run, selectedTags = state.selectedTags) {
  if (!Array.isArray(selectedTags) || selectedTags.length <= 1) {
    return [];
  }
  const runTagSet = new Set((Array.isArray(run && run.tags) ? run.tags : []).map((tag) => asTrimmedString(tag)).filter(Boolean));
  return selectedTags.filter((tag) => runTagSet.has(tag));
}

function getSelectedTagsForRuns(runs, selectedTags = state.selectedTags) {
  if (!Array.isArray(selectedTags) || selectedTags.length <= 1) {
    return [];
  }
  const presentTags = new Set(
    (runs || [])
      .flatMap((run) => (Array.isArray(run && run.tags) ? run.tags : []))
      .map((tag) => asTrimmedString(tag))
      .filter(Boolean)
  );
  return selectedTags.filter((tag) => presentTags.has(tag));
}

function createHtmlTagBadge(tag, color, className = "tag-badge") {
  const badge = document.createElement("span");
  badge.className = className;
  badge.textContent = tag;
  if (color) {
    badge.style.setProperty("--tag-badge-color", color);
  }
  return badge;
}

function estimateTimeSeriesTagBadgeWidth(label) {
  const text = asTrimmedString(label);
  return Math.max(28, text.length * 6.3 + 14);
}

function createTimeSeriesTagBadgeGroup(tags, tagColorMap, x, y, bounds = {}) {
  const visibleTags = uniqueNonEmptyStrings(tags);
  if (!visibleTags.length) {
    return null;
  }

  const gap = 4;
  const height = 16;
  const badgeRows = visibleTags.map((tag) => {
    const label = truncateTimeSeriesLabel(tag, 14);
    return {
      label,
      color: tagColorMap.get(tag),
      width: estimateTimeSeriesTagBadgeWidth(label),
    };
  });
  const totalWidth = badgeRows.reduce((sum, entry, index) => sum + entry.width + (index > 0 ? gap : 0), 0);
  const minX = typeof bounds.minX === "number" ? bounds.minX : 0;
  const maxX = typeof bounds.maxX === "number" ? bounds.maxX : Number.POSITIVE_INFINITY;
  const maxY = typeof bounds.maxY === "number" ? bounds.maxY : Number.POSITIVE_INFINITY;
  const clampedX = Math.max(minX, Math.min(x, maxX - totalWidth));
  const clampedY = Math.max(0, Math.min(y, maxY - height));

  const group = createSvgNode("g", {
    class: "time-series-tag-badge-group",
    transform: `translate(${clampedX} ${clampedY})`,
  });

  let offsetX = 0;
  badgeRows.forEach((entry, index) => {
    const fillColor = entry.color || SELECTED_TAG_BADGE_COLORS[index % SELECTED_TAG_BADGE_COLORS.length];
    const rect = createSvgNode("rect", {
      x: offsetX,
      y: 0,
      width: entry.width,
      height,
      rx: 8,
      class: "time-series-tag-badge-rect",
    });
    rect.style.fill = fillColor;
    rect.style.stroke = fillColor;
    group.appendChild(rect);

    const text = createSvgNode("text", {
      x: offsetX + entry.width / 2,
      y: 11.3,
      "text-anchor": "middle",
      class: "time-series-tag-badge-text",
    });
    text.textContent = entry.label;
    group.appendChild(text);

    offsetX += entry.width + gap;
  });

  return group;
}

function clampTimeSeriesViewport(viewport, fullDomain) {
  const normalized = normalizeTimeSeriesViewport(viewport);
  if (!normalized || !fullDomain) {
    return null;
  }

  const xMin = Math.max(fullDomain.xMin, Math.min(normalized.xMin, fullDomain.xMax));
  const xMax = Math.max(fullDomain.xMin, Math.min(normalized.xMax, fullDomain.xMax));
  const yMin = Math.max(fullDomain.yMin, Math.min(normalized.yMin, fullDomain.yMax));
  const yMax = Math.max(fullDomain.yMin, Math.min(normalized.yMax, fullDomain.yMax));

  if (xMax - xMin <= 1 || yMax - yMin <= 1e-9) {
    return null;
  }

  return { xMin, xMax, yMin, yMax };
}

function renderLeaderboardTimeSeries(container, runs) {
  const metricKey = state.sortBy;
  const metricLabel = METRIC_LABELS[metricKey] || metricKey;
  const showApproximateCi = supportsApproximateCi(metricKey);
  const showSelectedTagBadges = hasMultipleSelectedTags();
  const selectedTagColorMap = buildSelectedTagColorMap();
  const source = runs
    .map((run) => {
      const timestampMs = parseRunTimestampMs(run);
      const metricValue = getMetricValueForRun(run, metricKey);
      if (!Number.isFinite(timestampMs) || typeof metricValue !== "number" || !Number.isFinite(metricValue)) {
        return null;
      }
      return {
        run,
        timestampMs,
        metricValue,
        ci: showApproximateCi ? getRunMetricConfidence(run, metricKey) : null,
      };
    })
    .filter(Boolean)
    .sort((a, b) => {
      if (a.timestampMs !== b.timestampMs) {
        return a.timestampMs - b.timestampMs;
      }
      return compareMetricNumbers(a.metricValue, b.metricValue, metricKey);
    });

  if (!source.length) {
    container.innerHTML = `<p class="muted">No runs with both timestamp and ${metricLabel.toLowerCase()} in current filter.</p>`;
    return;
  }

  const tasks = [...new Set(source.map((entry) => asTrimmedString(entry.run.task)).filter(Boolean))].sort((a, b) =>
    a.localeCompare(b)
  );
  const models = [...new Set(source.map((entry) => asTrimmedString(entry.run.model)).filter(Boolean))].sort((a, b) =>
    a.localeCompare(b)
  );
  const taskColorByName = new Map(tasks.map((task, index) => [task, TIME_SERIES_TASK_COLORS[index % TIME_SERIES_TASK_COLORS.length]]));
  const modelShapeByName = new Map(
    models.map((model, index) => [model, TIME_SERIES_MODEL_SHAPES[index % TIME_SERIES_MODEL_SHAPES.length]])
  );

  const header = document.createElement("div");
  header.className = "time-series-head";
  const summary = document.createElement("p");
  summary.className = "muted";
  summary.textContent = `${formatNum(source.length, 0)} runs | ${formatNum(tasks.length, 0)} task(s) | ${formatNum(models.length, 0)} model(s)`;
  header.appendChild(summary);

  const controls = document.createElement("div");
  controls.className = "time-series-controls";

  const labelsButton = document.createElement("button");
  labelsButton.type = "button";
  labelsButton.className = `time-series-control-btn${state.timeSeriesShowLabels ? " active" : ""}`;
  labelsButton.textContent = state.timeSeriesShowLabels ? "Labels: On" : "Labels: Off";
  labelsButton.setAttribute("aria-pressed", state.timeSeriesShowLabels ? "true" : "false");
  labelsButton.addEventListener("click", () => setTimeSeriesShowLabels(!state.timeSeriesShowLabels));
  controls.appendChild(labelsButton);

  const zoomResetButton = document.createElement("button");
  zoomResetButton.type = "button";
  zoomResetButton.className = "time-series-control-btn";
  zoomResetButton.textContent = "Reset Zoom";
  zoomResetButton.disabled = !state.timeSeriesViewport;
  zoomResetButton.addEventListener("click", () => resetTimeSeriesZoom());
  controls.appendChild(zoomResetButton);

  header.appendChild(controls);
  container.appendChild(header);

  const noteWrap = document.createElement("div");
  noteWrap.className = "time-series-notes";
  if (showApproximateCi) {
    const ciNote = document.createElement("p");
    ciNote.className = "muted";
    ciNote.textContent = "95% CI is approximated from each run's evaluated examples.";
    noteWrap.appendChild(ciNote);
  }
  if (isLowerBetterMetric(metricKey)) {
    const note = document.createElement("p");
    note.className = "muted";
    note.textContent = `${metricLabel} is lower-is-better.`;
    noteWrap.appendChild(note);
  }
  const zoomNote = document.createElement("p");
  zoomNote.className = "muted";
  zoomNote.textContent = state.timeSeriesViewport
    ? "Drag on the chart to zoom again. Double-click to reset zoom."
    : "Drag on the chart to zoom into a selected region. Double-click to reset zoom.";
  noteWrap.appendChild(zoomNote);
  if (noteWrap.childElementCount) {
    container.appendChild(noteWrap);
  }

  const chartWrap = document.createElement("div");
  chartWrap.className = "time-series-wrap";
  container.appendChild(chartWrap);

  const width = 960;
  const height = 460;
  const margin = { top: 18, right: 26, bottom: 58, left: 68 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const svg = createSvgNode("svg", {
    viewBox: `0 0 ${width} ${height}`,
    class: "time-series-svg",
    "aria-label": `Time series scatterplot for ${metricLabel}`,
  });
  chartWrap.appendChild(svg);

  let fullXMin = Math.min(...source.map((entry) => entry.timestampMs));
  let fullXMax = Math.max(...source.map((entry) => entry.timestampMs));
  if (fullXMin === fullXMax) {
    fullXMin -= 1000 * 60 * 30;
    fullXMax += 1000 * 60 * 30;
  }
  const fullXSpan = Math.max(fullXMax - fullXMin, 1);

  const metricValues = source.map((entry) => entry.metricValue);
  let fullYMin = isPercentMetric(metricKey) ? 0 : Math.min(...metricValues);
  let fullYMax = isPercentMetric(metricKey) ? 100 : Math.max(...metricValues);
  if (fullYMin === fullYMax) {
    const pad = fullYMin === 0 ? 1 : Math.abs(fullYMin) * 0.05;
    fullYMin -= pad;
    fullYMax += pad;
  }
  if (!isPercentMetric(metricKey)) {
    const padding = (fullYMax - fullYMin) * 0.08;
    fullYMin -= padding;
    fullYMax += padding;
  }

  const clampedViewport = clampTimeSeriesViewport(state.timeSeriesViewport, {
    xMin: fullXMin,
    xMax: fullXMax,
    yMin: fullYMin,
    yMax: fullYMax,
  });
  const xMin = clampedViewport ? clampedViewport.xMin : fullXMin;
  const xMax = clampedViewport ? clampedViewport.xMax : fullXMax;
  const yMin = clampedViewport ? clampedViewport.yMin : fullYMin;
  const yMax = clampedViewport ? clampedViewport.yMax : fullYMax;
  const xSpan = Math.max(xMax - xMin, 1);
  const ySpan = Math.max(yMax - yMin, 1e-9);

  const xToPx = (value) => margin.left + ((value - xMin) / xSpan) * innerWidth;
  const yToPx = (value) => margin.top + innerHeight - ((value - yMin) / ySpan) * innerHeight;
  const pxToX = (px) => xMin + ((px - margin.left) / innerWidth) * xSpan;
  const pxToY = (px) => yMin + ((margin.top + innerHeight - px) / innerHeight) * ySpan;
  const yTicks = isPercentMetric(metricKey) ? [0, 25, 50, 75, 100] : buildNumericTicks(yMin, yMax, 5);
  const xTicks = buildNumericTicks(xMin, xMax, Math.min(6, Math.max(2, source.length)));

  yTicks.forEach((tickValue) => {
    const y = yToPx(tickValue);
    svg.appendChild(createSvgNode("line", { x1: margin.left, x2: width - margin.right, y1: y, y2: y, class: "time-series-grid-line" }));
    const label = createSvgNode("text", { x: margin.left - 10, y: y + 4, "text-anchor": "end", class: "time-series-tick" });
    label.textContent = isPercentMetric(metricKey) ? `${formatNum(tickValue, 0)}%` : formatNum(tickValue, 2);
    svg.appendChild(label);
  });

  xTicks.forEach((tickValue) => {
    const x = xToPx(tickValue);
    svg.appendChild(createSvgNode("line", { x1: x, x2: x, y1: margin.top, y2: height - margin.bottom, class: "time-series-grid-line" }));
    const label = createSvgNode("text", {
      x,
      y: height - margin.bottom + 20,
      "text-anchor": "middle",
      class: "time-series-tick",
    });
    label.textContent = formatTimeSeriesTick(tickValue, xSpan);
    svg.appendChild(label);
  });

  svg.appendChild(
    createSvgNode("line", {
      x1: margin.left,
      x2: width - margin.right,
      y1: height - margin.bottom,
      y2: height - margin.bottom,
      class: "time-series-axis",
    })
  );
  svg.appendChild(
    createSvgNode("line", {
      x1: margin.left,
      x2: margin.left,
      y1: margin.top,
      y2: height - margin.bottom,
      class: "time-series-axis",
    })
  );

  const xAxisLabel = createSvgNode("text", {
    x: margin.left + innerWidth / 2,
    y: height - 14,
    "text-anchor": "middle",
    class: "time-series-axis-label",
  });
  xAxisLabel.textContent = "Time";
  svg.appendChild(xAxisLabel);

  const yAxisLabel = createSvgNode("text", {
    x: 18,
    y: margin.top + innerHeight / 2,
    transform: `rotate(-90 18 ${margin.top + innerHeight / 2})`,
    "text-anchor": "middle",
    class: "time-series-axis-label",
  });
  yAxisLabel.textContent = metricLabel;
  svg.appendChild(yAxisLabel);

  const selectionOverlay = createSvgNode("rect", {
    x: margin.left,
    y: margin.top,
    width: innerWidth,
    height: innerHeight,
    class: "time-series-selection-overlay",
  });
  svg.appendChild(selectionOverlay);

  source.forEach((entry) => {
    if (entry.ci && typeof entry.ci.low === "number" && typeof entry.ci.high === "number") {
      const ciX = xToPx(entry.timestampMs);
      const lowY = yToPx(entry.ci.low);
      const highY = yToPx(entry.ci.high);
      svg.appendChild(
        createSvgNode("line", {
          x1: ciX,
          x2: ciX,
          y1: lowY,
          y2: highY,
          class: "time-series-ci-line",
        })
      );
      svg.appendChild(
        createSvgNode("line", {
          x1: ciX - 5,
          x2: ciX + 5,
          y1: lowY,
          y2: lowY,
          class: "time-series-ci-cap",
        })
      );
      svg.appendChild(
        createSvgNode("line", {
          x1: ciX - 5,
          x2: ciX + 5,
          y1: highY,
          y2: highY,
          class: "time-series-ci-cap",
        })
      );
    }

    const pointGroup = createSvgNode("g", {
      class: `time-series-point${entry.run.filePath === state.selectedRunPath ? " is-selected" : ""}`,
      tabindex: "0",
      role: "button",
      "aria-label": `${entry.run.task}, ${getRunModelDisplayName(entry.run)}, ${formatTs(entry.run.timestamp)}, ${formatMetric(metricKey, entry.metricValue)}`,
    });
    pointGroup.style.color = entry.run.filePath === state.selectedRunPath ? "var(--accent)" : "var(--ink)";
    const pointX = xToPx(entry.timestampMs);
    const pointY = yToPx(entry.metricValue);
    const point = buildTimeSeriesShape(
      modelShapeByName.get(entry.run.model) || TIME_SERIES_MODEL_SHAPES[0],
      pointX,
      pointY,
      10,
      taskColorByName.get(entry.run.task) || TIME_SERIES_TASK_COLORS[0],
      "currentColor",
      entry.run.filePath === state.selectedRunPath ? 2.2 : 1.4
    );
    const title = createSvgNode("title");
    title.textContent = `${entry.run.task} | ${getRunModelDisplayName(entry.run)} | ${formatTs(entry.run.timestamp)} | ${formatMetric(
      metricKey,
      entry.metricValue
    )}`;
    pointGroup.appendChild(title);
    pointGroup.appendChild(point);
    pointGroup.addEventListener("click", () => openRunModal(entry.run));
    pointGroup.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openRunModal(entry.run);
      }
    });
    svg.appendChild(pointGroup);

    if (state.timeSeriesShowLabels) {
      const label = createSvgNode("text", {
        x: pointX + 7,
        y: pointY - 8,
        class: "time-series-point-label",
      });
      label.textContent = truncateTimeSeriesLabel(`${entry.run.task} | ${getRunModelDisplayName(entry.run)}`, 26);
      svg.appendChild(label);
    }

    if (showSelectedTagBadges && state.timeSeriesShowLabels) {
      const pointTags = getSelectedTagsForRun(entry.run);
      const badgeGroup = createTimeSeriesTagBadgeGroup(pointTags, selectedTagColorMap, pointX + 7, pointY + 6, {
        minX: margin.left + 2,
        maxX: width - margin.right - 2,
        maxY: height - margin.bottom - 2,
      });
      if (badgeGroup) {
        svg.appendChild(badgeGroup);
      }
    }
  });

  const selectionRect = createSvgNode("rect", {
    class: "time-series-selection-box",
    visibility: "hidden",
  });
  svg.appendChild(selectionRect);

  const pointerState = {
    dragging: false,
    startX: 0,
    startY: 0,
    currentX: 0,
    currentY: 0,
  };

  const svgPointForEvent = (event) => {
    const rect = svg.getBoundingClientRect();
    const scaleX = width / Math.max(rect.width, 1);
    const scaleY = height / Math.max(rect.height, 1);
    return {
      x: (event.clientX - rect.left) * scaleX,
      y: (event.clientY - rect.top) * scaleY,
    };
  };

  const clampPlotPoint = (point) => ({
    x: Math.max(margin.left, Math.min(width - margin.right, point.x)),
    y: Math.max(margin.top, Math.min(height - margin.bottom, point.y)),
  });

  const updateSelectionBox = () => {
    const left = Math.min(pointerState.startX, pointerState.currentX);
    const top = Math.min(pointerState.startY, pointerState.currentY);
    const boxWidth = Math.abs(pointerState.currentX - pointerState.startX);
    const boxHeight = Math.abs(pointerState.currentY - pointerState.startY);
    selectionRect.setAttribute("x", String(left));
    selectionRect.setAttribute("y", String(top));
    selectionRect.setAttribute("width", String(boxWidth));
    selectionRect.setAttribute("height", String(boxHeight));
    selectionRect.setAttribute("visibility", boxWidth > 0 && boxHeight > 0 ? "visible" : "hidden");
  };

  const stopDragging = () => {
    pointerState.dragging = false;
    selectionRect.setAttribute("visibility", "hidden");
    svg.classList.remove("is-brushing");
  };

  const handleWindowMouseUp = (event) => {
    if (!pointerState.dragging) {
      window.removeEventListener("mouseup", handleWindowMouseUp);
      return;
    }
    const point = clampPlotPoint(svgPointForEvent(event));
    pointerState.currentX = point.x;
    pointerState.currentY = point.y;
    const dx = Math.abs(pointerState.currentX - pointerState.startX);
    const dy = Math.abs(pointerState.currentY - pointerState.startY);
    stopDragging();
    window.removeEventListener("mouseup", handleWindowMouseUp);

    if (dx < 12 || dy < 12) {
      return;
    }

    const left = Math.min(pointerState.startX, pointerState.currentX);
    const right = Math.max(pointerState.startX, pointerState.currentX);
    const top = Math.min(pointerState.startY, pointerState.currentY);
    const bottom = Math.max(pointerState.startY, pointerState.currentY);
    setTimeSeriesViewport({
      xMin: pxToX(left),
      xMax: pxToX(right),
      yMin: pxToY(bottom),
      yMax: pxToY(top),
    });
  };

  selectionOverlay.addEventListener("mousedown", (event) => {
    if (event.button !== 0) {
      return;
    }
    const point = clampPlotPoint(svgPointForEvent(event));
    pointerState.dragging = true;
    pointerState.startX = point.x;
    pointerState.startY = point.y;
    pointerState.currentX = point.x;
    pointerState.currentY = point.y;
    svg.classList.add("is-brushing");
    updateSelectionBox();
    window.addEventListener("mouseup", handleWindowMouseUp);
    event.preventDefault();
  });

  svg.addEventListener("mousemove", (event) => {
    if (!pointerState.dragging) {
      return;
    }
    const point = clampPlotPoint(svgPointForEvent(event));
    pointerState.currentX = point.x;
    pointerState.currentY = point.y;
    updateSelectionBox();
  });

  svg.addEventListener("dblclick", () => {
    resetTimeSeriesZoom();
  });

  const legends = document.createElement("div");
  legends.className = "time-series-legends";
  legends.appendChild(
    createTimeSeriesLegend(
      "Tasks (color)",
      tasks.map((task) => ({
        type: "task",
        color: taskColorByName.get(task),
        label: task,
      }))
    )
  );
  legends.appendChild(
    createTimeSeriesLegend(
      "Models (shape)",
      models.map((model) => ({
        type: "model",
        shape: modelShapeByName.get(model),
        label: model,
      }))
    )
  );
  container.appendChild(legends);
}

function cleanupLeaderboardMetricsScrollAffordance() {
  if (typeof leaderboardMetricsScrollCleanup === "function") {
    leaderboardMetricsScrollCleanup();
    leaderboardMetricsScrollCleanup = null;
  }
}

function setupLeaderboardMetricsScrollAffordance(wrap, hint) {
  cleanupLeaderboardMetricsScrollAffordance();

  const update = () => {
    const maxScrollLeft = Math.max(wrap.scrollWidth - wrap.clientWidth, 0);
    const hasOverflow = maxScrollLeft > 12;
    const atStart = wrap.scrollLeft <= 8;
    const atEnd = maxScrollLeft - wrap.scrollLeft <= 8;

    wrap.classList.toggle("has-horizontal-overflow", hasOverflow);
    wrap.classList.toggle("is-scroll-start", hasOverflow && atStart);
    wrap.classList.toggle("is-scroll-mid", hasOverflow && !atStart && !atEnd);
    wrap.classList.toggle("is-scroll-end", hasOverflow && atEnd);
    hint.classList.toggle("is-visible", hasOverflow && atStart);
  };

  const onScroll = () => update();
  const onResize = () => update();
  let resizeObserver = null;

  wrap.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", onResize);

  if (typeof ResizeObserver === "function") {
    resizeObserver = new ResizeObserver(update);
    resizeObserver.observe(wrap);
    const table = wrap.querySelector("table");
    if (table) {
      resizeObserver.observe(table);
    }
  }

  requestAnimationFrame(update);

  leaderboardMetricsScrollCleanup = () => {
    wrap.removeEventListener("scroll", onScroll);
    window.removeEventListener("resize", onResize);
    if (resizeObserver) {
      resizeObserver.disconnect();
    }
  };
}

function renderLeaderboardMetricsTable(container, runs) {
  const sortSpec = resolveLeaderboardMetricsTableSortSpec();
  const showSelectedTagBadges = hasMultipleSelectedTags();
  const selectedTagColorMap = buildSelectedTagColorMap();
  const source = runs.filter((run) =>
    LEADERBOARD_TABLE_METRICS.some((key) => getMetricValueForRun(run, key) !== null)
  );

  if (!source.length) {
    container.innerHTML = '<p class="muted">No run-level metric values in current filter.</p>';
    return;
  }

  const bestByMetric = {};
  LEADERBOARD_TABLE_METRICS.forEach((key) => {
    bestByMetric[key] = getPreferredMetricValue(
      source.map((run) => getMetricValueForRun(run, key)),
      key
    );
  });

  const sorted = [...source].sort((a, b) => compareLeaderboardMetricsTableRows(a, b, sortSpec));

  const summary = document.createElement("p");
  summary.className = "leaderboard-metrics-summary muted";
  summary.textContent =
    "Highlighted cells mark the preferred value per metric in the current selection. Click a column header to sort.";
  container.appendChild(summary);

  const scrollHint = document.createElement("p");
  scrollHint.className = "leaderboard-scroll-note";
  scrollHint.textContent = "Swipe sideways to reveal more metrics";
  container.appendChild(scrollHint);

  const wrap = document.createElement("div");
  wrap.className = "leaderboard-metrics-wrap";
  const table = document.createElement("table");
  table.className = "leaderboard-metrics-table";

  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  const tableColumns = [{ key: "run", label: "Run" }, { key: "timestamp", label: "Date" }].concat(
    LEADERBOARD_TABLE_METRICS.map((key) => ({
      key,
      label: METRIC_LABELS[key] || key,
    }))
  );
  tableColumns.forEach((column) => {
    const isActiveSort = sortSpec.key === column.key;
    const sortLabel = isActiveSort ? (sortSpec.direction === "asc" ? " ^" : " v") : "";
    const th = document.createElement("th");
    th.setAttribute("aria-sort", isActiveSort ? (sortSpec.direction === "asc" ? "ascending" : "descending") : "none");
    const button = document.createElement("button");
    button.type = "button";
    button.className = `sort-head${isActiveSort ? " active" : ""}`;
    button.textContent = `${column.label}${sortLabel}`;
    button.setAttribute("aria-label", `Sort by ${column.label}`);
    button.addEventListener("click", () => setLeaderboardMetricsTableSort(column.key));
    th.appendChild(button);
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
    const runLabel = getLeaderboardMetricsTableRunLabel(run);
    const runTags = showSelectedTagBadges ? getSelectedTagsForRun(run) : [];
    runCell.title = [runLabel, ...runTags, run.fileName].filter(Boolean).join(" / ");
    const runContent = document.createElement("span");
    runContent.className = "leaderboard-run-cell-content";
    const runLabelEl = document.createElement("span");
    runLabelEl.className = "leaderboard-run-label";
    runLabelEl.textContent = runLabel;
    runContent.appendChild(runLabelEl);
    if (runTags.length) {
      const badgesWrap = document.createElement("span");
      badgesWrap.className = "leaderboard-run-tag-badges";
      runTags.forEach((tag) => {
        badgesWrap.appendChild(createHtmlTagBadge(tag, selectedTagColorMap.get(tag), "tag-badge tag-badge-compact"));
      });
      runContent.appendChild(badgesWrap);
    }
    runCell.appendChild(runContent);
    tr.appendChild(runCell);

    const timestampCell = document.createElement("td");
    timestampCell.className = "mono";
    timestampCell.textContent = formatDateOnly(run.timestamp);
    tr.appendChild(timestampCell);

    LEADERBOARD_TABLE_METRICS.forEach((key) => {
      const value = getMetricValueForRun(run, key);
      const td = document.createElement("td");
      td.className = "mono";
      td.textContent = value == null ? "N/A" : formatMetric(key, value);
      if (value != null && bestByMetric[key] != null && Math.abs(value - bestByMetric[key]) < 1e-9) {
        td.classList.add("metric-best");
        td.title = `Preferred ${METRIC_LABELS[key]} in current selection`;
      }
      tr.appendChild(td);
    });

    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  wrap.appendChild(table);
  container.appendChild(wrap);
  setupLeaderboardMetricsScrollAffordance(wrap, scrollHint);
}

function renderLeaderboard(runs) {
  cleanupLeaderboardMetricsScrollAffordance();
  renderLeaderboardTabControls();
  els.leaderboardChart.innerHTML = "";

  const panel = document.createElement("div");
  panel.className = `leaderboard-panel${
    state.leaderboardTab === "radar"
      ? " leaderboard-panel-radar"
      : state.leaderboardTab === "time_series"
        ? " leaderboard-panel-time-series"
        : ""
  }`;
  els.leaderboardChart.appendChild(panel);

  if (state.leaderboardTab === "time_series") {
    renderLeaderboardTimeSeries(panel, runs);
    return;
  }
  if (state.leaderboardTab === "table") {
    renderLeaderboardMetricsTable(panel, runs);
    return;
  }
  if (state.leaderboardTab === "best_by_task") {
    renderBestByTask(panel, runs);
    return;
  }
  if (state.leaderboardTab === "radar") {
    renderRadarPanel(panel, runs);
    return;
  }
  renderLeaderboardChart(panel, runs);
}

function renderBestByTask(container, runs) {
  const metricKey = state.sortBy;
  const metricLabel = METRIC_LABELS[metricKey] || metricKey;
  const bestByTask = {};
  const metricIsLowerBetter = isLowerBetterMetric(metricKey);

  runs.forEach((run) => {
    const metricValue = getMetricValueForRun(run, metricKey);
    if (metricValue === null || Number.isNaN(metricValue)) return;
    const current = bestByTask[run.task];
    if (!current || compareMetricNumbers(metricValue, current.metricValue, metricKey) < 0) {
      bestByTask[run.task] = { run, metricValue };
    }
  });

  const items = Object.values(bestByTask)
    .sort((a, b) => compareMetricNumbers(a.metricValue, b.metricValue, metricKey));

  if (!items.length) {
    container.innerHTML = `<p class="muted">No task-level ${metricLabel.toLowerCase()} data found.</p>`;
    return;
  }

  if (metricIsLowerBetter) {
    const note = document.createElement("p");
    note.className = "leaderboard-ci-note muted";
    note.textContent = `${metricLabel} is lower-is-better.`;
    container.appendChild(note);
  }

  const max = resolveLeaderboardBarMax(
    metricKey,
    items.map((item) => item.metricValue)
  );
  const visibleCount = getVisibleItemCount("bestByTaskVisibleCount", BEST_BY_TASK_PAGE_SIZE, items.length);
  items.slice(0, visibleCount).forEach(({ run, metricValue }) => {
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

  if (items.length > visibleCount) {
    container.appendChild(
      createMetricRevealControls({
        shownCount: visibleCount,
        totalCount: items.length,
        pageSize: BEST_BY_TASK_PAGE_SIZE,
        itemLabel: "task-leading runs",
        onShowMore: () => {
          state.bestByTaskVisibleCount = Math.min(items.length, visibleCount + BEST_BY_TASK_PAGE_SIZE);
          renderLeaderboard(state.filtered);
        },
        onShowAll: () => {
          state.bestByTaskVisibleCount = items.length;
          renderLeaderboard(state.filtered);
        },
      })
    );
  }
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
const TIME_SERIES_TASK_COLORS = [
  "#6ea8ff",
  "#50e3c2",
  "#ffb36b",
  "#f472b6",
  "#facc15",
  "#22d3ee",
  "#fb7185",
  "#4ade80",
  "#f97316",
  "#38bdf8",
  "#e879f9",
  "#34d399",
];
const TIME_SERIES_MODEL_SHAPES = ["circle", "square", "diamond", "triangle", "triangle_down", "hexagon", "pentagon", "octagon"];
const SELECTED_TAG_BADGE_COLORS = ["#6ea8ff", "#50e3c2", "#ffb36b", "#f472b6", "#facc15", "#22d3ee", "#fb7185", "#4ade80"];

function toNonNegativeNumber(value) {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : 0;
}

function averageOrZero(total, denominator) {
  return denominator > 0 ? total / denominator : 0;
}

function getVisibleItemCount(stateKey, defaultCount, totalCount) {
  if (!Number.isFinite(totalCount) || totalCount <= 0) {
    return 0;
  }
  const rawValue = Number(state[stateKey]);
  const fallback = Number.isFinite(defaultCount) && defaultCount > 0 ? Math.floor(defaultCount) : totalCount;
  const normalized = Number.isFinite(rawValue) && rawValue > 0 ? Math.floor(rawValue) : fallback;
  return Math.max(1, Math.min(totalCount, normalized));
}

function createMetricRevealControls({
  shownCount,
  totalCount,
  pageSize,
  itemLabel = "items",
  summaryText = "",
  onShowMore,
  onShowAll,
}) {
  if (!Number.isFinite(totalCount) || totalCount <= shownCount) {
    return null;
  }

  const controls = document.createElement("div");
  controls.className = "metric-reveal-controls";

  const note = document.createElement("p");
  note.className = "muted";
  note.textContent =
    summaryText || `Showing ${formatNum(shownCount, 0)} of ${formatNum(totalCount, 0)} ${itemLabel}.`;
  controls.appendChild(note);

  const actions = document.createElement("div");
  actions.className = "metric-reveal-actions";

  const remaining = totalCount - shownCount;
  if (typeof onShowMore === "function" && remaining > 0) {
    const loadMoreBtn = document.createElement("button");
    loadMoreBtn.type = "button";
    loadMoreBtn.className = "btn metric-reveal-btn";
    loadMoreBtn.textContent = `Load ${formatNum(Math.min(pageSize, remaining), 0)} more`;
    loadMoreBtn.addEventListener("click", onShowMore);
    actions.appendChild(loadMoreBtn);
  }

  if (typeof onShowAll === "function") {
    const showAllBtn = document.createElement("button");
    showAllBtn.type = "button";
    showAllBtn.className = "btn metric-reveal-btn";
    showAllBtn.textContent = "Show all";
    showAllBtn.addEventListener("click", onShowAll);
    actions.appendChild(showAllBtn);
  }

  if (actions.childElementCount) {
    controls.appendChild(actions);
  }

  return controls;
}

function computeRunSignalRows(runs) {
  return runs
    .map((run) => {
      const prompts = toNonNegativeNumber(getPromptCountForRun(run));
      const inputTotal = toNonNegativeNumber(run.inputTokensTotal);
      const cachedTotal = toNonNegativeNumber(run.cachedInputTokensTotal ?? run.cachedTokens);
      const outputTotal = toNonNegativeNumber(run.outputTokensTotal);
      const thinkingTotal = toNonNegativeNumber(run.thinkingTokensTotal);
      const avgInput = averageOrZero(inputTotal, prompts);
      const avgCached = averageOrZero(cachedTotal, prompts);
      const avgOutput = averageOrZero(outputTotal, prompts);
      const avgThinking = averageOrZero(thinkingTotal, prompts);
      return {
        run,
        model: run.model || "unknown",
        modelDisplay: getRunModelDisplayName(run),
        task: run.task || "unknown",
        prompts,
        inputTotal,
        cachedTotal,
        outputTotal,
        thinkingTotal,
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

function getRadarSeriesColor(index, totalCount) {
  if (totalCount <= RADAR_COLORS.length) {
    return RADAR_COLORS[index % RADAR_COLORS.length];
  }
  const hue = Math.round((210 + (index * 360) / totalCount) % 360);
  return `hsl(${hue}, 76%, 60%)`;
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
        return null;
      }
      return numeric.reduce((sum, value) => sum + value, 0) / numeric.length;
    });
    const validValues = values.filter((value) => typeof value === "number" && Number.isFinite(value));
    const mean = validValues.length
      ? validValues.reduce((sum, value) => sum + value, 0) / validValues.length
      : null;
    return { model, values, mean };
  });
  const populatedSeries = series.filter((row) =>
    row.values.some((value) => typeof value === "number" && Number.isFinite(value))
  );

  const usesAutoRanking = state.selectedModels.length === 0;
  const orderedSeries = usesAutoRanking
    ? [...populatedSeries].sort((a, b) => {
        const diff = compareMetricNumbers(a.mean, b.mean, metricKey);
        if (diff !== 0) return diff;
        return a.model.localeCompare(b.model);
      })
    : populatedSeries;
  const visibleCount = usesAutoRanking
    ? getVisibleItemCount("radarVisibleSeriesCount", RADAR_MODEL_PAGE_SIZE, orderedSeries.length)
    : orderedSeries.length;
  const visibleSeries = orderedSeries.slice(0, visibleCount);

  return {
    axes: axesCandidates,
    series: visibleSeries,
    hiddenModelCount: Math.max(0, orderedSeries.length - visibleSeries.length),
    totalSeriesCount: orderedSeries.length,
  };
}

function normalizeRadarValue(rawValue, axisMax, metricKey, scaleMode) {
  if (typeof rawValue !== "number" || !Number.isFinite(rawValue)) {
    return isLowerBetterMetric(metricKey) ? 1 : 0;
  }

  const maxValue = Math.max(1, toNonNegativeNumber(axisMax));
  const clampedValue = Math.min(maxValue, toNonNegativeNumber(rawValue));
  const linearRatio = Math.max(0, Math.min(1, clampedValue / maxValue));
  if (scaleMode === "contrast") {
    const gamma = 2.5;
    return Math.pow(linearRatio, gamma);
  }
  return linearRatio;
}

function buildRadarSeriesTitle(row, tasks, metricKey) {
  const lines = [`Model: ${row.model}`, `Avg: ${formatMetric(metricKey, row.mean)}`];
  tasks.forEach((task, index) => {
    lines.push(`${task}: ${formatMetric(metricKey, row.values[index])}`);
  });
  return lines.join("\n");
}

function setActiveRadarSeries(svg, activePolygon) {
  const polygons = svg.querySelectorAll(".radar-series");
  const hasActive = Boolean(activePolygon);
  svg.classList.toggle("radar-hover-active", hasActive);
  polygons.forEach((polygon) => {
    polygon.classList.toggle("radar-series-active", polygon === activePolygon);
  });
}

function buildRadarSvg(tasks, seriesRows, metricKey, scaleMode, colorCount = seriesRows.length) {
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
  svg.setAttribute(
    "aria-label",
    `Radar chart comparing model metric profiles across selected dimensions using ${scaleMode === "contrast" ? "contrast" : "linear"} scaling`
  );

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
    const color = getRadarSeriesColor(idx, colorCount);
    const points = tasks
      .map((_, axisIndex) => {
        const rawValue = row.values[axisIndex];
        const axisMax = axisMaxima[axisIndex] || 1;
        const normalized = normalizeRadarValue(rawValue, axisMax, metricKey, scaleMode);
        return pointAt(angleFor(axisIndex), radius * normalized);
      })
      .map((xy) => xy.join(","))
      .join(" ");
    const polygon = document.createElementNS(ns, "polygon");
    polygon.setAttribute("class", "radar-series");
    polygon.setAttribute("points", points);
    polygon.setAttribute("fill", color);
    polygon.setAttribute("stroke", color);
    polygon.setAttribute("aria-label", buildRadarSeriesTitle(row, tasks, metricKey));
    const title = document.createElementNS(ns, "title");
    title.textContent = buildRadarSeriesTitle(row, tasks, metricKey);
    polygon.appendChild(title);
    polygon.addEventListener("mouseenter", () => setActiveRadarSeries(svg, polygon));
    polygon.addEventListener("mouseleave", () => setActiveRadarSeries(svg, null));
    svg.appendChild(polygon);
  });

  return svg;
}

function renderRadarPanel(panel, runs) {
  const metricKey = state.sortBy;
  const metricLabel = METRIC_LABELS[metricKey] || metricKey;
  const axisMode = state.radarAxis;
  const scaleMode = state.radarScale;
  const axisLabel = RADAR_AXIS_LABELS[axisMode] || "Axis";
  const header = document.createElement("h4");
  header.className = "token-radar-header";
  header.textContent = `Profile by ${axisLabel.toLowerCase()} (${metricLabel})`;
  panel.appendChild(header);

  if (isLowerBetterMetric(metricKey)) {
    const note = document.createElement("p");
    note.className = "muted";
    note.textContent = `${metricLabel} is lower-is-better. Smaller shapes indicate better calibration.`;
    panel.appendChild(note);
  }

  const controls = document.createElement("div");
  controls.className = "radar-controls";

  const axisControls = document.createElement("div");
  axisControls.className = "radar-control-group";
  const axisLabelEl = document.createElement("span");
  axisLabelEl.className = "radar-control-label";
  axisLabelEl.textContent = "Axes";
  axisControls.appendChild(axisLabelEl);

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
  axisControls.appendChild(toggleWrap);
  controls.appendChild(axisControls);

  const scaleControls = document.createElement("div");
  scaleControls.className = "radar-control-group";
  const scaleLabelEl = document.createElement("span");
  scaleLabelEl.className = "radar-control-label";
  scaleLabelEl.textContent = "Scale";
  scaleControls.appendChild(scaleLabelEl);

  const scaleToggle = document.createElement("div");
  scaleToggle.className = "radar-mode-toggle";
  ["linear", "contrast"].forEach((mode) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `radar-mode-btn${scaleMode === mode ? " active" : ""}`;
    button.textContent = RADAR_SCALE_LABELS[mode];
    button.setAttribute("aria-pressed", scaleMode === mode ? "true" : "false");
    button.addEventListener("click", () => setRadarScale(mode));
    scaleToggle.appendChild(button);
  });
  scaleControls.appendChild(scaleToggle);
  controls.appendChild(scaleControls);

  panel.appendChild(controls);

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
  wrap.appendChild(buildRadarSvg(dataset.axes, dataset.series, metricKey, scaleMode, dataset.totalSeriesCount));

  const legend = document.createElement("div");
  legend.className = "radar-legend";
  dataset.series.forEach((row, idx) => {
    const entry = document.createElement("div");
    entry.className = "radar-legend-row";
    const line = document.createElement("span");
    line.className = "radar-legend-line";
    line.style.background = getRadarSeriesColor(idx, dataset.totalSeriesCount);
    const text = document.createElement("span");
    text.className = "mono";
    text.textContent = `${row.model} | avg ${row.mean == null ? "N/A" : formatMetric(metricKey, row.mean)}`;
    entry.appendChild(line);
    entry.appendChild(text);
    legend.appendChild(entry);
  });
  wrap.appendChild(legend);

  if (dataset.hiddenModelCount > 0) {
    wrap.appendChild(
      createMetricRevealControls({
        shownCount: dataset.series.length,
        totalCount: dataset.totalSeriesCount,
        pageSize: RADAR_MODEL_PAGE_SIZE,
        itemLabel: "models",
        summaryText: `Showing ${formatNum(dataset.series.length, 0)} of ${formatNum(dataset.totalSeriesCount, 0)} models ranked by average ${metricLabel.toLowerCase()}.`,
        onShowMore: () => {
          state.radarVisibleSeriesCount = Math.min(
            dataset.totalSeriesCount,
            dataset.series.length + RADAR_MODEL_PAGE_SIZE
          );
          renderLeaderboard(state.filtered);
        },
        onShowAll: () => {
          state.radarVisibleSeriesCount = dataset.totalSeriesCount;
          renderLeaderboard(state.filtered);
        },
      })
    );
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

  const stackPanel = document.createElement("section");
  stackPanel.className = "token-stack-panel";
  const stackHeader = document.createElement("h4");
  stackHeader.className = "token-stack-header";
  stackHeader.textContent = "Average tokens per prompt by run";
  stackPanel.appendChild(stackHeader);
  appendTokenLegend(stackPanel);

  const rowsWrap = document.createElement("div");
  rowsWrap.className = "token-rows";
  const visibleCount = getVisibleItemCount("tokenSignalsVisibleCount", TOKEN_SIGNAL_PAGE_SIZE, runRows.length);
  runRows.slice(0, visibleCount).forEach((row) => {
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
    meta.textContent = `${row.task} | avg/prompt ${formatNum(row.totalAvg, 2)} | prompts ${formatNum(row.prompts, 0)}`;
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
      const totalKey = segment.key.replace(/^avg/, "").toLowerCase();
      const totalValue = row[`${totalKey}Total`];
      seg.title = `${segment.label}: ${formatNum(totalValue, 0)} total | ${formatNum(value, 2)} avg/prompt`;
      track.appendChild(seg);
    });
    rowEl.appendChild(track);

    const values = document.createElement("div");
    values.className = "token-row-values mono";
    values.textContent =
      `totals in ${formatNum(row.inputTotal, 0)} | cached ${formatNum(row.cachedTotal, 0)} | ` +
      `out ${formatNum(row.outputTotal, 0)} | think ${formatNum(row.thinkingTotal, 0)}`;
    rowEl.appendChild(values);

    rowEl.addEventListener("click", () => openRunModal(row.run));
    rowsWrap.appendChild(rowEl);
  });
  stackPanel.appendChild(rowsWrap);

  if (runRows.length > visibleCount) {
    stackPanel.appendChild(
      createMetricRevealControls({
        shownCount: visibleCount,
        totalCount: runRows.length,
        pageSize: TOKEN_SIGNAL_PAGE_SIZE,
        itemLabel: "runs",
        onShowMore: () => {
          state.tokenSignalsVisibleCount = Math.min(runRows.length, visibleCount + TOKEN_SIGNAL_PAGE_SIZE);
          renderTokenSignals(state.filtered);
        },
        onShowAll: () => {
          state.tokenSignalsVisibleCount = runRows.length;
          renderTokenSignals(state.filtered);
        },
      })
    );
  }
  els.tokenChart.appendChild(stackPanel);
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
      renderTableCell(
        "Calibration ECE",
        run.calibrationEce !== null ? `${formatNum(run.calibrationEce, 2)}%` : '<span class="muted">N/A</span>'
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
    ["Calibration ECE", run.calibrationEce == null ? "N/A" : `${formatNum(run.calibrationEce, 2)}%`],
    ["Calibration MCE", run.calibrationMce == null ? "N/A" : `${formatNum(run.calibrationMce, 2)}%`],
    ["Brier Score", run.calibrationBrierScore == null ? "N/A" : formatNum(run.calibrationBrierScore, 3)],
    ["Calibration Samples", formatNum(run.calibrationSampleCount, 0)],
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
  els.hideNoAccuracy.checked = state.hideNoAccuracy;
  renderTaskControls();
  renderModelControls();
  renderTagControls();
  renderTimeRangeControls();
  updateMobileFilterSummary();
  updateResetFiltersButton();
  renderKpis(state.filtered);
  renderLeaderboard(state.filtered);
  renderTokenSignals(state.filtered);
  renderTable(state.filtered);
  requestAnimationFrame(() => {
    updateSidebarScrollAffordances();
  });
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
  state.expandedLeaderboardGroups = new Set();
  state.tokenSignalsVisibleCount = TOKEN_SIGNAL_PAGE_SIZE;
  state.bestByTaskVisibleCount = BEST_BY_TASK_PAGE_SIZE;
  state.radarVisibleSeriesCount = RADAR_MODEL_PAGE_SIZE;

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
  try {
    const result = await runWithLoadingNotice(
      state.runs.length ? "Refreshing metrics from server source..." : "Loading metrics from server source...",
      (onProgress) => loadFromServer(onProgress)
    );
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
  try {
    const result = await runWithLoadingNotice(
      state.runs.length ? "Refreshing metrics from selected folder..." : "Loading metrics from selected folder...",
      (onProgress) => loadFromDirectoryHandle(dirHandle, onProgress)
    );
    applyLoadedResult(result);
    const warningInfo = result.warnings.length ? ` (${result.warnings.length} warning(s))` : "";
    els.heroSubtitle.textContent = `Loaded ${result.runs.length.toLocaleString()} runs from local folder${warningInfo}.`;
  } catch (error) {
    renderError(`Folder source failed: ${error.message}`, true);
    updateSourceStatus();
  }
}

async function activateFilesSource(files) {
  try {
    const result = await runWithLoadingNotice(
      state.runs.length ? "Refreshing metrics from selected files..." : "Loading metrics from selected files...",
      (onProgress) => loadFromFiles(files, onProgress)
    );
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
  applySidebarLayoutState();
  setupFilters();
  setupResponsiveShell();
  setupSourceControls();
  setupModalControls();
  renderTimeRangeControls();

  if (!supportsDirectoryPicker()) {
    els.btnOpenFolder.disabled = true;
    els.btnOpenFolder.title = "Directory picker is not supported in this browser.";
  }

  if (isFileProtocol()) {
    state.sourceMode = "none";
    state.sourceFileCount = 0;
    state.warnings = [];
    setLoadingState(false);
    updateSourceStatus();
    els.heroSubtitle.textContent = "Choose a local data source to load metrics.";
    return;
  }

  await activateServerSource();
}

init();

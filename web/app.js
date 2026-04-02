const STORAGE_KEY = "dhAIBench.metricsDashboard.state.v1";
const METRICS_MANIFEST_PATH = "./metrics-manifest.json";
const AGREEMENT_SUMMARY_FILENAME = "agreement_summary.json";
const AGREEMENT_CLUSTERS_FILENAME = "agreement_clusters.json";
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
const pricingApi =
  window.DHAIBenchPricing && typeof window.DHAIBenchPricing.estimateRunCost === "function"
    ? window.DHAIBenchPricing
    : null;
const metricsAggregationApi =
  window.DHAIBenchMetricsAggregation && typeof window.DHAIBenchMetricsAggregation.buildBalancedAggregate === "function"
    ? window.DHAIBenchMetricsAggregation
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
  filterSearchQuery: "",
  timeRanges: [{ from: "", to: "" }],
  selectedRunPath: null,
  sortBy: "accuracy",
  leaderboardTableSortKey: null,
  leaderboardTableSortDirection: null,
  leaderboardTab: "chart",
  agreementViewMode: "same_model",
  leaderboardChartGroupBy: "model",
  leaderboardScatterGroupBy: "none",
  leaderboardChartBestByTask: false,
  leaderboardScatterXAxis: "price",
  agreementRepresentativePolicy: "latest",
  scatterShowCi: true,
  timeSeriesShowLabels: false,
  timeSeriesViewport: null,
  priceScatterViewport: null,
  priceScatterCostMode: "total",
  radarAxis: "tag",
  radarScale: "contrast",
  tokenSignalsVisibleCount: TOKEN_SIGNAL_PAGE_SIZE,
  bestByTaskVisibleCount: BEST_BY_TASK_PAGE_SIZE,
  radarVisibleSeriesCount: RADAR_MODEL_PAGE_SIZE,
  hideNoAccuracy: false,
  theme: "dark",
  sourceMode: "none",
  sourceFileCount: 0,
  warnings: [],
  agreementSummary: null,
  agreementClusters: null,
  repeatAgreementByRunStem: new Map(),
  crossAgreementByPolicyAndRunStem: {
    latest: new Map(),
    best_accuracy: new Map(),
  },
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
  filterSearchInput: document.querySelector("#filterSearchInput"),
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
  leaderboardMetricField: document.querySelector(".leaderboard-metric-field"),
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
  leaderboardChartToggle: document.querySelector("#leaderboardChartToggle"),
  leaderboardChart: document.querySelector("#leaderboardChart"),
  tokenChart: document.querySelector("#tokenChart"),
  runsTableBody: document.querySelector("#runsTableBody"),
  tableMeta: document.querySelector("#tableMeta"),
  runModal: document.querySelector("#runModal"),
  runModalTitle: document.querySelector("#runModalTitle"),
  runModalMeta: document.querySelector("#runModalMeta"),
  runModalContent: document.querySelector("#runModalContent"),
  runModalClose: document.querySelector("#runModalClose"),
  clusterModal: document.querySelector("#clusterModal"),
  clusterModalTitle: document.querySelector("#clusterModalTitle"),
  clusterModalMeta: document.querySelector("#clusterModalMeta"),
  clusterModalContent: document.querySelector("#clusterModalContent"),
  clusterModalClose: document.querySelector("#clusterModalClose"),
  barRowTemplate: document.querySelector("#barRowTemplate"),
};

const METRIC_KEYS = new Set(["accuracy", "cohen_kappa", "macro_f1", "macro_precision", "macro_recall", "calibration_ece"]);
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
const LEADERBOARD_TAB_KEYS = new Set(["chart", "scatter", "table", "radar", "agreement"]);
const AGREEMENT_VIEW_MODE_KEYS = new Set(["same_model", "cross_model"]);
const LEADERBOARD_CHART_GROUP_BY_KEYS = new Set(["none", "model", "task"]);
const LEADERBOARD_SCATTER_X_AXIS_KEYS = new Set(["price", "time"]);
const PRICE_SCATTER_COST_MODE_KEYS = new Set(["total", "per_prompt"]);
const AGREEMENT_REPRESENTATIVE_POLICY_KEYS = new Set(["latest", "best_accuracy"]);
const LEADERBOARD_TABLE_METRICS = [
  "accuracy",
  "cohen_kappa",
  "repeat_alpha",
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
  cohen_kappa: "Cohen's Kappa",
  repeat_alpha: "Repeat α",
  macro_f1: "Macro F1",
  macro_precision: "Macro Precision",
  macro_recall: "Macro Recall",
  calibration_ece: "Calibration ECE",
};

const AGREEMENT_REPRESENTATIVE_POLICY_LABELS = {
  latest: "Latest",
  best_accuracy: "Best Accuracy",
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
  none: "No Group",
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

function computeCohenKappaFromMarginals(rowTotals, colTotals, correct, total) {
  if (!(total > 0)) {
    return 0;
  }
  const accuracy = correct / total;
  let expectedNumerator = 0;
  const count = Math.min(Array.isArray(rowTotals) ? rowTotals.length : 0, Array.isArray(colTotals) ? colTotals.length : 0);
  for (let index = 0; index < count; index += 1) {
    expectedNumerator += rowTotals[index] * colTotals[index];
  }
  const expectedAgreement = expectedNumerator / (total * total);
  if (Math.abs(1 - expectedAgreement) <= 1e-12) {
    return Math.abs(accuracy - 1) <= 1e-12 ? 1 : 0;
  }
  return (accuracy - expectedAgreement) / (1 - expectedAgreement);
}

function deriveCohenKappaFromPayload(payload) {
  if (!payload || typeof payload !== "object" || !Array.isArray(payload.labels) || !payload.labels.length) {
    return null;
  }
  const labelCount = payload.labels.length;
  const rowTotals = new Array(labelCount).fill(0);
  const colTotals = new Array(labelCount).fill(0);
  let correct = 0;
  let total = 0;

  if (Array.isArray(payload.confusion_matrix_sparse) && payload.confusion_matrix_sparse.length) {
    for (const triplet of payload.confusion_matrix_sparse) {
      if (!Array.isArray(triplet) || triplet.length !== 3) {
        return null;
      }
      const [trueIndex, predIndex, count] = triplet;
      if (
        !Number.isInteger(trueIndex) ||
        !Number.isInteger(predIndex) ||
        !Number.isInteger(count) ||
        trueIndex < 0 ||
        predIndex < 0 ||
        trueIndex >= labelCount ||
        predIndex >= labelCount ||
        count < 0
      ) {
        return null;
      }
      rowTotals[trueIndex] += count;
      colTotals[predIndex] += count;
      total += count;
      if (trueIndex === predIndex) {
        correct += count;
      }
    }
    return computeCohenKappaFromMarginals(rowTotals, colTotals, correct, total);
  }

  const confusion = payload.confusion_matrix;
  if (!confusion || typeof confusion !== "object") {
    return null;
  }
  const labelToIndex = new Map(payload.labels.map((label, index) => [label, index]));
  for (const [trueLabel, row] of Object.entries(confusion)) {
    const trueIndex = labelToIndex.get(trueLabel);
    if (trueIndex == null || !row || typeof row !== "object") {
      continue;
    }
    for (const [predLabel, rawCount] of Object.entries(row)) {
      const predIndex = labelToIndex.get(predLabel);
      if (predIndex == null || !Number.isInteger(rawCount) || rawCount < 0) {
        continue;
      }
      rowTotals[trueIndex] += rawCount;
      colTotals[predIndex] += rawCount;
      total += rawCount;
      if (trueIndex === predIndex) {
        correct += rawCount;
      }
    }
  }
  if (!(total > 0)) {
    return null;
  }
  return computeCohenKappaFromMarginals(rowTotals, colTotals, correct, total);
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
  if (state.selectedTasks.length === 1) {
    return getRunModelDisplayName(run);
  }
  return `${run.task} / ${getRunModelDisplayName(run)}`;
}

function getSelectionAwareScatterModelLabel(run) {
  if (!run) {
    return "";
  }
  if (state.selectedModels.length !== 1) {
    return getRunModelDisplayName(run);
  }
  return getRunEffortSuffix(run);
}

function getSelectionAwareScatterRunLabel(run) {
  if (!run) {
    return "";
  }
  const parts = [];
  if (state.selectedTasks.length !== 1) {
    const taskLabel = asTrimmedString(run.task);
    if (taskLabel) {
      parts.push(taskLabel);
    }
  }
  const modelLabel = getSelectionAwareScatterModelLabel(run);
  if (modelLabel) {
    parts.push(modelLabel);
  }
  return parts.join(" | ");
}

function getSelectionAwarePriceScatterBaseLabel(entry, groupBy) {
  if (!entry) {
    return "";
  }
  if (groupBy === "none") {
    return getSelectionAwareScatterRunLabel(entry.representativeRun);
  }
  if (groupBy === "task") {
    return state.selectedTasks.length === 1 ? "" : asTrimmedString(entry.label);
  }
  if (groupBy === "model") {
    if (state.selectedModels.length === 1) {
      return getSharedRunEffortSuffix(entry.runs);
    }
    return asTrimmedString(entry.label);
  }
  return asTrimmedString(entry.label);
}

function buildPriceScatterPointLabelText(entry, groupBy, priceLabelText) {
  const baseLabel = getSelectionAwarePriceScatterBaseLabel(entry, groupBy);
  if (baseLabel) {
    return `${baseLabel}${entry.count > 1 ? ` (${entry.count})` : ""} | ${priceLabelText}`;
  }
  if (entry.count > 1) {
    return `${formatNum(entry.count, 0)} run(s) | ${priceLabelText}`;
  }
  return priceLabelText;
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
  const runConfig = payload.run_config && typeof payload.run_config === "object" ? payload.run_config : {};
  const usage = payload.usage_metadata_summary || {};
  const tokenTotals = payload.token_usage_totals || {};
  const controls = payload.request_control_summary || {};
  const calibrationMetrics =
    payload.calibration_metrics && typeof payload.calibration_metrics === "object"
      ? payload.calibration_metrics
      : {};
  const ts = payload.first_prompt_timestamp || nameParts.timestamp;
  const taskNameFromMetrics = asTrimmedString(runConfig.task_name) || asTrimmedString(payload.task_name);
  const providerFromMetrics =
    asTrimmedString(modelDetails.provider) || asTrimmedString(payload.provider);
  const modelFromMetrics =
    asTrimmedString(modelDetails.model_requested) ||
    asTrimmedString(modelDetails.model_for_requests) ||
    asTrimmedString(payload.model_requested) ||
    asTrimmedString(payload.model);
  const taskDescription =
    asTrimmedString(runConfig.task_description) || asTrimmedString(payload.task_description);
  const tags = parseSemicolonTags(runConfig.tags).length
    ? parseSemicolonTags(runConfig.tags)
    : parseSemicolonTags(payload.tags);

  const accuracy = toPct(safeNum(payload.accuracy));
  const cohenKappa = safeNum(payload.cohen_kappa) ?? deriveCohenKappaFromPayload(payload);
  const macroPrecision = toPct(safeNum(payload.macro_precision));
  const macroRecall = toPct(safeNum(payload.macro_recall));
  const macroF1 = toPct(safeNum(payload.macro_f1));
  const calibrationEce = toPct(safeNum(calibrationMetrics.ece));
  const calibrationMce = toPct(safeNum(calibrationMetrics.mce));
  const calibrationBrierScore = safeNum(calibrationMetrics.brier_score);
  const normalizedRun = {
    filePath: normalizedFilePath,
    fileName,
    runStem,
    fileModelSlug: nameParts.model || "",
    task: taskNameFromMetrics || nameParts.task,
    taskDescription,
    tags,
    tagsDisplay: tags.join("; "),
    provider: providerFromMetrics || nameParts.provider || "",
    model: modelFromMetrics || nameParts.model,
    timestamp: ts,
    accuracy,
    cohenKappa,
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
    serviceTier: asTrimmedString(runConfig.service_tier) || "standard",
    truthLabelCount: safeNum(payload.truth_label_count),
    labelMetricsAvailable:
      payload.label_metrics_available !== false &&
      (
        safeNum(payload.accuracy) !== null ||
        safeNum(payload.cohen_kappa) !== null ||
        safeNum(payload.macro_precision) !== null ||
        safeNum(payload.macro_recall) !== null ||
        safeNum(payload.macro_f1) !== null
      ),
    labelMetricsReason: payload.label_metrics_reason || null,
    modelDetails,
    runConfig,
    usageSummary: usage,
    tokenUsageTotals: tokenTotals,
    controlSummary: controls,
    rawMetrics: payload,
  };

  const pricing = pricingApi ? pricingApi.estimateRunCost(window.MODEL_PRICING_CATALOG, normalizedRun) : null;
  normalizedRun.pricing = pricing;
  normalizedRun.estimatedCostUsd = pricing && safeNum(pricing.estimatedCostUsd) !== null ? pricing.estimatedCostUsd : null;
  normalizedRun.pricingStatus = pricing ? pricing.statusLabel : "catalog unavailable";

  return normalizedRun;
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

function normalizeAgreementRepresentative(rawRepresentative) {
  if (!rawRepresentative || typeof rawRepresentative !== "object") {
    return null;
  }
  return {
    provider: asTrimmedString(rawRepresentative.provider),
    model: asTrimmedString(rawRepresentative.model),
    runStem: asTrimmedString(rawRepresentative.run_stem),
    metricsFile: asTrimmedString(rawRepresentative.metrics_file),
    timestamp: asTrimmedString(rawRepresentative.timestamp),
    accuracy: safeNum(rawRepresentative.accuracy),
    cohenKappa: safeNum(rawRepresentative.cohen_kappa),
  };
}

function normalizeAgreementGroupEntry(rawEntry, policy = "") {
  if (!rawEntry || typeof rawEntry !== "object") {
    return null;
  }

  const runStems = uniqueNonEmptyStrings(rawEntry.run_stems || rawEntry.representative_run_stems || []);
  const representatives = Array.isArray(rawEntry.representatives)
    ? rawEntry.representatives.map(normalizeAgreementRepresentative).filter(Boolean)
    : [];

  return {
    groupId: asTrimmedString(rawEntry.group_id),
    policy: asTrimmedString(rawEntry.representative_policy) || policy,
    taskNameDisplay: asTrimmedString(rawEntry.task_name_display),
    taskNamesSeen: uniqueNonEmptyStrings(rawEntry.task_names_seen || []),
    tagsDisplay: asTrimmedString(rawEntry.tags_display),
    provider: asTrimmedString(rawEntry.provider),
    model: asTrimmedString(rawEntry.model),
    runCount: safeNum(rawEntry.run_count),
    modelCount: safeNum(rawEntry.model_count),
    alphaNominal: safeNum(rawEntry.alpha_nominal),
    pairableItemCount: safeNum(rawEntry.pairable_item_count),
    ratedItemCount: safeNum(rawEntry.rated_item_count),
    fullySharedItemCount: safeNum(rawEntry.fully_shared_item_count),
    categoryCount: safeNum(rawEntry.category_count),
    runStems,
    representatives,
  };
}

function normalizeAgreementSummary(payload) {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    return null;
  }

  const repeatGroups = Array.isArray(payload.repeat_groups)
    ? payload.repeat_groups.map((entry) => normalizeAgreementGroupEntry(entry)).filter(Boolean)
    : [];
  const crossModelRaw = payload.cross_model && typeof payload.cross_model === "object" ? payload.cross_model : {};
  const crossModel = {};
  AGREEMENT_REPRESENTATIVE_POLICY_KEYS.forEach((policy) => {
    crossModel[policy] = Array.isArray(crossModelRaw[policy])
      ? crossModelRaw[policy].map((entry) => normalizeAgreementGroupEntry(entry, policy)).filter(Boolean)
      : [];
  });

  return {
    generatedAt: asTrimmedString(payload.generated_at),
    runCount: safeNum(payload.run_count),
    repeatGroups,
    crossModel,
  };
}

function normalizeAgreementClusterPair(rawPair) {
  if (!rawPair || typeof rawPair !== "object") {
    return null;
  }
  const left = safeNum(rawPair.a);
  const right = safeNum(rawPair.b);
  if (!Number.isInteger(left) || !Number.isInteger(right) || left < 0 || right < 0 || left === right) {
    return null;
  }
  return {
    a: left,
    b: right,
    distance: safeNum(rawPair.distance),
    overlapCount: safeNum(rawPair.overlap_count),
    agreementCount: safeNum(rawPair.agreement_count),
    disagreementCount: safeNum(rawPair.disagreement_count),
  };
}

function normalizeAgreementClusterLinkageStep(rawStep) {
  if (!Array.isArray(rawStep) || rawStep.length < 4) {
    return null;
  }
  const left = safeNum(rawStep[0]);
  const right = safeNum(rawStep[1]);
  const distance = safeNum(rawStep[2]);
  const count = safeNum(rawStep[3]);
  if (
    !Number.isInteger(left) ||
    !Number.isInteger(right) ||
    !Number.isFinite(distance) ||
    !Number.isInteger(count)
  ) {
    return null;
  }
  return [left, right, distance, count];
}

function normalizeAgreementClusterEntry(rawEntry, policy = "") {
  if (!rawEntry || typeof rawEntry !== "object") {
    return null;
  }

  const representatives = Array.isArray(rawEntry.representatives)
    ? rawEntry.representatives.map(normalizeAgreementRepresentative).filter(Boolean)
    : [];
  const pairwise = Array.isArray(rawEntry.pairwise)
    ? rawEntry.pairwise.map(normalizeAgreementClusterPair).filter(Boolean)
    : [];
  const linkage = Array.isArray(rawEntry.linkage)
    ? rawEntry.linkage.map(normalizeAgreementClusterLinkageStep).filter(Boolean)
    : [];

  return {
    groupId: asTrimmedString(rawEntry.group_id),
    policy: asTrimmedString(rawEntry.representative_policy) || policy,
    taskNameDisplay: asTrimmedString(rawEntry.task_name_display),
    taskNamesSeen: uniqueNonEmptyStrings(rawEntry.task_names_seen || []),
    tagsDisplay: asTrimmedString(rawEntry.tags_display),
    modelCount: safeNum(rawEntry.model_count),
    distanceMetric: asTrimmedString(rawEntry.distance_metric),
    linkageMethod: asTrimmedString(rawEntry.linkage_method),
    comparablePairCount: safeNum(rawEntry.comparable_pair_count),
    linkageComplete: Boolean(rawEntry.linkage_complete),
    representatives,
    pairwise,
    linkage,
  };
}

function normalizeAgreementClusters(payload) {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    return null;
  }

  const crossModelRaw = payload.cross_model && typeof payload.cross_model === "object" ? payload.cross_model : {};
  const crossModel = {};
  AGREEMENT_REPRESENTATIVE_POLICY_KEYS.forEach((policy) => {
    crossModel[policy] = Array.isArray(crossModelRaw[policy])
      ? crossModelRaw[policy].map((entry) => normalizeAgreementClusterEntry(entry, policy)).filter(Boolean)
      : [];
  });

  return {
    generatedAt: asTrimmedString(payload.generated_at),
    runCount: safeNum(payload.run_count),
    crossModel,
  };
}

function parseAgreementSummaryText(filePath, rawText, warnings) {
  let payload;
  try {
    payload = JSON.parse(rawText);
  } catch (error) {
    warnings.push({
      file: filePath,
      message: `Agreement summary is invalid JSON (${error.message}).`,
    });
    return null;
  }

  const normalized = normalizeAgreementSummary(payload);
  if (!normalized) {
    warnings.push({
      file: filePath,
      message: "Agreement summary has an unexpected JSON structure.",
    });
    return null;
  }
  return normalized;
}

function parseAgreementClustersText(filePath, rawText, warnings) {
  let payload;
  try {
    payload = JSON.parse(rawText);
  } catch (error) {
    warnings.push({
      file: filePath,
      message: `Agreement clusters are invalid JSON (${error.message}).`,
    });
    return null;
  }

  const normalized = normalizeAgreementClusters(payload);
  if (!normalized) {
    warnings.push({
      file: filePath,
      message: "Agreement clusters have an unexpected JSON structure.",
    });
    return null;
  }
  return normalized;
}

function dedupeRuns(runs) {
  const byFile = new Map();
  runs.forEach((run) => {
    byFile.set(run.filePath, run);
  });
  return Array.from(byFile.values());
}

function rebuildAgreementLookups(summary) {
  const repeatAgreementByRunStem = new Map();
  const crossAgreementByPolicyAndRunStem = {
    latest: new Map(),
    best_accuracy: new Map(),
  };

  if (summary && Array.isArray(summary.repeatGroups)) {
    summary.repeatGroups.forEach((entry) => {
      (entry.runStems || []).forEach((runStem) => {
        repeatAgreementByRunStem.set(runStem, entry);
      });
    });
  }

  if (summary && summary.crossModel && typeof summary.crossModel === "object") {
    AGREEMENT_REPRESENTATIVE_POLICY_KEYS.forEach((policy) => {
      const bucket = crossAgreementByPolicyAndRunStem[policy];
      (summary.crossModel[policy] || []).forEach((entry) => {
        (entry.representatives || []).forEach((representative) => {
          if (representative && representative.runStem) {
            bucket.set(representative.runStem, entry);
          }
        });
      });
    });
  }

  state.repeatAgreementByRunStem = repeatAgreementByRunStem;
  state.crossAgreementByPolicyAndRunStem = crossAgreementByPolicyAndRunStem;
}

function getRepeatAgreementForRun(run) {
  if (!run || !run.runStem || !(state.repeatAgreementByRunStem instanceof Map)) {
    return null;
  }
  return state.repeatAgreementByRunStem.get(run.runStem) || null;
}

function getCrossModelAgreementForRun(run, policy = state.agreementRepresentativePolicy) {
  if (!run || !run.runStem || !AGREEMENT_REPRESENTATIVE_POLICY_KEYS.has(policy)) {
    return null;
  }
  const bucket = state.crossAgreementByPolicyAndRunStem
    ? state.crossAgreementByPolicyAndRunStem[policy]
    : null;
  if (!(bucket instanceof Map)) {
    return null;
  }
  return bucket.get(run.runStem) || null;
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

async function loadAgreementSummaryFromServer(manifestBaseDirs = [], warnings = []) {
  const candidatePaths = uniqueNonEmptyStrings([
    ...manifestBaseDirs.map((dir) => joinPath(trimTrailingSlash(dir), AGREEMENT_SUMMARY_FILENAME)),
    ...METRICS_SERVER_DIR_CANDIDATES.map((dir) => joinPath(trimTrailingSlash(dir), AGREEMENT_SUMMARY_FILENAME)),
  ]);

  for (const candidatePath of candidatePaths) {
    try {
      const response = await fetch(candidatePath, { cache: "no-store" });
      if (!response.ok) {
        if (response.status !== 404) {
          warnings.push({
            file: candidatePath,
            message: `Agreement summary fetch failed (HTTP ${response.status}).`,
          });
        }
        continue;
      }
      const text = await response.text();
      const summary = parseAgreementSummaryText(candidatePath, text, warnings);
      if (summary) {
        return summary;
      }
    } catch (error) {
      warnings.push({
        file: candidatePath,
        message: `Agreement summary fetch failed (${error.message}).`,
      });
    }
  }
  return null;
}

async function loadAgreementClustersFromServer(manifestBaseDirs = [], warnings = []) {
  const candidatePaths = uniqueNonEmptyStrings([
    ...manifestBaseDirs.map((dir) => joinPath(trimTrailingSlash(dir), AGREEMENT_CLUSTERS_FILENAME)),
    ...METRICS_SERVER_DIR_CANDIDATES.map((dir) => joinPath(trimTrailingSlash(dir), AGREEMENT_CLUSTERS_FILENAME)),
  ]);

  for (const candidatePath of candidatePaths) {
    try {
      const response = await fetch(candidatePath, { cache: "no-store" });
      if (!response.ok) {
        if (response.status !== 404) {
          warnings.push({
            file: candidatePath,
            message: `Agreement clusters fetch failed (HTTP ${response.status}).`,
          });
        }
        continue;
      }
      const text = await response.text();
      const clusters = parseAgreementClustersText(candidatePath, text, warnings);
      if (clusters) {
        return clusters;
      }
    } catch (error) {
      warnings.push({
        file: candidatePath,
        message: `Agreement clusters fetch failed (${error.message}).`,
      });
    }
  }
  return null;
}

async function loadFromServer(onProgress = null) {
  const discovery = await discoverMetricFilesFromServer();
  const files = discovery.files;
  const manifestBaseDirs = discovery.manifestBaseDirs || [];
  const warnings = [];
  let runs = await loadRunsFromServerFileList(files, manifestBaseDirs, warnings, onProgress);
  let fileCount = files.length;
  const agreementSummary = await loadAgreementSummaryFromServer(manifestBaseDirs, warnings);
  const agreementClusters = await loadAgreementClustersFromServer(manifestBaseDirs, warnings);

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
    agreementSummary,
    agreementClusters,
    warnings,
  };
}

async function collectMetricFilesFromDirectoryHandle(dirHandle, prefix = "") {
  const files = [];
  let agreementSummaryFile = null;
  let agreementClustersFile = null;
  for await (const [name, entry] of dirHandle.entries()) {
    if (entry.kind === "file" && name.toLowerCase().endsWith("_metrics.json")) {
      files.push({
        path: `${prefix}${name}`,
        handle: entry,
      });
      continue;
    }
    if (
      entry.kind === "file" &&
      name.toLowerCase() === AGREEMENT_SUMMARY_FILENAME &&
      agreementSummaryFile === null
    ) {
      agreementSummaryFile = {
        path: `${prefix}${name}`,
        handle: entry,
      };
      continue;
    }
    if (
      entry.kind === "file" &&
      name.toLowerCase() === AGREEMENT_CLUSTERS_FILENAME &&
      agreementClustersFile === null
    ) {
      agreementClustersFile = {
        path: `${prefix}${name}`,
        handle: entry,
      };
      continue;
    }
    if (entry.kind === "directory") {
      const nested = await collectMetricFilesFromDirectoryHandle(entry, `${prefix}${name}/`);
      files.push(...nested.metricFiles);
      if (!agreementSummaryFile && nested.agreementSummaryFile) {
        agreementSummaryFile = nested.agreementSummaryFile;
      }
      if (!agreementClustersFile && nested.agreementClustersFile) {
        agreementClustersFile = nested.agreementClustersFile;
      }
    }
  }
  return { metricFiles: files, agreementSummaryFile, agreementClustersFile };
}

async function loadAgreementSummaryFromDirectoryHandle(summaryFileEntry, warnings) {
  if (!summaryFileEntry || !summaryFileEntry.handle) {
    return null;
  }
  try {
    const fileObj = await summaryFileEntry.handle.getFile();
    const text = await fileObj.text();
    return parseAgreementSummaryText(summaryFileEntry.path, text, warnings);
  } catch (error) {
    warnings.push({
      file: summaryFileEntry.path,
      message: `Cannot read agreement summary (${error.message}).`,
    });
    return null;
  }
}

async function loadAgreementClustersFromDirectoryHandle(clustersFileEntry, warnings) {
  if (!clustersFileEntry || !clustersFileEntry.handle) {
    return null;
  }
  try {
    const fileObj = await clustersFileEntry.handle.getFile();
    const text = await fileObj.text();
    return parseAgreementClustersText(clustersFileEntry.path, text, warnings);
  } catch (error) {
    warnings.push({
      file: clustersFileEntry.path,
      message: `Cannot read agreement clusters (${error.message}).`,
    });
    return null;
  }
}

async function loadFromDirectoryHandle(dirHandle, onProgress = null) {
  const { metricFiles, agreementSummaryFile, agreementClustersFile } = await collectMetricFilesFromDirectoryHandle(dirHandle);
  if (!metricFiles.length) {
    throw new Error("No *_metrics.json files found in selected folder.");
  }

  const warnings = [];
  const agreementSummary = await loadAgreementSummaryFromDirectoryHandle(agreementSummaryFile, warnings);
  const agreementClusters = await loadAgreementClustersFromDirectoryHandle(agreementClustersFile, warnings);
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
    agreementSummary,
    agreementClusters,
    warnings,
  };
}

async function loadFromFiles(fileList, onProgress = null) {
  const files = Array.from(fileList || []).filter((file) =>
    String(file.name || "").toLowerCase().endsWith("_metrics.json")
  );
  const agreementFile =
    Array.from(fileList || []).find(
      (file) => String(file.name || "").toLowerCase() === AGREEMENT_SUMMARY_FILENAME
    ) || null;
  const agreementClustersFile =
    Array.from(fileList || []).find(
      (file) => String(file.name || "").toLowerCase() === AGREEMENT_CLUSTERS_FILENAME
    ) || null;

  if (!files.length) {
    throw new Error("No *_metrics.json files selected.");
  }

  const warnings = [];
  let agreementSummary = null;
  let agreementClusters = null;
  if (agreementFile) {
    try {
      agreementSummary = parseAgreementSummaryText(
        agreementFile.name,
        await agreementFile.text(),
        warnings
      );
    } catch (error) {
      warnings.push({
        file: agreementFile.name,
        message: `Cannot read agreement summary (${error.message}).`,
      });
    }
  }
  if (agreementClustersFile) {
    try {
      agreementClusters = parseAgreementClustersText(
        agreementClustersFile.name,
        await agreementClustersFile.text(),
        warnings
      );
    } catch (error) {
      warnings.push({
        file: agreementClustersFile.name,
        message: `Cannot read agreement clusters (${error.message}).`,
      });
    }
  }
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
    agreementSummary,
    agreementClusters,
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
    filterSearchQuery: state.filterSearchQuery,
    timeRanges: normalizeTimeRanges(state.timeRanges),
    desktopSidebarCollapsed: state.desktopSidebarCollapsed,
    sortBy: state.sortBy,
    leaderboardTableSortKey: state.leaderboardTableSortKey,
    leaderboardTableSortDirection: state.leaderboardTableSortDirection,
    leaderboardTab: state.leaderboardTab,
    agreementViewMode: state.agreementViewMode,
    leaderboardChartGroupBy: state.leaderboardChartGroupBy,
    leaderboardScatterGroupBy: state.leaderboardScatterGroupBy,
    leaderboardChartBestByTask: state.leaderboardChartBestByTask,
    leaderboardScatterXAxis: state.leaderboardScatterXAxis,
    agreementRepresentativePolicy: state.agreementRepresentativePolicy,
    scatterShowCi: state.scatterShowCi,
    timeSeriesShowLabels: state.timeSeriesShowLabels,
    timeSeriesViewport: state.timeSeriesViewport,
    priceScatterViewport: state.priceScatterViewport,
    priceScatterCostMode: state.priceScatterCostMode,
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
      if (typeof payload.filterSearchQuery === "string") {
        state.filterSearchQuery = asTrimmedString(payload.filterSearchQuery);
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
      if (typeof payload.leaderboardTab === "string") {
        if (LEADERBOARD_TAB_KEYS.has(payload.leaderboardTab)) {
          state.leaderboardTab = payload.leaderboardTab;
        } else if (payload.leaderboardTab === "price_scatter") {
          state.leaderboardTab = "scatter";
          state.leaderboardScatterXAxis = "price";
        } else if (payload.leaderboardTab === "time_series") {
          state.leaderboardTab = "scatter";
          state.leaderboardScatterXAxis = "time";
        } else if (payload.leaderboardTab === "best_by_task") {
          state.leaderboardTab = "chart";
          state.leaderboardChartBestByTask = true;
        }
      }
      if (
        typeof payload.agreementViewMode === "string" &&
        AGREEMENT_VIEW_MODE_KEYS.has(payload.agreementViewMode)
      ) {
        state.agreementViewMode = payload.agreementViewMode;
      }
      if (
        typeof payload.leaderboardChartGroupBy === "string" &&
        LEADERBOARD_CHART_GROUP_BY_KEYS.has(payload.leaderboardChartGroupBy)
      ) {
        state.leaderboardChartGroupBy = payload.leaderboardChartGroupBy;
      }
      if (
        typeof payload.leaderboardScatterGroupBy === "string" &&
        LEADERBOARD_CHART_GROUP_BY_KEYS.has(payload.leaderboardScatterGroupBy)
      ) {
        state.leaderboardScatterGroupBy = payload.leaderboardScatterGroupBy;
      }
      if (typeof payload.leaderboardChartBestByTask === "boolean") {
        state.leaderboardChartBestByTask = payload.leaderboardChartBestByTask;
      }
      if (
        typeof payload.leaderboardScatterXAxis === "string" &&
        LEADERBOARD_SCATTER_X_AXIS_KEYS.has(payload.leaderboardScatterXAxis)
      ) {
        state.leaderboardScatterXAxis = payload.leaderboardScatterXAxis;
      }
      if (
        typeof payload.agreementRepresentativePolicy === "string" &&
        AGREEMENT_REPRESENTATIVE_POLICY_KEYS.has(payload.agreementRepresentativePolicy)
      ) {
        state.agreementRepresentativePolicy = payload.agreementRepresentativePolicy;
      }
      if (typeof payload.scatterShowCi === "boolean") {
        state.scatterShowCi = payload.scatterShowCi;
      }
      if (typeof payload.timeSeriesShowLabels === "boolean") {
        state.timeSeriesShowLabels = payload.timeSeriesShowLabels;
      }
      if (payload.timeSeriesViewport && typeof payload.timeSeriesViewport === "object") {
        state.timeSeriesViewport = payload.timeSeriesViewport;
      }
      if (payload.priceScatterViewport && typeof payload.priceScatterViewport === "object") {
        state.priceScatterViewport = payload.priceScatterViewport;
      }
      if (
        typeof payload.priceScatterCostMode === "string" &&
        PRICE_SCATTER_COST_MODE_KEYS.has(payload.priceScatterCostMode)
      ) {
        state.priceScatterCostMode = payload.priceScatterCostMode;
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
  renderFilterSearchControl();
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

function hasResettableFilterState() {
  return getActiveFilterCount() > 0 || Boolean(asTrimmedString(state.filterSearchQuery));
}

function updateResetFiltersButton() {
  if (!els.resetFiltersBtn) {
    return;
  }
  els.resetFiltersBtn.disabled = !hasResettableFilterState();
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

function normalizeFilterSearchQuery(value) {
  return asTrimmedString(value).toLocaleLowerCase();
}

function getVisibleFilterOptions(options, selectedValues, query = state.filterSearchQuery) {
  const source = Array.isArray(options) ? options : [];
  const selectedSet = new Set(Array.isArray(selectedValues) ? selectedValues : []);
  const normalizedQuery = normalizeFilterSearchQuery(query);
  if (!normalizedQuery) {
    return source;
  }
  return source.filter((option) => {
    const normalizedOption = asTrimmedString(option);
    if (!normalizedOption) {
      return false;
    }
    return selectedSet.has(option) || normalizedOption.toLocaleLowerCase().includes(normalizedQuery);
  });
}

function renderFilterSearchControl() {
  if (!els.filterSearchInput) {
    return;
  }
  const nextValue = asTrimmedString(state.filterSearchQuery);
  if (els.filterSearchInput.value !== nextValue) {
    els.filterSearchInput.value = nextValue;
  }
}

function setFilterSearchQuery(value) {
  const nextValue = asTrimmedString(value);
  if (state.filterSearchQuery === nextValue) {
    return;
  }
  state.filterSearchQuery = nextValue;
  persistUiState();
  renderFilterSearchControl();
  renderTaskControls();
  renderModelControls();
  updateResetFiltersButton();
  requestAnimationFrame(() => {
    updateSidebarScrollAffordances();
  });
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
  state.filterSearchQuery = "";
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

function setLeaderboardChartBestByTask(nextValue) {
  const normalized = Boolean(nextValue);
  if (state.leaderboardChartBestByTask === normalized) {
    return;
  }
  state.leaderboardChartBestByTask = normalized;
  if (!normalized) {
    state.bestByTaskVisibleCount = BEST_BY_TASK_PAGE_SIZE;
  }
  persistUiState();
  renderLeaderboard(state.filtered);
}

function setLeaderboardScatterXAxis(nextAxis) {
  if (!LEADERBOARD_SCATTER_X_AXIS_KEYS.has(nextAxis) || state.leaderboardScatterXAxis === nextAxis) {
    return;
  }
  state.leaderboardScatterXAxis = nextAxis;
  persistUiState();
  renderLeaderboard(state.filtered);
}

function setScatterShowCi(nextValue) {
  const normalized = Boolean(nextValue);
  if (state.scatterShowCi === normalized) {
    return;
  }
  state.scatterShowCi = normalized;
  persistUiState();
  renderLeaderboard(state.filtered);
}

function setTimeSeriesShowLabels(nextValue) {
  state.timeSeriesShowLabels = Boolean(nextValue);
  persistUiState();
  renderLeaderboard(state.filtered);
}

function setPriceScatterCostMode(nextMode) {
  if (!PRICE_SCATTER_COST_MODE_KEYS.has(nextMode) || state.priceScatterCostMode === nextMode) {
    return;
  }
  state.priceScatterCostMode = nextMode;
  state.priceScatterViewport = null;
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

function setPriceScatterViewport(nextViewport) {
  state.priceScatterViewport = normalizeTimeSeriesViewport(nextViewport);
  persistUiState();
  renderLeaderboard(state.filtered);
}

function resetPriceScatterZoom() {
  if (!state.priceScatterViewport) {
    return;
  }
  state.priceScatterViewport = null;
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

  if (els.filterSearchInput) {
    els.filterSearchInput.addEventListener("input", (event) => {
      setFilterSearchQuery(event.target.value);
    });
  }

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
  if (key === "cohen_kappa") return run.cohenKappa;
  if (key === "repeat_alpha") {
    const agreement = getRepeatAgreementForRun(run);
    return agreement ? agreement.alphaNominal : null;
  }
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
  if (metricKey === "cohen_kappa" || metricKey === "repeat_alpha") {
    return 1;
  }
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

function combineApproximateConfidenceRows(ciRows, meanValue) {
  const validRows = (Array.isArray(ciRows) ? ciRows : []).filter(
    (row) => row && typeof row.low === "number" && typeof row.high === "number"
  );
  if (!validRows.length || typeof meanValue !== "number" || !Number.isFinite(meanValue)) {
    return null;
  }
  const standardErrorsSquared = validRows.map((ci) => {
    const margin = (ci.high - ci.low) / 2;
    const sePercent = margin / 1.96;
    const se = sePercent / 100;
    return se * se;
  });
  const meanSe = Math.sqrt(standardErrorsSquared.reduce((sum, value) => sum + value, 0)) / validRows.length;
  const margin = 1.96 * meanSe * 100;
  return {
    low: Math.max(0, meanValue - margin),
    high: Math.min(100, meanValue + margin),
    sampleSize: validRows.reduce((sum, ci) => sum + (ci.sampleSize || 0), 0),
  };
}

function getMeanMetricConfidence(runs, metricKey, meanValue) {
  if (!supportsApproximateCi(metricKey)) {
    return null;
  }
  return combineApproximateConfidenceRows(
    runs.map((run) => getRunMetricConfidence(run, metricKey)),
    meanValue
  );
}

function averageFiniteNumbers(values) {
  const numeric = (Array.isArray(values) ? values : []).filter(
    (value) => typeof value === "number" && Number.isFinite(value)
  );
  if (!numeric.length) {
    return null;
  }
  return numeric.reduce((sum, value) => sum + value, 0) / numeric.length;
}

function buildBalancedGroupAggregate(items, options = {}) {
  const source = Array.isArray(items) ? items : [];
  const getUnitKey = typeof options.getUnitKey === "function" ? options.getUnitKey : () => "";
  const getMetricValue = typeof options.getMetricValue === "function" ? options.getMetricValue : () => null;
  const getPriceValue = typeof options.getPriceValue === "function" ? options.getPriceValue : null;
  const getConfidenceRows =
    typeof options.getConfidenceRows === "function" ? options.getConfidenceRows : null;
  const aggregate = metricsAggregationApi
    ? metricsAggregationApi.buildBalancedAggregate(source, {
        getUnitKey,
        getMetricValue,
        getPriceValue,
      })
    : (() => {
        const fallbackUnits = new Map();
        source.forEach((item) => {
          const unitKey = asTrimmedString(getUnitKey(item));
          if (!fallbackUnits.has(unitKey)) {
            fallbackUnits.set(unitKey, []);
          }
          fallbackUnits.get(unitKey).push(item);
        });
        const units = Array.from(fallbackUnits.entries()).map(([key, unitItems]) => ({
          key,
          items: unitItems,
          metricValue: averageFiniteNumbers(unitItems.map((item) => getMetricValue(item))),
          priceValue: getPriceValue ? averageFiniteNumbers(unitItems.map((item) => getPriceValue(item))) : null,
        }));
        return {
          units,
          metricValue: averageFiniteNumbers(units.map((unit) => unit.metricValue)),
          priceValue: getPriceValue ? averageFiniteNumbers(units.map((unit) => unit.priceValue)) : null,
          unitCount: units.length,
          pricedUnitCount: units.filter((unit) => typeof unit.priceValue === "number" && Number.isFinite(unit.priceValue))
            .length,
          itemCount: source.length,
        };
      })();

  const units = aggregate.units.map((unit) => {
    const ciRows = getConfidenceRows ? getConfidenceRows(unit.items) : [];
    return {
      ...unit,
      ci: combineApproximateConfidenceRows(ciRows, unit.metricValue),
    };
  });

  return {
    ...aggregate,
    units,
    ci: combineApproximateConfidenceRows(
      units.map((unit) => unit.ci),
      aggregate.metricValue
    ),
  };
}

function formatGroupedAggregateCount(itemCount, unitCount, singularLabel, pluralLabel) {
  const safeItemCount = typeof itemCount === "number" && Number.isFinite(itemCount) ? itemCount : 0;
  const safeUnitCount = typeof unitCount === "number" && Number.isFinite(unitCount) ? unitCount : 0;
  const unitLabel = safeUnitCount === 1 ? singularLabel : pluralLabel;
  const runLabel = safeItemCount === 1 ? "run" : "runs";
  if (safeItemCount === safeUnitCount) {
    return `${formatNum(safeUnitCount, 0)} ${unitLabel}`;
  }
  return `${formatNum(safeUnitCount, 0)} ${unitLabel}, ${formatNum(safeItemCount, 0)} ${runLabel}`;
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

function formatUsd(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "N/A";
  }
  return `$${Number(value).toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
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
  const visibleTasks = getVisibleFilterOptions(state.tasks, state.selectedTasks);
  const tasks = ["ALL", ...visibleTasks];
  syncSelectOptions(els.taskSelect, tasks, (task) => (task === "ALL" ? "All Tasks" : task));

  state.selectedTasks = sanitizeSelections(state.selectedTasks, state.tasks);
  syncTaskSelectValue();

  renderChoiceChipList(els.taskChipList, visibleTasks, state.selectedTasks, "All Tasks", toggleTaskSelection);
}

function renderModelControls() {
  const visibleModels = getVisibleFilterOptions(state.models, state.selectedModels);
  const models = ["ALL", ...visibleModels];
  syncSelectOptions(els.modelSelect, models, (model) => (model === "ALL" ? "All Models" : model));

  state.selectedModels = sanitizeSelections(state.selectedModels, state.models);
  syncModelSelectValue();

  renderChoiceChipList(els.modelChipList, visibleModels, state.selectedModels, "All Models", toggleModelSelection);
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
  const labelSuffix = asTrimmedString(options.labelSuffix);
  const metricKey = asTrimmedString(options.metricKey);
  labelEl.textContent = labelText;
  labelEl.title = `${labelText}${labelSuffix}`;

  const rowClass = asTrimmedString(options.rowClass);
  if (rowClass) {
    node.classList.add(rowClass);
  }

  const normalizeTrackPosition = (rawValue) => {
    if (metricKey === "cohen_kappa") {
      const clamped = Math.max(-1, Math.min(1, rawValue));
      return (clamped + 1) / 2;
    }
    return max > 0 ? Math.max(0, Math.min(1, rawValue / max)) : 0;
  };
  const ratio = normalizeTrackPosition(value);
  fillEl.style.width = `${ratio * 100}%`;

  if (colorClass === "warm") {
    fillEl.style.background = "linear-gradient(90deg, #f59e0b, #ea580c)";
  } else if (colorClass === "blue") {
    fillEl.style.background = "linear-gradient(90deg, #2563eb, #1e3a8a)";
  }

  const badges = Array.isArray(options.badges) ? options.badges.filter((badge) => asTrimmedString(badge)) : [];
  if (badges.length || labelSuffix) {
    labelEl.textContent = "";
    const textNode = document.createElement("span");
    textNode.className = "bar-label-text";
    textNode.textContent = labelText;
    labelEl.appendChild(textNode);
    if (labelSuffix) {
      const suffixNode = document.createElement("span");
      suffixNode.className = "bar-label-suffix";
      suffixNode.textContent = labelSuffix;
      labelEl.appendChild(suffixNode);
    }
  }
  if (badges.length) {
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
    const lowClamped = metricKey === "cohen_kappa"
      ? Math.max(-1, Math.min(1, referenceBand.low))
      : Math.max(0, Math.min(max, referenceBand.low));
    const highClamped = metricKey === "cohen_kappa"
      ? Math.max(lowClamped, Math.min(1, referenceBand.high))
      : Math.max(lowClamped, Math.min(max, referenceBand.high));
    const bandEl = document.createElement("span");
    bandEl.className = "bar-reference-band";
    const left = normalizeTrackPosition(lowClamped);
    const right = normalizeTrackPosition(highClamped);
    bandEl.style.left = `${left * 100}%`;
    bandEl.style.width = `${Math.max((right - left) * 100, 0.8)}%`;
    bandEl.title = `Grouped IQR ${formatMetric(metricKey, lowClamped)}-${formatMetric(metricKey, highClamped)}`;
    trackEl.appendChild(bandEl);
  }

  const distribution = options.distribution;
  if (distribution && distribution.values && Array.isArray(distribution.values) && distribution.values.length && max > 0) {
    const distributionWrap = document.createElement("span");
    distributionWrap.className = "bar-distribution";
    const distributionItemLabel = asTrimmedString(distribution.itemLabel) || "runs";
    const stats = getDistributionStats(distribution.values);
    if (stats) {
      const lowClamped = metricKey === "cohen_kappa"
        ? Math.max(-1, Math.min(1, stats.min))
        : Math.max(0, Math.min(max, stats.min));
      const highClamped = metricKey === "cohen_kappa"
        ? Math.max(lowClamped, Math.min(1, stats.max))
        : Math.max(lowClamped, Math.min(max, stats.max));
      const whiskerEl = document.createElement("span");
      whiskerEl.className = "bar-distribution-whisker";
      const whiskerLeft = normalizeTrackPosition(lowClamped);
      const whiskerRight = normalizeTrackPosition(highClamped);
      whiskerEl.style.left = `${whiskerLeft * 100}%`;
      whiskerEl.style.width = `${Math.max((whiskerRight - whiskerLeft) * 100, 0.5)}%`;
      distributionWrap.appendChild(whiskerEl);

      const q1Clamped = metricKey === "cohen_kappa"
        ? Math.max(-1, Math.min(1, stats.q1))
        : Math.max(0, Math.min(max, stats.q1));
      const q3Clamped = metricKey === "cohen_kappa"
        ? Math.max(q1Clamped, Math.min(1, stats.q3))
        : Math.max(q1Clamped, Math.min(max, stats.q3));
      const boxEl = document.createElement("span");
      boxEl.className = "bar-distribution-box";
      const boxLeft = normalizeTrackPosition(q1Clamped);
      const boxRight = normalizeTrackPosition(q3Clamped);
      boxEl.style.left = `${boxLeft * 100}%`;
      boxEl.style.width = `${Math.max((boxRight - boxLeft) * 100, 0.8)}%`;
      distributionWrap.appendChild(boxEl);

      const medianClamped = metricKey === "cohen_kappa"
        ? Math.max(-1, Math.min(1, stats.median))
        : Math.max(0, Math.min(max, stats.median));
      const medianEl = document.createElement("span");
      medianEl.className = "bar-distribution-median";
      medianEl.style.left = `${normalizeTrackPosition(medianClamped) * 100}%`;
      distributionWrap.appendChild(medianEl);

      stats.values.forEach((entryValue) => {
        const clamped = metricKey === "cohen_kappa"
          ? Math.max(-1, Math.min(1, entryValue))
          : Math.max(0, Math.min(max, entryValue));
        const tick = document.createElement("span");
        tick.className = "bar-distribution-tick";
        tick.style.left = `${normalizeTrackPosition(clamped) * 100}%`;
        distributionWrap.appendChild(tick);
      });
      distributionWrap.title = `${distributionItemLabel}: min ${formatMetric(metricKey, stats.min)}, median ${formatMetric(metricKey, stats.median)}, max ${formatMetric(metricKey, stats.max)}`;
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
    const lowClamped = metricKey === "cohen_kappa"
      ? Math.max(-1, Math.min(1, ci.low))
      : Math.max(0, Math.min(max, ci.low));
    const highClamped = metricKey === "cohen_kappa"
      ? Math.max(lowClamped, Math.min(1, ci.high))
      : Math.max(lowClamped, Math.min(max, ci.high));
    const ciRange = document.createElement("span");
    ciRange.className = "bar-ci-range";
    const left = normalizeTrackPosition(lowClamped);
    const right = normalizeTrackPosition(highClamped);
    ciRange.style.left = `${left * 100}%`;
    ciRange.style.width = `${Math.max((right - left) * 100, 0.6)}%`;
    ciRange.title = `95% CI ${formatMetric(metricKey, lowClamped)}-${formatMetric(metricKey, highClamped)}`;
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
  if (state.leaderboardChartGroupBy === "task") {
    return state.selectedModels.length !== 1;
  }
  if (state.leaderboardChartGroupBy === "model") {
    return state.selectedTasks.length !== 1;
  }
  return true;
}

function setLeaderboardChartGroupBy(groupBy) {
  if (!LEADERBOARD_CHART_GROUP_BY_KEYS.has(groupBy) || state.leaderboardChartGroupBy === groupBy) {
    return;
  }
  state.leaderboardChartGroupBy = groupBy;
  persistUiState();
  renderLeaderboard(state.filtered);
}

function setLeaderboardScatterGroupBy(groupBy) {
  if (!LEADERBOARD_CHART_GROUP_BY_KEYS.has(groupBy) || state.leaderboardScatterGroupBy === groupBy) {
    return;
  }
  state.leaderboardScatterGroupBy = groupBy;
  persistUiState();
  renderLeaderboard(state.filtered);
}

function renderLeaderboardGroupSwitch() {
  if (!els.leaderboardGroupSwitch) {
    return;
  }
  const supportsGrouping =
    state.leaderboardTab === "chart"
    || state.leaderboardTab === "scatter"
    || state.leaderboardTab === "radar";
  els.leaderboardGroupSwitch.hidden = !supportsGrouping;
  els.leaderboardGroupSwitch.innerHTML = "";
  if (!supportsGrouping) {
    return;
  }

  const label = document.createElement("span");
  label.className = "leaderboard-group-switch-label";
  label.textContent = "Group By";
  els.leaderboardGroupSwitch.appendChild(label);

  const toggle = document.createElement("div");
  toggle.className = "leaderboard-group-toggle";
  toggle.setAttribute("role", "group");
  toggle.setAttribute("aria-label", state.leaderboardTab === "radar" ? "Radar grouping" : "Leaderboard grouping");

  const options =
    state.leaderboardTab === "radar"
      ? [
          { key: "task", label: "Task", active: state.radarAxis === "task", onClick: () => setRadarAxis("task") },
          { key: "tag", label: "Tag", active: state.radarAxis === "tag", onClick: () => setRadarAxis("tag") },
        ]
      : state.leaderboardTab === "scatter"
        ? Object.entries(LEADERBOARD_CHART_GROUP_BY_LABELS).map(([key, text]) => ({
            key,
            label: text,
            active: state.leaderboardScatterGroupBy === key,
            onClick: () => setLeaderboardScatterGroupBy(key),
          }))
      : Object.entries(LEADERBOARD_CHART_GROUP_BY_LABELS).map(([key, text]) => ({
          key,
          label: text,
          active: state.leaderboardChartGroupBy === key,
          onClick: () => setLeaderboardChartGroupBy(key),
        }));

  options.forEach((option) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `leaderboard-group-btn${option.active ? " active" : ""}`;
    button.textContent = option.label;
    button.setAttribute("aria-pressed", option.active ? "true" : "false");
    button.addEventListener("click", option.onClick);
    toggle.appendChild(button);
  });

  els.leaderboardGroupSwitch.appendChild(toggle);
}

function renderLeaderboardTabControls() {
  if (!els.leaderboardTabs) {
    return;
  }
  renderLeaderboardGroupSwitch();
  if (els.leaderboardChartToggle) {
    els.leaderboardChartToggle.hidden = true;
    els.leaderboardChartToggle.innerHTML = "";
  }
  if (els.leaderboardMetricField) {
    const hideMetricField = state.leaderboardTab === "agreement";
    els.leaderboardMetricField.hidden = hideMetricField;
    els.leaderboardMetricField.style.display = hideMetricField ? "none" : "";
  }
  els.leaderboardTabs.innerHTML = "";
  const tabs = [
    { key: "chart", label: "Chart" },
    { key: "scatter", label: "Scatter" },
    { key: "table", label: "Table" },
    { key: "radar", label: "Radar" },
    { key: "agreement", label: "Agreement" },
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
  const noGrouping = groupBy === "none";
  const showContextLabels = shouldShowLeaderboardContextLabels();
  const grouping =
    noGrouping
      ? {
          key: "run",
          getGroupValue: (run) => run.filePath,
          getGroupLabel: () => "",
          getSingleRowLabel: (run) => `${run.task} / ${getRunModelDisplayName(run)}`,
          getSingleRowContextLabel: (run) => formatTs(run.timestamp),
          getGroupedRunLabel: (run) => `${run.task} / ${getRunModelDisplayName(run)}`,
          getGroupedRunContextLabel: (run) => formatTs(run.timestamp),
          getGroupContextLabel: () => "",
          getAggregationUnitValue: null,
          aggregationUnitSingular: "run",
          aggregationUnitPlural: "runs",
          distributionItemLabel: "runs",
        }
      : groupBy === "task"
      ? {
          key: "task",
          getGroupValue: (run) => asTrimmedString(run.task) || "Unknown task",
          getGroupLabel: (run) => asTrimmedString(run.task) || "Unknown task",
          getSingleRowLabel: (run) => asTrimmedString(run.task) || "Unknown task",
          getSingleRowContextLabel: (run) => (showContextLabels ? getRunModelDisplayName(run) : ""),
          getGroupedRunLabel: (run) => getRunModelDisplayName(run),
          getGroupedRunContextLabel: () => "",
          getGroupContextLabel: (groupRuns) => (showContextLabels ? getConcatenatedModelLabel(groupRuns) : ""),
          getAggregationUnitValue: (run) => asTrimmedString(run.model) || "Unknown model",
          aggregationUnitSingular: "model",
          aggregationUnitPlural: "models",
          distributionItemLabel: "model means",
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
          getAggregationUnitValue: (run) => asTrimmedString(run.task) || "Unknown task",
          aggregationUnitSingular: "task",
          aggregationUnitPlural: "tasks",
          distributionItemLabel: "task means",
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

    const balancedAggregate = buildBalancedGroupAggregate(runsSorted, {
      getUnitKey: grouping.getAggregationUnitValue,
      getMetricValue: (run) => getMetricValueForRun(run, metricKey),
      getConfidenceRows: (unitRuns) => unitRuns.map((run) => getRunMetricConfidence(run, metricKey)),
    });
    const avgMetric = balancedAggregate.metricValue;
    return {
      type: "group",
      key: `${grouping.key}:${groupValue}`,
      label: grouping.getGroupLabel(runsSorted[0], runsSorted),
      metricValue: avgMetric,
      avgMetric,
      ci: showApproximateCi ? balancedAggregate.ci : null,
      distributionValues: balancedAggregate.units.map((unit) => unit.metricValue),
      unitCount: balancedAggregate.unitCount,
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
      noGrouping
        ? "Rows are shown per run without aggregation."
        : groupBy === "task"
        ? `Grouped task rows first average repeated runs per model, then average across models. Distribution overlays show those model means${showApproximateCi ? " with CI" : ""}.`
        : `Grouped model rows first average repeated runs per task, then average across tasks. Distribution overlays show those task means${showApproximateCi ? " with CI" : ""}.`;
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
          { badges, rowClass, trackBadges, trackBadgeColorMap: selectedTagColorMap, metricKey }
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
    const distributionValues = Array.isArray(entry.distributionValues) ? entry.distributionValues : [];
    summary.appendChild(
      createBarRow(
        entry.label,
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
          metricKey,
          labelSuffix: ` (${formatGroupedAggregateCount(
            entry.runs.length,
            entry.unitCount,
            grouping.aggregationUnitSingular,
            grouping.aggregationUnitPlural
          )})`,
          distribution: distributionValues.length
            ? { values: distributionValues, itemLabel: grouping.distributionItemLabel }
            : null,
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
            metricKey,
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

function appendTimeSeriesPointGlyph(parent, options) {
  const {
    shape,
    cx,
    cy,
    size,
    fillColor = "var(--paper-strong)",
    fillOpacity = 0.78,
    outlineColor,
    strokeWidth = 1.5,
    innerDotColor = "",
    innerDotRadius = null,
    isSelected = false,
  } = options || {};

  if (!parent) {
    return null;
  }

  if (isSelected) {
    const halo = buildTimeSeriesShape(
      shape || TIME_SERIES_MODEL_SHAPES[0],
      cx,
      cy,
      size + 5,
      "none",
      "var(--accent)",
      Math.max(strokeWidth + 1.8, 3)
    );
    halo.setAttribute("class", "time-series-point-halo");
    parent.appendChild(halo);
  }

  const point = buildTimeSeriesShape(
    shape || TIME_SERIES_MODEL_SHAPES[0],
    cx,
    cy,
    size,
    fillColor,
    outlineColor || "var(--ink)",
    strokeWidth
  );
  point.setAttribute("fill-opacity", String(fillOpacity));
  parent.appendChild(point);

  const dotColor = asTrimmedString(innerDotColor);
  if (dotColor) {
    const dot = createSvgNode("circle", {
      cx,
      cy,
      r: innerDotRadius != null ? innerDotRadius : Math.max(size * 0.22, 2.4),
      fill: dotColor,
      stroke: "var(--paper-strong)",
      "stroke-width": 0.8,
      class: "time-series-point-core",
    });
    parent.appendChild(dot);
  }

  return point;
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

function quantileSorted(values, quantile) {
  if (!Array.isArray(values) || !values.length) {
    return null;
  }
  const q = Math.max(0, Math.min(1, Number(quantile)));
  const position = (values.length - 1) * q;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) {
    return values[lower];
  }
  const weight = position - lower;
  return values[lower] * (1 - weight) + values[upper] * weight;
}

function computeSymlogConstant(values) {
  const positiveValues = (values || [])
    .filter((value) => typeof value === "number" && Number.isFinite(value) && value > 0)
    .sort((a, b) => a - b);
  if (!positiveValues.length) {
    return 0.1;
  }
  const lowerQuartile = quantileSorted(positiveValues, 0.25) || positiveValues[0];
  return Math.max(0.01, Math.min(0.5, lowerQuartile / 2));
}

function symlogTransform(value, constant) {
  const numeric = typeof value === "number" && Number.isFinite(value) ? value : 0;
  const linearConstant = Math.max(Number(constant) || 0.1, 1e-9);
  if (numeric === 0) {
    return 0;
  }
  return Math.sign(numeric) * Math.log1p(Math.abs(numeric) / linearConstant);
}

function symlogInverse(value, constant) {
  const numeric = typeof value === "number" && Number.isFinite(value) ? value : 0;
  const linearConstant = Math.max(Number(constant) || 0.1, 1e-9);
  if (numeric === 0) {
    return 0;
  }
  return Math.sign(numeric) * linearConstant * Math.expm1(Math.abs(numeric));
}

function dedupeNumericTicks(values) {
  const unique = [];
  const seen = new Set();
  values.forEach((value) => {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      return;
    }
    const key = value.toPrecision(12);
    if (seen.has(key)) {
      return;
    }
    seen.add(key);
    unique.push(value);
  });
  return unique.sort((a, b) => a - b);
}

function buildSymlogPriceTicks(minValue, maxValue, constant, maxTicks = 6) {
  if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) {
    return [];
  }
  if (minValue === maxValue) {
    return [minValue];
  }

  const candidates = [minValue, maxValue];
  if (minValue <= 0 && maxValue >= 0) {
    candidates.push(0);
  }

  for (let exponent = -4; exponent <= 5; exponent += 1) {
    const base = 10 ** exponent;
    [1, 2, 5].forEach((multiplier) => {
      const tick = multiplier * base;
      if (tick >= minValue && tick <= maxValue) {
        candidates.push(tick);
      }
    });
  }

  const sorted = dedupeNumericTicks(candidates);
  if (sorted.length <= maxTicks) {
    return sorted;
  }

  const minProjection = symlogTransform(minValue, constant);
  const maxProjection = symlogTransform(maxValue, constant);
  const chosen = new Map();
  sorted.forEach((value) => {
    const projection = symlogTransform(value, constant);
    if (value === minValue || value === maxValue || value === 0) {
      chosen.set(value.toPrecision(12), { value, projection });
    }
  });

  for (let index = 0; index < maxTicks; index += 1) {
    const target = minProjection + ((maxProjection - minProjection) * index) / Math.max(maxTicks - 1, 1);
    let best = null;
    sorted.forEach((value) => {
      const projection = symlogTransform(value, constant);
      const diff = Math.abs(projection - target);
      if (!best || diff < best.diff) {
        best = { value, projection, diff };
      }
    });
    if (best) {
      chosen.set(best.value.toPrecision(12), { value: best.value, projection: best.projection });
    }
  }

  const selected = Array.from(chosen.values())
    .sort((left, right) => left.value - right.value)
    .map((entry) => entry.value);
  if (selected.length <= maxTicks) {
    return selected;
  }

  return dedupeNumericTicks(
    Array.from({ length: maxTicks }, (_, index) => {
      const position = (selected.length - 1) * (index / Math.max(maxTicks - 1, 1));
      return selected[Math.round(position)];
    })
  );
}

function formatUsdAxisTick(value) {
  const numeric = safeNum(value);
  if (numeric == null) {
    return "$0.00";
  }
  const absolute = Math.abs(numeric);
  if (absolute >= 1) {
    return formatUsd(numeric);
  }
  if (absolute >= 0.1) {
    return `$${formatNum(numeric, 2)}`;
  }
  if (absolute >= 0.01) {
    return `$${formatNum(numeric, 3)}`;
  }
  if (absolute >= 0.001) {
    return `$${formatNum(numeric, 4)}`;
  }
  return `$${numeric.toExponential(1)}`;
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
    if (row.tooltip) {
      entry.title = row.tooltip;
    }
    if (row.type === "task") {
      const swatch = document.createElement("span");
      swatch.className = "time-series-task-swatch";
      swatch.style.background = row.color;
      entry.appendChild(swatch);
    } else {
      const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      icon.setAttribute("viewBox", "0 0 16 16");
      icon.setAttribute("class", "time-series-shape-icon");
      appendTimeSeriesPointGlyph(icon, {
        shape: row.shape,
        cx: 8,
        cy: 8,
        size: 9,
        fillColor: row.fillColor || "var(--paper-strong)",
        fillOpacity: row.fillOpacity != null ? row.fillOpacity : 0.72,
        outlineColor: row.strokeColor || row.color || "var(--ink)",
        strokeWidth: 1.4,
        innerDotColor: row.innerDotColor || "",
        innerDotRadius: 2.2,
      });
      entry.appendChild(icon);
    }

    const label = document.createElement("span");
    label.textContent = row.label;
    if (row.tooltip) {
      label.title = row.tooltip;
    }
    entry.appendChild(label);
    grid.appendChild(entry);
  });
  wrap.appendChild(grid);
  return wrap;
}

function getResolvedPricingCatalogEntry(providerKey, resolvedKey) {
  const catalog = pricingApi ? pricingApi.getPricingCatalog(window.MODEL_PRICING_CATALOG) : null;
  if (
    !catalog ||
    !catalog.providers ||
    typeof catalog.providers !== "object" ||
    !catalog.providers[providerKey] ||
    !catalog.providers[providerKey].models ||
    typeof catalog.providers[providerKey].models !== "object"
  ) {
    return null;
  }
  const providerModels = catalog.providers[providerKey].models;
  const entry = providerModels[resolvedKey];
  return entry && typeof entry === "object" ? entry : null;
}

function formatPricingTierSummary(tierName, tierRates) {
  if (!tierRates || typeof tierRates !== "object") {
    return `${tierName} unavailable`;
  }
  return `${tierName} ~ in ${formatUsd(tierRates.input_usd_per_mtokens)} | cache ${formatUsd(
    tierRates.cached_input_usd_per_mtokens
  )} | out ${formatUsd(tierRates.output_usd_per_mtokens)} / 1M`;
}

function buildModelLegendPricingTooltip(modelName, runsForModel) {
  if (!pricingApi || !Array.isArray(runsForModel) || !runsForModel.length) {
    return "";
  }
  const summaries = new Map();
  runsForModel.forEach((run) => {
    const pricing = run && run.pricing ? run.pricing : null;
    const providerKey = asTrimmedString(pricing && pricing.providerKey)
      || (pricingApi.normalizeProviderKey ? pricingApi.normalizeProviderKey(run && run.provider) : asTrimmedString(run && run.provider));
    const resolvedKey = asTrimmedString(pricing && pricing.resolvedKey) || asTrimmedString(run && run.model);
    const tierName = asTrimmedString(pricing && pricing.pricingTier) || "standard";
    const entry = providerKey && resolvedKey ? getResolvedPricingCatalogEntry(providerKey, resolvedKey) : null;
    let line = "";
    const entryKind = entry && pricingApi && typeof pricingApi.classifyCatalogEntry === "function"
      ? pricingApi.classifyCatalogEntry(entry)
      : "";
    if (entryKind === "priced") {
      const serviceTiers = entry.service_tiers && typeof entry.service_tiers === "object" ? entry.service_tiers : {};
      const preferredTier = serviceTiers[tierName] && typeof serviceTiers[tierName] === "object" ? serviceTiers[tierName] : null;
      const standardTier = serviceTiers.standard && typeof serviceTiers.standard === "object" ? serviceTiers.standard : null;
      const tierSummary = preferredTier
        ? formatPricingTierSummary(tierName, preferredTier)
        : standardTier
        ? `${tierName} unavailable; ${formatPricingTierSummary("standard", standardTier)}`
        : `${tierName} unavailable`;
      line = `${providerKey || "unknown"} / ${resolvedKey}: ${tierSummary}`;
    } else {
      const statusLabel = asTrimmedString(pricing && pricing.statusLabel) || "no pricing match";
      line = `${providerKey || "unknown"} / ${resolvedKey || modelName}: ${statusLabel}`;
    }
    summaries.set(line, line);
  });
  if (!summaries.size) {
    return "";
  }
  return [`Pricing for ${modelName}`, ...summaries.keys()].join("\n");
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

function clampTimeSeriesViewport(viewport, fullDomain, minimumSpan = {}) {
  const normalized = normalizeTimeSeriesViewport(viewport);
  if (!normalized || !fullDomain) {
    return null;
  }

  const minXSpan =
    typeof minimumSpan.x === "number" && Number.isFinite(minimumSpan.x) ? minimumSpan.x : 1;
  const minYSpan =
    typeof minimumSpan.y === "number" && Number.isFinite(minimumSpan.y) ? minimumSpan.y : 1e-9;

  const xMin = Math.max(fullDomain.xMin, Math.min(normalized.xMin, fullDomain.xMax));
  const xMax = Math.max(fullDomain.xMin, Math.min(normalized.xMax, fullDomain.xMax));
  const yMin = Math.max(fullDomain.yMin, Math.min(normalized.yMin, fullDomain.yMax));
  const yMax = Math.max(fullDomain.yMin, Math.min(normalized.yMax, fullDomain.yMax));

  if (xMax - xMin <= minXSpan || yMax - yMin <= minYSpan) {
    return null;
  }

  return { xMin, xMax, yMin, yMax };
}

function createTimeSeriesSegmentedControl(ariaLabel, options) {
  const control = document.createElement("div");
  control.className = "time-series-segmented-control";
  control.setAttribute("role", "group");
  control.setAttribute("aria-label", ariaLabel);

  options.forEach((option) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `time-series-segment-btn${option.active ? " active" : ""}`;
    button.textContent = option.label;
    button.setAttribute("aria-pressed", option.active ? "true" : "false");
    button.addEventListener("click", option.onClick);
    control.appendChild(button);
  });

  return control;
}

function appendLeaderboardScatterAxisControls(controls) {
  if (!controls) {
    return;
  }

  controls.appendChild(
    createTimeSeriesSegmentedControl("Scatter x-axis", [
      {
        label: "Price",
        active: state.leaderboardScatterXAxis === "price",
        onClick: () => setLeaderboardScatterXAxis("price"),
      },
      {
        label: "Time",
        active: state.leaderboardScatterXAxis === "time",
        onClick: () => setLeaderboardScatterXAxis("time"),
      },
    ])
  );
}

function createTimeSeriesToggleControl(label, checked, onToggle) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `time-series-toggle${checked ? " active" : ""}`;
  button.setAttribute("role", "switch");
  button.setAttribute("aria-checked", checked ? "true" : "false");
  button.setAttribute("aria-label", `${label}: ${checked ? "on" : "off"}`);
  button.addEventListener("click", onToggle);

  const labelEl = document.createElement("span");
  labelEl.className = "time-series-toggle-label";
  labelEl.textContent = label;
  button.appendChild(labelEl);

  const statusEl = document.createElement("span");
  statusEl.className = "time-series-toggle-status";
  statusEl.textContent = checked ? "On" : "Off";
  button.appendChild(statusEl);

  const switchEl = document.createElement("span");
  switchEl.className = "time-series-toggle-switch";
  switchEl.setAttribute("aria-hidden", "true");
  const thumbEl = document.createElement("span");
  thumbEl.className = "time-series-toggle-thumb";
  switchEl.appendChild(thumbEl);
  button.appendChild(switchEl);

  return button;
}

function appendScatterCiToggleControl(controls, showControl) {
  if (!controls || !showControl) {
    return;
  }

  controls.appendChild(createTimeSeriesToggleControl("CI", state.scatterShowCi, () => setScatterShowCi(!state.scatterShowCi)));
}

function appendTimeSeriesLabelsToggleControl(controls) {
  if (!controls) {
    return;
  }
  controls.appendChild(
    createTimeSeriesToggleControl("Labels", state.timeSeriesShowLabels, () => setTimeSeriesShowLabels(!state.timeSeriesShowLabels))
  );
}

function createTimeSeriesResetButton(label, onClick, disabled = false) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "time-series-reset-btn";
  button.textContent = label;
  button.disabled = disabled;
  button.addEventListener("click", onClick);
  return button;
}

function renderChartTabControls(container) {
  if (!container) {
    return;
  }

  const controls = document.createElement("div");
  controls.className = "chart-tab-controls";

  const group = document.createElement("div");
  group.className = "chart-tab-controls-group";
  group.appendChild(
    createTimeSeriesToggleControl("Best run per task", state.leaderboardChartBestByTask, () =>
      setLeaderboardChartBestByTask(!state.leaderboardChartBestByTask)
    )
  );

  controls.appendChild(group);
  container.appendChild(controls);
}

function renderLeaderboardTimeSeries(container, runs, options = {}) {
  const { includeScatterAxisSwitch = false } = options;
  const metricKey = state.sortBy;
  const metricLabel = METRIC_LABELS[metricKey] || metricKey;
  const supportsCi = supportsApproximateCi(metricKey);
  const showApproximateCi = supportsCi && state.scatterShowCi;
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
  const taskColorByName = new Map(tasks.map((task) => [task, getTaskSeriesColor(task)]));
  const modelColorByName = new Map(models.map((model) => [model, getModelSeriesColor(model)]));
  const modelShapeByName = new Map(models.map((model) => [model, getModelSeriesShape(model)]));

  const header = document.createElement("div");
  header.className = "time-series-head";

  const primaryControls = document.createElement("div");
  primaryControls.className = "time-series-controls time-series-controls-primary";
  const secondaryControls = document.createElement("div");
  secondaryControls.className = "time-series-controls time-series-controls-secondary";

  if (includeScatterAxisSwitch) {
    appendLeaderboardScatterAxisControls(primaryControls);
  }
  appendScatterCiToggleControl(secondaryControls, supportsCi);
  appendTimeSeriesLabelsToggleControl(secondaryControls);
  secondaryControls.appendChild(createTimeSeriesResetButton("Reset Zoom", () => resetTimeSeriesZoom(), !state.timeSeriesViewport));

  if (primaryControls.childElementCount) {
    header.appendChild(primaryControls);
  }
  header.appendChild(secondaryControls);
  container.appendChild(header);

  const noteWrap = document.createElement("div");
  noteWrap.className = "time-series-notes";
  if (isLowerBetterMetric(metricKey)) {
    const note = document.createElement("p");
    note.className = "muted";
    note.textContent = `${metricLabel} is lower-is-better.`;
    noteWrap.appendChild(note);
  }
  if (supportsCi && showApproximateCi) {
    const ciNote = document.createElement("p");
    ciNote.className = "muted";
    ciNote.textContent = "95% CI is approximated from each run's evaluated examples.";
    noteWrap.appendChild(ciNote);
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
      "aria-label": `${entry.run.task}, ${getRunModelDisplayName(entry.run)}, ${formatTs(entry.run.timestamp)}, ${formatMetric(metricKey, entry.metricValue)}${formatCiRange(entry.ci)}`,
    });
    const pointX = xToPx(entry.timestampMs);
    const pointY = yToPx(entry.metricValue);
    appendTimeSeriesPointGlyph(pointGroup, {
      shape: modelShapeByName.get(entry.run.model) || TIME_SERIES_MODEL_SHAPES[0],
      cx: pointX,
      cy: pointY,
      size: 8.8,
      fillColor: "var(--paper-strong)",
      fillOpacity: 0.72,
      outlineColor: modelColorByName.get(entry.run.model) || getModelSeriesColor(entry.run.model),
      strokeWidth: 1.45,
      innerDotColor: taskColorByName.get(entry.run.task) || TIME_SERIES_TASK_COLORS[0],
      innerDotRadius: 2.3,
      isSelected: entry.run.filePath === state.selectedRunPath,
    });
    const title = createSvgNode("title");
    title.textContent = `${entry.run.task} | ${getRunModelDisplayName(entry.run)} | ${formatTs(entry.run.timestamp)} | ${formatMetric(
      metricKey,
      entry.metricValue
    )}${formatCiRange(entry.ci)}`;
    pointGroup.appendChild(title);
    pointGroup.addEventListener("click", () => openRunModal(entry.run));
    pointGroup.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openRunModal(entry.run);
      }
    });
    svg.appendChild(pointGroup);

    const visibleLabelText = getSelectionAwareScatterRunLabel(entry.run);
    if (state.timeSeriesShowLabels && visibleLabelText) {
      const label = createSvgNode("text", {
        x: pointX + 7,
        y: pointY - 8,
        class: "time-series-point-label",
      });
      label.textContent = truncateTimeSeriesLabel(visibleLabelText, 26);
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
      "Models (shape + outline)",
      models.map((model) => ({
        type: "model",
        shape: modelShapeByName.get(model),
        fillOpacity: 0.72,
        strokeColor: modelColorByName.get(model),
        label: model,
      }))
    )
  );
  container.appendChild(legends);
}

function expandNumericDomain(minValue, maxValue, options = {}) {
  const clampMin = typeof options.clampMin === "number" ? options.clampMin : null;
  if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) {
    return { min: clampMin ?? 0, max: clampMin != null ? clampMin + 1 : 1 };
  }
  if (minValue === maxValue) {
    const center = minValue;
    const pad = Math.max(Math.abs(center) * 0.15, options.fallbackPad ?? 1);
    const nextMin = clampMin != null ? Math.max(clampMin, center - pad) : center - pad;
    const nextMax = center + pad;
    if (nextMin === nextMax) {
      return { min: nextMin, max: nextMax + 1 };
    }
    return { min: nextMin, max: nextMax };
  }
  const pad = (maxValue - minValue) * 0.08;
  const nextMin = clampMin != null ? Math.max(clampMin, minValue - pad) : minValue - pad;
  const nextMax = maxValue + pad;
  if (nextMin === nextMax) {
    return { min: nextMin, max: nextMax + 1 };
  }
  return { min: nextMin, max: nextMax };
}

function getPriceScatterCostModeLabel(mode = state.priceScatterCostMode) {
  return mode === "per_prompt" ? "Average Cost per Prediction (USD)" : "Estimated Cost (USD)";
}

function getPriceScatterUnknownLabel(mode = state.priceScatterCostMode) {
  return mode === "per_prompt" ? "Unknown Avg/Prediction" : "Unknown Cost";
}

function getPriceScatterUnknownValueText(mode = state.priceScatterCostMode) {
  return mode === "per_prompt" ? "Unknown avg/prediction cost" : "Unknown cost";
}

function getPriceScatterValueForRun(run, mode = state.priceScatterCostMode) {
  const estimatedCostUsd = safeNum(run && run.estimatedCostUsd);
  if (estimatedCostUsd == null) {
    return null;
  }
  if (mode !== "per_prompt") {
    return estimatedCostUsd;
  }
  const predictions = toNonNegativeNumber(getPredictionCountForRun(run));
  return predictions > 0 ? estimatedCostUsd / predictions : null;
}

function renderLeaderboardPriceScatter(container, runs, options = {}) {
  const { includeScatterAxisSwitch = false } = options;
  const metricKey = state.sortBy;
  const metricLabel = METRIC_LABELS[metricKey] || metricKey;
  const supportsCi = supportsApproximateCi(metricKey);
  const showApproximateCi = supportsCi && state.scatterShowCi;
  const priceScatterCostMode = state.priceScatterCostMode;
  const priceAxisLabel = getPriceScatterCostModeLabel(priceScatterCostMode);
  const unknownBandLabelText = getPriceScatterUnknownLabel(priceScatterCostMode);
  const unknownValueText = getPriceScatterUnknownValueText(priceScatterCostMode);
  const averagedCostLabel = priceScatterCostMode === "per_prompt" ? "avg/prediction cost" : "cost";
  const entriesWithMetric = runs
    .map((run) => {
      const metricValue = getMetricValueForRun(run, metricKey);
      if (typeof metricValue !== "number" || !Number.isFinite(metricValue)) {
        return null;
      }
      const knownPriceValue = getPriceScatterValueForRun(run, priceScatterCostMode);
      const hasKnownPrice = typeof knownPriceValue === "number" && Number.isFinite(knownPriceValue);
      return {
        run,
        metricValue,
        priceValue: hasKnownPrice ? knownPriceValue : null,
        hasKnownPrice,
      };
    })
    .filter(Boolean);
  if (!entriesWithMetric.length) {
    container.innerHTML = `<p class="muted">No runs with ${metricLabel.toLowerCase()} in current filter.</p>`;
    return;
  }

  const tasks = [...new Set(entriesWithMetric.map((entry) => asTrimmedString(entry.run.task)).filter(Boolean))].sort((a, b) =>
    a.localeCompare(b)
  );
  const models = [...new Set(entriesWithMetric.map((entry) => asTrimmedString(entry.run.model)).filter(Boolean))].sort((a, b) =>
    a.localeCompare(b)
  );
  const modelRunsByName = new Map();
  entriesWithMetric.forEach((entry) => {
    const modelName = asTrimmedString(entry.run.model);
    if (!modelName) {
      return;
    }
    if (!modelRunsByName.has(modelName)) {
      modelRunsByName.set(modelName, []);
    }
    modelRunsByName.get(modelName).push(entry.run);
  });
  const groupBy = state.leaderboardScatterGroupBy;
  const noGrouping = groupBy === "none";
  const taskColorByName = new Map(tasks.map((task) => [task, getTaskSeriesColor(task)]));
  const modelColorByName = new Map(models.map((model) => [model, getModelSeriesColor(model)]));
  const modelShapeByName = new Map(models.map((model) => [model, getModelSeriesShape(model)]));
  const groups = new Map();
  entriesWithMetric.forEach((entry) => {
    const groupKey =
      noGrouping
        ? entry.run.filePath
        : groupBy === "task"
        ? asTrimmedString(entry.run.task) || "Unknown task"
        : asTrimmedString(entry.run.model) || "Unknown model";
    if (!groups.has(groupKey)) {
      groups.set(groupKey, []);
    }
    groups.get(groupKey).push(entry);
  });
  const source = Array.from(groups.entries())
    .map(([groupKey, groupEntries]) => {
      const representative = [...groupEntries].sort((a, b) => {
        const diff = compareMetricNumbers(a.metricValue, b.metricValue, metricKey);
        if (diff !== 0) {
          return diff;
        }
        return parseRunTimestampMs(b.run) - parseRunTimestampMs(a.run);
      })[0];
      const knownPriceEntries = groupEntries.filter((entry) => entry.hasKnownPrice);
      const averagePrice = knownPriceEntries.length
        ? knownPriceEntries.reduce((sum, entry) => sum + entry.priceValue, 0) / knownPriceEntries.length
        : null;
      const groupRuns = groupEntries.map((entry) => entry.run);
      const aggregationUnitKey =
        noGrouping
          ? null
          : groupBy === "task"
          ? (entry) => asTrimmedString(entry.run.model) || "Unknown model"
          : (entry) => asTrimmedString(entry.run.task) || "Unknown task";
      const balancedAggregate =
        noGrouping
          ? null
          : buildBalancedGroupAggregate(groupEntries, {
              getUnitKey: aggregationUnitKey,
              getMetricValue: (entry) => entry.metricValue,
              getPriceValue: (entry) => entry.priceValue,
              getConfidenceRows: (unitEntries) =>
                unitEntries.map((entry) => getRunMetricConfidence(entry.run, metricKey)),
            });
      const averageMetric = noGrouping
        ? groupEntries.reduce((sum, entry) => sum + entry.metricValue, 0) / Math.max(groupEntries.length, 1)
        : balancedAggregate.metricValue;
      const groupUnknownPriceCount = groupEntries.filter((entry) => !entry.hasKnownPrice).length;
      const sharedSuffix = groupBy === "model" ? getSharedRunEffortSuffix(groupRuns) : "";
      const label =
        noGrouping
          ? `${representative.run.task} / ${getRunModelDisplayName(representative.run)}`
          : groupBy === "model"
          ? `${groupKey}${sharedSuffix ? ` ${sharedSuffix}` : ""}`
          : groupKey;
      return {
        groupKey,
        label,
        count: groupEntries.length,
        priceValue: noGrouping ? averagePrice : balancedAggregate.priceValue,
        hasKnownPrice: knownPriceEntries.length > 0,
        metricValue: averageMetric,
        ci:
          showApproximateCi && groupEntries.length === 1
            ? getRunMetricConfidence(representative.run, metricKey)
            : showApproximateCi
            ? noGrouping
              ? getMeanMetricConfidence(groupRuns, metricKey, averageMetric)
              : balancedAggregate.ci
            : null,
        unknownPriceCount: groupUnknownPriceCount,
        representativeRun: representative.run,
        unitCount: noGrouping ? groupEntries.length : balancedAggregate.unitCount,
        runs: groupRuns,
      };
    })
    .sort((a, b) => {
      if (a.hasKnownPrice !== b.hasKnownPrice) {
        return a.hasKnownPrice ? 1 : -1;
      }
      if (a.priceValue !== b.priceValue) {
        return a.priceValue - b.priceValue;
      }
      const diff = compareMetricNumbers(a.metricValue, b.metricValue, metricKey);
      if (diff !== 0) {
        return diff;
      }
      return a.label.localeCompare(b.label);
    });
  const numericSource = source.filter((entry) => entry.hasKnownPrice);
  const unknownBandSource = source.filter((entry) => !entry.hasKnownPrice);

  const header = document.createElement("div");
  header.className = "time-series-head";
  const primaryControls = document.createElement("div");
  primaryControls.className = "time-series-controls time-series-controls-primary";
  const secondaryControls = document.createElement("div");
  secondaryControls.className = "time-series-controls time-series-controls-secondary";
  if (includeScatterAxisSwitch) {
    appendLeaderboardScatterAxisControls(primaryControls);
  }
  primaryControls.appendChild(
    createTimeSeriesSegmentedControl("Price aggregation mode", [
      {
        label: "Total",
        active: priceScatterCostMode === "total",
        onClick: () => setPriceScatterCostMode("total"),
      },
      {
        label: "Avg Price",
        active: priceScatterCostMode === "per_prompt",
        onClick: () => setPriceScatterCostMode("per_prompt"),
      },
    ])
  );
  appendScatterCiToggleControl(secondaryControls, supportsCi);
  appendTimeSeriesLabelsToggleControl(secondaryControls);
  secondaryControls.appendChild(
    createTimeSeriesResetButton("Reset Zoom", () => resetPriceScatterZoom(), !state.priceScatterViewport || !numericSource.length)
  );
  if (primaryControls.childElementCount) {
    header.appendChild(primaryControls);
  }
  header.appendChild(secondaryControls);
  container.appendChild(header);

  const noteWrap = document.createElement("div");
  noteWrap.className = "time-series-notes";
  if (supportsCi && showApproximateCi) {
    const ciNote = document.createElement("p");
    ciNote.className = "muted";
    ciNote.textContent = "95% CI is approximated from each run's evaluated examples.";
    noteWrap.appendChild(ciNote);
  }
  const zoomNote = document.createElement("p");
  zoomNote.className = "muted";
  zoomNote.textContent = numericSource.length
    ? state.priceScatterViewport
      ? "Drag within the numeric price area to zoom again. Double-click to reset zoom."
      : "Drag within the numeric price area to zoom into a selected region. Double-click to reset zoom."
    : "Zoom is unavailable because none of the visible points have a known numeric price.";
  noteWrap.appendChild(zoomNote);
  if (noteWrap.childElementCount) {
    container.appendChild(noteWrap);
  }

  const chartWrap = document.createElement("div");
  chartWrap.className = "time-series-wrap";
  container.appendChild(chartWrap);

  const width = 960;
  const height = 460;
  const margin = { top: 18, right: 26, bottom: 58, left: 78 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const svg = createSvgNode("svg", {
    viewBox: `0 0 ${width} ${height}`,
    class: "time-series-svg",
    "aria-label": `${priceAxisLabel} versus ${metricLabel} scatterplot`,
  });
  chartWrap.appendChild(svg);

  const hasNumericAxis = numericSource.length > 0;
  const unknownBandWidth = unknownBandSource.length ? 88 : 0;
  const unknownBandGap = unknownBandSource.length && hasNumericAxis ? 14 : 0;
  const plotLeft = margin.left + unknownBandWidth + unknownBandGap;
  const numericInnerWidth = width - margin.right - plotLeft;
  const priceValues = numericSource.map((entry) => entry.priceValue);
  const metricValues = source.map((entry) => entry.metricValue);
  const fullXDomain = hasNumericAxis
    ? expandNumericDomain(Math.min(...priceValues), Math.max(...priceValues), {
        clampMin: 0,
        fallbackPad: 0.01,
      })
    : { min: 0, max: 1 };
  const fullYDomain = expandNumericDomain(Math.min(...metricValues), Math.max(...metricValues), {
    fallbackPad: isPercentMetric(metricKey) ? 1 : 0.5,
  });
  const symlogConstant = hasNumericAxis ? computeSymlogConstant(priceValues) : 1;
  const clampedViewport = hasNumericAxis
    ? clampTimeSeriesViewport(
        state.priceScatterViewport,
        {
          xMin: fullXDomain.min,
          xMax: fullXDomain.max,
          yMin: fullYDomain.min,
          yMax: fullYDomain.max,
        },
        {
          x: 1e-9,
          y: 1e-9,
        }
      )
    : null;
  const xDomain = clampedViewport
    ? { min: clampedViewport.xMin, max: clampedViewport.xMax }
    : fullXDomain;
  const yDomain = clampedViewport
    ? { min: clampedViewport.yMin, max: clampedViewport.yMax }
    : fullYDomain;
  const xDomainProjectedMin = symlogTransform(xDomain.min, symlogConstant);
  const xDomainProjectedMax = symlogTransform(xDomain.max, symlogConstant);
  const xProjectedSpan = Math.max(xDomainProjectedMax - xDomainProjectedMin, 1e-9);
  const xToPx = (value) =>
    plotLeft + ((symlogTransform(value, symlogConstant) - xDomainProjectedMin) / xProjectedSpan) * numericInnerWidth;
  const yToPx = (value) => margin.top + innerHeight - ((value - yDomain.min) / (yDomain.max - yDomain.min)) * innerHeight;
  const pxToX = (px) =>
    symlogInverse(xDomainProjectedMin + ((px - plotLeft) / Math.max(numericInnerWidth, 1)) * xProjectedSpan, symlogConstant);
  const pxToY = (px) => yDomain.min + ((margin.top + innerHeight - px) / innerHeight) * (yDomain.max - yDomain.min);
  const unknownBandCenterX = margin.left + unknownBandWidth / 2;
  const unknownBandJitter = Math.min(unknownBandWidth * 0.23, 18);

  const yTicks = buildNumericTicks(yDomain.min, yDomain.max, 5);
  const xTicks = hasNumericAxis ? buildSymlogPriceTicks(xDomain.min, xDomain.max, symlogConstant, 6) : [];

  yTicks.forEach((tickValue) => {
    const y = yToPx(tickValue);
    svg.appendChild(createSvgNode("line", { x1: margin.left, x2: width - margin.right, y1: y, y2: y, class: "time-series-grid-line" }));
    const label = createSvgNode("text", { x: margin.left - 10, y: y + 4, "text-anchor": "end", class: "time-series-tick" });
    label.textContent = isPercentMetric(metricKey) ? `${formatNum(tickValue, 0)}%` : formatNum(tickValue, 2);
    svg.appendChild(label);
  });

  if (unknownBandSource.length) {
    svg.appendChild(
      createSvgNode("rect", {
        x: margin.left,
        y: margin.top,
        width: unknownBandWidth,
        height: innerHeight,
        class: "time-series-unknown-band",
      })
    );
    const unknownBandLabel = createSvgNode("text", {
      x: unknownBandCenterX,
      y: margin.top + 14,
      "text-anchor": "middle",
      class: "time-series-unknown-band-label",
    });
    unknownBandLabel.textContent = unknownBandLabelText;
    svg.appendChild(unknownBandLabel);
    const unknownBandTick = createSvgNode("text", {
      x: unknownBandCenterX,
      y: height - margin.bottom + 20,
      "text-anchor": "middle",
      class: "time-series-tick",
    });
    unknownBandTick.textContent = "Unknown";
    svg.appendChild(unknownBandTick);
  }

  xTicks.forEach((tickValue) => {
    const x = xToPx(tickValue);
    svg.appendChild(createSvgNode("line", { x1: x, x2: x, y1: margin.top, y2: height - margin.bottom, class: "time-series-grid-line" }));
    const label = createSvgNode("text", {
      x,
      y: height - margin.bottom + 20,
      "text-anchor": "middle",
      class: "time-series-tick",
    });
    label.textContent = formatUsdAxisTick(tickValue);
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
    x: hasNumericAxis
      ? plotLeft + numericInnerWidth / 2
      : margin.left + (unknownBandWidth ? unknownBandWidth / 2 : innerWidth / 2),
    y: height - 18,
    "text-anchor": "middle",
    class: "time-series-axis-label",
  });
  xAxisLabel.textContent = priceAxisLabel;
  svg.appendChild(xAxisLabel);
  const xAxisSubLabel = createSvgNode("text", {
    x: hasNumericAxis
      ? plotLeft + numericInnerWidth / 2
      : margin.left + (unknownBandWidth ? unknownBandWidth / 2 : innerWidth / 2),
    y: height - 4,
    "text-anchor": "middle",
    class: "time-series-tick",
  });
  xAxisSubLabel.textContent = "(compressed symlog scale)";
  svg.appendChild(xAxisSubLabel);

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
    x: plotLeft,
    y: margin.top,
    width: hasNumericAxis ? numericInnerWidth : 0,
    height: innerHeight,
    class: "time-series-selection-overlay",
  });
  svg.appendChild(selectionOverlay);

  const getUnknownBandX = (entry) => {
    const jitterSeed = hashOrdinalKey(`${entry.groupKey}|${entry.label}|${entry.count}`);
    const jitterRatio = (jitterSeed % 1000) / 999;
    return unknownBandCenterX + (jitterRatio - 0.5) * 2 * unknownBandJitter;
  };

  source.forEach((entry) => {
    const isSelected = entry.runs.some((run) => run.filePath === state.selectedRunPath);
    const priceLabelText = entry.hasKnownPrice ? formatUsd(entry.priceValue) : unknownValueText;
    const priceDetailText = entry.hasKnownPrice
      ? entry.unknownPriceCount
        ? `${formatNum(entry.unknownPriceCount, 0)} run(s) omitted from average ${averagedCostLabel}`
        : ""
      : `Shown in ${unknownBandLabelText} band`;
    const pointGroup = createSvgNode("g", {
      class: `time-series-point${isSelected ? " is-selected" : ""}`,
      tabindex: "0",
      role: "button",
      "aria-label": [
        entry.label,
        priceLabelText,
        `${formatMetric(metricKey, entry.metricValue)}${formatCiRange(entry.ci)}`,
        noGrouping
          ? `${formatNum(entry.count, 0)} run(s)`
          : formatGroupedAggregateCount(
              entry.count,
              entry.unitCount,
              groupBy === "task" ? "model" : "task",
              groupBy === "task" ? "models" : "tasks"
            ),
        priceDetailText,
      ]
        .filter(Boolean)
        .join(", "),
    });
    const pointX = entry.hasKnownPrice ? xToPx(entry.priceValue) : getUnknownBandX(entry);
    const pointY = yToPx(entry.metricValue);
    if (entry.ci && typeof entry.ci.low === "number" && typeof entry.ci.high === "number") {
      const lowY = yToPx(entry.ci.low);
      const highY = yToPx(entry.ci.high);
      svg.appendChild(
        createSvgNode("line", {
          x1: pointX,
          x2: pointX,
          y1: lowY,
          y2: highY,
          class: "time-series-ci-line",
        })
      );
      svg.appendChild(
        createSvgNode("line", {
          x1: pointX - 5,
          x2: pointX + 5,
          y1: lowY,
          y2: lowY,
          class: "time-series-ci-cap",
        })
      );
      svg.appendChild(
        createSvgNode("line", {
          x1: pointX - 5,
          x2: pointX + 5,
          y1: highY,
          y2: highY,
          class: "time-series-ci-cap",
        })
      );
    }
    const fillColor =
      noGrouping
        ? "var(--paper-strong)"
        : groupBy === "task"
        ? taskColorByName.get(entry.groupKey) || TIME_SERIES_TASK_COLORS[0]
        : "var(--paper-strong)";
    const outlineColor =
      noGrouping
        ? modelColorByName.get(entry.representativeRun.model) || getModelSeriesColor(entry.representativeRun.model)
        : groupBy === "model"
        ? modelColorByName.get(entry.groupKey) || getModelSeriesColor(entry.groupKey)
        : "var(--ink)";
    const shape =
      noGrouping
        ? modelShapeByName.get(entry.representativeRun.model) || TIME_SERIES_MODEL_SHAPES[0]
        : groupBy === "model"
        ? modelShapeByName.get(entry.groupKey) || TIME_SERIES_MODEL_SHAPES[0]
        : "circle";
    appendTimeSeriesPointGlyph(pointGroup, {
      shape,
      cx: pointX,
      cy: pointY,
      size: entry.count > 1 ? 10.5 : 8.6,
      fillColor,
      fillOpacity: groupBy === "task" ? 0.88 : 0.74,
      outlineColor,
      strokeWidth: groupBy === "task" ? 1.35 : 1.55,
      innerDotColor:
        noGrouping
          ? taskColorByName.get(asTrimmedString(entry.representativeRun.task)) || TIME_SERIES_TASK_COLORS[0]
          : "",
      innerDotRadius: noGrouping ? 2.2 : null,
      isSelected,
    });
    const title = createSvgNode("title");
    title.textContent = [
      entry.label,
      entry.count > 1 ? `avg ${priceLabelText}` : priceLabelText,
      entry.count > 1
        ? `avg ${formatMetric(metricKey, entry.metricValue)}${formatCiRange(entry.ci)}`
        : `${formatMetric(metricKey, entry.metricValue)}${formatCiRange(entry.ci)}`,
      noGrouping
        ? `${entry.count} run(s)`
        : formatGroupedAggregateCount(
            entry.count,
            entry.unitCount,
            groupBy === "task" ? "model" : "task",
            groupBy === "task" ? "models" : "tasks"
          ),
      priceDetailText,
    ]
      .filter(Boolean)
      .join(" | ");
    pointGroup.appendChild(title);
    pointGroup.addEventListener("click", () => openRunModal(entry.representativeRun));
    pointGroup.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openRunModal(entry.representativeRun);
      }
    });
    svg.appendChild(pointGroup);

    if (state.timeSeriesShowLabels) {
      const visibleLabelText = buildPriceScatterPointLabelText(entry, groupBy, priceLabelText);
      const label = createSvgNode("text", {
        x: pointX + 7,
        y: pointY - 8,
        class: "time-series-point-label",
      });
      label.textContent = truncateTimeSeriesLabel(visibleLabelText, 30);
      svg.appendChild(label);
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
    x: Math.max(plotLeft, Math.min(width - margin.right, point.x)),
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
    setPriceScatterViewport({
      xMin: pxToX(left),
      xMax: pxToX(right),
      yMin: pxToY(bottom),
      yMax: pxToY(top),
    });
  };

  if (hasNumericAxis) {
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
  }

  svg.addEventListener("dblclick", () => {
    resetPriceScatterZoom();
  });

  const legends = document.createElement("div");
  legends.className = "time-series-legends";
  if (noGrouping) {
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
        "Models (shape + outline, hover to see pricing)",
        models.map((model) => ({
          type: "model",
          shape: modelShapeByName.get(model),
          fillOpacity: 0.72,
          strokeColor: modelColorByName.get(model),
          label: model,
          tooltip: buildModelLegendPricingTooltip(model, modelRunsByName.get(model) || []),
        }))
      )
    );
  } else if (groupBy === "task") {
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
  } else {
    legends.appendChild(
      createTimeSeriesLegend(
        "Models (shape + outline, hover to see pricing)",
        models.map((model) => ({
          type: "model",
          shape: modelShapeByName.get(model),
          fillOpacity: 0.72,
          strokeColor: modelColorByName.get(model),
          label: model,
          tooltip: buildModelLegendPricingTooltip(model, modelRunsByName.get(model) || []),
        }))
      )
    );
  }
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
    const runTooltipParts = [runLabel];
    if (run.tagsDisplay) {
      runTooltipParts.push(`Tags: ${run.tagsDisplay}`);
    }
    if (run.fileName) {
      runTooltipParts.push(run.fileName);
    }
    runCell.title = runTooltipParts.join("\n");
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

function renderLeaderboardScatter(container, runs) {
  if (state.leaderboardScatterXAxis === "time") {
    renderLeaderboardTimeSeries(container, runs, { includeScatterAxisSwitch: true });
    return;
  }
  renderLeaderboardPriceScatter(container, runs, { includeScatterAxisSwitch: true });
}

function renderLeaderboard(runs) {
  cleanupLeaderboardMetricsScrollAffordance();
  renderLeaderboardTabControls();
  els.leaderboardChart.innerHTML = "";

  const panel = document.createElement("div");
  panel.className = `leaderboard-panel${
    state.leaderboardTab === "radar"
      ? " leaderboard-panel-radar"
      : state.leaderboardTab === "scatter"
        ? " leaderboard-panel-time-series"
        : ""
  }`;
  els.leaderboardChart.appendChild(panel);

  if (state.leaderboardTab === "scatter") {
    renderLeaderboardScatter(panel, runs);
    return;
  }
  if (state.leaderboardTab === "table") {
    renderLeaderboardMetricsTable(panel, runs);
    return;
  }
  if (state.leaderboardTab === "radar") {
    renderRadarPanel(panel, runs);
    return;
  }
  if (state.leaderboardTab === "agreement") {
    renderAgreement(panel, runs);
    return;
  }
  if (state.leaderboardTab === "chart") {
    renderChartTabControls(panel);
    if (state.leaderboardChartBestByTask) {
      renderBestByTask(panel, runs);
      return;
    }
    renderLeaderboardChart(panel, runs);
    return;
  }
  renderLeaderboardChart(panel, runs);
}

function setAgreementRepresentativePolicy(policy) {
  if (!AGREEMENT_REPRESENTATIVE_POLICY_KEYS.has(policy) || state.agreementRepresentativePolicy === policy) {
    return;
  }
  state.agreementRepresentativePolicy = policy;
  persistUiState();
  renderLeaderboard(state.filtered);
  const selectedRun = findRunByPath(state.selectedRunPath);
  if (selectedRun && els.runModal && !els.runModal.classList.contains("hidden")) {
    fillRunDetailsContent(selectedRun);
  }
}

function setAgreementViewMode(mode) {
  if (!AGREEMENT_VIEW_MODE_KEYS.has(mode) || state.agreementViewMode === mode) {
    return;
  }
  state.agreementViewMode = mode;
  persistUiState();
  renderLeaderboard(state.filtered);
}

function getVisibleRepeatAgreementEntries(runs) {
  const visibleRunStems = new Set((runs || []).map((run) => run.runStem).filter(Boolean));
  const source =
    state.agreementSummary && Array.isArray(state.agreementSummary.repeatGroups)
      ? state.agreementSummary.repeatGroups
      : [];
  return source.filter(
    (entry) =>
      Array.isArray(entry.runStems) &&
      entry.runStems.length >= 2 &&
      entry.runStems.every((runStem) => visibleRunStems.has(runStem))
  );
}

function getVisibleCrossModelAgreementEntries(runs, policy = state.agreementRepresentativePolicy) {
  const visibleRunStems = new Set((runs || []).map((run) => run.runStem).filter(Boolean));
  const crossModel =
    state.agreementSummary && state.agreementSummary.crossModel
      ? state.agreementSummary.crossModel
      : {};
  const source = Array.isArray(crossModel[policy]) ? crossModel[policy] : [];
  return source.filter(
    (entry) =>
      Array.isArray(entry.representatives) &&
      entry.representatives.length >= 2 &&
      entry.representatives.every((representative) => visibleRunStems.has(representative.runStem))
  );
}

function getVisibleCrossModelClusterEntries(runs, policy = state.agreementRepresentativePolicy) {
  const visibleRunStems = new Set((runs || []).map((run) => run.runStem).filter(Boolean));
  const crossModel =
    state.agreementClusters && state.agreementClusters.crossModel
      ? state.agreementClusters.crossModel
      : {};
  const source = Array.isArray(crossModel[policy]) ? crossModel[policy] : [];
  return source
    .map((entry) => {
      const indexMap = new Map();
      const visibleRepresentatives = [];
      (entry.representatives || []).forEach((representative, originalIndex) => {
        if (!representative || !visibleRunStems.has(representative.runStem)) {
          return;
        }
        indexMap.set(originalIndex, visibleRepresentatives.length);
        visibleRepresentatives.push({
          ...representative,
          originalIndex,
        });
      });

      if (visibleRepresentatives.length < 2) {
        return null;
      }

      const visiblePairwise = (entry.pairwise || [])
        .filter(
          (pair) =>
            pair &&
            indexMap.has(pair.a) &&
            indexMap.has(pair.b) &&
            typeof pair.distance === "number" &&
            Number.isFinite(pair.distance)
        )
        .map((pair) => ({
          ...pair,
          a: indexMap.get(pair.a),
          b: indexMap.get(pair.b),
        }));

      return {
        ...entry,
        visibleRepresentatives,
        visiblePairwise,
        visibleModelCount: visibleRepresentatives.length,
        fullVisible: visibleRepresentatives.length === (entry.representatives || []).length,
      };
    })
    .filter(Boolean);
}

function computeAverageLinkageDendrogram(leafCount, pairwiseEntries) {
  if (!Number.isInteger(leafCount) || leafCount < 2) {
    return [];
  }

  const distanceLookup = new Map();
  (pairwiseEntries || []).forEach((pair) => {
    if (
      !pair ||
      !Number.isInteger(pair.a) ||
      !Number.isInteger(pair.b) ||
      pair.a < 0 ||
      pair.b < 0 ||
      pair.a === pair.b ||
      typeof pair.distance !== "number" ||
      !Number.isFinite(pair.distance)
    ) {
      return;
    }
    const left = Math.min(pair.a, pair.b);
    const right = Math.max(pair.a, pair.b);
    distanceLookup.set(`${left}|${right}`, pair.distance);
  });

  const activeClusterIds = [];
  const clusterMembers = new Map();
  for (let leafIndex = 0; leafIndex < leafCount; leafIndex += 1) {
    activeClusterIds.push(leafIndex);
    clusterMembers.set(leafIndex, [leafIndex]);
  }

  const linkage = [];
  let nextClusterId = leafCount;

  const getClusterDistance = (leftClusterId, rightClusterId) => {
    const distances = [];
    const leftLeaves = clusterMembers.get(leftClusterId) || [];
    const rightLeaves = clusterMembers.get(rightClusterId) || [];
    leftLeaves.forEach((leftLeaf) => {
      rightLeaves.forEach((rightLeaf) => {
        const left = Math.min(leftLeaf, rightLeaf);
        const right = Math.max(leftLeaf, rightLeaf);
        const distance = distanceLookup.get(`${left}|${right}`);
        if (typeof distance === "number" && Number.isFinite(distance)) {
          distances.push(distance);
        }
      });
    });
    if (!distances.length) {
      return null;
    }
    return distances.reduce((sum, value) => sum + value, 0) / distances.length;
  };

  while (activeClusterIds.length > 1) {
    let bestPair = null;
    let bestSortKey = null;

    for (let leftIndex = 0; leftIndex < activeClusterIds.length; leftIndex += 1) {
      const leftClusterId = activeClusterIds[leftIndex];
      for (let rightIndex = leftIndex + 1; rightIndex < activeClusterIds.length; rightIndex += 1) {
        const rightClusterId = activeClusterIds[rightIndex];
        const distance = getClusterDistance(leftClusterId, rightClusterId);
        if (distance == null) {
          continue;
        }
        const sortKey = [distance, Math.min(leftClusterId, rightClusterId), Math.max(leftClusterId, rightClusterId)];
        if (
          !bestSortKey ||
          sortKey[0] < bestSortKey[0] ||
          (sortKey[0] === bestSortKey[0] &&
            (sortKey[1] < bestSortKey[1] || (sortKey[1] === bestSortKey[1] && sortKey[2] < bestSortKey[2])))
        ) {
          bestSortKey = sortKey;
          bestPair = [leftClusterId, rightClusterId];
        }
      }
    }

    if (!bestPair || !bestSortKey) {
      return null;
    }

    const [leftClusterId, rightClusterId] = bestPair;
    const mergedMembers = [...(clusterMembers.get(leftClusterId) || []), ...(clusterMembers.get(rightClusterId) || [])]
      .slice()
      .sort((a, b) => a - b);
    linkage.push([leftClusterId, rightClusterId, bestSortKey[0], mergedMembers.length]);
    clusterMembers.set(nextClusterId, mergedMembers);
    for (let index = activeClusterIds.length - 1; index >= 0; index -= 1) {
      if (activeClusterIds[index] === leftClusterId || activeClusterIds[index] === rightClusterId) {
        activeClusterIds.splice(index, 1);
      }
    }
    activeClusterIds.push(nextClusterId);
    nextClusterId += 1;
  }

  return linkage;
}

function createAgreementCell(label, primaryText, secondaryText = "") {
  const cell = document.createElement("td");
  cell.dataset.label = label;
  const primary = document.createElement("div");
  primary.className = "agreement-cell-primary";
  primary.textContent = primaryText || "N/A";
  cell.appendChild(primary);
  if (secondaryText) {
    const secondary = document.createElement("div");
    secondary.className = "agreement-cell-secondary muted";
    secondary.textContent = secondaryText;
    cell.appendChild(secondary);
  }
  return cell;
}

function createAgreementMetricCell(label, value) {
  const cell = document.createElement("td");
  cell.dataset.label = label;
  cell.className = "mono";
  cell.textContent = value == null ? "N/A" : formatNum(value, 3);
  return cell;
}

function createAgreementCountCell(label, value) {
  const cell = document.createElement("td");
  cell.dataset.label = label;
  cell.className = "mono";
  cell.textContent = value == null ? "N/A" : formatNum(value, 0);
  return cell;
}

function buildAgreementProviderModelLabel(provider, model) {
  const modelText = asTrimmedString(model);
  const providerText = asTrimmedString(provider);
  if (!providerText) {
    return modelText || "Unknown model";
  }
  return `${modelText || "Unknown model"} (${providerText})`;
}

function getAgreementClusterLeafOrder(leafCount, linkage) {
  if (!Number.isInteger(leafCount) || leafCount <= 0) {
    return [];
  }
  if (!Array.isArray(linkage) || !linkage.length) {
    return Array.from({ length: leafCount }, (_, index) => index);
  }

  const childrenByClusterId = new Map();
  linkage.forEach((step, index) => {
    if (!Array.isArray(step) || step.length < 4) {
      return;
    }
    childrenByClusterId.set(leafCount + index, {
      left: step[0],
      right: step[1],
    });
  });

  const rootClusterId = leafCount + linkage.length - 1;
  const orderedLeaves = [];
  const visit = (clusterId) => {
    if (clusterId < leafCount) {
      orderedLeaves.push(clusterId);
      return;
    }
    const children = childrenByClusterId.get(clusterId);
    if (!children) {
      return;
    }
    visit(children.left);
    visit(children.right);
  };
  visit(rootClusterId);
  if (orderedLeaves.length !== leafCount) {
    return Array.from({ length: leafCount }, (_, index) => index);
  }
  return orderedLeaves;
}

function createAgreementClusterSvg(entry, linkage, options = {}) {
  const leafCount = Array.isArray(entry.visibleRepresentatives) ? entry.visibleRepresentatives.length : 0;
  if (!leafCount || !Array.isArray(linkage) || linkage.length !== leafCount - 1) {
    return null;
  }

  const orderedLeafIndices = getAgreementClusterLeafOrder(leafCount, linkage);
  const leafYByIndex = new Map();
  const branchXByClusterId = new Map();
  const branchYByClusterId = new Map();

  const top = Number.isFinite(options.top) ? options.top : 20;
  const bottom = Number.isFinite(options.bottom) ? options.bottom : 42;
  const rowGap = Number.isFinite(options.rowGap) ? options.rowGap : 26;
  const width = Number.isFinite(options.width) ? options.width : 980;
  const leftLabelWidth = Number.isFinite(options.leftLabelWidth) ? options.leftLabelWidth : 370;
  const rightPadding = Number.isFinite(options.rightPadding) ? options.rightPadding : 32;
  const tickSteps = Number.isInteger(options.tickSteps) && options.tickSteps > 0 ? options.tickSteps : 4;
  const branchStartX = leftLabelWidth + 16;
  const branchWidth = Math.max(220, width - branchStartX - rightPadding);
  const labelOffsetX = 12;
  const labelX = branchStartX - labelOffsetX;
  const height = top + bottom + Math.max(leafCount - 1, 0) * rowGap;
  const maxDistance = linkage.reduce((max, step) => Math.max(max, safeNum(step[2]) || 0), 0);
  const normalizedMaxDistance = maxDistance > 0 ? maxDistance : 1;
  const toBranchX = (distance) => branchStartX + (Math.max(0, distance) / normalizedMaxDistance) * branchWidth;

  orderedLeafIndices.forEach((leafIndex, orderIndex) => {
    const y = top + orderIndex * rowGap;
    leafYByIndex.set(leafIndex, y);
    branchXByClusterId.set(leafIndex, branchStartX);
    branchYByClusterId.set(leafIndex, y);
  });

  const svg = createSvgNode("svg", {
    viewBox: `0 0 ${width} ${height}`,
    class: "agreement-cluster-svg",
    "aria-label": `${entry.taskNameDisplay || "Agreement"} similarity tree`,
  });
  const leafAreaBottomY = top + Math.max(leafCount - 1, 0) * rowGap;
  const tickLabelY = leafAreaBottomY + 16;

  for (let stepIndex = 0; stepIndex <= tickSteps; stepIndex += 1) {
    const tickDistance = (maxDistance * stepIndex) / tickSteps;
    const tickX = toBranchX(tickDistance);
    const similarity = Math.max(0, 1 - tickDistance);

    svg.appendChild(
      createSvgNode("line", {
        x1: tickX,
        x2: tickX,
        y1: top - 6,
        y2: leafAreaBottomY + 4,
        class: "agreement-cluster-grid",
      })
    );

    const tickLabel = createSvgNode("text", {
      x: tickX,
      y: tickLabelY,
      "text-anchor": "middle",
      class: "agreement-cluster-tick-label",
    });
    tickLabel.textContent = `${formatNum(similarity * 100, 0)}%`;
    svg.appendChild(tickLabel);
  }

  linkage.forEach((step, mergeIndex) => {
    const [leftClusterId, rightClusterId, distance, mergedLeafCount] = step;
    const clusterId = leafCount + mergeIndex;
    const leftX = branchXByClusterId.get(leftClusterId);
    const rightX = branchXByClusterId.get(rightClusterId);
    const leftY = branchYByClusterId.get(leftClusterId);
    const rightY = branchYByClusterId.get(rightClusterId);
    if (
      typeof leftX !== "number" ||
      typeof rightX !== "number" ||
      typeof leftY !== "number" ||
      typeof rightY !== "number"
    ) {
      return;
    }

    const mergeX = toBranchX(distance);
    const mergeTopY = Math.min(leftY, rightY);
    const mergeBottomY = Math.max(leftY, rightY);
    const mergeY = (leftY + rightY) / 2;

    svg.appendChild(
      createSvgNode("line", {
        x1: mergeX,
        x2: mergeX,
        y1: mergeTopY,
        y2: mergeBottomY,
        class: "agreement-cluster-branch",
      })
    );
    svg.appendChild(
      createSvgNode("line", {
        x1: leftX,
        x2: mergeX,
        y1: leftY,
        y2: leftY,
        class: "agreement-cluster-branch",
      })
    );
    svg.appendChild(
      createSvgNode("line", {
        x1: rightX,
        x2: mergeX,
        y1: rightY,
        y2: rightY,
        class: "agreement-cluster-branch",
      })
    );

    const mergeSimilarity = Math.max(0, 1 - Math.max(0, distance));
    const mergeDot = createSvgNode("circle", {
      cx: mergeX,
      cy: mergeY,
      r: 5.6,
      class: "agreement-cluster-merge-dot",
      tabindex: "0",
    });
    const mergeTitle = createSvgNode("title");
    mergeTitle.textContent = [
      `Merge similarity: ${formatNum(mergeSimilarity * 100, 1)}%`,
      `Disagreement distance: ${formatNum(distance, 3)}`,
      `Merged representatives: ${formatNum(mergedLeafCount, 0)}`,
    ].join("\n");
    mergeDot.appendChild(mergeTitle);
    svg.appendChild(mergeDot);

    branchXByClusterId.set(clusterId, mergeX);
    branchYByClusterId.set(clusterId, mergeY);
  });

  orderedLeafIndices.forEach((leafIndex) => {
    const representative = entry.visibleRepresentatives[leafIndex];
    const y = leafYByIndex.get(leafIndex);
    const style = getModelSeriesStyle(representative.model);
    svg.appendChild(
      createSvgNode("circle", {
        cx: branchStartX,
        cy: y,
        r: 4.5,
        class: "agreement-cluster-leaf-dot",
        fill: style.color,
      })
    );

    const label = createSvgNode("text", {
      x: labelX,
      y: y + 4,
      "text-anchor": "end",
      class: "agreement-cluster-label",
    });
    const labelParts = [buildAgreementProviderModelLabel(representative.provider, representative.model)];
    if (representative.cohenKappa != null) {
      labelParts.push(`κ ${formatNum(representative.cohenKappa, 3)}`);
    }
    label.textContent = labelParts.join(" | ");
    svg.appendChild(label);
  });

  const axisLabel = createSvgNode("text", {
    x: branchStartX + branchWidth / 2,
    y: height - 4,
    "text-anchor": "middle",
    class: "agreement-cluster-axis-label",
  });
  axisLabel.textContent = "Similarity at cut (1 - disagreement distance)";
  svg.appendChild(axisLabel);

  return svg;
}

function createCrossModelClusterCard(entry) {
  const card = document.createElement("section");
  card.className = "agreement-cluster-card";

  const heading = document.createElement("h4");
  heading.textContent = entry.taskNameDisplay || "Cross-Model Similarity";
  card.appendChild(heading);

  const subtitle = document.createElement("p");
  subtitle.className = "agreement-cluster-subtitle muted";
  const subsetNote = entry.fullVisible
    ? `${formatNum(entry.visibleModelCount, 0)} visible model(s).`
    : `Recomputed from ${formatNum(entry.visibleModelCount, 0)} visible of ${formatNum(entry.modelCount, 0)} total models.`;
  subtitle.textContent = [
    entry.tagsDisplay,
    subsetNote,
    "Distance = disagreement rate on overlapping rated items.",
    "Cut labels show similarity at that threshold.",
    "Hover merge dots for exact merge similarity.",
    "Label suffix = representative run Cohen's kappa."
  ]
    .filter(Boolean)
    .join(" ");
  card.appendChild(subtitle);

  const linkage = computeAverageLinkageDendrogram(entry.visibleRepresentatives.length, entry.visiblePairwise);
  if (!Array.isArray(linkage) || linkage.length !== entry.visibleRepresentatives.length - 1) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "Unable to build a similarity tree for the visible representatives because the pairwise distance graph is incomplete.";
    card.appendChild(empty);
    return card;
  }

  const svg = createAgreementClusterSvg(entry, linkage);
  if (!svg) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "Unable to build a similarity tree for this group.";
    card.appendChild(empty);
    return card;
  }
  card.appendChild(svg);
  return card;
}

function createAgreementTreeButton(clusterEntry) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "btn icon-btn agreement-tree-btn";
  button.title = "Open similarity tree";
  button.setAttribute("aria-label", "Open similarity tree");
  button.innerHTML = `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M6 4v8"></path>
      <path d="M6 12h6"></path>
      <path d="M12 12v8"></path>
      <path d="M12 20h6"></path>
      <circle cx="6" cy="4" r="1.7"></circle>
      <circle cx="6" cy="12" r="1.7"></circle>
      <circle cx="12" cy="20" r="1.7"></circle>
      <circle cx="18" cy="20" r="1.7"></circle>
    </svg>
  `;
  button.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    openAgreementClusterModal(clusterEntry);
  });
  return button;
}

function createCrossModelClustersSection(entries) {
  const section = document.createElement("section");
  section.className = "agreement-section agreement-clusters-section";

  const heading = document.createElement("h4");
  heading.textContent = "Similarity Trees";
  section.appendChild(heading);

  entries.forEach((entry) => {
    section.appendChild(createCrossModelClusterCard(entry));
  });

  return section;
}

function createAgreementTable(title, entries, mode, options = {}) {
  const section = document.createElement("section");
  section.className = "agreement-section";

  const heading = document.createElement("h4");
  heading.textContent = title;
  section.appendChild(heading);

  const wrap = document.createElement("div");
  wrap.className = "table-wrap agreement-table-wrap";
  const table = document.createElement("table");
  table.className = "agreement-table";
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  const columns =
    mode === "repeat"
      ? ["Task", "Model", "Runs", "Repeat α", "Pairable Items"]
      : ["Task", "Models", "Cross-Model α", "Pairable Items", "Representatives"];
  if (mode !== "repeat") {
    columns.push("Tree");
  }
  columns.forEach((labelText) => {
    const th = document.createElement("th");
    th.textContent = labelText;
    if (labelText === "Tree") {
      th.className = "agreement-tree-col";
    }
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  entries.forEach((entry) => {
    const tr = document.createElement("tr");
    tr.appendChild(createAgreementCell("Task", entry.taskNameDisplay, entry.tagsDisplay));

    if (mode === "repeat") {
      tr.appendChild(
        createAgreementCell(
          "Model",
          buildAgreementProviderModelLabel(entry.provider, entry.model),
          entry.taskNamesSeen && entry.taskNamesSeen.length > 1
            ? `Names seen: ${entry.taskNamesSeen.join(" | ")}`
            : ""
        )
      );
      tr.appendChild(createAgreementCountCell("Runs", entry.runCount));
      tr.appendChild(createAgreementMetricCell("Repeat α", entry.alphaNominal));
      tr.appendChild(createAgreementCountCell("Pairable Items", entry.pairableItemCount));
    } else {
      const clusterEntry =
        options.clusterEntriesByGroupId instanceof Map ? options.clusterEntriesByGroupId.get(entry.groupId) || null : null;
      tr.appendChild(createAgreementCountCell("Models", entry.modelCount));
      tr.appendChild(createAgreementMetricCell("Cross-Model α", entry.alphaNominal));
      tr.appendChild(createAgreementCountCell("Pairable Items", entry.pairableItemCount));
      tr.appendChild(
        createAgreementCell(
          "Representatives",
          (entry.representatives || [])
            .map((representative) =>
              buildAgreementProviderModelLabel(representative.provider, representative.model)
            )
            .join(", "),
          (entry.representatives || [])
            .map((representative) => {
              const parts = [representative.model || "Unknown model"];
              if (representative.accuracy != null) {
                parts.push(`${formatNum(representative.accuracy, 2)}%`);
              }
              if (representative.timestamp) {
                parts.push(formatDateOnly(representative.timestamp));
              }
              return parts.join(" | ");
            })
            .join(" ; ")
        )
      );
      const actionCell = document.createElement("td");
      actionCell.className = "agreement-tree-col";
      actionCell.dataset.label = "Tree";
      if (clusterEntry) {
        actionCell.appendChild(createAgreementTreeButton(clusterEntry));
      } else {
        actionCell.innerHTML = '<span class="muted">N/A</span>';
      }
      tr.appendChild(actionCell);
    }

    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  wrap.appendChild(table);
  section.appendChild(wrap);
  return section;
}

function createAgreementTableV2(title, entries, mode, options = {}) {
  const section = document.createElement("section");
  section.className = "agreement-section";

  const heading = document.createElement("h4");
  heading.textContent = title;
  section.appendChild(heading);

  const wrap = document.createElement("div");
  wrap.className = "table-wrap agreement-table-wrap";
  const table = document.createElement("table");
  table.className = "agreement-table";
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  const columns =
    mode === "repeat"
      ? ["Task", "Model", "Runs", "Repeat α", "Pairable Items"]
      : ["Tree", "Task", "Models", "Cross-Model α", "Pairable Items", "Representatives"];
  columns.forEach((labelText) => {
    const th = document.createElement("th");
    th.textContent = labelText;
    if (labelText === "Tree") {
      th.className = "agreement-tree-col";
    }
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  entries.forEach((entry) => {
    const tr = document.createElement("tr");

    if (mode === "repeat") {
      tr.appendChild(createAgreementCell("Task", entry.taskNameDisplay, entry.tagsDisplay));
      tr.appendChild(
        createAgreementCell(
          "Model",
          buildAgreementProviderModelLabel(entry.provider, entry.model),
          entry.taskNamesSeen && entry.taskNamesSeen.length > 1
            ? `Names seen: ${entry.taskNamesSeen.join(" | ")}`
            : ""
        )
      );
      tr.appendChild(createAgreementCountCell("Runs", entry.runCount));
      tr.appendChild(createAgreementMetricCell("Repeat α", entry.alphaNominal));
      tr.appendChild(createAgreementCountCell("Pairable Items", entry.pairableItemCount));
    } else {
      const clusterEntry =
        options.clusterEntriesByGroupId instanceof Map ? options.clusterEntriesByGroupId.get(entry.groupId) || null : null;
      const actionCell = document.createElement("td");
      actionCell.className = "agreement-tree-col";
      actionCell.dataset.label = "Tree";
      if (clusterEntry) {
        actionCell.appendChild(createAgreementTreeButton(clusterEntry));
      } else {
        actionCell.innerHTML = '<span class="muted">N/A</span>';
      }
      tr.appendChild(actionCell);
      tr.appendChild(createAgreementCell("Task", entry.taskNameDisplay, entry.tagsDisplay));
      tr.appendChild(createAgreementCountCell("Models", entry.modelCount));
      tr.appendChild(createAgreementMetricCell("Cross-Model α", entry.alphaNominal));
      tr.appendChild(createAgreementCountCell("Pairable Items", entry.pairableItemCount));
      tr.appendChild(
        createAgreementCell(
          "Representatives",
          (entry.representatives || [])
            .map((representative) =>
              buildAgreementProviderModelLabel(representative.provider, representative.model)
            )
            .join(", "),
          (entry.representatives || [])
            .map((representative) => {
              const parts = [representative.model || "Unknown model"];
              if (representative.accuracy != null) {
                parts.push(`${formatNum(representative.accuracy, 2)}%`);
              }
              if (representative.timestamp) {
                parts.push(formatDateOnly(representative.timestamp));
              }
              return parts.join(" | ");
            })
            .join(" ; ")
        )
      );
    }

    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  wrap.appendChild(table);
  section.appendChild(wrap);
  return section;
}

function renderAgreementControls(container) {
  if (!container) {
    return;
  }

  const controls = document.createElement("div");
  controls.className = "agreement-controls";

  const modeToggle = createTimeSeriesSegmentedControl("Agreement mode", [
    {
      label: "Same model",
      active: state.agreementViewMode === "same_model",
      onClick: () => setAgreementViewMode("same_model"),
    },
    {
      label: "Cross-Model",
      active: state.agreementViewMode === "cross_model",
      onClick: () => setAgreementViewMode("cross_model"),
    },
  ]);
  modeToggle.classList.add("agreement-mode-toggle");
  controls.appendChild(modeToggle);

  if (state.agreementViewMode === "cross_model") {
    const representativeControls = document.createElement("div");
    representativeControls.className = "agreement-control-group";
    const representativeLabel = document.createElement("span");
    representativeLabel.className = "agreement-control-label";
    representativeLabel.textContent = "Compare by";
    representativeControls.appendChild(representativeLabel);
    const representativeToggle = createTimeSeriesSegmentedControl("Cross-model representative policy", [
      {
        label: AGREEMENT_REPRESENTATIVE_POLICY_LABELS.latest,
        active: state.agreementRepresentativePolicy === "latest",
        onClick: () => setAgreementRepresentativePolicy("latest"),
      },
      {
        label: AGREEMENT_REPRESENTATIVE_POLICY_LABELS.best_accuracy,
        active: state.agreementRepresentativePolicy === "best_accuracy",
        onClick: () => setAgreementRepresentativePolicy("best_accuracy"),
      },
    ]);
    representativeToggle.classList.add("agreement-mode-toggle");
    representativeControls.appendChild(representativeToggle);
    controls.appendChild(representativeControls);
  }

  container.appendChild(controls);
}

function renderAgreement(container, runs) {
  if (!container) {
    return;
  }
  container.innerHTML = "";

  if (!state.agreementSummary) {
    container.innerHTML =
      '<p class="muted">Agreement summary not loaded. Recalculate metrics locally so <code>agreement_summary.json</code> is published with the metrics artifacts.</p>';
    return;
  }

  renderAgreementControls(container);

  const showCrossModel = state.agreementViewMode === "cross_model";
  const repeatEntries = showCrossModel ? [] : getVisibleRepeatAgreementEntries(runs);
  const crossEntries = showCrossModel
    ? getVisibleCrossModelAgreementEntries(runs, state.agreementRepresentativePolicy)
    : [];
  const clusterEntries = showCrossModel
    ? getVisibleCrossModelClusterEntries(runs, state.agreementRepresentativePolicy)
    : [];
  const clusterEntriesByGroupId = new Map((clusterEntries || []).map((entry) => [entry.groupId, entry]));

  const meta = document.createElement("p");
  meta.className = "agreement-note muted";
  const generatedText = state.agreementSummary.generatedAt
    ? ` Generated ${formatTs(state.agreementSummary.generatedAt)}.`
    : "";
  meta.textContent =
    showCrossModel
      ? `Cross-model agreement uses one representative run per provider/model (${AGREEMENT_REPRESENTATIVE_POLICY_LABELS[state.agreementRepresentativePolicy]}). Similarity trees are recomputed from the currently visible representatives.${generatedText}`
      : `Same-model agreement uses all repeated runs inside a comparable task variant.${generatedText}`;
  container.appendChild(meta);

  if (!repeatEntries.length && !crossEntries.length && !clusterEntries.length) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent =
      "No agreement groups are fully represented in the current filter. Expand the visible run set or switch the representative policy.";
    container.appendChild(empty);
    return;
  }

  if (repeatEntries.length) {
    container.appendChild(
      createAgreementTableV2("Same model", repeatEntries, "repeat")
    );
  }
  if (crossEntries.length) {
    container.appendChild(
      createAgreementTableV2("Cross-Model", crossEntries, "cross", { clusterEntriesByGroupId })
    );
  }
  if (showCrossModel && !crossEntries.length && clusterEntries.length) {
    const note = document.createElement("p");
    note.className = "muted";
    note.textContent =
      "No full cross-model alpha groups are fully represented in the current filter. The similarity trees below still use the visible representatives.";
    container.appendChild(note);
  }
  if (clusterEntries.length) {
    container.appendChild(createCrossModelClustersSection(clusterEntries));
  }
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
        { metricKey }
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

function getPredictionCountForRun(run) {
  return run.predictionCount ?? run.totalExamples ?? run.truthLabelCount ?? 0;
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
const TIME_SERIES_MODEL_OUTLINE_COLORS = [
  "#93c5fd",
  "#fca5a5",
  "#fde047",
  "#86efac",
  "#c4b5fd",
  "#fb7185",
  "#67e8f9",
  "#fdba74",
  "#bef264",
  "#818cf8",
  "#f0abfc",
  "#6ee7b7",
  "#f87171",
  "#f59e0b",
  "#2dd4bf",
  "#d8b4fe",
  "#60a5fa",
  "#fb923c",
  "#84cc16",
  "#e879f9",
  "#38bdf8",
  "#ef4444",
  "#facc15",
  "#22c55e",
];
const TIME_SERIES_MODEL_SHAPES = ["circle", "square", "diamond", "triangle", "triangle_down", "hexagon", "pentagon", "octagon"];
const SELECTED_TAG_BADGE_COLORS = ["#6ea8ff", "#50e3c2", "#ffb36b", "#f472b6", "#facc15", "#22d3ee", "#fb7185", "#4ade80"];
let cachedModelSeriesStyleUniverseKey = "";
let cachedModelSeriesStyleMap = new Map();

function hashOrdinalKey(value) {
  const text = String(value || "");
  let hash = 2166136261;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function getStableOrdinalIndex(value, paletteSize, universe = [], namespace = "") {
  if (!Number.isFinite(paletteSize) || paletteSize <= 0) {
    return 0;
  }
  const normalized = asTrimmedString(value);
  if (!normalized) {
    return 0;
  }
  if (Array.isArray(universe) && universe.length) {
    const universeIndex = universe.findIndex((entry) => asTrimmedString(entry) === normalized);
    if (universeIndex >= 0) {
      return universeIndex % paletteSize;
    }
  }
  return hashOrdinalKey(`${namespace}:${normalized}`) % paletteSize;
}

function getTaskSeriesColor(taskName) {
  return TIME_SERIES_TASK_COLORS[
    getStableOrdinalIndex(taskName, TIME_SERIES_TASK_COLORS.length, state.tasks, "task-color")
  ] || TIME_SERIES_TASK_COLORS[0];
}

function buildModelSeriesStyleMap(universe = state.models) {
  const normalizedUniverse = uniqueNonEmptyStrings(universe);
  const universeKey = normalizedUniverse.join("\u001f");
  if (cachedModelSeriesStyleUniverseKey === universeKey) {
    return cachedModelSeriesStyleMap;
  }

  const shapeBuckets = new Map();
  normalizedUniverse.forEach((modelName) => {
    const shapeIndex = getStableOrdinalIndex(modelName, TIME_SERIES_MODEL_SHAPES.length, normalizedUniverse, "model-shape");
    if (!shapeBuckets.has(shapeIndex)) {
      shapeBuckets.set(shapeIndex, []);
    }
    shapeBuckets.get(shapeIndex).push(modelName);
  });

  const nextMap = new Map();
  shapeBuckets.forEach((bucketModels, shapeIndex) => {
    bucketModels.forEach((modelName, bucketIndex) => {
      const colorIndex = (shapeIndex * 5 + bucketIndex * 11) % TIME_SERIES_MODEL_OUTLINE_COLORS.length;
      nextMap.set(modelName, {
        color: TIME_SERIES_MODEL_OUTLINE_COLORS[colorIndex] || TIME_SERIES_MODEL_OUTLINE_COLORS[0],
        shape: TIME_SERIES_MODEL_SHAPES[shapeIndex] || TIME_SERIES_MODEL_SHAPES[0],
      });
    });
  });

  cachedModelSeriesStyleUniverseKey = universeKey;
  cachedModelSeriesStyleMap = nextMap;
  return cachedModelSeriesStyleMap;
}

function getModelSeriesStyle(modelName) {
  const normalized = asTrimmedString(modelName);
  if (!normalized) {
    return {
      color: TIME_SERIES_MODEL_OUTLINE_COLORS[0],
      shape: TIME_SERIES_MODEL_SHAPES[0],
    };
  }

  const styleMap = buildModelSeriesStyleMap(state.models);
  if (styleMap.has(normalized)) {
    return styleMap.get(normalized);
  }

  const shapeIndex = getStableOrdinalIndex(normalized, TIME_SERIES_MODEL_SHAPES.length, [], "model-shape");
  const colorOffset = getStableOrdinalIndex(normalized, TIME_SERIES_MODEL_OUTLINE_COLORS.length, [], "model-color");
  const colorIndex = (shapeIndex * 5 + colorOffset * 11) % TIME_SERIES_MODEL_OUTLINE_COLORS.length;
  return {
    color: TIME_SERIES_MODEL_OUTLINE_COLORS[colorIndex] || TIME_SERIES_MODEL_OUTLINE_COLORS[0],
    shape: TIME_SERIES_MODEL_SHAPES[shapeIndex] || TIME_SERIES_MODEL_SHAPES[0],
  };
}

function getModelSeriesColor(modelName) {
  return getModelSeriesStyle(modelName).color;
}

function getModelSeriesShape(modelName) {
  return getModelSeriesStyle(modelName).shape;
}

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
      const predictions = toNonNegativeNumber(getPredictionCountForRun(run));
      const inputTotal = toNonNegativeNumber(run.inputTokensTotal);
      const cachedTotal = toNonNegativeNumber(run.cachedInputTokensTotal ?? run.cachedTokens);
      const outputTotal = toNonNegativeNumber(run.outputTokensTotal);
      const thinkingTotal = toNonNegativeNumber(run.thinkingTokensTotal);
      const avgInput = averageOrZero(inputTotal, predictions);
      const avgCached = averageOrZero(cachedTotal, predictions);
      const avgOutput = averageOrZero(outputTotal, predictions);
      const avgThinking = averageOrZero(thinkingTotal, predictions);
      const estimatedCostUsd = safeNum(run.estimatedCostUsd);
      return {
        run,
        model: run.model || "unknown",
        modelDisplay: getRunModelDisplayName(run),
        task: run.task || "unknown",
        predictions,
        inputTotal,
        cachedTotal,
        outputTotal,
        thinkingTotal,
        avgInput,
        avgCached,
        avgOutput,
        avgThinking,
        totalAvg: avgInput + avgCached + avgOutput + avgThinking,
        estimatedCostUsd,
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

function truncateRadarAxisLabel(label, maxChars = 22) {
  const text = String(label || "").replace(/\s+/g, " ").trim();
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

  if (metricKey === "cohen_kappa") {
    const clamped = Math.max(-1, Math.min(1, rawValue));
    const linearRatio = (clamped + 1) / 2;
    if (scaleMode === "contrast") {
      const gamma = 2.5;
      return Math.pow(linearRatio, gamma);
    }
    return linearRatio;
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
    if (metricKey === "cohen_kappa") {
      return 1;
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
    text.textContent = truncateRadarAxisLabel(task);
    const title = document.createElementNS(ns, "title");
    title.textContent = String(task || "");
    text.appendChild(title);
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

  const scaleControls = document.createElement("div");
  scaleControls.className = "radar-control-group";
  const scaleLabelEl = document.createElement("span");
  scaleLabelEl.className = "radar-control-label";
  scaleLabelEl.textContent = "Scale";
  scaleControls.appendChild(scaleLabelEl);

  const scaleToggle = createTimeSeriesSegmentedControl("Radar scale mode", [
    {
      label: RADAR_SCALE_LABELS.linear,
      active: scaleMode === "linear",
      onClick: () => setRadarScale("linear"),
    },
    {
      label: RADAR_SCALE_LABELS.contrast,
      active: scaleMode === "contrast",
      onClick: () => setRadarScale("contrast"),
    },
  ]);
  scaleToggle.classList.add("radar-mode-toggle");
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
  const runRows = computeRunSignalRows(runs).filter((row) => row.predictions > 0 || row.totalAvg > 0);
  if (!runRows.length) {
    els.tokenChart.innerHTML = '<p class="muted">No token/request metadata for current filter.</p>';
    return;
  }

  const stackPanel = document.createElement("section");
  stackPanel.className = "token-stack-panel";
  const stackHeader = document.createElement("h4");
  stackHeader.className = "token-stack-header";
  stackHeader.textContent = "Average tokens per prediction by run";
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
    meta.textContent =
      `${row.task} | avg/prediction ${formatNum(row.totalAvg, 2)} | predictions ${formatNum(row.predictions, 0)} | ` +
      `~ ${formatUsd(row.estimatedCostUsd)}`;
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
      seg.title = `${segment.label}: ${formatNum(totalValue, 0)} total | ${formatNum(value, 2)} avg/prediction`;
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
        "Cohen's Kappa",
        run.cohenKappa !== null ? formatNum(run.cohenKappa, 3) : '<span class="muted">N/A</span>'
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
  const repeatAgreement = getRepeatAgreementForRun(run);
  const crossModelAgreement = getCrossModelAgreementForRun(run, state.agreementRepresentativePolicy);
  const runThinkingLevel = getConfiguredControl(run, "thinking_level");
  const runReasoningEffort =
    getConfiguredControl(run, "reasoning_effort") || getConfiguredControl(run, "effort");
  const representativePolicyLabel =
    AGREEMENT_REPRESENTATIVE_POLICY_LABELS[state.agreementRepresentativePolicy] ||
    state.agreementRepresentativePolicy;
  const detailPairs = [
    ["Task", run.task],
    ["Task Description", run.taskDescription],
    ["Tags", run.tagsDisplay],
    ["Model", run.model],
    ["Thinking Level", runThinkingLevel],
    ["Reasoning/Effort", runReasoningEffort],
    ["Timestamp", formatTs(run.timestamp)],
    ["Accuracy", run.accuracy == null ? "N/A" : `${formatNum(run.accuracy, 2)}%`],
    ["Cohen's Kappa", run.cohenKappa == null ? "N/A" : formatNum(run.cohenKappa, 3)],
    ["Repeat α", repeatAgreement && repeatAgreement.alphaNominal != null ? formatNum(repeatAgreement.alphaNominal, 3) : "N/A"],
    ["Repeat Runs", repeatAgreement ? formatNum(repeatAgreement.runCount, 0) : "N/A"],
    [
      `Cross-Model α (${representativePolicyLabel})`,
      crossModelAgreement && crossModelAgreement.alphaNominal != null
        ? formatNum(crossModelAgreement.alphaNominal, 3)
        : "N/A",
    ],
    ["Cross-Model Models", crossModelAgreement ? formatNum(crossModelAgreement.modelCount, 0) : "N/A"],
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
    ["Estimated Cost", formatUsd(run.estimatedCostUsd)],
    ["Pricing Tier", run.serviceTier || "standard"],
    ["Pricing Status", run.pricingStatus || "N/A"],
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

function fillAgreementClusterModalContent(entry) {
  if (!entry) {
    els.clusterModalTitle.textContent = "Similarity Tree";
    els.clusterModalMeta.textContent = "No cluster selected.";
    els.clusterModalContent.innerHTML = "";
    return;
  }

  const linkage = computeAverageLinkageDendrogram(entry.visibleRepresentatives.length, entry.visiblePairwise);
  const subsetNote = entry.fullVisible
    ? `${formatNum(entry.visibleModelCount, 0)} visible model(s).`
    : `Recomputed from ${formatNum(entry.visibleModelCount, 0)} visible of ${formatNum(entry.modelCount, 0)} total models.`;

  els.clusterModalTitle.textContent = entry.taskNameDisplay || "Similarity Tree";
  els.clusterModalMeta.textContent = [entry.tagsDisplay, subsetNote].filter(Boolean).join(" ");
  els.clusterModalContent.innerHTML = "";

  const note = document.createElement("p");
  note.className = "muted";
  note.textContent =
    "Branch distance is disagreement rate on overlapping rated items. Vertical cut labels show similarity as 1 minus that disagreement distance. Hover the merge dots for exact merge similarity. The κ value next to each run is that representative run's Cohen's kappa against ground truth.";
  els.clusterModalContent.appendChild(note);

  if (!Array.isArray(linkage) || linkage.length !== entry.visibleRepresentatives.length - 1) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent =
      "Unable to build a similarity tree for the visible representatives because the pairwise distance graph is incomplete.";
    els.clusterModalContent.appendChild(empty);
    return;
  }

  const svg = createAgreementClusterSvg(entry, linkage, {
    width: 1280,
    leftLabelWidth: 520,
    rightPadding: 36,
    rowGap: 30,
    bottom: 48,
  });
  if (!svg) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "Unable to build a similarity tree for this group.";
    els.clusterModalContent.appendChild(empty);
    return;
  }
  els.clusterModalContent.appendChild(svg);
}

function openAgreementClusterModal(entry) {
  if (!entry || !els.clusterModal) {
    return;
  }
  fillAgreementClusterModalContent(entry);
  els.clusterModal.classList.remove("hidden");
}

function closeAgreementClusterModal() {
  if (!els.clusterModal) {
    return;
  }
  els.clusterModal.classList.add("hidden");
}

function setupModalControls() {
  els.runModalClose.addEventListener("click", closeRunModal);
  els.runModal.addEventListener("click", (event) => {
    if (event.target === els.runModal) {
      closeRunModal();
    }
  });
  if (els.clusterModalClose) {
    els.clusterModalClose.addEventListener("click", closeAgreementClusterModal);
  }
  if (els.clusterModal) {
    els.clusterModal.addEventListener("click", (event) => {
      if (event.target === els.clusterModal) {
        closeAgreementClusterModal();
      }
    });
  }
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !els.runModal.classList.contains("hidden")) {
      closeRunModal();
    }
    if (event.key === "Escape" && els.clusterModal && !els.clusterModal.classList.contains("hidden")) {
      closeAgreementClusterModal();
    }
  });
}

function render() {
  state.filtered = getFilteredRuns();
  els.hideNoAccuracy.checked = state.hideNoAccuracy;
  renderFilterSearchControl();
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
  state.agreementSummary = result.agreementSummary || null;
  state.agreementClusters = result.agreementClusters || null;
  rebuildAgreementLookups(state.agreementSummary);
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

(function initMetricsAggregationApi(global) {
  "use strict";

  function asTrimmedString(value) {
    return typeof value === "string" ? value.trim() : "";
  }

  function isFiniteNumber(value) {
    return typeof value === "number" && Number.isFinite(value);
  }

  function averageFiniteNumbers(values) {
    const numeric = (Array.isArray(values) ? values : []).filter(isFiniteNumber);
    if (!numeric.length) {
      return null;
    }
    return numeric.reduce((sum, value) => sum + value, 0) / numeric.length;
  }

  // Any explicit temperature overrides the provider default, including 0 and 1.
  // Missing historical metadata is unknown; tags recover recorded sweeps.
  function temperatureOverride(payload) {
    const source = payload && typeof payload === "object" ? payload : {};
    const raw = source.rawMetrics || source;
    const config = source.runConfig || raw.run_config || {};
    const summary = source.controlSummary || raw.request_control_summary || {};
    const candidates = [
      ["run_config.temperature", config.temperature],
      ["request_control_summary.configured.temperature", summary.configured?.temperature],
      ["request_control_summary.per_control.temperature.configured_value", summary.per_control?.temperature?.configured_value],
      ["temperature", raw.temperature],
    ];
    for (const [sourceName, value] of candidates) {
      if (value == null || (typeof value === "string" && ["", "default", "null", "none"].includes(value.trim().toLowerCase()))) continue;
      const numeric = typeof value === "string" ? Number(value) : value;
      return { overridden: true, temperature: typeof numeric === "number" && Number.isFinite(numeric) ? numeric : value,
        source: sourceName, status: "explicit" };
    }
    const tags = config.tags || source.tags || raw.tags || "";
    const text = Array.isArray(tags) ? tags.join(";") : String(tags);
    const match = text.match(/(?:^|[;,\s])temp(?:erature)?[_=:]\s*(-?\d+(?:\.\d+)?)(?=$|[;,\s])/i);
    if (match) return { overridden: true, temperature: Number(match[1]), source: "tags", status: "explicit" };
    const known = Object.hasOwn(config, "temperature") || Object.hasOwn(summary.configured || {}, "temperature") || Object.hasOwn(raw, "temperature");
    return { overridden: false, temperature: null, source: "", status: known ? "unset" : "unknown" };
  }

  function buildBalancedAggregate(items, options = {}) {
    const source = Array.isArray(items) ? items : [];
    const getUnitKey = typeof options.getUnitKey === "function" ? options.getUnitKey : () => "";
    const getMetricValue =
      typeof options.getMetricValue === "function" ? options.getMetricValue : (item) => item;
    const getPriceValue = typeof options.getPriceValue === "function" ? options.getPriceValue : null;
    const units = new Map();

    source.forEach((item, index) => {
      const unitKey = asTrimmedString(getUnitKey(item, index));
      if (!units.has(unitKey)) {
        units.set(unitKey, []);
      }
      units.get(unitKey).push(item);
    });

    const summaries = Array.from(units.entries())
      .map(([key, unitItems]) => {
        const metricValue = averageFiniteNumbers(unitItems.map((item, index) => getMetricValue(item, index)));
        if (!isFiniteNumber(metricValue)) {
          return null;
        }
        const priceValues = getPriceValue
          ? unitItems.map((item, index) => getPriceValue(item, index)).filter(isFiniteNumber)
          : [];
        return {
          key,
          items: unitItems,
          metricValue,
          priceValue: priceValues.length ? averageFiniteNumbers(priceValues) : null,
          knownPriceCount: priceValues.length,
        };
      })
      .filter(Boolean);

    return {
      units: summaries,
      metricValue: averageFiniteNumbers(summaries.map((summary) => summary.metricValue)),
      priceValue: getPriceValue
        ? averageFiniteNumbers(summaries.map((summary) => summary.priceValue))
        : null,
      unitCount: summaries.length,
      pricedUnitCount: summaries.filter((summary) => isFiniteNumber(summary.priceValue)).length,
      itemCount: source.length,
    };
  }

  global.DHAIBenchMetricsAggregation = {
    buildBalancedAggregate,
    temperatureOverride,
  };
})(typeof globalThis !== "undefined" ? globalThis : this);

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
  };
})(typeof globalThis !== "undefined" ? globalThis : this);

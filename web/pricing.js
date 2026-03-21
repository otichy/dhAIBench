(function initPricingApi(global) {
  "use strict";

  const PROVIDER_ALIASES = {
    einfra: "e-infra",
  };
  const SERVICE_TIERS = new Set(["standard", "flex", "priority", "batch"]);

  function asTrimmedString(value) {
    return typeof value === "string" ? value.trim() : "";
  }

  function safeNum(value) {
    return typeof value === "number" && Number.isFinite(value) ? value : null;
  }

  function toNonNegativeNumber(value) {
    const numeric = safeNum(value);
    return numeric == null ? 0 : Math.max(numeric, 0);
  }

  function sanitizeModelIdentifier(value) {
    const normalized = asTrimmedString(value).toLowerCase().replace(/[^0-9a-z]+/g, "");
    return normalized || "model";
  }

  function normalizeProviderKey(value) {
    const normalized = asTrimmedString(value).toLowerCase();
    return PROVIDER_ALIASES[normalized] || normalized;
  }

  function normalizeServiceTier(value) {
    const normalized = asTrimmedString(value).toLowerCase();
    return SERVICE_TIERS.has(normalized) ? normalized : "standard";
  }

  function getPricingCatalog(explicitCatalog) {
    const catalog = explicitCatalog || global.MODEL_PRICING_CATALOG;
    return catalog && typeof catalog === "object" ? catalog : null;
  }

  function pushUnique(list, seen, value) {
    const normalized = asTrimmedString(value);
    if (!normalized || seen.has(normalized)) {
      return;
    }
    seen.add(normalized);
    list.push(normalized);
  }

  function buildLookupCandidates(run) {
    const candidates = [];
    const seen = new Set();
    const values = [
      run && run.modelDetails ? run.modelDetails.model_requested : "",
      run && run.modelDetails ? run.modelDetails.model_for_requests : "",
      run ? run.model : "",
      run ? run.fileModelSlug : "",
    ];

    values.forEach((value) => {
      const trimmed = asTrimmedString(value);
      if (!trimmed) {
        return;
      }
      pushUnique(candidates, seen, trimmed);

      if (trimmed.includes("/models/")) {
        const modelTail = trimmed.split("/models/").pop();
        pushUnique(candidates, seen, modelTail);
      }

      if (trimmed.startsWith("models/")) {
        pushUnique(candidates, seen, trimmed.slice("models/".length));
      }

      if (trimmed.includes("/") && !(run && normalizeProviderKey(run.provider) === "requesty")) {
        pushUnique(candidates, seen, trimmed.split("/").pop());
      }

      pushUnique(candidates, seen, sanitizeModelIdentifier(trimmed));
    });

    return candidates;
  }

  function resolveEntryForKey(providerModels, key) {
    const trail = [];
    let currentKey = key;
    let safety = 0;
    while (providerModels && providerModels[currentKey] && safety < 12) {
      const entry = providerModels[currentKey];
      trail.push(currentKey);
      if (!entry || typeof entry !== "object") {
        return null;
      }
      if (entry.status !== "alias") {
        return {
          entry,
          resolvedKey: currentKey,
          aliasTrail: trail,
        };
      }
      currentKey = asTrimmedString(entry.pricing_ref);
      if (!currentKey) {
        return null;
      }
      safety += 1;
    }
    return null;
  }

  function uniqueProviderMatch(catalog, candidates) {
    const providers = catalog && catalog.providers && typeof catalog.providers === "object" ? catalog.providers : {};
    const matches = [];
    Object.entries(providers).forEach(([providerKey, providerEntry]) => {
      const models = providerEntry && providerEntry.models && typeof providerEntry.models === "object"
        ? providerEntry.models
        : {};
      candidates.forEach((candidate) => {
        if (Object.prototype.hasOwnProperty.call(models, candidate)) {
          const resolved = resolveEntryForKey(models, candidate);
          if (resolved) {
            matches.push({
              providerKey,
              matchedKey: candidate,
              resolvedKey: resolved.resolvedKey,
              entry: resolved.entry,
            });
          }
        }
      });
    });

    const uniqueResolved = new Map();
    matches.forEach((match) => {
      uniqueResolved.set(`${match.providerKey}::${match.resolvedKey}`, match);
    });
    return uniqueResolved.size === 1 ? [...uniqueResolved.values()][0] : null;
  }

  function resolvePricingMatch(catalog, run, candidates) {
    const providers = catalog && catalog.providers && typeof catalog.providers === "object" ? catalog.providers : {};
    const preferredProvider = normalizeProviderKey(
      (run && run.provider) || (run && run.modelDetails ? run.modelDetails.provider : "")
    );

    if (preferredProvider && providers[preferredProvider]) {
      const providerModels = providers[preferredProvider].models || {};
      for (const candidate of candidates) {
        if (!Object.prototype.hasOwnProperty.call(providerModels, candidate)) {
          continue;
        }
        const resolved = resolveEntryForKey(providerModels, candidate);
        if (resolved) {
          return {
            providerKey: preferredProvider,
            matchedKey: candidate,
            resolvedKey: resolved.resolvedKey,
            entry: resolved.entry,
          };
        }
      }
    }

    if (preferredProvider === "requesty") {
      return null;
    }

    return uniqueProviderMatch(catalog, candidates);
  }

  function statusLabel(status) {
    switch (status) {
      case "priced":
        return "priced";
      case "unsupported":
        return "unsupported model";
      case "tier_unavailable":
        return "tier unavailable";
      case "rate_missing":
        return "missing rate";
      case "ambiguous":
        return "ambiguous match";
      case "catalog_missing":
        return "catalog unavailable";
      case "model_missing":
      default:
        return "no pricing match";
    }
  }

  function estimateRunCost(catalogInput, run) {
    const catalog = getPricingCatalog(catalogInput);
    const serviceTier = normalizeServiceTier(
      run && run.rawMetrics && run.rawMetrics.run_config ? run.rawMetrics.run_config.service_tier : run && run.serviceTier
    );
    const inputTotal = toNonNegativeNumber(run ? run.inputTokensTotal : null);
    const cachedTotal = toNonNegativeNumber(run ? (run.cachedInputTokensTotal != null ? run.cachedInputTokensTotal : run.cachedTokens) : null);
    const nonCachedInputTotal = run && safeNum(run.nonCachedInputTokensTotal) != null
      ? toNonNegativeNumber(run.nonCachedInputTokensTotal)
      : Math.max(inputTotal - cachedTotal, 0);
    const outputTotal = toNonNegativeNumber(run ? run.outputTokensTotal : null);

    const baseResult = {
      pricingTier: serviceTier,
      estimatedCostUsd: null,
      status: "model_missing",
      statusLabel: statusLabel("model_missing"),
      providerKey: "",
      matchedKey: "",
      resolvedKey: "",
      missingBuckets: [],
    };

    if (!catalog) {
      return {
        ...baseResult,
        status: "catalog_missing",
        statusLabel: statusLabel("catalog_missing"),
      };
    }

    const candidates = buildLookupCandidates(run);
    const match = resolvePricingMatch(catalog, run, candidates);
    if (!match) {
      return baseResult;
    }

    if (!match.entry || match.entry.status === "unsupported") {
      return {
        ...baseResult,
        providerKey: match.providerKey,
        matchedKey: match.matchedKey,
        resolvedKey: match.resolvedKey,
        status: "unsupported",
        statusLabel: statusLabel("unsupported"),
      };
    }

    const serviceTiers = match.entry.service_tiers && typeof match.entry.service_tiers === "object"
      ? match.entry.service_tiers
      : {};
    const tierRates = serviceTiers[serviceTier];
    if (!tierRates) {
      return {
        ...baseResult,
        providerKey: match.providerKey,
        matchedKey: match.matchedKey,
        resolvedKey: match.resolvedKey,
        status: "tier_unavailable",
        statusLabel: statusLabel("tier_unavailable"),
      };
    }

    const missingBuckets = [];
    if (nonCachedInputTotal > 0 && safeNum(tierRates.input_usd_per_mtokens) == null) {
      missingBuckets.push("input");
    }
    if (cachedTotal > 0 && safeNum(tierRates.cached_input_usd_per_mtokens) == null) {
      missingBuckets.push("cached_input");
    }
    if (outputTotal > 0 && safeNum(tierRates.output_usd_per_mtokens) == null) {
      missingBuckets.push("output");
    }
    if (missingBuckets.length) {
      return {
        ...baseResult,
        providerKey: match.providerKey,
        matchedKey: match.matchedKey,
        resolvedKey: match.resolvedKey,
        status: "rate_missing",
        statusLabel: statusLabel("rate_missing"),
        missingBuckets,
      };
    }

    const estimatedCostUsd =
      ((nonCachedInputTotal * toNonNegativeNumber(tierRates.input_usd_per_mtokens)) +
        (cachedTotal * toNonNegativeNumber(tierRates.cached_input_usd_per_mtokens)) +
        (outputTotal * toNonNegativeNumber(tierRates.output_usd_per_mtokens))) /
      1000000;

    return {
      ...baseResult,
      providerKey: match.providerKey,
      matchedKey: match.matchedKey,
      resolvedKey: match.resolvedKey,
      status: "priced",
      statusLabel: statusLabel("priced"),
      estimatedCostUsd,
    };
  }

  global.DHAIBenchPricing = {
    buildLookupCandidates,
    estimateRunCost,
    getPricingCatalog,
    normalizeProviderKey,
    normalizeServiceTier,
    sanitizeModelIdentifier,
  };
})(typeof window !== "undefined" ? window : globalThis);

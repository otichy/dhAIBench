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

  function getServiceTiers(entry) {
    return entry && entry.service_tiers && typeof entry.service_tiers === "object"
      ? entry.service_tiers
      : {};
  }

  function hasPricingRef(entry) {
    return !!asTrimmedString(entry && entry.pricing_ref);
  }

  function hasReason(entry) {
    return !!asTrimmedString(entry && entry.reason);
  }

  function hasAnyUsableRates(entry) {
    return Object.values(getServiceTiers(entry)).some((tierRates) => {
      if (!tierRates || typeof tierRates !== "object") {
        return false;
      }
      return (
        safeNum(tierRates.input_usd_per_mtokens) != null ||
        safeNum(tierRates.cached_input_usd_per_mtokens) != null ||
        safeNum(tierRates.output_usd_per_mtokens) != null
      );
    });
  }

  function classifyCatalogEntry(entry) {
    if (!entry || typeof entry !== "object") {
      return "missing";
    }
    if (hasPricingRef(entry)) {
      return "alias";
    }
    if (hasAnyUsableRates(entry)) {
      return "priced";
    }
    if (entry.needs_manual_update === true) {
      return "unpriced";
    }
    if (hasReason(entry)) {
      return "unsupported";
    }
    return "unpriced";
  }

  function pushUnique(list, seen, value) {
    const normalized = asTrimmedString(value);
    if (!normalized || seen.has(normalized)) {
      return;
    }
    seen.add(normalized);
    list.push(normalized);
  }

  function buildLookupCandidates(run, options = {}) {
    const includeFileModelSlug = options.includeFileModelSlug !== false;
    const candidates = [];
    const seen = new Set();
    const values = [
      run && run.modelDetails ? run.modelDetails.model_requested : "",
      run && run.modelDetails ? run.modelDetails.model_for_requests : "",
      run ? run.model : "",
      includeFileModelSlug && run ? run.fileModelSlug : "",
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

  function getRequestyRoutePriority(modelKey) {
    const prefix = asTrimmedString(modelKey).split("/", 1)[0].toLowerCase();
    switch (prefix) {
      case "policy":
        return 100;
      case "bedrock":
        return 40;
      case "vertex":
        return 30;
      case "openai-responses":
        return 20;
      default:
        return 0;
    }
  }

  function deriveRequestyPrimaryRouteCandidates(providerModels, candidate) {
    const trimmed = asTrimmedString(candidate);
    if (!trimmed || trimmed.includes("/")) {
      return [];
    }

    const matches = Object.keys(providerModels || {})
      .map((modelKey) => {
        const normalizedKey = asTrimmedString(modelKey);
        if (!normalizedKey.includes("/")) {
          return null;
        }
        const tail = normalizedKey.split("/", 2)[1];
        if (!tail) {
          return null;
        }
        const tailBase = tail.split("@", 1)[0];
        if (tail !== trimmed && tailBase !== trimmed) {
          return null;
        }
        const resolved = resolveEntryForKey(providerModels, normalizedKey);
        if (!resolved || classifyCatalogEntry(resolved.entry) !== "priced") {
          return null;
        }
        return {
          key: normalizedKey,
          resolvedKey: resolved.resolvedKey,
        };
      })
      .filter(Boolean);

    if (!matches.length) {
      return [];
    }

    matches.sort((left, right) => {
      const priorityDiff = getRequestyRoutePriority(left.resolvedKey) - getRequestyRoutePriority(right.resolvedKey);
      if (priorityDiff !== 0) {
        return priorityDiff;
      }
      const leftRegionPenalty = left.resolvedKey.includes("@") ? 1 : 0;
      const rightRegionPenalty = right.resolvedKey.includes("@") ? 1 : 0;
      if (leftRegionPenalty !== rightRegionPenalty) {
        return leftRegionPenalty - rightRegionPenalty;
      }
      return left.resolvedKey.localeCompare(right.resolvedKey);
    });

    const best = matches[0];
    const bestPriority = getRequestyRoutePriority(best.resolvedKey);
    const bestRegionPenalty = best.resolvedKey.includes("@") ? 1 : 0;
    const competingBest = matches.filter((entry) => {
      return (
        getRequestyRoutePriority(entry.resolvedKey) === bestPriority &&
        (entry.resolvedKey.includes("@") ? 1 : 0) === bestRegionPenalty
      );
    });
    if (competingBest.length !== 1) {
      return [];
    }

    return [best.key, best.resolvedKey, sanitizeModelIdentifier(best.resolvedKey)];
  }

  function buildProviderScopedCandidates(providerKey, candidate, providerModels) {
    const scoped = [];
    const seen = new Set();
    const trimmed = asTrimmedString(candidate);
    if (!trimmed) {
      return scoped;
    }

    pushUnique(scoped, seen, trimmed);

    if (normalizeProviderKey(providerKey) === "google" && !trimmed.startsWith("models/")) {
      const prefixed = `models/${trimmed}`;
      pushUnique(scoped, seen, prefixed);
      pushUnique(scoped, seen, sanitizeModelIdentifier(prefixed));
    }

    if (normalizeProviderKey(providerKey) === "requesty") {
      deriveRequestyPrimaryRouteCandidates(providerModels, trimmed).forEach((value) => {
        pushUnique(scoped, seen, value);
      });
    }

    return scoped;
  }

  function resolveProviderMatch(providerKey, providerModels, candidates) {
    for (const candidate of candidates) {
      const scopedCandidates = buildProviderScopedCandidates(providerKey, candidate, providerModels);
      for (const scopedCandidate of scopedCandidates) {
        if (!Object.prototype.hasOwnProperty.call(providerModels, scopedCandidate)) {
          continue;
        }
        const resolved = resolveEntryForKey(providerModels, scopedCandidate);
        if (resolved) {
          return {
            providerKey,
            matchedKey: scopedCandidate,
            resolvedKey: resolved.resolvedKey,
            entry: resolved.entry,
          };
        }
      }
    }
    return null;
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
      if (!hasPricingRef(entry)) {
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

  function resolvePricingMatch(catalog, run, candidates, options = {}) {
    const providers = catalog && catalog.providers && typeof catalog.providers === "object" ? catalog.providers : {};
    const preferredProvider = normalizeProviderKey(
      (run && run.provider) || (run && run.modelDetails ? run.modelDetails.provider : "")
    );
    const allowUniqueProviderFallback = options.allowUniqueProviderFallback !== false;

    if (preferredProvider && providers[preferredProvider]) {
      const providerModels = providers[preferredProvider].models || {};
      const preferredMatch = resolveProviderMatch(preferredProvider, providerModels, candidates);
      if (preferredMatch) {
        return preferredMatch;
      }
    }

    if (preferredProvider === "requesty") {
      return null;
    }

    if (!allowUniqueProviderFallback) {
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
      case "unpriced":
        return "manual pricing needed";
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

    const strongCandidates = buildLookupCandidates(run, { includeFileModelSlug: false });
    let match = resolvePricingMatch(catalog, run, strongCandidates);
    if (!match) {
      const allCandidates = buildLookupCandidates(run);
      const hasWeakOnlyCandidates = allCandidates.some((candidate) => !strongCandidates.includes(candidate));
      if (hasWeakOnlyCandidates) {
        match = resolvePricingMatch(catalog, run, allCandidates, {
          allowUniqueProviderFallback: strongCandidates.length === 0,
        });
      }
    }
    if (!match) {
      return baseResult;
    }

    const entryKind = classifyCatalogEntry(match.entry);
    if (entryKind === "unsupported" || entryKind === "unpriced") {
      return {
        ...baseResult,
        providerKey: match.providerKey,
        matchedKey: match.matchedKey,
        resolvedKey: match.resolvedKey,
        status: entryKind,
        statusLabel: statusLabel(entryKind),
      };
    }

    const serviceTiers = getServiceTiers(match.entry);
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
    classifyCatalogEntry,
    estimateRunCost,
    getPricingCatalog,
    hasAnyUsableRates,
    normalizeProviderKey,
    normalizeServiceTier,
    sanitizeModelIdentifier,
  };
})(typeof window !== "undefined" ? window : globalThis);

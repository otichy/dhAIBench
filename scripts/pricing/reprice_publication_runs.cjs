// Recalculate only explicitly reviewed publication models with the web app engine.
// JSON stdin: [{file, payload, inventory}]. JSON stdout: {filename: pricing}.
const fs = require('fs');
const path = require('path');
const root = path.resolve(__dirname, '../..');
global.window = global;
require(path.join(root, 'config_prices.js'));
require(path.join(root, 'web/pricing.js'));
const policy = JSON.parse(fs.readFileSync(path.join(root, 'benchmark_price_overrides.json'), 'utf8'));
const targets = Object.fromEntries(Object.entries(policy.publication_reprice_models || {}).map(
  ([provider, names]) => [provider, new Set(names.map(DHAIBenchPricing.sanitizeModelIdentifier))]
));
const result = {};
for (const {file, payload: p, inventory} of JSON.parse(fs.readFileSync(0, 'utf8'))) {
  const details = p.model_details || {};
  const provider = DHAIBenchPricing.normalizeProviderKey(details.provider || p.provider || inventory.provider);
  const model = details.model_requested || details.model_for_requests || p.model_requested || p.model || inventory.model;
  if (!targets[provider]?.has(DHAIBenchPricing.sanitizeModelIdentifier(model))) continue;
  const t = p.token_usage_totals || {}, u = p.usage_metadata_summary || {};
  if (![t.input_tokens_total, t.output_tokens_total].every(Number.isFinite)) {
    throw new Error(`Missing token totals for repricing: ${file}`);
  }
  const estimate = DHAIBenchPricing.estimateRunCost(MODEL_PRICING_CATALOG, {
    provider, model, modelDetails: details, rawMetrics: p,
    fileModelSlug: file.split('__')[2],
    inputTokensTotal: t.input_tokens_total, outputTokensTotal: t.output_tokens_total,
    cachedInputTokensTotal: t.cached_input_tokens_total ?? u.cached_tokens_total_estimate ?? u.cache_read_tokens_total,
    nonCachedInputTokensTotal: t.non_cached_input_tokens_total,
  });
  if (estimate.status !== 'priced') throw new Error(`Cannot price ${file}: ${estimate.status}`);
  result[file] = {
    status: estimate.status, estimated_cost_usd: estimate.estimatedCostUsd,
    catalog_updated_at: MODEL_PRICING_CATALOG.updated_at,
    resolved_key: estimate.resolvedKey, pricing_tier: estimate.pricingTier,
    reference_note: estimate.pricingCalculation.referenceNote,
    sources: estimate.pricingCalculation.sources, calculation: estimate.pricingCalculation,
  };
}
process.stdout.write(JSON.stringify(result));

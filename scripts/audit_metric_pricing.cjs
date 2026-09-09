// Audit the same model lookup, service tiers, and token buckets as the dashboard.
// Usage: node scripts/audit_metric_pricing.cjs [output.json]
const fs = require('fs');
const path = require('path');
const root = path.resolve(__dirname, '..');
global.window = global;
require(path.join(root, 'config_prices.js'));
require(path.join(root, 'web/pricing.js'));
const rows = [];
for (const file of fs.readdirSync(path.join(root, 'data/metrics')).sort()) {
  if (!/__metrics(?:_old)?\.json$/.test(file)) continue;
  const p = JSON.parse(fs.readFileSync(path.join(root, 'data/metrics', file), 'utf8'));
  const d = p.model_details || {}, n = file.split('__');
  const t = p.token_usage_totals || {}, u = p.usage_metadata_summary || {};
  const run = {
    provider: d.provider || p.provider || n[1] || '',
    model: d.model_requested || d.model_for_requests || p.model_requested || p.model || n[2],
    modelDetails: d, fileModelSlug: n[2], rawMetrics: p,
    inputTokensTotal: t.input_tokens_total,
    outputTokensTotal: t.output_tokens_total,
    cachedInputTokensTotal: t.cached_input_tokens_total ?? u.cached_tokens_total_estimate ?? u.cache_read_tokens_total,
    nonCachedInputTokensTotal: t.non_cached_input_tokens_total,
  };
  // Require both base rates even for a run with absent or zero usage totals.
  const base = DHAIBenchPricing.estimateRunCost(MODEL_PRICING_CATALOG,
    {...run, inputTokensTotal: 1, nonCachedInputTokensTotal: 1, outputTokensTotal: 1, cachedInputTokensTotal: 0});
  const actual = DHAIBenchPricing.estimateRunCost(MODEL_PRICING_CATALOG, run);
  const result = base.status === 'priced' ? actual : base;
  rows.push({file, provider: run.provider, model: run.model, tier: result.pricingTier,
    status: result.status, resolvedProvider: result.providerKey,
    resolvedModel: result.resolvedKey, missingBuckets: result.missingBuckets});
}
const summary = {files: rows.length, providerModelPairs: new Set(rows.map(r => `${r.provider}|${r.model}`)).size,
  statuses: rows.reduce((a, r) => { a[r.status] = (a[r.status] || 0) + 1; return a; }, {})};
if (process.argv[2]) fs.writeFileSync(process.argv[2], JSON.stringify({summary, rows}, null, 2) + '\n');
console.log(JSON.stringify(summary, null, 2));

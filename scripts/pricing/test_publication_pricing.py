"""Reviewed publication rates must resolve through the dashboard calculator."""
import json
from pathlib import Path
import subprocess
import unittest

from pricing_catalog import apply_benchmark_price_overrides

ROOT = Path(__file__).resolve().parents[2]


class PublicationPricingTests(unittest.TestCase):
    def test_kimi_k31_reference_prices_cached_input_and_resolves_aliases(self):
        script = r'''
global.window = global;
require('./config_prices.js');
require('./web/pricing.js');
const results = [];
for (const [provider, model] of [['e-infra', 'kimi-k3.1-pro'], ['einfra', 'kimik31pro'],
    ['requesty', 'moonshot/kimi-k3.1-pro'], ['requesty', 'moonshotkimik31pro']]) {
  for (const serviceTier of provider.startsWith('e') ? ['standard', 'flex'] : ['standard']) {
  results.push(DHAIBenchPricing.estimateRunCost(MODEL_PRICING_CATALOG, {
    provider, model, inputTokensTotal: 1000000, cachedInputTokensTotal: 200000,
    outputTokensTotal: 1000000, serviceTier,
  }));
  }
}
process.stdout.write(JSON.stringify(results));
'''
        completed = subprocess.run(['node', '-e', script], cwd=ROOT,
                                   capture_output=True, text=True, check=True)
        for result in json.loads(completed.stdout):
            self.assertEqual(result['status'], 'priced')
            self.assertAlmostEqual(result['estimatedCostUsd'], 11.64)
            self.assertIn('Requesty reference', result['pricingCalculation']['referenceNote'])

    def test_reviewed_overrides_preserve_unrelated_entries_without_mutating_input(self):
        base = {'providers': {'p': {'models': {'existing': {'x': 1}}}}}
        overrides = {'verified_date': '2026-11-05', 'providers': {'p': {'models': {
            'new': {'service_tiers': {'standard': {'input_usd_per_mtokens': .1}}},
            'alias': {'pricing_ref': 'new'}}}}}
        merged = apply_benchmark_price_overrides(base, overrides)
        self.assertEqual(merged['providers']['p']['models']['existing'], {'x': 1})
        self.assertEqual(merged['providers']['p']['models']['alias'], {'pricing_ref': 'new'})
        self.assertNotIn('new', base['providers']['p']['models'])
        merged['providers']['p']['models']['new']['service_tiers']['standard']['input_usd_per_mtokens'] = 99
        self.assertEqual(overrides['providers']['p']['models']['new']['service_tiers']['standard']['input_usd_per_mtokens'], .1)

    def test_current_rates_aliases_caching_and_explicit_flex_reference(self):
        cases = [('gpt-5.6-luna', 'openrouter', 'standard', 1.364),
                 ('gpt-6-luna', 'openrouter', 'standard', .582),
                 ('gpt-6-luna', 'openai', 'flex', .291),
                 ('claude-haiku-5.5', 'openrouter', 'flex', .582),
                 ('qwen/qwen3.6-plus:free', 'openrouter', 'standard', 2.275)]
        inputs = []
        for i, (model, provider, tier, _) in enumerate(cases):
            inputs.append(dict(file=f'case__{provider}__model__{i}__metrics.json',
                inventory=dict(model=model, provider=provider), payload={
                    'model_details': {'provider': provider, 'model_requested': model},
                    'run_config': {'service_tier': tier},
                    'token_usage_totals': {'input_tokens_total': 1_000_000,
                                           'cached_input_tokens_total': 200_000,
                                           'output_tokens_total': 1_000_000}}))
        completed = subprocess.run(['node', str(ROOT / 'scripts/pricing/reprice_publication_runs.cjs')],
                                   input=json.dumps(inputs), capture_output=True, text=True, check=True)
        actual = json.loads(completed.stdout)
        for row, case in zip(inputs, cases):
            self.assertAlmostEqual(actual[row['file']]['estimated_cost_usd'], case[3])
        self.assertIn('no public', actual[inputs[3]['file']]['reference_note'])
        self.assertIn('current paid', actual[inputs[4]['file']]['reference_note'])


if __name__ == '__main__':
    unittest.main()

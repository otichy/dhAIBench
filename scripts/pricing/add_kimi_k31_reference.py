"""Install the reviewed Kimi K3.1 Pro reference quote without refreshing other models."""
import copy
import json
from pathlib import Path

from pricing_catalog import load_pricing_catalog_js, write_pricing_catalog_js

ROOT = Path(__file__).resolve().parents[2]
policy_path = ROOT / 'benchmark_price_overrides.json'
policy = json.loads(policy_path.read_text(encoding='utf-8'))
quote = {
    'service_tiers': {'standard': {
        'input_usd_per_mtokens': 2.0,
        'cached_input_usd_per_mtokens': 0.2,
        'output_usd_per_mtokens': 10.0,
        'reference_note': 'Requesty reference rates: moonshot/kimi-k3.1-pro; token rates exclude gateway fees/tax.',
    }},
    'sources': [{
        'label': 'Requesty public model pricing',
        'url': 'https://www.requesty.ai/models/moonshot/kimi-k3.1-pro',
        'model_id': 'moonshot/kimi-k3.1-pro',
        'checked_at': '2026-11-22',
        'requested_tier': 'standard',
        'quoted_tier': 'standard',
    }],
    'notes': ['Token rates only; routing fees, taxes and cache storage/writes are excluded.'],
    'pricing_basis': 'reference',
    'verified_at': '2026-11-22',
}
einfra_quote = copy.deepcopy(quote)
einfra_quote['notes'].append('User-requested reference pricing, not actual e-infra hosting charges; hosting and quantization can differ.')
for provider, model, value in [('e-infra', 'kimi-k3.1-pro', einfra_quote),
                               ('requesty', 'moonshot/kimi-k3.1-pro', quote)]:
    policy['providers'][provider]['models'][model] = value
policy['providers']['e-infra']['models']['kimik31pro'] = {
    'pricing_ref': 'kimi-k3.1-pro', 'alias_kind': 'verified_benchmark_alias'}
policy['providers']['requesty']['models']['moonshotkimik31pro'] = {
    'pricing_ref': 'moonshot/kimi-k3.1-pro', 'alias_kind': 'verified_benchmark_alias'}
policy['verified_date'] = '2026-11-22'
note = '2026-11-22: added Kimi K3.1 Pro Requesty reference pricing, including cached input, for e-infra and the Requesty publisher route. Other frozen entries retain their individual verification dates.'
if note not in policy['notes']:
    policy['notes'].append(note)
policy_path.write_text(json.dumps(policy, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
catalog_path = ROOT / 'config_prices.js'
catalog = load_pricing_catalog_js(str(catalog_path))
write_pricing_catalog_js(catalog, str(catalog_path))
assert catalog_path.read_bytes() == (ROOT / 'web/config_prices.js').read_bytes()
print('Updated reviewed overrides and both pricing catalogs.')

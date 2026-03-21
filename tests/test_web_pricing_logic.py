import json
import subprocess
import textwrap
import unittest
from pathlib import Path


PRICING_JS_PATH = Path(__file__).resolve().parents[1] / "web" / "pricing.js"


def _run_node_pricing(catalog: dict, run: dict) -> dict:
    script = textwrap.dedent(
        f"""
        const fs = require("fs");
        const vm = require("vm");
        const code = fs.readFileSync({json.dumps(str(PRICING_JS_PATH))}, "utf8");
        const context = {{}};
        context.globalThis = context;
        vm.createContext(context);
        vm.runInContext(code, context);
        const result = context.DHAIBenchPricing.estimateRunCost(
          {json.dumps(catalog)},
          {json.dumps(run)}
        );
        process.stdout.write(JSON.stringify(result));
        """
    )
    completed = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


class WebPricingLogicTests(unittest.TestCase):
    def test_openai_gpt54mini_flex_pricing(self) -> None:
        catalog = {
            "providers": {
                "openai": {
                    "models": {
                        "gpt-5.4-mini": {
                            "status": "priced",
                            "service_tiers": {
                                "standard": {
                                    "input_usd_per_mtokens": 0.75,
                                    "cached_input_usd_per_mtokens": 0.075,
                                    "output_usd_per_mtokens": 4.5,
                                },
                                "flex": {
                                    "input_usd_per_mtokens": 0.375,
                                    "cached_input_usd_per_mtokens": 0.0375,
                                    "output_usd_per_mtokens": 2.25,
                                },
                            },
                        }
                    }
                }
            }
        }
        run = {
            "provider": "openai",
            "model": "gpt-5.4-mini",
            "modelDetails": {"provider": "openai", "model_requested": "gpt-5.4-mini"},
            "rawMetrics": {"run_config": {"service_tier": "flex"}},
            "inputTokensTotal": 1000,
            "cachedInputTokensTotal": 200,
            "nonCachedInputTokensTotal": 800,
            "outputTokensTotal": 100,
        }
        result = _run_node_pricing(catalog, run)
        self.assertEqual(result["status"], "priced")
        self.assertEqual(result["pricingTier"], "flex")
        self.assertAlmostEqual(result["estimatedCostUsd"], 0.0005325)

    def test_standard_openai_run_ignores_missing_cached_rate_when_cached_tokens_are_zero(self) -> None:
        catalog = {
            "providers": {
                "openai": {
                    "models": {
                        "gpt-5.2-pro": {
                            "status": "priced",
                            "service_tiers": {
                                "standard": {
                                    "input_usd_per_mtokens": 21.0,
                                    "cached_input_usd_per_mtokens": None,
                                    "output_usd_per_mtokens": 168.0,
                                }
                            },
                        }
                    }
                }
            }
        }
        run = {
            "provider": "openai",
            "model": "gpt-5.2-pro",
            "modelDetails": {"provider": "openai", "model_requested": "gpt-5.2-pro"},
            "rawMetrics": {"run_config": {"service_tier": "standard"}},
            "inputTokensTotal": 1000,
            "cachedInputTokensTotal": 0,
            "nonCachedInputTokensTotal": 1000,
            "outputTokensTotal": 100,
        }
        result = _run_node_pricing(catalog, run)
        self.assertEqual(result["status"], "priced")
        self.assertAlmostEqual(result["estimatedCostUsd"], 0.0378)

    def test_requesty_exact_match_uses_requesty_rates(self) -> None:
        catalog = {
            "providers": {
                "requesty": {
                    "models": {
                        "anthropic/claude-sonnet-4-6": {
                            "status": "priced",
                            "service_tiers": {
                                "standard": {
                                    "input_usd_per_mtokens": 3.0,
                                    "cached_input_usd_per_mtokens": 0.3,
                                    "output_usd_per_mtokens": 15.0,
                                }
                            },
                        }
                    }
                }
            }
        }
        run = {
            "provider": "requesty",
            "model": "anthropic/claude-sonnet-4-6",
            "modelDetails": {"provider": "requesty", "model_requested": "anthropic/claude-sonnet-4-6"},
            "rawMetrics": {"run_config": {"service_tier": "standard"}},
            "inputTokensTotal": 1200,
            "cachedInputTokensTotal": 300,
            "nonCachedInputTokensTotal": 900,
            "outputTokensTotal": 200,
        }
        result = _run_node_pricing(catalog, run)
        self.assertEqual(result["status"], "priced")
        self.assertAlmostEqual(result["estimatedCostUsd"], 0.00579)

    def test_requesty_legacy_file_slug_resolves_provider_qualified_model(self) -> None:
        catalog = {
            "providers": {
                "requesty": {
                    "models": {
                        "anthropic/claude-sonnet-4-6": {
                            "status": "priced",
                            "service_tiers": {
                                "standard": {
                                    "input_usd_per_mtokens": 3.0,
                                    "cached_input_usd_per_mtokens": 0.3,
                                    "output_usd_per_mtokens": 15.0,
                                }
                            },
                        },
                        "anthropicclaudesonnet46": {
                            "status": "alias",
                            "pricing_ref": "anthropic/claude-sonnet-4-6",
                            "alias_kind": "legacy_slug",
                        },
                    }
                },
                "e-infra": {
                    "models": {
                        "claudesonnet46": {
                            "status": "unsupported",
                            "reason": "No official compatible per-token pricing source is configured for this provider.",
                        }
                    }
                },
            }
        }
        run = {
            "provider": "requesty",
            "model": "claude-sonnet-4-6",
            "fileModelSlug": "anthropicclaudesonnet46",
            "modelDetails": {"provider": "requesty", "model_requested": "claude-sonnet-4-6"},
            "rawMetrics": {"run_config": {}},
            "inputTokensTotal": 1200,
            "cachedInputTokensTotal": 300,
            "nonCachedInputTokensTotal": 900,
            "outputTokensTotal": 200,
        }
        result = _run_node_pricing(catalog, run)
        self.assertEqual(result["status"], "priced")
        self.assertEqual(result["providerKey"], "requesty")
        self.assertEqual(result["resolvedKey"], "anthropic/claude-sonnet-4-6")

    def test_requesty_does_not_fall_back_to_other_provider_on_miss(self) -> None:
        catalog = {
            "providers": {
                "requesty": {
                    "models": {}
                },
                "e-infra": {
                    "models": {
                        "kimik25": {
                            "status": "unsupported",
                            "reason": "No official compatible per-token pricing source is configured for this provider.",
                        }
                    }
                },
            }
        }
        run = {
            "provider": "requesty",
            "model": "kimi-k2.5",
            "modelDetails": {"provider": "requesty", "model_requested": "kimi-k2.5"},
            "rawMetrics": {"run_config": {}},
            "inputTokensTotal": 1000,
            "cachedInputTokensTotal": 0,
            "nonCachedInputTokensTotal": 1000,
            "outputTokensTotal": 100,
        }
        result = _run_node_pricing(catalog, run)
        self.assertEqual(result["status"], "model_missing")
        self.assertEqual(result["providerKey"], "")

    def test_unsupported_einfra_run_returns_unsupported_status(self) -> None:
        catalog = {
            "providers": {
                "e-infra": {
                    "models": {
                        "qwen3.5": {
                            "status": "unsupported",
                            "reason": "No official compatible per-token pricing source is configured for this provider.",
                        }
                    }
                }
            }
        }
        run = {
            "provider": "e-infra",
            "model": "qwen3.5",
            "modelDetails": {"provider": "e-infra", "model_requested": "qwen3.5"},
            "rawMetrics": {"run_config": {"service_tier": "standard"}},
            "inputTokensTotal": 1000,
            "cachedInputTokensTotal": 0,
            "nonCachedInputTokensTotal": 1000,
            "outputTokensTotal": 100,
        }
        result = _run_node_pricing(catalog, run)
        self.assertEqual(result["status"], "unsupported")

    def test_older_run_defaults_to_standard_and_uses_slug_alias(self) -> None:
        catalog = {
            "providers": {
                "openai": {
                    "models": {
                        "gpt-5.2-pro": {
                            "status": "priced",
                            "service_tiers": {
                                "standard": {
                                    "input_usd_per_mtokens": 21.0,
                                    "cached_input_usd_per_mtokens": None,
                                    "output_usd_per_mtokens": 168.0,
                                }
                            },
                        },
                        "gpt52pro": {
                            "status": "alias",
                            "pricing_ref": "gpt-5.2-pro",
                            "alias_kind": "legacy_slug",
                        },
                    }
                }
            }
        }
        run = {
            "provider": "",
            "model": "gpt52pro",
            "modelDetails": {"provider": "", "model_requested": "gpt52pro"},
            "rawMetrics": {"run_config": {}},
            "inputTokensTotal": 1000,
            "cachedInputTokensTotal": 0,
            "nonCachedInputTokensTotal": 1000,
            "outputTokensTotal": 100,
        }
        result = _run_node_pricing(catalog, run)
        self.assertEqual(result["status"], "priced")
        self.assertEqual(result["pricingTier"], "standard")
        self.assertEqual(result["providerKey"], "openai")


if __name__ == "__main__":
    unittest.main()

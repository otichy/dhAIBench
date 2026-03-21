import json
import tempfile
import unittest
from pathlib import Path

import pricing_catalog as pc


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "pricing"


def _fixture_text(name: str) -> str:
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


def _fake_fetcher(mapping: dict[str, tuple[str, str] | str]):
    normalized = {}
    for key, value in mapping.items():
        normalized[key] = value if isinstance(value, tuple) else (key, value)

    def fetch(url: str):
        if url not in normalized:
            raise AssertionError(f"Unexpected URL fetch: {url}")
        return normalized[url]

    return fetch


class PricingParserTests(unittest.TestCase):
    def test_parse_openai_pricing_page_extracts_flex(self) -> None:
        parsed = pc.parse_openai_pricing_page(pc.html_to_text(_fixture_text("openai_pricing_excerpt.html")))
        self.assertIn("gpt-5.4-mini", parsed)
        self.assertEqual(
            parsed["gpt-5.4-mini"]["service_tiers"]["flex"],
            {
                "input_usd_per_mtokens": 0.375,
                "cached_input_usd_per_mtokens": 0.0375,
                "output_usd_per_mtokens": 2.25,
            },
        )

    def test_parse_google_pricing_page_extracts_long_context(self) -> None:
        parsed = pc.build_google_source_entries(
            _fake_fetcher({pc.GOOGLE_PRICING_URL: _fixture_text("google_pricing_excerpt.html")})
        )
        self.assertEqual(
            parsed["models/gemini-3.1-pro-preview"]["service_tiers"]["standard"],
            {
                "input_usd_per_mtokens": 2.0,
                "cached_input_usd_per_mtokens": 0.2,
                "output_usd_per_mtokens": 12.0,
            },
        )
        self.assertEqual(
            parsed["models/gemini-3.1-pro-preview"]["long_context"]["service_tiers"]["batch"],
            {
                "input_usd_per_mtokens": 2.0,
                "cached_input_usd_per_mtokens": 0.4,
                "output_usd_per_mtokens": 9.0,
            },
        )

    def test_parse_vertex_pricing_page_extracts_priority_and_flex(self) -> None:
        parsed = pc.build_vertex_source_entries(
            _fake_fetcher({pc.VERTEX_PRICING_URL: _fixture_text("vertex_pricing_excerpt.html")})
        )
        self.assertEqual(
            parsed["gemini-3-flash-preview"]["service_tiers"]["priority"],
            {
                "input_usd_per_mtokens": 0.9,
                "cached_input_usd_per_mtokens": 0.09,
                "output_usd_per_mtokens": 5.4,
            },
        )
        self.assertEqual(
            parsed["gemini-3-flash-preview"]["service_tiers"]["flex"],
            {
                "input_usd_per_mtokens": 0.25,
                "cached_input_usd_per_mtokens": None,
                "output_usd_per_mtokens": 1.5,
            },
        )

    def test_parse_inception_pricing_page_extracts_cached_input(self) -> None:
        parsed = pc.build_inception_source_entries(
            _fake_fetcher({pc.INCEPTION_PRICING_URL: _fixture_text("inception_pricing_excerpt.html")})
        )
        self.assertEqual(
            parsed["mercury-2"]["service_tiers"]["standard"],
            {
                "input_usd_per_mtokens": 0.25,
                "cached_input_usd_per_mtokens": 0.025,
                "output_usd_per_mtokens": 0.75,
            },
        )

    def test_parse_requesty_detail_page_extracts_cache_read(self) -> None:
        parsed = pc.parse_requesty_detail_page(pc.html_to_text(_fixture_text("requesty_detail_excerpt.html")))
        self.assertEqual(
            parsed["standard"],
            {
                "input_usd_per_mtokens": 3.0,
                "cached_input_usd_per_mtokens": 0.3,
                "output_usd_per_mtokens": 15.0,
            },
        )


class PricingCatalogGenerationTests(unittest.TestCase):
    def test_snapshot_alias_maps_to_canonical_openai_model(self) -> None:
        model_catalog = {
            "openai": {
                "models": ["gpt-5.4-mini", "gpt-5.4-mini-2026-03-17"],
            }
        }
        model_page = """
        <html><body>
          Text tokens Per 1M tokens ∙ Batch API price Input $0.75 Cached input $0.075 Output $4.50
        </body></html>
        """
        fetcher = _fake_fetcher(
            {
                pc.OPENAI_PRICING_URL: _fixture_text("openai_pricing_excerpt.html"),
                pc.OPENAI_PRIORITY_URL: "<html></html>",
                pc.OPENAI_MODEL_DOC_URL_TEMPLATE.format(model="gpt-5.4-mini"): model_page,
                pc.OPENAI_MODEL_DOC_URL_TEMPLATE.format(model="gpt-5.4-mini-2026-03-17"): model_page,
            }
        )

        catalog = pc.build_pricing_catalog(model_catalog, fetch_html=fetcher, updated_at="2026-03-21T00:00:00Z")
        snapshot_entry = catalog["providers"]["openai"]["models"]["gpt-5.4-mini-2026-03-17"]
        self.assertEqual(snapshot_entry["status"], "alias")
        self.assertEqual(snapshot_entry["pricing_ref"], "gpt-5.4-mini")

    def test_unsupported_stub_is_created_for_provider_without_official_source(self) -> None:
        model_catalog = {
            "e-infra": {
                "models": ["qwen3.5"],
            }
        }
        catalog = pc.build_pricing_catalog(model_catalog, fetch_html=_fake_fetcher({}), updated_at="2026-03-21T00:00:00Z")
        entry = catalog["providers"]["e-infra"]["models"]["qwen3.5"]
        self.assertEqual(entry["status"], "unsupported")
        self.assertIn("No official compatible", entry["reason"])

    def test_legacy_slug_alias_is_generated(self) -> None:
        model_catalog = {
            "openai": {
                "models": ["gpt-5.4-mini"],
            }
        }
        model_page = """
        <html><body>
          Text tokens Per 1M tokens ∙ Batch API price Input $0.75 Cached input $0.075 Output $4.50
        </body></html>
        """
        fetcher = _fake_fetcher(
            {
                pc.OPENAI_PRICING_URL: _fixture_text("openai_pricing_excerpt.html"),
                pc.OPENAI_PRIORITY_URL: "<html></html>",
                pc.OPENAI_MODEL_DOC_URL_TEMPLATE.format(model="gpt-5.4-mini"): model_page,
            }
        )
        catalog = pc.build_pricing_catalog(model_catalog, fetch_html=fetcher, updated_at="2026-03-21T00:00:00Z")
        alias_entry = catalog["providers"]["openai"]["models"]["gpt54mini"]
        self.assertEqual(alias_entry["status"], "alias")
        self.assertEqual(alias_entry["pricing_ref"], "gpt-5.4-mini")

    def test_provider_filtering_limits_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            models_path = Path(tmpdir) / "config_models.js"
            output_path = Path(tmpdir) / "config_prices.js"
            models_path.write_text(
                (
                    "window.MODEL_CATALOG = "
                    + json.dumps(
                        {
                            "openai": {"models": ["gpt-5.4-mini"]},
                            "e-infra": {"models": ["qwen3.5"]},
                        }
                    )
                    + ";"
                ),
                encoding="utf-8",
            )
            model_page = """
            <html><body>
              Text tokens Per 1M tokens ∙ Batch API price Input $0.75 Cached input $0.075 Output $4.50
            </body></html>
            """
            fetcher = _fake_fetcher(
                {
                    pc.OPENAI_PRICING_URL: _fixture_text("openai_pricing_excerpt.html"),
                    pc.OPENAI_PRIORITY_URL: "<html></html>",
                    pc.OPENAI_MODEL_DOC_URL_TEMPLATE.format(model="gpt-5.4-mini"): model_page,
                }
            )
            pc.update_model_prices(str(models_path), str(output_path), providers=["openai"], fetch_html=fetcher)

            written = output_path.read_text(encoding="utf-8")
            payload = json.loads(written.split("=", 1)[1].rsplit(";", 1)[0].strip())
            self.assertEqual(set(payload["providers"].keys()), {"openai"})


if __name__ == "__main__":
    unittest.main()

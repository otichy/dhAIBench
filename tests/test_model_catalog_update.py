import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import benchmark_agent as ba


class ModelCatalogUpdateTests(unittest.TestCase):
    def test_provider_filtered_update_preserves_existing_catalog_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "config_models.js"
            output_path.write_text(
                (
                    "window.MODEL_CATALOG = "
                    + json.dumps(
                        {
                            "openai": {
                                "models": ["gpt-test"],
                                "api_base": "https://api.openai.com/v1",
                            }
                        }
                    )
                    + ";\n"
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(
                    ba,
                    "parse_env_file",
                    return_value={
                        "E-INFRA_API_KEY": "test-key",
                        "E-INFRA_BASE_URL": "https://llm.ai.e-infra.cz/v1",
                    },
                ),
                mock.patch.object(
                    ba,
                    "fetch_provider_models",
                    return_value=(
                        ["glm-5.1"],
                        {"glm-5.1": {"input_cost_per_token": 0.0000006}},
                        None,
                    ),
                ),
                mock.patch.object(ba, "fetch_provider_model_metadata", return_value=({}, None)),
            ):
                exit_code = ba.update_model_catalog(["e-infra"], str(output_path))

            self.assertEqual(exit_code, 0)
            catalog = ba.load_model_catalog_js(str(output_path))
            self.assertEqual(set(catalog.keys()), {"openai", "e-infra"})
            self.assertEqual(catalog["openai"]["models"], ["gpt-test"])
            self.assertEqual(catalog["e-infra"]["models"], ["glm-5.1"])


if __name__ == "__main__":
    unittest.main()

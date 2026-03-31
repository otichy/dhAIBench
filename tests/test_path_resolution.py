import os
import tempfile
import unittest
from unittest.mock import patch

import benchmark_agent as ba


class PathResolutionTests(unittest.TestCase):
    def test_resolve_user_path_maps_legacy_data_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(ba, "DATA_ROOT_DIR", tmpdir):
                resolved = ba.resolve_user_path("/data/input/examples.csv")
                self.assertEqual(resolved, os.path.join(tmpdir, "input", "examples.csv"))

                resolved_root = ba.resolve_user_path("/data")
                self.assertEqual(resolved_root, tmpdir)

    def test_resolve_user_path_keeps_legacy_root_when_active(self) -> None:
        with patch.object(ba, "DATA_ROOT_DIR", ba.LEGACY_DATA_ROOT_DIR):
            legacy_path = "/data/input/examples.csv"
            self.assertEqual(ba.resolve_user_path(legacy_path), legacy_path)

    def test_resolve_output_path_maps_legacy_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.csv")
            with patch.object(ba, "DATA_ROOT_DIR", tmpdir):
                output_path = ba.resolve_output_path(
                    input_path=input_path,
                    provider="openai",
                    model="gpt-4o-mini",
                    output_argument="/data/output",
                    timestamp_tag="2026-03-10-14-52",
                    multiple_inputs=False,
                )
            self.assertTrue(output_path.startswith(os.path.join(tmpdir, "output")))
            self.assertTrue(output_path.endswith(".csv"))

    def test_resolve_user_path_finds_existing_script_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_dir = os.path.join(tmpdir, "repo")
            target_dir = os.path.join(script_dir, "nested")
            os.makedirs(target_dir, exist_ok=True)
            target_path = os.path.join(target_dir, "examples.csv")
            with open(target_path, "w", encoding="utf-8") as handle:
                handle.write("ID;node\n1;test\n")

            with patch.object(ba, "SCRIPT_DIR", script_dir):
                resolved = ba.resolve_user_path("nested/examples.csv")

            self.assertEqual(resolved, os.path.normpath(target_path))

    def test_normalize_metrics_path_reference_prefers_repo_data_path_for_legacy_absolute(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_dir = os.path.join(tmpdir, "repo")
            data_root = os.path.join(script_dir, "data")
            input_dir = os.path.join(data_root, "input")
            os.makedirs(input_dir, exist_ok=True)
            target_path = os.path.join(input_dir, "examples.csv")
            with open(target_path, "w", encoding="utf-8") as handle:
                handle.write("ID;node\n1;test\n")

            with (
                patch.object(ba, "SCRIPT_DIR", script_dir),
                patch.object(ba, "DATA_ROOT_DIR", data_root),
                patch.object(ba, "DEFAULT_INPUT_DIR", input_dir),
            ):
                normalized = ba.normalize_metrics_path_reference(
                    "/home/ondrej/dhAIBench/data/input/examples.csv",
                    ba.DEFAULT_INPUT_DIR,
                )

            self.assertEqual(normalized, "data/input/examples.csv")


if __name__ == "__main__":
    unittest.main()

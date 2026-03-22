from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "docs" / "cli-reference.md"


def normalize_help_text(help_text: str) -> str:
    normalized = help_text.replace("\r\n", "\n")
    replacements = {
        str(ROOT / "data" / "input"): "<repo>/data/input",
        str(ROOT): "<repo>",
    }
    for raw, replacement in replacements.items():
        normalized = normalized.replace(raw, replacement)
    return normalized


def main() -> int:
    result = subprocess.run(
        [sys.executable, str(ROOT / "benchmark_agent.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    help_text = normalize_help_text(result.stdout).rstrip()
    content = (
        "# CLI Reference\n\n"
        "This file is generated from `python benchmark_agent.py --help` "
        "by `python scripts/generate_cli_reference.py`.\n\n"
        "```text\n"
        f"{help_text}\n"
        "```\n"
    )
    TARGET.write_text(content, encoding="utf-8")
    print(f"Wrote {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

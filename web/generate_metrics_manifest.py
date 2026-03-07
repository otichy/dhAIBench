#!/usr/bin/env python3
"""Generate web/metrics-manifest.json for the local dashboard."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    web_dir = Path(__file__).resolve().parent
    metrics_dir = web_dir.parent / "data" / "metrics"
    output_path = web_dir / "metrics-manifest.json"

    if not metrics_dir.exists():
        raise FileNotFoundError(f"Missing metrics directory: {metrics_dir}")

    files = sorted(
        f"../data/metrics/{path.name}"
        for path in metrics_dir.glob("*_metrics.json")
        if path.is_file()
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics_files": files,
    }

    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(files)} entries to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

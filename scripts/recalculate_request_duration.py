"""Backfill successful-request latency, preserving other metrics and backing up changes."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import benchmark_agent as benchmark


def records(path):
    # Strict parsing prevents silently computing a mean from a truncated log.
    with path.open(encoding="utf-8-sig") as handle:
        first = next((c for c in iter(lambda: handle.read(1), "") if not c.isspace()), "")
        handle.seek(0)
        entries = json.load(handle) if first == "[" else (
            json.loads(line) for line in handle if line.strip()
        )
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Expected an object record")
            yield entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write changes after audit; otherwise preview only")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    audit_dir = ROOT / "data" / "backup" / f"request_duration_{stamp}"
    audit_dir.mkdir(parents=True)
    report = []
    changes = []
    paths = sorted((ROOT / "data" / "metrics").glob("*_metrics.json"))
    for index, path in enumerate(paths, 1):
        original = path.read_bytes()
        payload = json.loads(original.decode("utf-8-sig"))
        suffix = "__metrics.json" if path.name.endswith("__metrics.json") else "_metrics.json"
        log = ROOT / "data" / "logs" / (path.name[:-len(suffix)] + ".log")
        item = {"metrics_file": path.name, "log_file": log.name}
        try:
            duration = benchmark.compute_mean_request_duration(records(log), payload)
        except (OSError, ValueError, TypeError) as exc:
            item["log_error"] = str(exc)
            duration = benchmark.compute_mean_request_duration([], payload)
        item.update(duration)
        item["changed"] = any(payload.get(key) != value or key not in payload for key, value in duration.items())
        if item["changed"]:
            changes.append((path, original, {**payload, **duration}, set(duration)))
        report.append(item)
        if index % 10 == 0 or index == len(paths):
            print(f"Audited {index}/{len(paths)}", flush=True)

    result = {
        "applied": False,
        "method": "mean_successful_physical_request_duration_v1",
        "summary": dict(Counter(item["mean_request_duration_source"] for item in report)),
        "changed": len(changes),
        "runs": report,
    }
    report_path = audit_dir / "audit.json"
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.apply:
        for path, original, updated, keys in changes:
            if path.read_bytes() != original:
                raise RuntimeError(f"File changed during audit: {path}")
        for path, original, updated, keys in changes:
            backup = audit_dir / "metrics" / path.name
            backup.parent.mkdir(exist_ok=True)
            shutil.copy2(path, backup)
            assert backup.read_bytes() == original
            before = json.loads(original.decode("utf-8-sig"))
            assert {k: v for k, v in before.items() if k not in keys} == {
                k: v for k, v in updated.items() if k not in keys
            }
            temporary = path.with_suffix(".request-duration.tmp")
            temporary.write_text(json.dumps(updated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            temporary.replace(path)
            assert json.loads(path.read_text(encoding="utf-8")) == updated
        result["applied"] = True
        report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary": result["summary"], "changed": len(changes), "applied": args.apply, "audit": str(report_path)}), flush=True)


if __name__ == "__main__":
    main()

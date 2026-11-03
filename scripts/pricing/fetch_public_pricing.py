"""Download public price catalogs only; no credentials or account calls."""
import json
import pathlib
import urllib.request

OUT = pathlib.Path(__file__).resolve().parents[1] / "analysis" / "pricing_sources"
OUT.mkdir(parents=True, exist_ok=True)
SOURCES = {
    "requesty_models.html": "https://www.requesty.ai/models",
    "openrouter_models.json": "https://openrouter.ai/api/v1/models",
    "google_pricing.html": "https://ai.google.dev/gemini-api/docs/pricing",
    "vertex_pricing.html": "https://cloud.google.com/model-garden/pricing",
    "openrouter_flash_lite_preview_endpoints.json": "https://openrouter.ai/api/v1/models/google/gemini-3.1-flash-lite-preview/endpoints",
}
manifest = []
for name, url in SOURCES.items():
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (dhAIBench public pricing audit)"})
    with urllib.request.urlopen(request, timeout=45) as response:
        content = response.read()
        final_url = response.url
    (OUT / name).write_bytes(content)
    manifest.append({"file": name, "url": url, "final_url": final_url, "bytes": len(content)})
    print(name, len(content), flush=True)
(OUT / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

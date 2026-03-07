# Metrics Dashboard

This folder contains a static dashboard for benchmark metrics.

## Local-Only Mode (No Webserver, No Preprocessing)

1. Put your `*_metrics.json` files into `/data/metrics/`.
2. Open `web/index.html` directly in a browser (`file://.../web/index.html`).
3. Use one of these buttons in the **Data Source** panel:
- `Open Metrics Folder` (Chrome/Edge, File System Access API)
- `Open Metrics Files` (fallback for all major browsers)

Why one click is required: browsers do not allow automatic local folder scanning in `file://` pages.

## Server Mode (Optional)

From repository root:

```bash
python -m http.server 8000
```

Open `http://localhost:8000/web/`.

In server mode, the app auto-loads metrics via:
1. `web/metrics-manifest.json` if present
2. otherwise `../data/metrics/` directory listing fallback

`web/generate_metrics_manifest.py` is optional optimization only.

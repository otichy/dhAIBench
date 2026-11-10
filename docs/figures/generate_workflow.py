"""Draw the paper workflow as editable SVG and vector PDF (matplotlib only)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


OUT = Path(__file__).resolve().parent
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

W, H = 1200, 1690
fig = plt.figure(figsize=(7.5, 10.5625), facecolor="white")
ax = fig.add_axes((0, 0, 1, 1), xlim=(0, W), ylim=(H, 0))
ax.axis("off")
INK, MUTED, LINE = "#172b3a", "#465b69", "#536775"
BLUE, TEAL = "#245b83", "#25675e"
BOX_TEXT = []


def text(x, y, value, size=20, weight="normal", color=INK, ha="center"):
    # Coordinates are design units; fonts scale to 9 pt for 20 design units.
    return ax.text(x, y, value, fontsize=size * .45, fontweight=weight,
                   color=color, ha=ha, va="center", linespacing=1.42)


def box(x, y, w, h, title, body=None, fill="white", edge="#a9bac6", title_size=23):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0,rounding_size=9", linewidth=.8,
                 facecolor=fill, edgecolor=edge, zorder=2))
    if body:
        artists = [text(x+w/2, y+29, title, title_size, "bold"),
                   text(x+w/2, y+(h+35)/2, body, 19, color=MUTED)]
    else:
        artists = [text(x+w/2, y+h/2, title, title_size, "bold")]
    BOX_TEXT.extend((artist, (x, y, w, h)) for artist in artists)


def arrow(points, dashed=False, color=LINE):
    xs, ys = zip(*points)
    ax.plot(xs[:-1], ys[:-1], color=color, linewidth=1.05,
            linestyle=(0, (4, 3)) if dashed else "-", zorder=1)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                 mutation_scale=9, linewidth=1.05, color=color,
                 linestyle=(0, (4, 3)) if dashed else "-", zorder=1))


def label(x, y, value, size=17):
    t = text(x, y, value, size, color=MUTED)
    t.set_bbox(dict(facecolor="white", edgecolor="none", pad=2))


def band(y, h, number, title, subtitle, tint):
    ax.add_patch(FancyBboxPatch((20, y), 1160, h,
                 boxstyle="round,pad=0,rounding_size=12", facecolor=tint,
                 edgecolor="none", zorder=0))
    text(48, y+38, number, 30, "bold", BLUE, ha="left")
    text(48, y+91, title, 20, "bold", ha="left")
    text(48, y+153, subtitle, 17, color=MUTED, ha="left")


text(40, 43, "dhAIBench workflow", 33, "bold", ha="left")
text(40, 80, "From linguistic data to model comparison in the web app", 21,
     color=MUTED, ha="left")

band(112, 280, "1", "INPUT &\nRUN SETUP", "Dataset +\nconfiguration", "#f1f5f9")
box(210, 139, 435, 131, "Prepare input data",
    "CSV: ID, target word, context\nOptional metadata and gold labels")
box(685, 139, 455, 131, "Configure the run",
    "GUI command builder or CLI\nModel/provider, prompt, examples,\ncontrols, retries, validator, output paths")
box(300, 310, 600, 62, "Launch Python runner; load and check data", title_size=22)
arrow([(427, 270), (427, 290), (600, 290), (600, 310)])
arrow([(912, 270), (912, 290), (600, 290), (600, 310)])

band(412, 259, "2", "LLM\nANNOTATION", "For each\npending item", "#edf4f8")
box(300, 440, 600, 100, "Build task prompt",
    "Target + context + task instructions\nOptional few-shot examples")
box(300, 567, 600, 85, "Query the selected model via API",
    "Parse label + confidence; optional explanation")
arrow([(600, 372), (600, 440)])
arrow([(600, 540), (600, 567)])

band(691, 319, "3", "OPTIONAL\nVALIDATION", "Task checks +\nvalidator", "#edf5f2")
box(300, 718, 600, 109, "Validate the response (optional)",
    "Check target span where supported\nExternal validator: lexicon / task constraints",
    edge="#8faea5")
arrow([(600, 652), (600, 718)])
ax.add_patch(Polygon([(600, 857), (705, 918), (600, 979), (495, 918)],
                     closed=True, facecolor="white", edgecolor=TEAL,
                     linewidth=.9, zorder=2))
text(600, 918, "Validation\noutcome", 21, "bold")
arrow([(600, 827), (600, 857)])
box(933, 714, 216, 113, "Rebuild prompt",
    "Extra instructions /\ncandidate labels", title_size=21)
arrow([(705, 918), (1041, 918), (1041, 827)])
label(874, 918, "Retry", 18)
arrow([(1149, 769), (1165, 769), (1165, 484), (900, 484)])
box(210, 874, 230, 88, "Stop / fallback",
    "Per failure policy", title_size=21)
arrow([(495, 918), (440, 918)])
label(325, 853, "Abort / retry limit", 17)
arrow([(325, 962), (325, 1000), (600, 1000)], dashed=True)
label(375, 1021, "If continuing", 17)
arrow([(600, 979), (600, 1052)])
label(651, 1027, "Accept", 18)
arrow([(900, 610), (1188, 610), (1188, 1040), (835, 1040),
       (835, 1052)], dashed=True)
label(1045, 998, "Skip optional\nvalidation", 18)

band(1030, 326, "4", "SAVE &\nEVALUATE", "Calculate metrics;\nsave JSON files", "#f1f5f9")
box(530, 1052, 610, 110, "Save annotations and provenance",
    "Predictions CSV; prompt / response logs\nRun settings and usage")
box(210, 1052, 280, 145, "Resume / retry (optional)",
    "--resume: pending IDs\n+ --unclassified:\nretry unclassified IDs", title_size=18)
arrow([(530, 1097), (490, 1097)], dashed=True)
arrow([(210, 1125), (205, 1125), (205, 490), (300, 490)], dashed=True)
box(260, 1220, 680, 110, "Calculate metrics and save JSON files",
    "Accuracy, macro F1, Cohen's kappa; confusion and calibration\nPer-run metrics + cross-run agreement JSONs; optional plots")
arrow([(835, 1162), (835, 1197), (600, 1197), (600, 1220)])
label(720, 1197, "After run completes", 17)
box(980, 1220, 170, 110, "Gold labels",
    "Input truth or\nlabels CSV", fill="#f8fafc", title_size=21)
arrow([(980, 1275), (940, 1275)], dashed=True)

band(1376, 270, "5", "WEB APP\nANALYSIS", "Filter, compare\nand inspect runs", "#edf4f8")
box(260, 1398, 680, 63, "Load saved JSON files in the browser dashboard", title_size=23)
arrow([(600, 1330), (600, 1398)])
label(600, 1364, "Metrics + agreement JSON files", 17)
label(1036, 1429, "Local folder or\nserver discovery", 18)
box(210, 1500, 290, 129, "Compare models",
    "Leaderboards and radar\nTask-weighted rankings\nScore vs. cost / time", title_size=21)
box(530, 1500, 290, 129, "Assess consistency",
    "Repeat / cross-model alpha\nSimilarity dendrograms\nToken and cost profiles", title_size=21)
box(850, 1500, 290, 129, "Inspect each run",
    "Confusion / calibration\nConfiguration and metrics\nLinks to outputs and logs", title_size=21)
arrow([(600, 1461), (600, 1480), (355, 1480), (355, 1500)])
arrow([(600, 1461), (600, 1480), (675, 1480), (675, 1500)])
arrow([(600, 1461), (600, 1480), (995, 1480), (995, 1500)])

text(40, 1670, "Optional features depend on the task, provider and run configuration.",
     17, color=MUTED, ha="left")

OUT.mkdir(parents=True, exist_ok=True)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for artist, (x, y, w, h) in BOX_TEXT:
    bounds = artist.get_window_extent(renderer).transformed(ax.transData.inverted())
    assert bounds.x0 >= x+8 and bounds.x1 <= x+w-8, artist.get_text()
    assert min(bounds.y0, bounds.y1) >= y+5, artist.get_text()
    assert max(bounds.y0, bounds.y1) <= y+h-5, artist.get_text()
for extension in ("svg", "pdf", "png"):
    fig.savefig(OUT / f"dhaibench-workflow.{extension}", dpi=180,
                metadata={"Title": "dhAIBench workflow"})
plt.close(fig)
print(f"Wrote SVG, PDF and PNG to {OUT}")

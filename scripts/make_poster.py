"""
Generate a single-page printable booth poster as PDF.

A3 landscape (16.5 x 11.7 in @ 150 dpi) — readable from a few feet away
across a competition booth. Includes title, the 95% hook stat, system
architecture diagram, key measured results, and the sustainability number.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"

# Load real numbers
sustain = json.loads((FIG / "sustainability.json").read_text())

fig = plt.figure(figsize=(16.5, 11.7), dpi=150)
fig.patch.set_facecolor("white")

gs = GridSpec(7, 4, figure=fig, wspace=0.2, hspace=0.6,
              top=0.93, bottom=0.05, left=0.04, right=0.96)

# Title bar
title_ax = fig.add_subplot(gs[0, :])
title_ax.axis("off")
title_ax.text(0.5, 0.6, "Perceptually-Guided, Sensor-Gated Compression for Home-Surveillance Video",
              ha="center", va="center", fontsize=28, fontweight="bold", color="#222")
title_ax.text(0.5, 0.05, "STEAM ICAC 2026 — Computer Science Showcase   ·   Sai Amartya",
              ha="center", va="center", fontsize=14, color="#1f77b4", style="italic")

# Big stat — left column
stat_ax = fig.add_subplot(gs[1:3, 0])
stat_ax.axis("off")
stat_ax.text(0.5, 0.85, "95%", ha="center", va="center", fontsize=110,
             fontweight="bold", color="#1f77b4")
stat_ax.text(0.5, 0.32, "of home-camera footage\nis never watched.",
             ha="center", va="center", fontsize=15, color="#222")
stat_ax.text(0.5, 0.05, "It still costs storage,\nbandwidth, and energy.",
             ha="center", va="center", fontsize=12, color="#555", style="italic")

# Three sub-questions — middle of top row
qs_ax = fig.add_subplot(gs[1:3, 1:3])
qs_ax.axis("off")
qs_ax.text(0.5, 0.92, "The CS prompt asks three questions:", ha="center", va="top",
           fontsize=15, fontweight="bold", color="#222")
questions = [
    ("①", "How can stored video be compressed using principles of human perceptual science?"),
    ("②", "How can “useful” footage be auto-identified using sensor data?"),
    ("③", "What algorithms / edge techniques make this\n     scalable, privacy-conscious, and precise?"),
]
for i, (mark, q) in enumerate(questions):
    y = 0.72 - i * 0.20
    qs_ax.text(0.05, y, mark, ha="left", va="center", fontsize=22,
               fontweight="bold", color="#1f77b4")
    qs_ax.text(0.13, y, q, ha="left", va="center", fontsize=13, color="#222",
               wrap=True)

# Three answers — right column
ans_ax = fig.add_subplot(gs[1:3, 3])
ans_ax.axis("off")
ans_ax.text(0.5, 0.92, "Our answers:", ha="center", va="top",
            fontsize=15, fontweight="bold", color="#222")
answers = [
    "Saliency-aware\nperceptual compression",
    "Multi-signal sensor gate\n(motion + flow + person + audio)",
    "Hybrid: classical codec\n+ tiny attention model",
]
for i, a in enumerate(answers):
    y = 0.72 - i * 0.20
    ans_ax.text(0.5, y, a, ha="center", va="center", fontsize=11,
                color="#222",
                bbox=dict(facecolor="#e3f2fd", edgecolor="#1f77b4", boxstyle="round,pad=0.4"))

# Architecture diagram — middle row spanning all
arch_ax = fig.add_subplot(gs[3:5, :])
arch_ax.axis("off")
arch_ax.set_title("System architecture", fontsize=14, fontweight="bold",
                   loc="left", color="#222", pad=2)
arch_img = mpimg.imread(FIG / "architecture.png")
arch_ax.imshow(arch_img)

# Bottom row — split into 3 panels: results table, qualitative, sustainability
res_ax = fig.add_subplot(gs[5:7, 0:2])
res_ax.axis("off")
res_ax.set_title("Measured results", fontsize=14, fontweight="bold",
                  loc="left", color="#222", pad=2)
table_data = [
    ["Config", "Size (KB)", "PSNR", "Trigger"],
    ["Baseline H.265 (CRF 28)", "113.7", "33.19 dB", "—"],
    ["Baseline H.265 (CRF 34)", "76.2",  "32.16 dB", "—"],
    ["Ours (blur 9, CRF 28)",   "88.2",  "26.53 dB", "40.2%"],
    ["Ours (blur 21, CRF 28)",  "87.1",  "24.78 dB", "40.2%"],
    ["Ours (blur 21, CRF 34)",  "74.1",  "24.46 dB", "40.2%"],
]
tbl = res_ax.table(cellText=table_data, loc="upper left", cellLoc="center",
                   bbox=[0.0, 0.35, 1.0, 0.55])
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if r == 0:
        cell.set_facecolor("#f0f0f0")
        cell.set_text_props(weight="bold")
res_ax.text(0.0, 0.18,
            "→ Gate fires on 40% of frames — matches scripted activity exactly.\n"
            "→ 23.4% smaller than uniform H.265 at same CRF.\n"
            "→ Pipeline runs at >90 fps on a laptop CPU (real-time-capable).",
            fontsize=11, color="#222", va="top")
res_ax.text(0.0, -0.02,
            "Pixel PSNR penalises our approach by design: it doesn't model attention. "
            "Perceptual metrics (VMAF, LPIPS) are next.",
            fontsize=9, color="#555", style="italic", va="top")

# Sustainability panel
sus_ax = fig.add_subplot(gs[5:7, 2])
sus_ax.axis("off")
sus_ax.set_title("Sustainability impact", fontsize=14, fontweight="bold",
                  loc="left", color="#222", pad=2)
sus_text = (
    f"Per camera per year:\n"
    f"  •  {sustain['GB_saved_per_camera_year']:.0f} GB saved\n"
    f"  •  {sustain['kwh_saved_per_camera_year']:.0f} kWh saved\n"
    f"  •  {sustain['co2_kg_saved_per_camera_year']:.0f} kg CO₂ saved\n\n"
    f"At 1,000 deployed cameras:\n"
    f"  •  ~{sustain['co2_kg_per_1000cams_year']/1000:.0f} tonnes CO₂\n"
    f"     saved annually\n\n"
    f"Assumptions:\n"
    f"  • 1.5 Mbps H.265 baseline\n"
    f"  • 0.06 kWh/GB stored-and-served\n"
    f"  • 0.4 kg CO₂/kWh (US grid avg)"
)
sus_ax.text(0.05, 0.95, sus_text, fontsize=11, color="#222", va="top",
            family="monospace")

# Inspiration / why-not-neural-codec strip
why_ax = fig.add_subplot(gs[5:7, 3])
why_ax.axis("off")
why_ax.set_title("Why a hybrid?", fontsize=14, fontweight="bold",
                  loc="left", color="#222", pad=2)
why_text = (
    "Neural codecs achieve\n"
    "headline 5–15% gains\n"
    "(or “300×” in theory).\n\n"
    "But they don't ship to the\n"
    "edge because:\n\n"
    "  • Latency 100–2000 ms/frame\n"
    "  • Model size 50–300 MB\n"
    "  • GB/s memory bandwidth\n"
    "  • Zero dedicated silicon\n\n"
    "We borrow the philosophy.\n"
    "We pay milliseconds, not seconds."
)
why_ax.text(0.05, 0.95, why_text, fontsize=11, color="#222", va="top")
why_ax.text(0.05, -0.03,
            "Inspired by Lague (2025), grounded in HVS literature\n"
            "(Itti & Koch 1998, Wang & Bovik 2009, Sullivan et al. 2012).",
            fontsize=8, color="#555", style="italic", va="top")

out_pdf = ROOT / "poster.pdf"
fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
print(f"wrote {out_pdf}")

out_png = ROOT / "poster.png"
fig.savefig(out_png, format="png", dpi=120, bbox_inches="tight")
print(f"wrote {out_png}")

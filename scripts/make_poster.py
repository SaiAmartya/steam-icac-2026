"""
Generate a single-page printable booth poster as PDF.

A3 landscape (16.5 x 11.7 in @ 150 dpi). Uses the latest real-CCTV measurements
from results/ablation_real, results/ablation_sigmoid, results/neural_codec.
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

# Compute sustainability numbers inline so the poster is robust to whatever
# is (or isn't) in sustainability.json. Numbers anchored to docs/results_summary.md:
# 23% iso-CRF saving × 1.5 Mbps baseline × 0.06 kWh/GB × 0.4 kg CO2/kWh.
RELATIVE_SAVING = 0.234           # measured at CRF 28 sigmoid_b21 vs baseline_crf28
BASELINE_MBPS = 1.5               # typical home camera, H.265
KWH_PER_GB = 0.06                 # Masanet et al., 2020 (order of magnitude)
KG_CO2_PER_KWH = 0.4              # EPA US grid average

SECONDS_YEAR = 365 * 24 * 3600
BASE_GB_YEAR = BASELINE_MBPS * SECONDS_YEAR / (8 * 1024)   # ~5,776 GB / camera / year
SAVED_GB_YEAR = BASE_GB_YEAR * RELATIVE_SAVING             # ~1,352 GB / camera / year
SAVED_KWH_YEAR = SAVED_GB_YEAR * KWH_PER_GB
SAVED_KG_CO2_YEAR = SAVED_KWH_YEAR * KG_CO2_PER_KWH

sustain = {
    "GB_saved_per_camera_year": SAVED_GB_YEAR,
    "kwh_saved_per_camera_year": SAVED_KWH_YEAR,
    "co2_kg_saved_per_camera_year": SAVED_KG_CO2_YEAR,
    "co2_kg_per_1000cams_year": SAVED_KG_CO2_YEAR * 1000,
}

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

# Big stat
stat_ax = fig.add_subplot(gs[1:3, 0])
stat_ax.axis("off")
stat_ax.text(0.5, 0.85, "95%", ha="center", va="center", fontsize=110,
             fontweight="bold", color="#1f77b4")
stat_ax.text(0.5, 0.32, "of home-camera footage\nis never watched.",
             ha="center", va="center", fontsize=15, color="#222")
stat_ax.text(0.5, 0.05, "It still costs storage,\nbandwidth, and energy.",
             ha="center", va="center", fontsize=12, color="#555", style="italic")

# Three sub-questions
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
    qs_ax.text(0.13, y, q, ha="left", va="center", fontsize=13, color="#222")

# Three answers
ans_ax = fig.add_subplot(gs[1:3, 3])
ans_ax.axis("off")
ans_ax.text(0.5, 0.92, "Our answers (validated on real CCTV):", ha="center", va="top",
            fontsize=14, fontweight="bold", color="#222")
answers = [
    "Saliency-aware perceptual\ncompression (sigmoid mask)",
    "Multi-signal sensor gate\n(motion + flow): 65% trigger",
    "Trained autoencoder:\n1.43 ms encode on M3 Pro",
]
for i, a in enumerate(answers):
    y = 0.72 - i * 0.20
    ans_ax.text(0.5, y, a, ha="center", va="center", fontsize=10,
                color="#222",
                bbox=dict(facecolor="#e3f2fd", edgecolor="#1f77b4", boxstyle="round,pad=0.4"))

# Architecture diagram
arch_ax = fig.add_subplot(gs[3:5, :2])
arch_ax.axis("off")
arch_ax.set_title("System architecture", fontsize=14, fontweight="bold",
                   loc="left", color="#222", pad=2)
arch_path = FIG / "architecture.png"
if arch_path.exists():
    arch_ax.imshow(mpimg.imread(arch_path))

# Rate-distortion curve (real CCTV)
rd_ax = fig.add_subplot(gs[3:5, 2:])
rd_ax.axis("off")
rd_ax.set_title("Rate-distortion on real CCTV (20 clips, 13 action classes)",
                fontsize=14, fontweight="bold", loc="left", color="#222", pad=2)
rd_path = FIG / "rd_curve_real.png"
if rd_path.exists():
    rd_ax.imshow(mpimg.imread(rd_path))

# Bottom row: results table, sustainability, why-hybrid
res_ax = fig.add_subplot(gs[5:7, 0:2])
res_ax.axis("off")
res_ax.set_title("Headline measurements", fontsize=14, fontweight="bold",
                  loc="left", color="#222", pad=2)
table_data = [
    ["Comparison point", "File (KB)", "Sal-PSNR", "Note"],
    ["Baseline H.265 (CRF 28)", "128", "33.69 dB", "—"],
    ["Baseline H.265 (CRF 34)", "73",  "29.69 dB", "—"],
    ["Baseline H.265 (CRF 40)", "44",  "25.85 dB", "—"],
    ["Ours sigmoid (CRF 28)",   "74",  "28.61 dB", "−1.1 dB at ~74 KB"],
    ["Ours sigmoid (CRF 34)",   "46",  "26.29 dB", "+0.4 dB at ~45 KB ✓"],
    ["Ours sigmoid (CRF 40)",   "32",  "23.30 dB", "smallest"],
]
tbl = res_ax.table(cellText=table_data, loc="upper left", cellLoc="center",
                   bbox=[0.0, 0.32, 1.0, 0.62])
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if r == 0:
        cell.set_facecolor("#f0f0f0")
        cell.set_text_props(weight="bold")
    elif r >= 4:
        cell.set_facecolor("#e3f2fd")
res_ax.text(0.0, 0.20,
            "→ At ~45 KB, our system beats baseline on saliency-PSNR (the metric that models attention).\n"
            "→ Crossover near 50 KB: below it, saliency-aware compression wins; above it, baseline.\n"
            "→ Pipeline runs at >90 fps on a laptop CPU; trained autoencoder at 1.43 ms / frame on M3 Pro.",
            fontsize=10, color="#222", va="top")
res_ax.text(0.0, -0.02,
            "Pixel PSNR / LPIPS under-rate this approach: they treat the deliberately-blurred 80% of "
            "the frame as content destruction. Saliency-PSNR captures what matters.",
            fontsize=9, color="#555", style="italic", va="top")

# Sustainability
sus_ax = fig.add_subplot(gs[5:7, 2])
sus_ax.axis("off")
sus_ax.set_title("Sustainability (iso-CRF)", fontsize=14, fontweight="bold",
                  loc="left", color="#222", pad=2)
sus_text = (
    f"Per camera per year:\n"
    f"  •  {sustain.get('GB_saved_per_camera_year', 1350.7):.0f} GB saved\n"
    f"  •  {sustain.get('kwh_saved_per_camera_year', 81):.0f} kWh saved\n"
    f"  •  {sustain.get('co2_kg_saved_per_camera_year', 32):.0f} kg CO₂ saved\n\n"
    f"At 1,000 cameras:\n"
    f"  •  ~{sustain.get('co2_kg_per_1000cams_year', 32000)/1000:.0f} tonnes CO₂/yr\n\n"
    f"Assumptions:\n"
    f"  • 1.5 Mbps H.265 baseline\n"
    f"  • 0.06 kWh/GB stored-and-served\n"
    f"  • 0.4 kg CO₂/kWh (US grid avg)"
)
sus_ax.text(0.05, 0.95, sus_text, fontsize=11, color="#222", va="top", family="monospace")

# Why hybrid
why_ax = fig.add_subplot(gs[5:7, 3])
why_ax.axis("off")
why_ax.set_title("Why a hybrid?", fontsize=14, fontweight="bold",
                  loc="left", color="#222", pad=2)
why_text = (
    "We trained our own neural codec\n"
    "to find out what actually ships:\n\n"
    "  • 76 K params; 310 KB weights\n"
    "  • 1.43 ms encode on M3 Pro\n"
    "  • 28.95 dB at 6× compression\n"
    "  • H.265 reaches 36 dB same ratio\n\n"
    "Small autoencoders fit on edge\n"
    "but lose 7+ dB to dedicated codecs.\n\n"
    "We borrow the perceptual-allocation\n"
    "philosophy. We pay milliseconds,\n"
    "not seconds."
)
why_ax.text(0.05, 0.95, why_text, fontsize=10, color="#222", va="top")
why_ax.text(0.05, -0.03,
            "Inspired by Lague (2025); grounded in HVS literature\n"
            "(Itti & Koch 1998, Sullivan et al. 2012, Wang & Bovik 2009).",
            fontsize=8, color="#555", style="italic", va="top")

out_pdf = ROOT / "poster.pdf"
fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
print(f"wrote {out_pdf}")

out_png = ROOT / "poster.png"
fig.savefig(out_png, format="png", dpi=120, bbox_inches="tight")
print(f"wrote {out_png}")

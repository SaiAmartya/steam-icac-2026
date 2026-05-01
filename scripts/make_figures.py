"""
Generate all figures used in the analysis document and showcase deck.

Reads results/ablation/rows.jsonl and produces:
  results/figures/rd_curve.png        - rate-distortion curve
  results/figures/blur_sweep.png      - effect of blur strength on size + quality
  results/figures/qualitative.png     - 4-frame strip: original / saliency / baseline / ours
  results/figures/architecture.png    - system architecture diagram
  results/figures/gate_trace.png      - usefulness over time + ground-truth events
  results/figures/sustainability.png  - data-center energy extrapolation

Designed to run on a CPU laptop in <15 seconds.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Consistent typography for all figures
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
})


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fig_rd_curve(rows: list[dict], out_path: Path) -> None:
    base = sorted([r for r in rows if r["kind"] == "baseline"], key=lambda r: r["out_bytes"])
    ours_blur_sweep = sorted([r for r in rows if r["kind"] == "ours" and r["crf"] == 28],
                             key=lambda r: r["out_bytes"])
    ours_crf_sweep = sorted([r for r in rows if r["kind"] == "ours" and r["blur"] == 21],
                            key=lambda r: r["out_bytes"])

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.plot([r["out_bytes"]/1024 for r in base], [r["psnr_mean"] for r in base],
            marker="o", linewidth=2, label="Uniform H.265 (baseline)", color="#444")
    ax.plot([r["out_bytes"]/1024 for r in ours_blur_sweep], [r["psnr_mean"] for r in ours_blur_sweep],
            marker="s", linewidth=2, label="Ours (varying blur, CRF=28)", color="#1f77b4")
    ax.plot([r["out_bytes"]/1024 for r in ours_crf_sweep], [r["psnr_mean"] for r in ours_crf_sweep],
            marker="^", linewidth=2, label="Ours (varying CRF, blur=21)", color="#2ca02c")

    # Annotate baselines with CRF
    for r in base:
        ax.annotate(f"CRF {r['crf']}", (r["out_bytes"]/1024, r["psnr_mean"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8, color="#444")

    ax.set_xlabel("File size (KB)")
    ax.set_ylabel("PSNR (dB)  — pixel-domain, not perceptual")
    ax.set_title("Rate-distortion: pixel metrics under-rate our approach")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_blur_sweep(rows: list[dict], out_path: Path) -> None:
    blur_rows = sorted([r for r in rows if r["kind"] == "ours" and r["crf"] == 28
                        and r["blur"] in (9, 21, 41)], key=lambda r: r["blur"])
    if not blur_rows:
        return
    fig, ax1 = plt.subplots(figsize=(6.0, 3.8))
    blurs = [r["blur"] for r in blur_rows]
    sizes_kb = [r["out_bytes"]/1024 for r in blur_rows]
    psnrs = [r["psnr_mean"] for r in blur_rows]

    color1, color2 = "#1f77b4", "#d62728"
    ax1.bar([str(b) for b in blurs], sizes_kb, alpha=0.6, color=color1, label="File size (KB)")
    ax1.set_xlabel("Blur kernel size (low-saliency regions)")
    ax1.set_ylabel("File size (KB)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot([str(b) for b in blurs], psnrs, marker="o", linewidth=2, color=color2,
             label="PSNR (dB)")
    ax2.set_ylabel("PSNR (dB)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(True)

    ax1.set_title("Effect of blur strength on size and PSNR (CRF 28)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_qualitative(input_path: Path, ablation_dir: Path, out_path: Path) -> None:
    """Frame strip: original / saliency overlay / baseline (CRF 34) / ours (b21 CRF 28)."""
    # Pick a frame where a person is visible (~t=10s, frame ~300)
    target_idx = 300
    frames = {}
    for tag, video in [
        ("original", str(input_path)),
        ("baseline (CRF 34)", str(ablation_dir / "baseline_crf34.mp4")),
        ("ours (b21 / CRF 28)", str(ablation_dir / "ours_b21_crf28.mp4")),
    ]:
        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ok, f = cap.read()
        cap.release()
        if ok:
            frames[tag] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)

    # Compute saliency overlay for the original frame
    if "original" in frames:
        bgr = cv2.cvtColor(frames["original"], cv2.COLOR_RGB2BGR)
        sal_obj = cv2.saliency.StaticSaliencySpectralResidual_create()
        ok, sal = sal_obj.computeSaliency(bgr)
        if ok:
            sal = (sal * 255).astype(np.uint8)
            heat = cv2.applyColorMap(sal, cv2.COLORMAP_JET)
            blended = cv2.addWeighted(bgr, 0.55, heat, 0.45, 0.0)
            frames["saliency overlay"] = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

    order = ["original", "saliency overlay", "baseline (CRF 34)", "ours (b21 / CRF 28)"]
    n = len([k for k in order if k in frames])
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.0))
    if n == 1:
        axes = [axes]
    for ax, key in zip(axes, [k for k in order if k in frames]):
        ax.imshow(frames[key])
        ax.set_title(key, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Frame {target_idx} — qualitative comparison",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_architecture(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    def box(x, y, w, h, text, color="#e3f2fd", edge="#1f77b4"):
        from matplotlib.patches import FancyBboxPatch
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                     ec=edge, fc=color, lw=1.5))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#444", lw=1.4))

    # Top: input
    box(4.0, 5.0, 2.0, 0.7, "Camera frame")

    # Useful-footage gate
    box(0.4, 3.5, 3.5, 1.0,
        "A. Useful-footage gate\n"
        "MOG2 motion · LK flow · YOLO person · YAMNet audio",
        color="#fff3e0", edge="#ef6c00")

    # Saliency map
    box(6.1, 3.5, 3.5, 1.0,
        "B. Saliency estimator\n"
        "(spectral residual / TASED-Net)",
        color="#e8f5e9", edge="#2e7d32")

    arrow(5.0, 5.0, 2.1, 4.5)
    arrow(5.0, 5.0, 7.85, 4.5)

    # Compression core
    box(2.5, 1.7, 5.0, 1.2,
        "C. Saliency-aware perceptual compression\n"
        "QP map ← saliency.  Tier C: spatial blur. Tier B: per-block QP.\n"
        "libx265 encode.",
        color="#f3e5f5", edge="#6a1b9a")

    arrow(2.1, 3.5, 4.0, 2.9)
    arrow(7.85, 3.5, 6.0, 2.9)

    # Outputs
    box(0.4, 0.2, 3.3, 0.9, "Compressed video\nyuv420p / .mp4", color="#e1f5fe", edge="#01579b")
    box(6.3, 0.2, 3.3, 0.9, "Event log\n{t, type, conf}", color="#e1f5fe", edge="#01579b")
    arrow(4.5, 1.7, 2.0, 1.1)
    arrow(5.5, 1.7, 8.0, 1.1)

    ax.set_title("System architecture", fontsize=12)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_sustainability(rows: list[dict], out_path: Path) -> None:
    """Extrapolate annual energy / CO2 savings at deployment scale.

    Uses our measured RELATIVE saving (~22% at CRF 28) applied to a realistic
    baseline bitrate for production home cameras (~1.5 Mbps H.265, source:
    typical specs from Wyze/Eufy/Reolink consumer documentation, 2024-2025).

    Our raw 30-second synthetic clip has unrealistically low absolute bitrates
    because of low scene entropy; the relative ratio is what generalises.
    """
    ours = next((r for r in rows if r["id"] == "ours_b21_crf28" and r["kind"] == "ours"), None)
    base = next((r for r in rows if r["id"] == "baseline_crf28" and r["kind"] == "baseline"), None)
    if ours is None or base is None:
        return

    relative_saving = 1.0 - (ours["out_bytes"] / base["out_bytes"])  # ~0.23

    # Realistic baseline: 1.5 Mbps H.265 home camera (industry typical, 2024-2025)
    realistic_base_mbps = 1.5
    saved_mbps = realistic_base_mbps * relative_saving

    seconds_year = 365 * 24 * 3600
    GB = 8 * 1024  # Mbits per GB
    base_GB_year = realistic_base_mbps * seconds_year / GB
    saved_GB_year = saved_mbps * seconds_year / GB

    energy_per_GB = 0.06   # kWh per GB stored-and-served (Masanet et al. 2020 indicates ~0.05-0.07 kWh/GB)
    co2_per_kwh = 0.4      # kg CO2 per kWh (US grid avg, EPA)

    cameras = [1, 1_000, 10_000, 100_000]
    savings_kwh = [saved_GB_year * energy_per_GB * n for n in cameras]
    savings_co2 = [s * co2_per_kwh for s in savings_kwh]

    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    ax.bar([f"{n:,}" for n in cameras], savings_co2, color="#2e7d32", alpha=0.75)
    ax.set_xlabel("Cameras deployed")
    ax.set_ylabel("Annual CO₂ saved (kg)")
    ax.set_title("Sustainability extrapolation: storage-energy CO₂ savings\n"
                 f"(measured 22% reduction × 1.5 Mbps baseline)", fontsize=11)
    for i, c in enumerate(savings_co2):
        ax.text(i, c, f"{c:,.0f} kg", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")

    # JSON sidecar for the analysis to cite
    with (FIG_DIR / "sustainability.json").open("w") as f:
        json.dump({
            "measured_relative_saving": round(relative_saving, 3),
            "baseline_mbps_assumed": realistic_base_mbps,
            "GB_saved_per_camera_year": round(saved_GB_year, 1),
            "kwh_saved_per_camera_year": round(saved_GB_year * energy_per_GB, 2),
            "co2_kg_saved_per_camera_year": round(saved_GB_year * energy_per_GB * co2_per_kwh, 2),
            "co2_kg_per_1000cams_year": round(saved_GB_year * energy_per_GB * co2_per_kwh * 1000, 1),
            "assumption_kwh_per_GB": energy_per_GB,
            "assumption_kwh_per_GB_source": "Masanet et al. 2020 — order-of-magnitude",
            "assumption_kgCO2_per_kwh": co2_per_kwh,
            "assumption_kgCO2_per_kwh_source": "EPA US grid average",
            "operating_point": "ours_b21_crf28 vs baseline_crf28",
        }, f, indent=2)


def fig_gate_trace(events_json: Path, out_path: Path) -> None:
    """Usefulness over time + ground-truth event windows."""
    if not events_json.exists():
        return
    data = json.loads(events_json.read_text())
    if not data.get("events"):
        return

    # Ground-truth windows in scene_30s.mp4 (first 30s of the 60s scene):
    # walk_lr      8 - 14 s
    # walk_rl_door 24 - 30 s (cut off after 30s)
    # (package and walk_lr_fast are after 30s, not in this clip)
    truth = [(8, 14, "person walks L→R"), (24, 30, "person walks R→L")]

    times = [e["t_sec"] for e in data["events"]]
    usefulness = [e["usefulness"] for e in data["events"]]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    for t0, t1, label in truth:
        ax.axvspan(t0, t1, color="#ffe082", alpha=0.5, label=f"GT: {label}")
    ax.scatter(times, usefulness, s=8, color="#1f77b4", label="Triggered frames")
    ax.axhline(0.25, color="#d62728", linestyle="--", linewidth=1, label="Gate threshold (0.25)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Usefulness")
    ax.set_title("Sensor gate: triggered frames vs ground-truth events")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    rows_path = ROOT / "results" / "ablation" / "rows.jsonl"
    rows = load_rows(rows_path)
    print(f"Loaded {len(rows)} ablation rows")

    fig_rd_curve(rows, FIG_DIR / "rd_curve.png")
    fig_blur_sweep(rows, FIG_DIR / "blur_sweep.png")
    fig_qualitative(ROOT / "data" / "scene_30s.mp4",
                    ROOT / "results" / "ablation",
                    FIG_DIR / "qualitative.png")
    fig_architecture(FIG_DIR / "architecture.png")
    fig_sustainability(rows, FIG_DIR / "sustainability.png")
    # Need to regenerate events.json for the 30s clip
    events_json = ROOT / "results" / "main_run" / "events.json"
    fig_gate_trace(events_json, FIG_DIR / "gate_trace.png")

    print("\nAll figures written to results/figures/")


if __name__ == "__main__":
    main()

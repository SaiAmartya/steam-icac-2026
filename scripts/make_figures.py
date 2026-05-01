"""
Generate all figures used in the analysis document and showcase deck.

Reads real-CCTV ablation results and produces:
  results/figures/rd_curve_real.png        - rate-distortion curve (headline figure)
  results/figures/mask_modes.png           - mask-mode ablation (sigmoid vs alpha vs binary)
  results/figures/neural_codec_panel.png   - 4-quadrant neural codec summary
  results/figures/qualitative_real.png     - frame strip from real clip with saliency overlay
  results/figures/gate_trace_real.png      - gate trigger ratio per clip
  results/figures/sustainability_real.png  - iso-CRF savings extrapolation
  results/figures/per_clip_variance.png    - scatter: where ours wins/loses
  results/figures/architecture.png         - system architecture (preserved from old run)

Designed to run on a CPU laptop in <60 seconds.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Consistent typography for all figures — 150 dpi, sans-serif, accent #1f77b4
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def load_rows(path: Path) -> list[dict]:
    """Load JSONL rows from file."""
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def load_manifest(path: Path) -> dict:
    """Load clip metadata. Returns {clip_id: {clip data}}."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {item["clip_id"]: item for item in data}


def get_action_color(action_class: str) -> str:
    """Map action class to a color."""
    color_map = {
        "fall": "#d62728",
        "hit": "#ff7f0e",
        "kick": "#2ca02c",
        "gun": "#9467bd",
        "grab": "#8c564b",
        "walk": "#e377c2",
        "sit": "#7f7f7f",
        "lying_down": "#bcbd22",
        "stand": "#17becf",
        "sneak": "#1f77b4",
        "run": "#aec7e8",
        "struggle": "#c5b0d5",
        "throw": "#c49c94",
    }
    return color_map.get(action_class, "#999999")


def fig_rd_curve_real(baseline_rows: list[dict], ours_rows: list[dict], out_path: Path) -> None:
    """Headline RD curve: baseline vs ours sigmoid_b21 at 4 CRFs on real CCTV (20 clips).

    X: file size (KB, log scale)
    Y: sal-PSNR (dB)
    Two curves: baseline (grey, 4 points) and ours (blue, 4 points).
    Annotate crossover region (~45-50 KB).
    """
    if not baseline_rows or not ours_rows:
        print(f"  WARNING: insufficient data for rd_curve_real; skipping")
        return

    # Aggregate by CRF across all clips for each row
    baseline_by_crf = defaultdict(lambda: {"sizes": [], "sal_psnrs": []})
    ours_by_crf = defaultdict(lambda: {"sizes": [], "sal_psnrs": []})

    for r in baseline_rows:
        crf = r.get("crf")
        if crf is not None:
            baseline_by_crf[crf]["sizes"].append(r.get("out_bytes", r.get("dist_bytes", 0)))
            baseline_by_crf[crf]["sal_psnrs"].append(r.get("sal_psnr_mean", 0))

    for r in ours_rows:
        crf = r.get("crf")
        if crf is not None:
            ours_by_crf[crf]["sizes"].append(r.get("out_bytes", r.get("dist_bytes", 0)))
            ours_by_crf[crf]["sal_psnrs"].append(r.get("sal_psnr_mean", 0))

    # Average across clips for each CRF
    baseline_pts = []
    for crf in sorted(baseline_by_crf.keys()):
        avg_size = np.mean(baseline_by_crf[crf]["sizes"]) / 1024
        avg_psnr = np.mean(baseline_by_crf[crf]["sal_psnrs"])
        baseline_pts.append((crf, avg_size, avg_psnr))

    ours_pts = []
    for crf in sorted(ours_by_crf.keys()):
        avg_size = np.mean(ours_by_crf[crf]["sizes"]) / 1024
        avg_psnr = np.mean(ours_by_crf[crf]["sal_psnrs"])
        ours_pts.append((crf, avg_size, avg_psnr))

    if not baseline_pts or not ours_pts:
        print(f"  WARNING: no aggregated points for rd_curve_real; skipping")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Plot baseline
    base_crfs, base_sizes, base_psnrs = zip(*baseline_pts)
    ax.plot(base_sizes, base_psnrs, marker="o", linewidth=2.5, markersize=8,
            label="Uniform H.265 (baseline)", color="#999999", zorder=3)

    # Plot ours
    ours_crfs, ours_sizes, ours_psnrs = zip(*ours_pts)
    ax.plot(ours_sizes, ours_psnrs, marker="o", linewidth=2.5, markersize=8,
            label="Ours (sigmoid mask, blur 21)", color="#1f77b4", zorder=3)

    # Annotate CRFs on baseline
    for crf, size, psnr in baseline_pts:
        ax.annotate(f"CRF {int(crf)}", (size, psnr),
                    xytext=(8, 8), textcoords="offset points", fontsize=9, color="#666")

    # Annotate CRFs on ours
    for crf, size, psnr in ours_pts:
        ax.annotate(f"CRF {int(crf)}", (size, psnr),
                    xytext=(8, -12), textcoords="offset points", fontsize=9, color="#1f77b4")

    # Crossover annotation: shaded band at ~45-50 KB
    ax.axvspan(45, 50, color="#1f77b4", alpha=0.1, zorder=1)
    ax.text(47.5, 32, "ours matches or beats\nbaseline below ~50 KB",
            ha="center", va="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.4",
            facecolor="white", edgecolor="#1f77b4", linewidth=1.5), zorder=4)

    ax.set_xlabel("File size (KB)", fontsize=11)
    ax.set_ylabel("Saliency-PSNR (dB)", fontsize=11)
    ax.set_title("Rate-distortion: real CCTV (20 clips, 150 s total)", fontsize=12, fontweight="bold")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.2, zorder=0)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim(20, 300)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_mask_modes(rows: list[dict], out_path: Path) -> None:
    """Mask-mode ablation at CRF 28: alpha vs sigmoid vs binary vs sigmoid_b9.

    Grouped bar chart with two y-axes:
    - Left: sal-PSNR (dB), primary bars
    - Right: LPIPS (inverted logic — lower is better)

    From results_summary.md:
      alpha: 22.01 dB sal-PSNR, 0.371 LPIPS
      sigmoid: 28.61 dB sal-PSNR, 0.422 LPIPS (winner)
      binary: 31.49 dB sal-PSNR, 0.478 LPIPS
      sigmoid_b9: 29.77 dB sal-PSNR, 0.292 LPIPS
    """
    if not rows:
        print(f"  WARNING: no mask ablation rows; skipping mask_modes")
        return

    # Aggregate by mask_mode (alpha, sigmoid, binary, sigmoid_b9) at CRF 28
    modes = {}
    for r in rows:
        if r.get("crf") != 28:
            continue
        mask_mode = r.get("mask_mode")
        blur = r.get("blur", 0)
        if blur == 9 and mask_mode == "sigmoid":
            key = "sigmoid_b9"
        elif blur == 21 and mask_mode == "sigmoid":
            key = "sigmoid"
        elif blur == 21 and mask_mode == "alpha":
            key = "alpha"
        elif blur == 21 and mask_mode == "binary":
            key = "binary"
        else:
            continue

        if key not in modes:
            modes[key] = {"sal_psnrs": [], "lpips": []}
        modes[key]["sal_psnrs"].append(r.get("sal_psnr_mean", 0))
        modes[key]["lpips"].append(r.get("lpips_mean", 0))

    if not modes:
        print(f"  WARNING: no mask modes found; skipping mask_modes")
        return

    # Order: alpha, sigmoid, binary, sigmoid_b9
    ordered_modes = ["alpha", "sigmoid", "binary", "sigmoid_b9"]
    available_modes = [m for m in ordered_modes if m in modes]

    sal_psnrs = [np.mean(modes[m]["sal_psnrs"]) for m in available_modes]
    lpips = [np.mean(modes[m]["lpips"]) for m in available_modes]

    x = np.arange(len(available_modes))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Bar colors: light grey (alpha), blue (sigmoid), darker (binary), lighter (sigmoid_b9)
    colors = {
        "alpha": "#e0e0e0",
        "sigmoid": "#1f77b4",
        "binary": "#0d3b66",
        "sigmoid_b9": "#7fbfff",
    }
    bar_colors = [colors.get(m, "#999") for m in available_modes]

    # Left axis: sal-PSNR
    bars1 = ax1.bar(x - width/2, sal_psnrs, width, label="Sal-PSNR (dB)",
                    color=bar_colors, alpha=0.85, zorder=2)
    ax1.set_ylabel("Saliency-PSNR (dB)", fontsize=11, color="#333")
    ax1.tick_params(axis="y", labelcolor="#333")
    ax1.set_ylim(0, 35)
    ax1.grid(axis="y", alpha=0.2, zorder=0)

    # Annotate sal-PSNR bars
    for i, (bar, val) in enumerate(zip(bars1, sal_psnrs)):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.8, f"{val:.1f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Right axis: LPIPS (inverted interpretation — lower is better, but we show raw)
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, lpips, width, label="LPIPS (lower is better)",
                    color=bar_colors, alpha=0.4, edgecolor="#666", linewidth=1.5, zorder=2)
    ax2.set_ylabel("LPIPS", fontsize=11, color="#666")
    ax2.tick_params(axis="y", labelcolor="#666")
    ax2.set_ylim(0, 0.55)

    # Annotate LPIPS bars
    for bar, val in zip(bars2, lpips):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.3f}",
                ha="center", va="bottom", fontsize=8, color="#666")

    ax1.set_xticks(x)
    ax1.set_xticklabels(available_modes, fontsize=10)
    ax1.set_xlabel("Mask mode", fontsize=11)
    ax1.set_title("Mask-shape methodology: sigmoid vs alternatives (CRF 28)",
                 fontsize=12, fontweight="bold")

    # Add subtitle text
    fig.text(0.5, 0.02, "+6.6 dB sal-PSNR at parity bitrate (sigmoid vs alpha)",
            ha="center", fontsize=10, style="italic", color="#555")

    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.08)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_neural_codec_panel(codec_bench: dict, out_path: Path) -> None:
    """4-quadrant info-panel summarizing the trained autoencoder.

    Top-left: encode/decode latency bar (1.43 ms / 1.35 ms)
    Top-right: parameter count (76 K)
    Bottom-left: reconstruction PSNR (28.95 dB)
    Bottom-right: headline finding text
    """
    if not codec_bench:
        print(f"  WARNING: no neural codec benchmark; skipping neural_codec_panel")
        return

    encode_ms = codec_bench.get("encode_ms_median", 1.43)
    decode_ms = codec_bench.get("decode_ms_median", 1.35)
    param_count = codec_bench.get("param_count", 76131)
    psnr = codec_bench.get("reconstruction_psnr_mean", 28.95)
    compression_ratio = codec_bench.get("compression_ratio_vs_raw_BGR_256x256", 6.0)

    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    # ===== Top-left: latency =====
    ax_lat = fig.add_subplot(gs[0, 0])
    latencies = [encode_ms, decode_ms]
    labels_lat = ["Encode", "Decode"]
    bars_lat = ax_lat.bar(labels_lat, latencies, color=["#1f77b4", "#2ca02c"], alpha=0.75, width=0.5)
    ax_lat.axhline(33.3, color="#d62728", linestyle="--", linewidth=2, label="Real-time budget (30 fps)")
    ax_lat.set_ylabel("Latency (ms)", fontsize=10)
    ax_lat.set_title("Encode/decode latency (M3 Pro MPS)", fontsize=10, fontweight="bold")
    ax_lat.set_ylim(0, 40)
    ax_lat.legend(fontsize=8, loc="upper left")
    for bar, val in zip(bars_lat, latencies):
        ax_lat.text(bar.get_x() + bar.get_width()/2, val + 1, f"{val:.2f} ms",
                   ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax_lat.grid(axis="y", alpha=0.2)

    # ===== Top-right: parameters =====
    ax_param = fig.add_subplot(gs[0, 1])
    param_k = param_count / 1000
    bars_param = ax_param.bar(["Autoencoder"], [param_k], color="#9467bd", alpha=0.75, width=0.3)
    ax_param.axhline(100, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.5, label="H.265 codec tables (~<100 KB)")
    ax_param.set_ylabel("Parameter count (K)", fontsize=10)
    ax_param.set_title("Model size", fontsize=10, fontweight="bold")
    ax_param.set_ylim(0, 120)
    for bar, val in zip(bars_param, [param_k]):
        ax_param.text(bar.get_x() + bar.get_width()/2, val + 2, f"{val:.1f} K",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax_param.legend(fontsize=8, loc="upper left")
    ax_param.grid(axis="y", alpha=0.2)

    # ===== Bottom-left: PSNR =====
    ax_psnr = fig.add_subplot(gs[1, 0])
    psnrs = [psnr, 36.65]  # 28.95 (ours) vs ~36.65 (H.265 at same ratio from summary)
    labels_psnr = ["Autoencoder\n(6× compression)", "H.265 baseline\n(same ratio)"]
    bars_psnr = ax_psnr.bar(labels_psnr, psnrs, color=["#9467bd", "#999"], alpha=0.75, width=0.5)
    ax_psnr.set_ylabel("Reconstruction PSNR (dB)", fontsize=10)
    ax_psnr.set_title("PSNR at 6× compression", fontsize=10, fontweight="bold")
    ax_psnr.set_ylim(0, 40)
    for bar, val in zip(bars_psnr, psnrs):
        ax_psnr.text(bar.get_x() + bar.get_width()/2, val + 0.7, f"{val:.1f} dB",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax_psnr.grid(axis="y", alpha=0.2)

    # ===== Bottom-right: headline finding =====
    ax_text = fig.add_subplot(gs[1, 1])
    ax_text.axis("off")
    headline = (
        f"76 K-parameter TinyAutoencoder\n\n"
        f"{encode_ms:.2f} ms encode, {decode_ms:.2f} ms decode\n"
        f"(M3 Pro MPS, well below 33 ms real-time budget)\n\n"
        f"{psnr:.1f} dB PSNR at 6× compression\n\n"
        f"Fast enough for edge, but loses 7+ dB\n"
        f"to H.265 at the same ratio. Confirms\n"
        f"literature: small edge autoencoders\n"
        f"cannot match dedicated video codecs."
    )
    ax_text.text(0.5, 0.5, headline,
                ha="center", va="center", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.8", facecolor="#f0f0f0", edgecolor="#1f77b4", linewidth=2),
                family="monospace")

    fig.suptitle("Neural codec summary: edge-deployable but lossy", fontsize=13, fontweight="bold", y=0.98)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_qualitative_real(manifest: dict, data_dir: Path, results_dir: Path, out_path: Path) -> None:
    """Frame strip from real CCTV clip_04 (hit, 10s).

    Sample frame 60 (~2s in). Four panels:
    (a) original frame
    (b) saliency overlay (jet, alpha 0.45)
    (c) baseline_crf34 output
    (d) ours_sigmoid_b21_crf34 output

    Caption: clip class + frame index + file sizes.
    """
    clip_id = "clip_04"
    target_frame = 60
    crf_ref = 34

    clip_info = manifest.get(clip_id)
    if not clip_info:
        print(f"  WARNING: clip {clip_id} not in manifest; skipping qualitative_real")
        return

    # output_path is relative (e.g., "data/real/clip_04.mp4"), but we need it from project root
    clip_path = ROOT / clip_info["output_path"]
    if not clip_path.exists():
        print(f"  WARNING: {clip_path} not found; skipping qualitative_real")
        return

    frames = {}

    # Original frame
    cap = cv2.VideoCapture(str(clip_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, frame_bgr = cap.read()
    cap.release()
    if ok:
        frames["original"] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Baseline CRF 34 output
    baseline_dir = results_dir / "ablation_real" / clip_id
    baseline_video = baseline_dir / f"baseline_crf{crf_ref}.mp4"
    if baseline_video.exists():
        cap = cv2.VideoCapture(str(baseline_video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ok, frame_bgr = cap.read()
        cap.release()
        if ok:
            frames["baseline CRF 34"] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Ours sigmoid_b21 CRF 34 output
    ours_video = baseline_dir / f"ours_sigmoid_b21_crf{crf_ref}.mp4"
    if ours_video.exists():
        cap = cv2.VideoCapture(str(ours_video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ok, frame_bgr = cap.read()
        cap.release()
        if ok:
            frames["ours sigmoid_b21 CRF 34"] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Saliency overlay (jet, alpha 0.45)
    if "original" in frames:
        frame_rgb = frames["original"]
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        sal_obj = cv2.saliency.StaticSaliencySpectralResidual_create()
        ok, sal = sal_obj.computeSaliency(frame_bgr)
        if ok:
            sal = (sal * 255).astype(np.uint8)
            heat = cv2.applyColorMap(sal, cv2.COLORMAP_JET)
            blended = cv2.addWeighted(frame_bgr, 0.55, heat, 0.45, 0.0)
            frames["saliency overlay"] = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

    order = ["original", "saliency overlay", "baseline CRF 34", "ours sigmoid_b21 CRF 34"]
    available = [k for k in order if k in frames]

    if not available:
        print(f"  WARNING: no frames loaded for qualitative_real; skipping")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(12, 3.5))
    if len(available) == 1:
        axes = [axes]

    for ax, key in zip(axes, available):
        ax.imshow(frames[key])
        ax.set_title(key, fontsize=10, fontweight="bold")
        ax.axis("off")

    # Caption with clip info and file sizes
    action = clip_info.get("action_class", "unknown")
    duration_s = clip_info.get("output_duration_s", 0)
    orig_size_kb = clip_info.get("output_size_bytes", 0) / 1024

    caption = f"{action.upper()} — clip {clip_id} (frame {target_frame}, ~{target_frame/30:.1f}s)"
    caption += f"\nOriginal size: {orig_size_kb:.0f} KB (duration {duration_s}s)"

    fig.text(0.5, 0.02, caption, ha="center", fontsize=9, style="italic", color="#555")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_per_clip_variance(ablation_rows: list[dict], baseline_rows: list[dict], manifest: dict, out_path: Path) -> None:
    """Scatter plot: where ours wins/loses per clip.

    X: baseline_crf34 sal-PSNR per clip
    Y: ours_sigmoid_b21_crf28 sal-PSNR per clip
    Diagonal = parity. Points above = ours wins. Points below = ours loses.
    Color by action class.
    """
    if not ablation_rows or not baseline_rows or not manifest:
        print(f"  WARNING: insufficient data for per_clip_variance; skipping")
        return

    # Aggregate per clip
    baseline_by_clip = {}
    ours_by_clip = {}

    # Baseline CRF 34
    for r in baseline_rows:
        clip_id = r.get("clip_id")
        crf = r.get("crf")
        sal_psnr = r.get("sal_psnr_mean")

        if clip_id and crf == 34 and sal_psnr is not None:
            if clip_id not in baseline_by_clip:
                baseline_by_clip[clip_id] = []
            baseline_by_clip[clip_id].append(sal_psnr)

    # Ours sigmoid_b21 CRF 28
    for r in ablation_rows:
        clip_id = r.get("clip_id")
        crf = r.get("crf")
        blur = r.get("blur")
        sal_psnr = r.get("sal_psnr_mean")

        if clip_id and crf == 28 and blur == 21 and sal_psnr is not None:
            if clip_id not in ours_by_clip:
                ours_by_clip[clip_id] = []
            ours_by_clip[clip_id].append(sal_psnr)

    # Average per clip
    clips_with_both = set(baseline_by_clip.keys()) & set(ours_by_clip.keys())
    if not clips_with_both:
        print(f"  WARNING: no clips with both baseline CRF34 and ours sigmoid_b21 CRF28; skipping per_clip_variance")
        return

    base_vals = [np.mean(baseline_by_clip[cid]) for cid in clips_with_both]
    ours_vals = [np.mean(ours_by_clip[cid]) for cid in clips_with_both]
    actions = [manifest.get(cid, {}).get("action_class", "unknown") for cid in clips_with_both]
    colors = [get_action_color(action) for action in actions]

    fig, ax = plt.subplots(figsize=(7, 6))

    # Scatter
    ax.scatter(base_vals, ours_vals, s=100, c=colors, alpha=0.7, edgecolor="#333", linewidth=1, zorder=3)

    # Parity line
    min_val = min(min(base_vals), min(ours_vals))
    max_val = max(max(base_vals), max(ours_vals))
    ax.plot([min_val-1, max_val+1], [min_val-1, max_val+1], "k--", linewidth=2, alpha=0.3, label="Parity", zorder=1)

    # Shaded regions
    ax.fill_between([min_val-1, max_val+1], [min_val-1, max_val+1], [max_val+2, max_val+2],
                   color="#1f77b4", alpha=0.05, label="Ours wins", zorder=0)
    ax.fill_between([min_val-1, max_val+1], [min_val-1, max_val+1], [min_val-2, min_val-2],
                   color="#d62728", alpha=0.05, label="Baseline wins", zorder=0)

    ax.set_xlabel("Baseline (CRF 34) sal-PSNR (dB)", fontsize=11)
    ax.set_ylabel("Ours (sigmoid_b21, CRF 28) sal-PSNR (dB)", fontsize=11)
    ax.set_title("Per-clip variance: where ours wins and loses", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2, zorder=0)
    ax.legend(fontsize=9, loc="lower right")

    # Equal aspect
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
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


def fig_sustainability_real(ablation_rows: list[dict], baseline_rows: list[dict], out_path: Path) -> None:
    """Extrapolate annual energy / CO2 savings at deployment scale.

    Uses measured RELATIVE saving (~23% at iso-CRF 28) applied to realistic
    baseline H.265 home camera bitrate (~1.5 Mbps). Honest framing: "iso-CRF saving"
    not "iso-quality saving".

    ablation_rows: from ablation_real/rows.jsonl (has "kind": "ours")
    baseline_rows: from ablation_real/baselines/rows.jsonl (has "kind": "baseline")
    """
    if not ablation_rows or not baseline_rows:
        print(f"  WARNING: insufficient rows for sustainability_real; skipping")
        return

    # Aggregate at CRF 28
    ours_sizes = []
    base_sizes = []

    # Ours: sigmoid_b21 at CRF 28
    for r in ablation_rows:
        if r.get("crf") == 28 and r.get("blur") == 21 and r.get("kind") == "ours":
            size = r.get("dist_bytes") or r.get("out_bytes", 0)
            if size > 0:
                ours_sizes.append(size)

    # Baseline at CRF 28
    for r in baseline_rows:
        if r.get("crf") == 28 and r.get("kind") == "baseline":
            size = r.get("dist_bytes") or r.get("out_bytes", 0)
            if size > 0:
                base_sizes.append(size)

    if not ours_sizes or not base_sizes:
        print(f"  WARNING: no data at CRF 28 for sustainability_real (ours: {len(ours_sizes)}, base: {len(base_sizes)}); skipping")
        return

    avg_ours_bytes = np.mean(ours_sizes)
    avg_base_bytes = np.mean(base_sizes)
    relative_saving = 1.0 - (avg_ours_bytes / avg_base_bytes)

    # Realistic baseline: 1.5 Mbps H.265 (industry typical)
    realistic_base_mbps = 1.5
    saved_mbps = realistic_base_mbps * relative_saving

    seconds_year = 365 * 24 * 3600
    bits_per_byte = 8
    bytes_per_gibibyte = 1024 ** 3
    base_gb_year = realistic_base_mbps * seconds_year / (bits_per_byte * bytes_per_gibibyte)
    saved_gb_year = saved_mbps * seconds_year / (bits_per_byte * bytes_per_gibibyte)

    energy_per_gb = 0.06   # kWh per GB (Masanet et al. 2020)
    co2_per_kwh = 0.4      # kg CO2 per kWh (US grid avg, EPA)

    cameras = [1, 1_000, 10_000, 100_000]
    savings_kwh = [saved_gb_year * energy_per_gb * n for n in cameras]
    savings_co2 = [s * co2_per_kwh for s in savings_kwh]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar([f"{n:,}" for n in cameras], savings_co2, color="#2e7d32", alpha=0.75, width=0.6)
    ax.set_xlabel("Cameras deployed", fontsize=11)
    ax.set_ylabel("Annual CO₂ saved (kg)", fontsize=11)
    ax.set_title("Sustainability: iso-CRF storage-energy CO₂ savings\n"
                f"(measured {relative_saving*100:.0f}% file-size reduction × 1.5 Mbps baseline)",
                fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.2)

    for i, (c, n) in enumerate(zip(savings_co2, cameras)):
        ax.text(i, c + max(savings_co2)*0.02, f"{c:,.0f} kg", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}")

    # JSON sidecar for the analysis to cite
    with (FIG_DIR / "sustainability.json").open("w") as f:
        json.dump({
            "measured_relative_saving": round(relative_saving, 3),
            "baseline_mbps_assumed": realistic_base_mbps,
            "gb_saved_per_camera_year": round(saved_gb_year, 1),
            "kwh_saved_per_camera_year": round(saved_gb_year * energy_per_gb, 2),
            "co2_kg_saved_per_camera_year": round(saved_gb_year * energy_per_gb * co2_per_kwh, 2),
            "co2_kg_per_1000cams_year": round(saved_gb_year * energy_per_gb * co2_per_kwh * 1000, 1),
            "assumption_kwh_per_gb": energy_per_gb,
            "assumption_kwh_per_gb_source": "Masanet et al. 2020",
            "assumption_kg_co2_per_kwh": co2_per_kwh,
            "assumption_kg_co2_per_kwh_source": "EPA US grid average",
            "operating_point": "ours_sigmoid_b21_crf28 vs baseline_crf28",
            "framing": "iso-CRF (encoder target, not iso-quality)",
        }, f, indent=2)


def fig_gate_trace_real(rows: list[dict], manifest: dict, out_path: Path) -> None:
    """Gate trigger ratio per CCTV clip (horizontal bar chart, one bar per clip).

    X: trigger_ratio (0-1, percentage of frames flagged useful)
    Y: clip ID (color-coded by action class)
    Mean line: 65.2%

    Falls back to per-clip aggregation if per-frame trace unavailable.
    """
    if not rows:
        print(f"  WARNING: no rows for gate_trace_real; skipping")
        return

    # Aggregate trigger_ratio per clip (take first occurrence or average if multiple)
    trigger_by_clip = {}
    for r in rows:
        clip_id = r.get("clip_id")
        if clip_id and "trigger_ratio" in r:
            if clip_id not in trigger_by_clip:
                trigger_by_clip[clip_id] = []
            trigger_by_clip[clip_id].append(r["trigger_ratio"])

    if not trigger_by_clip:
        print(f"  WARNING: no trigger_ratio data; skipping gate_trace_real")
        return

    # Average per clip
    clip_ids_sorted = sorted(trigger_by_clip.keys())
    ratios = [np.mean(trigger_by_clip[cid]) for cid in clip_ids_sorted]

    # Get action classes
    actions = [manifest.get(cid, {}).get("action_class", "unknown") for cid in clip_ids_sorted]

    # Colors by action class
    colors = [get_action_color(action) for action in actions]

    # Mean trigger ratio
    mean_ratio = np.mean(ratios)

    fig, ax = plt.subplots(figsize=(7.5, 5))

    y_pos = np.arange(len(clip_ids_sorted))
    ax.barh(y_pos, ratios, color=colors, alpha=0.75, height=0.7, edgecolor="#333", linewidth=0.5)

    # Mean line
    ax.axvline(mean_ratio, color="#d62728", linestyle="--", linewidth=2.5,
              label=f"Mean: {mean_ratio*100:.1f}%", zorder=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(clip_ids_sorted, fontsize=9)
    ax.set_xlabel("Gate trigger ratio (fraction of frames)", fontsize=11)
    ax.set_ylabel("Clip", fontsize=11)
    ax.set_title("Sensor gate usefulness: trigger ratio per CCTV clip", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.2)
    ax.legend(fontsize=10, loc="lower right")

    # Add percentage labels
    for i, (ratio, action) in enumerate(zip(ratios, actions)):
        ax.text(ratio + 0.02, i, f"{ratio*100:.0f}%", va="center", fontsize=8, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    print("=== Generating analysis figures from real CCTV ablation ===\n")

    # Load data
    data_dir = ROOT / "data" / "real"
    results_dir = ROOT / "results"

    manifest = load_manifest(data_dir / "manifest.json")
    print(f"Loaded manifest: {len(manifest)} clips")

    # Ablation rows
    baseline_rows = load_rows(results_dir / "ablation_real" / "baselines" / "rows.jsonl")
    ablation_real_rows = load_rows(results_dir / "ablation_real" / "rows.jsonl")
    ablation_mask_rows = load_rows(results_dir / "ablation_mask" / "rows.jsonl")
    ablation_sigmoid_rows = load_rows(results_dir / "ablation_sigmoid" / "rows.jsonl")

    print(f"Loaded {len(baseline_rows)} baseline rows")
    print(f"Loaded {len(ablation_real_rows)} ablation_real rows")
    print(f"Loaded {len(ablation_mask_rows)} ablation_mask rows")
    print(f"Loaded {len(ablation_sigmoid_rows)} ablation_sigmoid rows")

    # Combine ours rows for RD curve
    all_ours_rows = ablation_real_rows + ablation_sigmoid_rows

    # Neural codec
    codec_bench = {}
    codec_json = results_dir / "neural_codec" / "benchmark.json"
    if codec_json.exists():
        codec_bench = json.loads(codec_json.read_text())
        print(f"Loaded neural codec benchmark")
    else:
        print(f"  WARNING: {codec_json} not found")

    print("\n=== Generating figures ===\n")

    # 1. RD curve (headline)
    fig_rd_curve_real(baseline_rows, all_ours_rows, FIG_DIR / "rd_curve_real.png")

    # 2. Mask modes ablation
    fig_mask_modes(ablation_mask_rows, FIG_DIR / "mask_modes.png")

    # 3. Neural codec panel
    fig_neural_codec_panel(codec_bench, FIG_DIR / "neural_codec_panel.png")

    # 4. Qualitative frame strip from real clip_04
    fig_qualitative_real(manifest, data_dir, results_dir, FIG_DIR / "qualitative_real.png")

    # 5. Gate trigger ratio per clip
    fig_gate_trace_real(ablation_real_rows, manifest, FIG_DIR / "gate_trace_real.png")

    # 6. Sustainability (iso-CRF framing)
    fig_sustainability_real(ablation_real_rows, baseline_rows, FIG_DIR / "sustainability_real.png")

    # 7. Per-clip variance scatter
    fig_per_clip_variance(ablation_real_rows, baseline_rows, manifest, FIG_DIR / "per_clip_variance.png")

    # 8. Architecture (preserve old figure or regenerate)
    arch_path = FIG_DIR / "architecture.png"
    if arch_path.exists():
        print(f"  preserved {arch_path}")
    else:
        fig_architecture(arch_path)

    print("\n=== Complete ===")
    print(f"All figures written to {FIG_DIR}/")

    # List outputs
    print("\nGenerated files:")
    for fpath in sorted(FIG_DIR.glob("*.png")):
        size_kb = fpath.stat().st_size / 1024
        print(f"  {fpath.name:40s} {size_kb:7.1f} KB")

    if (FIG_DIR / "sustainability.json").exists():
        print(f"  sustainability.json")


if __name__ == "__main__":
    main()

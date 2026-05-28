"""
Background/foreground codec ablation — head-to-head against H.265 on every real clip.

For each clip × CRF, encodes the source two ways:
  - baseline_h265  : uniform H.265 (the comparison)
  - ours_bgfg      : the background/foreground decomposition codec
                     (saliency backend configurable: spectral / yolo / yolo+spectral)

Reports per row:
  bytes, psnr_mean, ssim_mean, sal_psnr_mean (saliency-weighted PSNR)

The sal_psnr_mean metric is the *honest* one for our codec:
plain PSNR penalises us for intentionally replacing non-salient pixels with
the background, but sal_psnr only measures error where saliency >= 0.5 — i.e.,
the regions we promised to preserve.

Outputs:
  results/ablation_bgfg/encoded/<config>_<clip>.mp4
  results/ablation_bgfg/rows.jsonl
  results/ablation_bgfg/summary.json
  results/ablation_bgfg/summary.md
  results/ablation_bgfg/rd_curve.png

Usage:
  python scripts/ablation_bgfg.py                          # all 20 clips, default CRFs
  python scripts/ablation_bgfg.py --clips clip_04 clip_08  # subset
  python scripts/ablation_bgfg.py --saliency yolo+spectral # YOLO-backed bg/fg
  python scripts/ablation_bgfg.py --quick                  # 3 clips × 1 CRF for smoke test
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

from src.bg_fg_codec import BgFgCodec, BgFgConfig
from src.compress import encode_uniform
from src.metrics import saliency_weighted_psnr
from src.saliency import SaliencyEstimator

OUT_DIR = ROOT / "results" / "ablation_bgfg"
ENCODED_DIR = OUT_DIR / "encoded"
DATA_DIR = ROOT / "data" / "real"


# ---------- Metric sampling (shared across configs) ----------

def sample_metrics(ref_path: str, dist_path: str, sal_estimator, every: int = 15) -> dict:
    """Compute PSNR, SSIM, and saliency-weighted PSNR at sampled frames."""
    cap_r = cv2.VideoCapture(ref_path)
    cap_d = cv2.VideoCapture(dist_path)
    psnrs, ssims, sal_psnrs = [], [], []
    i = 0
    while True:
        ok_r, fr = cap_r.read()
        ok_d, fd = cap_d.read()
        if not (ok_r and ok_d):
            break
        if i % every == 0:
            if fr.shape != fd.shape:
                fd = cv2.resize(fd, (fr.shape[1], fr.shape[0]))
            psnrs.append(float(sk_psnr(fr, fd, data_range=255)))
            ssims.append(float(sk_ssim(fr, fd, channel_axis=2, data_range=255)))
            if sal_estimator is not None:
                sal = sal_estimator.predict(fr)
                sp = saliency_weighted_psnr(fr, fd, sal, threshold=0.5)
                if not np.isnan(sp):
                    sal_psnrs.append(sp)
        i += 1
    cap_r.release()
    cap_d.release()
    return {
        "psnr_mean": float(np.mean(psnrs)) if psnrs else float("nan"),
        "ssim_mean": float(np.mean(ssims)) if ssims else float("nan"),
        "sal_psnr_mean": float(np.mean(sal_psnrs)) if sal_psnrs else float("nan"),
        "n_eval": len(psnrs),
    }


# ---------- Per-config encoders ----------

def encode_baseline(input_path: str, out_path: str, crf: int) -> dict:
    encode_uniform(input_path, out_path, crf=crf)
    return {"bytes": os.path.getsize(out_path)}


def encode_ours_bgfg(input_path: str, out_path: str, crf: int, saliency_backend: str,
                     saliency_yolo_kwargs: dict | None) -> dict:
    cfg = BgFgConfig(
        saliency_backend=saliency_backend,
        saliency_yolo_kwargs=saliency_yolo_kwargs,
        crf=crf,
        mask_mode="sigmoid",
        mask_threshold=0.30,
        mask_steepness=12.0,
        smooth_window=7,
        bg_sample_count=30,
    )
    stats = BgFgCodec(cfg).encode(input_path, out_path)
    return {"bytes": stats["bytes"], "background_bytes": stats["background_bytes"]}


# ---------- Main runner ----------

def discover_clips() -> list[str]:
    return sorted(p.stem for p in DATA_DIR.glob("clip_*.mp4"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", nargs="*", default=None,
                        help="Specific clip IDs (e.g. clip_04 clip_08). Default: all 20.")
    parser.add_argument("--crfs", type=int, nargs="*", default=[22, 28, 34])
    parser.add_argument("--saliency", default="spectral",
                        choices=["spectral", "finegrained", "yolo", "yolo+spectral"],
                        help="Saliency backend for ours_bgfg")
    parser.add_argument("--every", type=int, default=15,
                        help="Sample every N frames for metrics (lower = slower, more accurate)")
    parser.add_argument("--quick", action="store_true",
                        help="3 clips × 1 CRF for smoke test")
    args = parser.parse_args()

    if args.quick:
        args.clips = ["clip_04", "clip_08", "clip_10"]
        args.crfs = [28]

    clips = args.clips or discover_clips()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ENCODED_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = OUT_DIR / "rows.jsonl"

    sal_for_metrics = SaliencyEstimator(backend=args.saliency)
    rows = []

    n_total = len(clips) * len(args.crfs) * 2
    n_done = 0
    t0 = time.time()

    print(f"\n=== ablation_bgfg: {len(clips)} clips × {len(args.crfs)} CRFs ===")
    print(f"    saliency backend = {args.saliency}")
    print(f"    output           = {OUT_DIR}\n")

    with open(rows_path, "w") as f_rows:
        for clip_id in clips:
            input_path = str(DATA_DIR / f"{clip_id}.mp4")
            if not Path(input_path).exists():
                print(f"  [skip] {clip_id}: not found")
                continue
            for crf in args.crfs:

                # --- baseline ---
                base_path = str(ENCODED_DIR / f"baseline_crf{crf}_{clip_id}.mp4")
                enc = encode_baseline(input_path, base_path, crf)
                m = sample_metrics(input_path, base_path, sal_for_metrics, every=args.every)
                row = {"clip": clip_id, "config": "baseline_h265", "crf": crf, **enc, **m}
                rows.append(row); f_rows.write(json.dumps(row) + "\n"); f_rows.flush()
                n_done += 1
                print(f"  [{n_done}/{n_total}] {clip_id} crf{crf} baseline   "
                      f"{enc['bytes']/1024:7.1f} KB  sal-PSNR {m['sal_psnr_mean']:.2f} dB")

                # --- ours_bgfg ---
                bgfg_path = str(ENCODED_DIR / f"ours_bgfg_crf{crf}_{clip_id}.mp4")
                enc = encode_ours_bgfg(input_path, bgfg_path, crf, args.saliency, None)
                m = sample_metrics(input_path, bgfg_path, sal_for_metrics, every=args.every)
                row = {"clip": clip_id, "config": "ours_bgfg", "crf": crf, **enc, **m}
                rows.append(row); f_rows.write(json.dumps(row) + "\n"); f_rows.flush()
                n_done += 1
                print(f"  [{n_done}/{n_total}] {clip_id} crf{crf} bgfg       "
                      f"{enc['bytes']/1024:7.1f} KB  sal-PSNR {m['sal_psnr_mean']:.2f} dB")

    # Summary aggregations
    summary = aggregate(rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT_DIR / "summary.md").write_text(render_markdown(summary))
    try:
        plot_rd_curve(rows, OUT_DIR / "rd_curve.png")
    except Exception as e:
        print(f"  (rd_curve plot skipped: {e})")

    dt = time.time() - t0
    print(f"\n=== done in {dt:.1f}s ===")
    print(f"    rows.jsonl    -> {rows_path}")
    print(f"    summary.md    -> {OUT_DIR / 'summary.md'}")
    print(f"    rd_curve.png  -> {OUT_DIR / 'rd_curve.png'}")


# ---------- Aggregation + reporting ----------

def aggregate(rows: list[dict]) -> dict:
    """Average per (config, crf) across clips."""
    by_key: dict[tuple[str, int], list[dict]] = {}
    for r in rows:
        by_key.setdefault((r["config"], r["crf"]), []).append(r)

    out = []
    for (cfg, crf), rs in sorted(by_key.items()):
        out.append({
            "config": cfg, "crf": crf, "n_clips": len(rs),
            "bytes_mean_KB": float(np.mean([r["bytes"] for r in rs])) / 1024,
            "psnr_mean": float(np.mean([r["psnr_mean"] for r in rs])),
            "ssim_mean": float(np.mean([r["ssim_mean"] for r in rs])),
            "sal_psnr_mean": float(np.mean(
                [r["sal_psnr_mean"] for r in rs if not np.isnan(r["sal_psnr_mean"])]
            )),
        })
    return {"per_config": out}


def render_markdown(summary: dict) -> str:
    lines = [
        "# ablation_bgfg — three-way comparison",
        "",
        "Averages across all clips per (config, crf).  `sal_psnr` is the honest",
        "metric for our codec: it only counts error in pixels we promised to preserve.",
        "",
        "| config | CRF | size (KB) | PSNR | SSIM | **sal-PSNR** |",
        "|---|---|---|---|---|---|",
    ]
    for r in summary["per_config"]:
        lines.append(
            f"| {r['config']} | {r['crf']} | {r['bytes_mean_KB']:.1f} | "
            f"{r['psnr_mean']:.2f} | {r['ssim_mean']:.3f} | "
            f"**{r['sal_psnr_mean']:.2f}** |"
        )
    return "\n".join(lines)


def plot_rd_curve(rows: list[dict], out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Build per-config (bytes, sal_psnr) points averaged across clips at each CRF
    points = {}
    for r in rows:
        key = r["config"]
        points.setdefault(key, []).append(r)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"baseline_h265": "#888", "ours_bgfg": "#d62728"}
    markers = {"baseline_h265": "o", "ours_bgfg": "^"}

    for cfg, rs in points.items():
        # group by CRF
        by_crf = {}
        for r in rs:
            by_crf.setdefault(r["crf"], []).append(r)
        crf_sorted = sorted(by_crf.keys())
        xs = [float(np.mean([r["bytes"] for r in by_crf[c]])) / 1024 for c in crf_sorted]
        ys = [float(np.mean([r["sal_psnr_mean"] for r in by_crf[c]
                             if not np.isnan(r["sal_psnr_mean"])])) for c in crf_sorted]
        ax.plot(xs, ys, marker=markers.get(cfg, "o"), label=cfg,
                color=colors.get(cfg, None), linewidth=2, markersize=8)
        for x, y, c in zip(xs, ys, crf_sorted):
            ax.annotate(f"CRF {c}", (x, y), textcoords="offset points",
                        xytext=(6, 6), fontsize=8)

    ax.set_xlabel("Bytes per clip (KB) — lower is better")
    ax.set_ylabel("Saliency-weighted PSNR (dB) — higher is better")
    ax.set_title("RD curve — sal-PSNR vs bytes (averaged across clips)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()

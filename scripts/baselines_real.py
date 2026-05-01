"""
Run uniform H.265 baselines (no saliency, no gate) on every real CCTV clip,
sweeping CRF across {22, 28, 34, 40}. Computes PSNR, SSIM, saliency-weighted
PSNR, and LPIPS so the rate-distortion plots have a fair comparison curve.

Output:
  results/ablation_real/baselines/baseline_crf{NN}_{clip_id}.mp4
  results/ablation_real/baselines/rows.jsonl   (append)
  results/ablation_real/baselines/summary.json
  results/ablation_real/baselines/summary.md

Usage:
  python scripts/baselines_real.py --inputs "data/real/*.mp4" --use-lpips
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compress import encode_uniform
from src.metrics import video_metrics_with_saliency
from src.saliency import SaliencyEstimator

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

CRFS = [22, 28, 34, 40]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="data/real/*.mp4")
    ap.add_argument("--out", default="results/ablation_real/baselines")
    ap.add_argument("--every-metric", type=int, default=15)
    ap.add_argument("--use-lpips", action="store_true")
    args = ap.parse_args()

    clips = sorted(glob.glob(args.inputs))
    if not clips:
        log.error(f"No clips matched {args.inputs}")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"

    sal_est = SaliencyEstimator(backend="spectral")

    print(f"Found {len(clips)} clips × {len(CRFS)} CRFs = {len(clips)*len(CRFS)} encodes")
    rows = []
    for clip in clips:
        clip_id = Path(clip).stem
        for crf in CRFS:
            cfg_id = f"baseline_crf{crf}"
            out_path = out_dir / f"{cfg_id}_{clip_id}.mp4"
            t0 = time.perf_counter()
            try:
                encode_uniform(clip, str(out_path), crf=crf, codec="libx265", preset="superfast")
            except Exception as e:
                log.warning(f"{clip_id} {cfg_id} encode failed: {e}")
                continue
            t1 = time.perf_counter()
            try:
                m = video_metrics_with_saliency(clip, str(out_path), sal_est,
                                                every=args.every_metric, use_lpips=args.use_lpips)
            except Exception as e:
                log.warning(f"{clip_id} {cfg_id} metrics failed: {e}")
                continue

            row = {
                "clip_id": clip_id,
                "id": cfg_id,
                "kind": "baseline",
                "crf": crf,
                "blur": None,
                "idle_blur": None,
                "gate": None,
                "out_path": str(out_path),
                "out_bytes": os.path.getsize(out_path),
                "ref_bytes": os.path.getsize(clip),
                "ratio_vs_input": os.path.getsize(clip) / max(os.path.getsize(out_path), 1),
                "wall_encode_s": round(t1 - t0, 2),
                **m,
            }
            rows.append(row)
            with rows_path.open("a") as f:
                f.write(json.dumps(row, default=str) + "\n")
            lp = row.get("lpips_mean")
            print(f"  {clip_id} {cfg_id}: bytes={row['out_bytes']:>7}  "
                  f"PSNR={m['psnr_mean']:.2f}  sal-PSNR={m['sal_psnr_mean']:.2f}"
                  + (f"  LPIPS={lp:.3f}" if lp is not None else ""))

    # Aggregate by config
    from collections import defaultdict
    by_cfg = defaultdict(list)
    for r in rows:
        by_cfg[r["id"]].append(r)

    summary = []
    for cfg_id, group in by_cfg.items():
        n = len(group)
        agg = {
            "id": cfg_id,
            "kind": "baseline",
            "crf": group[0]["crf"],
            "n_clips": n,
            "out_bytes_mean": sum(r["out_bytes"] for r in group) / n,
            "psnr_mean": sum(r["psnr_mean"] for r in group) / n,
            "ssim_mean": sum(r["ssim_mean"] for r in group) / n,
            "sal_psnr_mean": sum(r["sal_psnr_mean"] for r in group) / n,
        }
        if args.use_lpips and group[0].get("lpips_mean") is not None:
            agg["lpips_mean"] = sum(r["lpips_mean"] for r in group) / n
        summary.append(agg)
    summary.sort(key=lambda r: r["crf"])

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    md_lines = ["# Baseline (uniform H.265) Results on Real CCTV", "",
                "| Config | CRF | Clips | File Size (KB) | PSNR (dB) | Sal-PSNR (dB) | LPIPS |",
                "|---|---|---|---|---|---|---|"]
    for s in summary:
        lp = f"{s['lpips_mean']:.3f}" if "lpips_mean" in s else "—"
        md_lines.append(f"| {s['id']} | {s['crf']} | {s['n_clips']} | "
                        f"{s['out_bytes_mean']/1024:.0f} | {s['psnr_mean']:.2f} | "
                        f"{s['sal_psnr_mean']:.2f} | {lp} |")
    (out_dir / "summary.md").write_text("\n".join(md_lines) + "\n")

    print()
    print((out_dir / "summary.md").read_text())


if __name__ == "__main__":
    main()

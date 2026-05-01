"""
Backfill LPIPS into the existing ours/ ablation rows.

The ablation_real.py run was supposed to compute LPIPS but produced 0 rows with
lpips_mean. This script reads results/ablation_real/rows.jsonl, computes LPIPS
between each output and its source clip in data/real/, and writes:
  results/ablation_real/rows_with_lpips.jsonl
  results/ablation_real/summary.md  (overwritten with LPIPS column)

Usage:
  python scripts/fill_lpips.py
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from src.metrics import LPIPSMetric

ROWS = Path("results/ablation_real/rows.jsonl")
OUT_ROWS = Path("results/ablation_real/rows_with_lpips.jsonl")
OUT_SUMMARY = Path("results/ablation_real/summary.md")


def lpips_for_video(lpips_model: LPIPSMetric, ref_path: str, dist_path: str,
                    every: int = 30) -> float:
    cap_r = cv2.VideoCapture(ref_path)
    cap_d = cv2.VideoCapture(dist_path)
    if not (cap_r.isOpened() and cap_d.isOpened()):
        return float("nan")
    lps = []
    i = 0
    while True:
        ok_r, fr = cap_r.read()
        ok_d, fd = cap_d.read()
        if not (ok_r and ok_d):
            break
        if i % every == 0:
            if fr.shape != fd.shape:
                fd = cv2.resize(fd, (fr.shape[1], fr.shape[0]))
            lps.append(lpips_model.compute(fr, fd))
        i += 1
    cap_r.release(); cap_d.release()
    return float(np.mean(lps)) if lps else float("nan")


def main() -> None:
    rows = [json.loads(l) for l in ROWS.open()]
    print(f"Loaded {len(rows)} rows. Computing LPIPS (this takes ~3-5 min)...")

    lpips_model = LPIPSMetric()  # default alex backbone, cpu

    t0 = time.perf_counter()
    missing = 0
    for i, r in enumerate(rows):
        src = f"data/real/{r['clip_id']}.mp4"
        # Try multiple known path conventions (different ablation scripts wrote different shapes)
        dst_candidates = [
            r.get("out_path"),
            f"results/ablation_real/{r['clip_id']}/{r['id']}.mp4",
            f"results/ablation_real/{r['id']}_{r['clip_id']}.mp4",
        ]
        dst = next((p for p in dst_candidates if p and Path(p).exists()), None)
        if dst is None:
            missing += 1
            r["lpips_mean"] = None
            if missing <= 3:
                print(f"  [{i+1}/{len(rows)}] {r['clip_id']} {r['id']}: output file not found "
                      f"(tried: {[c for c in dst_candidates if c]})")
            continue
        try:
            r["lpips_mean"] = lpips_for_video(lpips_model, src, dst, every=30)
        except Exception as e:
            print(f"  [{i+1}/{len(rows)}] {r['clip_id']} {r['id']}: FAILED — {e}")
            r["lpips_mean"] = None
            continue
        if (i + 1) % 10 == 0 or i == len(rows) - 1:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            eta = (len(rows) - i - 1) / rate
            print(f"  [{i+1}/{len(rows)}] {r['clip_id']} {r['id']}: "
                  f"LPIPS={r['lpips_mean']:.3f}  ({rate:.1f}/s, ETA {eta:.0f}s)")

    with OUT_ROWS.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\nWrote {OUT_ROWS}")

    by_cfg = defaultdict(list)
    for r in rows:
        by_cfg[r["id"]].append(r)

    md_lines = ["# Ablation Results Summary (real CCTV, with LPIPS)", "",
                "| Config | Clips | File Size (KB) | PSNR (dB) | Sal-PSNR (dB) | LPIPS | Trigger |",
                "|---|---|---|---|---|---|---|"]
    for cfg_id in sorted(by_cfg.keys()):
        group = by_cfg[cfg_id]
        n = len(group)
        size_kb = np.mean([r.get("dist_bytes", 0) for r in group]) / 1024
        psnr = np.mean([r["psnr_mean"] for r in group])
        sal = np.mean([r["sal_psnr_mean"] for r in group])
        lp_vals = [r["lpips_mean"] for r in group
                   if r.get("lpips_mean") is not None and not np.isnan(r["lpips_mean"])]
        lp = np.mean(lp_vals) if lp_vals else float("nan")
        trig = np.mean([r.get("trigger_ratio", 0) for r in group])
        md_lines.append(f"| {cfg_id} | {n} | {size_kb:.0f} | {psnr:.2f} | {sal:.2f} | "
                        f"{lp:.3f} | {trig:.3f} |")
    OUT_SUMMARY.write_text("\n".join(md_lines) + "\n")
    print(f"Wrote {OUT_SUMMARY}\n")
    print(OUT_SUMMARY.read_text())


if __name__ == "__main__":
    main()

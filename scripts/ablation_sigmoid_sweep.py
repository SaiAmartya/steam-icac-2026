"""
Fill the rate-distortion curve for the winning sigmoid-mask config.

Runs ours_sigmoid_b21 at CRF {22, 34, 40} on all 20 real clips and combines
with the existing CRF 28 result (already in results/ablation_mask/).

Output:
  results/ablation_sigmoid/rows.jsonl
  results/ablation_sigmoid/summary.md
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from src.compress import CompressorConfig, SaliencyCompressor
from src.gate import FootageGate
from src.metrics import LPIPSMetric, video_metrics_with_saliency
from src.saliency import SaliencyEstimator, TemporalSmoother

CRFS_TO_FILL = [22, 34, 40]   # 28 already done in ablation_mask
GATE = 0.25
BLUR = 21
IDLE_BLUR = 51
MASK_MODE = "sigmoid"
THRESHOLD = 0.4
STEEPNESS = 12.0


def run_one(input_path, out_dir, crf, sal_est, lpips_model):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    gate = FootageGate(threshold=GATE, enable_person=False)
    smoother = TemporalSmoother(window=5)
    active = SaliencyCompressor(CompressorConfig(
        tier="C", codec="libx265", crf=crf, blur_strength=BLUR, preset="superfast",
        mask_mode=MASK_MODE, mask_threshold=THRESHOLD, mask_steepness=STEEPNESS,
    ))
    idle = SaliencyCompressor(CompressorConfig(
        tier="C", codec="libx265", crf=crf, blur_strength=IDLE_BLUR, preset="superfast",
        mask_mode=MASK_MODE, mask_threshold=THRESHOLD, mask_steepness=STEEPNESS,
    ))

    triggered = total = 0
    processed = []
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok: break
        score = gate.step(frame)
        sal = smoother.smooth(sal_est.predict(frame))
        out = active.process_frame(frame, sal) if score.triggered else idle.process_frame(frame, sal)
        processed.append(out)
        triggered += int(score.triggered); total += 1
    cap.release()

    cfg_id = f"ours_sigmoid_b{BLUR}_crf{crf}"
    out_path = out_dir / f"{cfg_id}_{Path(input_path).stem}.mp4"
    active.encode(iter(processed), str(out_path), fps=fps, size=(w, h))
    m = video_metrics_with_saliency(input_path, str(out_path), sal_est, every=30, use_lpips=False)

    cap_r = cv2.VideoCapture(input_path); cap_d = cv2.VideoCapture(str(out_path))
    lps, i = [], 0
    while True:
        ok_r, fr = cap_r.read(); ok_d, fd = cap_d.read()
        if not (ok_r and ok_d): break
        if i % 30 == 0:
            if fr.shape != fd.shape: fd = cv2.resize(fd, (fr.shape[1], fr.shape[0]))
            lps.append(lpips_model.compute(fr, fd))
        i += 1
    cap_r.release(); cap_d.release()
    return {
        "id": cfg_id, "clip_id": Path(input_path).stem,
        "crf": crf, "blur": BLUR, "idle_blur": IDLE_BLUR,
        "mask_mode": MASK_MODE, "mask_threshold": THRESHOLD,
        "out_bytes": os.path.getsize(out_path),
        "ref_bytes": os.path.getsize(input_path),
        "trigger_ratio": triggered / max(total, 1),
        "psnr_mean": m["psnr_mean"], "ssim_mean": m["ssim_mean"],
        "sal_psnr_mean": m["sal_psnr_mean"],
        "lpips_mean": float(np.mean(lps)) if lps else float("nan"),
    }


def main():
    clips = sorted(glob.glob("data/real/*.mp4"))
    out_dir = Path("results/ablation_sigmoid")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"

    sal_est = SaliencyEstimator(backend="spectral")
    print("Loading LPIPS model...")
    lpips_model = LPIPSMetric()

    print(f"Running {len(CRFS_TO_FILL)} CRFs × {len(clips)} clips = {len(CRFS_TO_FILL) * len(clips)} runs")
    rows = []
    for crf in CRFS_TO_FILL:
        print(f"\n--- CRF {crf} ---")
        for clip in clips:
            row = run_one(clip, out_dir, crf, sal_est, lpips_model)
            rows.append(row)
            with rows_path.open("a") as f:
                f.write(json.dumps(row, default=str) + "\n")
            print(f"  {row['clip_id']}: {row['out_bytes']:>6} B  "
                  f"PSNR={row['psnr_mean']:.2f}  sal-PSNR={row['sal_psnr_mean']:.2f}  "
                  f"LPIPS={row['lpips_mean']:.3f}")

    by_cfg = defaultdict(list)
    for r in rows:
        by_cfg[r["id"]].append(r)

    md = ["# Sigmoid-mask RD curve (real CCTV, 20 clips)", "",
          "**Operating point:** sigmoid mask, threshold 0.4, steepness 12, blur kernel 21, idle blur 51, gate 0.25.",
          "",
          "| Config | CRF | Clips | KB | PSNR | Sal-PSNR | LPIPS |",
          "|---|---|---|---|---|---|---|"]
    for cfg_id in sorted(by_cfg.keys()):
        g = by_cfg[cfg_id]
        n = len(g)
        sz = np.mean([r["out_bytes"] for r in g]) / 1024
        psnr = np.mean([r["psnr_mean"] for r in g])
        sal = np.mean([r["sal_psnr_mean"] for r in g])
        lp = np.mean([r["lpips_mean"] for r in g])
        md.append(f"| {cfg_id} | {g[0]['crf']} | {n} | {sz:.0f} | {psnr:.2f} | {sal:.2f} | {lp:.3f} |")
    md.append("")
    md.append("**Plus the CRF 28 point already measured in results/ablation_mask/:**")
    md.append("| ours_sigmoid_b21_crf28 | 28 | 20 | 74 | 23.39 | 28.61 | 0.422 |")
    md.append("")
    md.append("**Reference baselines (uniform H.265 on the same 20 clips):**")
    md.append("| baseline_crf22 | 22 | 20 | 239 | 40.23 | 37.97 | 0.013 |")
    md.append("| baseline_crf28 | 28 | 20 | 128 | 36.65 | 33.69 | 0.029 |")
    md.append("| baseline_crf34 | 34 | 20 |  73 | 33.12 | 29.69 | 0.060 |")
    md.append("| baseline_crf40 | 40 | 20 |  44 | 29.60 | 25.85 | 0.117 |")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")
    print()
    print((out_dir / "summary.md").read_text())


if __name__ == "__main__":
    main()

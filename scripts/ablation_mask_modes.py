"""
Targeted experiment: does a sharper saliency mask close the LPIPS gap to baseline?

Hypothesis: the existing "alpha" mode (continuous saliency as alpha) softly blurs
even moderately-salient pixels. Switching to a sigmoid or binary mask should keep
salient interiors fully sharp while still aggressively blurring non-salient ones.

Sweeps (all on the 20 real CCTV clips, gate threshold 0.25, libx265 superfast):
  ours_alpha_b21_crf28      — control: existing soft alpha mode
  ours_sigmoid_b21_crf28    — sigmoid mask, threshold 0.4, steepness 12
  ours_sigmoid_b9_crf28     — same but lighter blur
  ours_binary_b21_crf28     — hard binary mask, threshold 0.4

Output:
  results/ablation_mask/{config_id}/{clip_id}.mp4
  results/ablation_mask/rows.jsonl
  results/ablation_mask/summary.md

Computes PSNR + sal-PSNR + LPIPS for each.
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


CONFIGS = [
    {"id": "ours_alpha_b21_crf28",   "blur": 21, "idle_blur": 51, "mask_mode": "alpha",   "mask_threshold": 0.4, "mask_steepness": 12.0},
    {"id": "ours_sigmoid_b21_crf28", "blur": 21, "idle_blur": 51, "mask_mode": "sigmoid", "mask_threshold": 0.4, "mask_steepness": 12.0},
    {"id": "ours_sigmoid_b9_crf28",  "blur": 9,  "idle_blur": 31, "mask_mode": "sigmoid", "mask_threshold": 0.4, "mask_steepness": 12.0},
    {"id": "ours_binary_b21_crf28",  "blur": 21, "idle_blur": 51, "mask_mode": "binary",  "mask_threshold": 0.4, "mask_steepness": 12.0},
]
CRF = 28
GATE = 0.25


def run_ours(input_path: str, out_dir: Path, cfg: dict, sal_est, lpips_model):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    gate = FootageGate(threshold=GATE, enable_person=False)
    smoother = TemporalSmoother(window=5)

    active = SaliencyCompressor(CompressorConfig(
        tier="C", codec="libx265", crf=CRF,
        blur_strength=cfg["blur"], preset="superfast",
        mask_mode=cfg["mask_mode"],
        mask_threshold=cfg["mask_threshold"],
        mask_steepness=cfg["mask_steepness"],
    ))
    idle = SaliencyCompressor(CompressorConfig(
        tier="C", codec="libx265", crf=CRF,
        blur_strength=cfg["idle_blur"], preset="superfast",
        mask_mode=cfg["mask_mode"],  # idle uses same mask mode for consistency
        mask_threshold=cfg["mask_threshold"],
        mask_steepness=cfg["mask_steepness"],
    ))

    triggered = total = 0
    processed = []
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        score = gate.step(frame)
        sal = smoother.smooth(sal_est.predict(frame))
        out_frame = active.process_frame(frame, sal) if score.triggered else idle.process_frame(frame, sal)
        processed.append(out_frame)
        triggered += int(score.triggered)
        total += 1
    cap.release()
    t_proc = time.perf_counter() - t0

    out_path = out_dir / f"{cfg['id']}_{Path(input_path).stem}.mp4"
    active.encode(iter(processed), str(out_path), fps=fps, size=(w, h))
    m = video_metrics_with_saliency(input_path, str(out_path), sal_est, every=30, use_lpips=False)

    # LPIPS separately (more reliable than ablation_real.py's path)
    cap_r = cv2.VideoCapture(input_path); cap_d = cv2.VideoCapture(str(out_path))
    lps = []
    i = 0
    while True:
        ok_r, fr = cap_r.read(); ok_d, fd = cap_d.read()
        if not (ok_r and ok_d): break
        if i % 30 == 0:
            if fr.shape != fd.shape: fd = cv2.resize(fd, (fr.shape[1], fr.shape[0]))
            lps.append(lpips_model.compute(fr, fd))
        i += 1
    cap_r.release(); cap_d.release()
    lpips_mean = float(np.mean(lps)) if lps else float("nan")

    return {
        "id": cfg["id"],
        "clip_id": Path(input_path).stem,
        "blur": cfg["blur"],
        "idle_blur": cfg["idle_blur"],
        "mask_mode": cfg["mask_mode"],
        "mask_threshold": cfg["mask_threshold"],
        "crf": CRF,
        "gate": GATE,
        "out_path": str(out_path),
        "out_bytes": os.path.getsize(out_path),
        "ref_bytes": os.path.getsize(input_path),
        "trigger_ratio": triggered / max(total, 1),
        "wall_proc_s": round(t_proc, 2),
        "fps_proc": round(total / t_proc, 1) if t_proc > 0 else 0,
        "psnr_mean": m["psnr_mean"],
        "ssim_mean": m["ssim_mean"],
        "sal_psnr_mean": m["sal_psnr_mean"],
        "lpips_mean": lpips_mean,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="data/real/*.mp4")
    ap.add_argument("--out", default="results/ablation_mask")
    args = ap.parse_args()

    clips = sorted(glob.glob(args.inputs))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"

    sal_est = SaliencyEstimator(backend="spectral")
    print("Loading LPIPS model...")
    lpips_model = LPIPSMetric()  # alex on cpu

    print(f"Running {len(CONFIGS)} configs × {len(clips)} clips = {len(CONFIGS) * len(clips)} runs")
    rows = []
    for cfg in CONFIGS:
        print(f"\n--- {cfg['id']} (mask={cfg['mask_mode']}) ---")
        for clip in clips:
            row = run_ours(clip, out_dir, cfg, sal_est, lpips_model)
            rows.append(row)
            with rows_path.open("a") as f:
                f.write(json.dumps(row, default=str) + "\n")
            print(f"  {row['clip_id']}: bytes={row['out_bytes']:>6}  "
                  f"PSNR={row['psnr_mean']:.2f}  sal-PSNR={row['sal_psnr_mean']:.2f}  "
                  f"LPIPS={row['lpips_mean']:.3f}")

    by_cfg = defaultdict(list)
    for r in rows:
        by_cfg[r["id"]].append(r)

    md = ["# Mask-mode ablation (real CCTV, 20 clips, CRF 28)", "",
          "| Config | Mask | Blur | Clips | KB | PSNR | Sal-PSNR | LPIPS | Trigger |",
          "|---|---|---|---|---|---|---|---|---|"]
    for cfg_id in sorted(by_cfg.keys()):
        g = by_cfg[cfg_id]
        n = len(g)
        sz = np.mean([r["out_bytes"] for r in g]) / 1024
        psnr = np.mean([r["psnr_mean"] for r in g])
        sal = np.mean([r["sal_psnr_mean"] for r in g])
        lp = np.mean([r["lpips_mean"] for r in g])
        trig = np.mean([r["trigger_ratio"] for r in g])
        md.append(f"| {cfg_id} | {g[0]['mask_mode']} | {g[0]['blur']} | {n} | "
                  f"{sz:.0f} | {psnr:.2f} | {sal:.2f} | {lp:.3f} | {trig:.3f} |")
    md.append("")
    md.append("**Reference points (uniform H.265 baselines, same 20 clips):**")
    md.append("- baseline_crf28: 128 KB · PSNR 36.65 · sal-PSNR 33.69 · LPIPS 0.029")
    md.append("- baseline_crf34:  73 KB · PSNR 33.12 · sal-PSNR 29.69 · LPIPS 0.060")
    md.append("- baseline_crf40:  44 KB · PSNR 29.60 · sal-PSNR 25.85 · LPIPS 0.117")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")
    print()
    print((out_dir / "summary.md").read_text())


if __name__ == "__main__":
    main()

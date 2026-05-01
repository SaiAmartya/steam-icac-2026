"""
Real-video ablation: sweep across multiple real clips instead of one synthetic.

Produces:
  results/ablation_real/<clip_id>/<config_id>.mp4
  results/ablation_real/rows.jsonl (append-only)
  results/ablation_real/summary.json
  results/ablation_real/summary.md
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

from src.gate import FootageGate
from src.saliency import SaliencyEstimator, TemporalSmoother
from src.compress import SaliencyCompressor, CompressorConfig, encode_uniform
from src.metrics import saliency_weighted_psnr

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import lpips as _lpips_pkg
    _LPIPS_AVAILABLE = _TORCH_AVAILABLE
except ImportError:
    _LPIPS_AVAILABLE = False


def sample_metrics(ref_path, dist_path, every=30, sal_estimator=None):
    """Sample metrics at given interval, optionally including sal_psnr."""
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
            
            # Saliency-weighted PSNR if estimator is provided
            if sal_estimator is not None:
                sal = sal_estimator.predict(fr)
                sal_p = saliency_weighted_psnr(fr, fd, sal, threshold=0.5)
                if not np.isnan(sal_p):
                    sal_psnrs.append(sal_p)
        i += 1
    cap_r.release()
    cap_d.release()
    
    out = {
        "psnr_mean": float(np.mean(psnrs)) if psnrs else 0.0,
        "ssim_mean": float(np.mean(ssims)) if ssims else 0.0,
        "sal_psnr_mean": float(np.mean(sal_psnrs)) if sal_psnrs else float("nan"),
        "n_evaluated": len(psnrs),
    }
    return out


def run_baseline(input_path: str, out_dir: Path, crf: int) -> dict:
    """Run baseline encode."""
    bp = out_dir / f"baseline_crf{crf}.mp4"
    t0 = time.perf_counter()
    encode_uniform(input_path, str(bp), crf=crf, codec="libx265", preset="superfast")
    t1 = time.perf_counter()
    sal_est = SaliencyEstimator(backend="spectral")
    m = sample_metrics(input_path, str(bp), every=30, sal_estimator=sal_est)
    return {
        "id": f"baseline_crf{crf}",
        "kind": "baseline",
        "crf": crf,
        "blur": None,
        "idle_blur": None,
        "gate": None,
        "ref_bytes": os.path.getsize(input_path),
        "dist_bytes": os.path.getsize(bp),
        "ratio_vs_input": os.path.getsize(input_path) / max(os.path.getsize(bp), 1),
        "trigger_ratio": None,
        "wall_proc_s": None,
        "wall_encode_s": round(t1 - t0, 2),
        "fps_proc": None,
        **m,
    }


def run_ours(input_path: str, out_dir: Path, cfg: dict) -> dict:
    """Run saliency-aware encode."""
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    gate = FootageGate(threshold=cfg["gate"], enable_person=False)
    sal_est = SaliencyEstimator(backend="spectral")
    smoother = TemporalSmoother(window=5)
    active = SaliencyCompressor(CompressorConfig(
        tier="C", codec="libx265", crf=cfg["crf"],
        blur_strength=cfg["blur"], preset="superfast"))
    idle = SaliencyCompressor(CompressorConfig(
        tier="C", codec="libx265", crf=cfg["crf"],
        blur_strength=cfg["idle_blur"], preset="superfast"))

    triggered = 0
    total = 0
    usum = 0.0
    processed = []
    t0 = time.perf_counter()
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        score = gate.step(frame)
        sal = smoother.smooth(sal_est.predict(frame))
        out_frame = (active.process_frame(frame, sal) if score.triggered
                     else idle.process_frame(frame, sal))
        processed.append(out_frame)
        triggered += int(score.triggered)
        total += 1
        usum += score.usefulness
    
    cap.release()
    t_proc = time.perf_counter() - t0

    out_path = out_dir / f"{cfg['id']}.mp4"
    t_enc0 = time.perf_counter()
    active.encode(iter(processed), str(out_path), fps=fps, size=(w, h))
    t_encode = time.perf_counter() - t_enc0
    
    m = sample_metrics(input_path, str(out_path), every=30, sal_estimator=sal_est)
    
    return {
        **cfg,
        "kind": "ours",
        "ref_bytes": os.path.getsize(input_path),
        "dist_bytes": os.path.getsize(out_path),
        "ratio_vs_input": os.path.getsize(input_path) / max(os.path.getsize(out_path), 1),
        "trigger_ratio": triggered / max(total, 1),
        "wall_proc_s": round(t_proc, 2),
        "wall_encode_s": round(t_encode, 2),
        "fps_proc": round(total / t_proc, 1) if t_proc > 0 else 0,
        **m,
    }


def get_configs():
    """Define the sweep: only 'ours' configs (skip redundant baselines)."""
    rows = []
    # Ours: vary blur at CRF 28
    for blur in [9, 21, 41]:
        rows.append({
            "id": f"ours_b{blur}_crf28",
            "crf": 28,
            "blur": blur,
            "idle_blur": blur * 2 + 1,
            "gate": 0.25,
        })
    # Ours: vary CRF at blur=21
    for crf in [22, 28, 34]:
        rows.append({
            "id": f"ours_b21_crf{crf}",
            "crf": crf,
            "blur": 21,
            "idle_blur": 51,
            "gate": 0.25,
        })
    return rows


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", default="data/real/*.mp4", help="Glob for input clips")
    p.add_argument("--out", default="results/ablation_real")
    p.add_argument("--start", type=int, default=0, help="Config start index")
    p.add_argument("--end", type=int, default=999, help="Config end index")
    p.add_argument("--every-metric", type=int, default=30, help="Sample every N frames for metrics")
    p.add_argument("--use-lpips", action="store_true", help="Compute LPIPS if available")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "rows.jsonl"

    # Gather input clips
    clip_paths = sorted(glob.glob(args.inputs))
    if not clip_paths:
        print(f"WARNING: No clips found matching {args.inputs}")
        return

    configs = get_configs()
    end = min(args.end, len(configs))
    print(f"Found {len(clip_paths)} clips; running configs [{args.start}:{end}] of {len(configs)}")

    # Per-clip-per-config processing
    all_rows = []
    for clip_path in clip_paths:
        clip_id = Path(clip_path).stem
        clip_out_dir = out_dir / clip_id
        clip_out_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(args.start, end):
            cfg = configs[idx]
            print(f"  {clip_id} [{idx}] {cfg['id']}...", end="", flush=True)
            
            try:
                row = run_ours(clip_path, clip_out_dir, cfg)
                row["clip_id"] = clip_id
                row["config_idx"] = idx
                all_rows.append(row)
                
                print(f" PSNR={row['psnr_mean']:.2f} sal-PSNR={row['sal_psnr_mean']:.2f}", flush=True)
                
                with log_path.open("a") as f:
                    f.write(json.dumps(row, default=str) + "\n")
            except Exception as e:
                print(f" ERROR: {e}", flush=True)

    # Write summary.json: per-config aggregates
    if all_rows:
        by_config = {}
        for row in all_rows:
            cid = row["id"]
            if cid not in by_config:
                by_config[cid] = []
            by_config[cid].append(row)

        summary = {}
        for cid, rows_list in by_config.items():
            summary[cid] = {
                "clips_evaluated": len(rows_list),
                "mean_file_size_kb": float(np.mean([r["dist_bytes"] for r in rows_list])) / 1024,
                "mean_psnr": float(np.mean([r["psnr_mean"] for r in rows_list])),
                "mean_sal_psnr": float(np.mean([r["sal_psnr_mean"] for r in rows_list if not np.isnan(r["sal_psnr_mean"])])),
                "mean_trigger_ratio": float(np.mean([r["trigger_ratio"] for r in rows_list if r["trigger_ratio"] is not None])),
            }
            if args.use_lpips and any("lpips_mean" in r for r in rows_list):
                summary[cid]["mean_lpips"] = float(np.mean([r.get("lpips_mean", float("nan")) for r in rows_list]))

        with (out_dir / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)

        # Write summary.md: markdown table
        md_lines = [
            "# Ablation Results Summary",
            "",
            "| Config | Clips | File Size (KB) | PSNR (dB) | Sal-PSNR (dB) | Trigger Ratio |",
            "|--------|-------|---|---|---|---|",
        ]
        for cid in sorted(summary.keys()):
            s = summary[cid]
            md_lines.append(
                f"| {cid} | {s['clips_evaluated']} | "
                f"{s['mean_file_size_kb']:.0f} | "
                f"{s['mean_psnr']:.2f} | "
                f"{s['mean_sal_psnr']:.2f} | "
                f"{s['mean_trigger_ratio']:.3f} |"
            )
        
        with (out_dir / "summary.md").open("w") as f:
            f.write("\n".join(md_lines))

        print(f"\nWrote summary to {out_dir}/summary.json and summary.md")


if __name__ == "__main__":
    main()

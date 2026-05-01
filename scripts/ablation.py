"""
Ablation runner. Sweeps gate threshold, baseline CRF, blur strength.

For each config it produces an encoded video and records:
  - file size (bytes)
  - PSNR/SSIM vs the original
  - mean usefulness score, gate-trigger ratio
  - encode wall time

Outputs:
  results/ablation/<config_id>.mp4 (if --keep-videos)
  results/ablation/results.csv
  results/ablation/results.json   (full structured)

Designed to fit in tight time budgets — uses lightweight metric sampling.
"""
from __future__ import annotations

import csv
import itertools
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

from src.gate import FootageGate
from src.saliency import SaliencyEstimator, TemporalSmoother
from src.compress import SaliencyCompressor, CompressorConfig, encode_uniform


def sample_metrics(ref_path: str, dist_path: str, every: int = 30) -> dict:
    """Fast PSNR + SSIM sampling (every Nth frame)."""
    cap_r = cv2.VideoCapture(ref_path)
    cap_d = cv2.VideoCapture(dist_path)
    psnrs, ssims = [], []
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
        i += 1
    cap_r.release(); cap_d.release()
    return {
        "psnr_mean": float(np.mean(psnrs)) if psnrs else float("nan"),
        "ssim_mean": float(np.mean(ssims)) if ssims else float("nan"),
        "n_evaluated": len(psnrs),
    }


def run_one(input_path: str, out_dir: Path, config: dict) -> dict:
    """Run a single ablation config end-to-end. Returns row dict."""
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    gate = FootageGate(threshold=config["gate"], enable_person=False)
    sal_est = SaliencyEstimator(backend="spectral")
    smoother = TemporalSmoother(window=5)

    active = SaliencyCompressor(CompressorConfig(
        tier="C", codec="libx265", crf=config["crf"],
        blur_strength=config["blur"], preset="superfast"
    ))
    idle = SaliencyCompressor(CompressorConfig(
        tier="C", codec="libx265", crf=config["crf"],
        blur_strength=config["idle_blur"], preset="superfast"
    ))

    triggered_count = 0
    total = 0
    usefulness_sum = 0.0
    processed = []

    t_start = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        score = gate.step(frame)
        sal = smoother.smooth(sal_est.predict(frame))
        if score.triggered:
            triggered_count += 1
            out_frame = active.process_frame(frame, sal)
        else:
            out_frame = idle.process_frame(frame, sal)
        processed.append(out_frame)
        usefulness_sum += score.usefulness
        total += 1
    cap.release()
    t_proc = time.perf_counter() - t_start

    out_path = out_dir / f"{config['id']}.mp4"
    t_enc_start = time.perf_counter()
    active.encode(iter(processed), str(out_path), fps=fps, size=(w, h))
    t_encode = time.perf_counter() - t_enc_start

    metrics = sample_metrics(input_path, str(out_path), every=30)

    row = {
        **config,
        "frames": total,
        "trigger_ratio": triggered_count / max(total, 1),
        "mean_usefulness": usefulness_sum / max(total, 1),
        "out_path": str(out_path),
        "out_bytes": os.path.getsize(out_path),
        "ref_bytes": os.path.getsize(input_path),
        "ratio_vs_input": os.path.getsize(input_path) / max(os.path.getsize(out_path), 1),
        "wall_proc_s": round(t_proc, 2),
        "wall_encode_s": round(t_encode, 2),
        "fps_proc": round(total / t_proc, 1) if t_proc > 0 else 0,
        **metrics,
    }
    return row


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", default="results/ablation")
    p.add_argument("--quick", action="store_true", help="Smaller sweep for fast smoke test")
    args = p.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        configs = [
            {"id": "qc28_b21_g0.25", "crf": 28, "blur": 21, "idle_blur": 51, "gate": 0.25},
            {"id": "qc34_b21_g0.25", "crf": 34, "blur": 21, "idle_blur": 51, "gate": 0.25},
        ]
    else:
        crfs = [22, 28, 34]
        blurs = [11, 21, 41]
        gates = [0.20, 0.35]
        configs = []
        for crf, blur, gate in itertools.product(crfs, blurs, gates):
            configs.append({
                "id": f"crf{crf}_b{blur}_g{gate}",
                "crf": crf, "blur": blur, "idle_blur": blur * 2 + 1,
                "gate": gate,
            })

    print(f"Running {len(configs)} configs...")
    rows = []
    # Baselines (uniform CRF, no saliency)
    for crf in [22, 28, 34]:
        bp = out_dir / f"baseline_crf{crf}.mp4"
        t0 = time.perf_counter()
        encode_uniform(args.input, str(bp), crf=crf, codec="libx265", preset="superfast")
        t1 = time.perf_counter()
        m = sample_metrics(args.input, str(bp), every=30)
        rows.append({
            "id": f"baseline_crf{crf}", "kind": "baseline",
            "crf": crf, "blur": None, "idle_blur": None, "gate": None,
            "out_path": str(bp),
            "out_bytes": os.path.getsize(bp),
            "ref_bytes": os.path.getsize(args.input),
            "ratio_vs_input": os.path.getsize(args.input) / max(os.path.getsize(bp), 1),
            "wall_encode_s": round(t1 - t0, 2),
            **m,
        })
        print(f"  baseline_crf{crf}: {os.path.getsize(bp)} bytes  PSNR={m['psnr_mean']:.2f}  SSIM={m['ssim_mean']:.3f}")

    for cfg in configs:
        cfg["kind"] = "ours"
        row = run_one(args.input, out_dir, cfg)
        rows.append(row)
        print(f"  {cfg['id']}: {row['out_bytes']} bytes  PSNR={row['psnr_mean']:.2f}  "
              f"SSIM={row['ssim_mean']:.3f}  trig={row['trigger_ratio']:.2f}")

    # Write CSV + JSON
    keys = sorted({k for r in rows for k in r})
    with (out_dir / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with (out_dir / "results.json").open("w") as f:
        json.dump(rows, f, indent=2, default=str)

    print(f"\nWrote {out_dir}/results.csv  and  results.json  ({len(rows)} rows)")


if __name__ == "__main__":
    main()

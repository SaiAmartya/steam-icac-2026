"""
Focused ablation that produces a clean rate-distortion story.

Six baselines (varying CRF) + six 'ours' configs (varying blur + CRF).
Designed to fit in tight time budgets: each call accepts --start and --end
indices so we can chunk the work across multiple bash calls.

Outputs:
  results/ablation/rows.jsonl  (append-only, one JSON per config)
  results/ablation/<id>.mp4
"""
from __future__ import annotations

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


def sample_metrics(ref_path, dist_path, every=30):
    cap_r = cv2.VideoCapture(ref_path); cap_d = cv2.VideoCapture(dist_path)
    psnrs, ssims = [], []
    i = 0
    while True:
        ok_r, fr = cap_r.read(); ok_d, fd = cap_d.read()
        if not (ok_r and ok_d): break
        if i % every == 0:
            if fr.shape != fd.shape: fd = cv2.resize(fd, (fr.shape[1], fr.shape[0]))
            psnrs.append(float(sk_psnr(fr, fd, data_range=255)))
            ssims.append(float(sk_ssim(fr, fd, channel_axis=2, data_range=255)))
        i += 1
    cap_r.release(); cap_d.release()
    return {"psnr_mean": float(np.mean(psnrs)) if psnrs else 0.0,
            "ssim_mean": float(np.mean(ssims)) if ssims else 0.0,
            "n_evaluated": len(psnrs)}


def run_baseline(input_path: str, out_dir: Path, crf: int) -> dict:
    bp = out_dir / f"baseline_crf{crf}.mp4"
    t0 = time.perf_counter()
    encode_uniform(input_path, str(bp), crf=crf, codec="libx265", preset="superfast")
    t1 = time.perf_counter()
    m = sample_metrics(input_path, str(bp), every=30)
    return {
        "id": f"baseline_crf{crf}", "kind": "baseline",
        "crf": crf, "blur": None, "idle_blur": None, "gate": None,
        "out_bytes": os.path.getsize(bp),
        "ref_bytes": os.path.getsize(input_path),
        "ratio_vs_input": os.path.getsize(input_path) / max(os.path.getsize(bp), 1),
        "trigger_ratio": None, "mean_usefulness": None,
        "wall_total_s": round(t1 - t0, 2),
        **m,
    }


def run_ours(input_path: str, out_dir: Path, cfg: dict) -> dict:
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    gate = FootageGate(threshold=cfg["gate"], enable_person=False)
    sal_est = SaliencyEstimator(backend="spectral")
    smoother = TemporalSmoother(window=5)
    active = SaliencyCompressor(CompressorConfig(tier="C", codec="libx265",
                                                 crf=cfg["crf"], blur_strength=cfg["blur"], preset="superfast"))
    idle = SaliencyCompressor(CompressorConfig(tier="C", codec="libx265",
                                               crf=cfg["crf"], blur_strength=cfg["idle_blur"], preset="superfast"))

    triggered = 0; total = 0; usum = 0.0; processed = []
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok: break
        score = gate.step(frame)
        sal = smoother.smooth(sal_est.predict(frame))
        out_frame = active.process_frame(frame, sal) if score.triggered else idle.process_frame(frame, sal)
        processed.append(out_frame)
        triggered += int(score.triggered); total += 1; usum += score.usefulness
    cap.release()
    t_proc = time.perf_counter() - t0

    out_path = out_dir / f"{cfg['id']}.mp4"
    t_enc0 = time.perf_counter()
    active.encode(iter(processed), str(out_path), fps=fps, size=(w, h))
    t_encode = time.perf_counter() - t_enc0
    m = sample_metrics(input_path, str(out_path), every=30)
    return {
        **cfg, "kind": "ours",
        "frames": total,
        "trigger_ratio": triggered / max(total, 1),
        "mean_usefulness": usum / max(total, 1),
        "out_bytes": os.path.getsize(out_path),
        "ref_bytes": os.path.getsize(input_path),
        "ratio_vs_input": os.path.getsize(input_path) / max(os.path.getsize(out_path), 1),
        "wall_proc_s": round(t_proc, 2),
        "wall_encode_s": round(t_encode, 2),
        "fps_proc": round(total / t_proc, 1) if t_proc > 0 else 0,
        **m,
    }


def get_configs():
    """Define the full sweep."""
    rows = []
    # Baselines: varying CRF
    for crf in [22, 28, 34, 40]:
        rows.append({"_kind": "baseline", "crf": crf})
    # Ours: vary blur at CRF 28 (perceptual sweep at fixed quality target)
    for blur in [9, 21, 41]:
        rows.append({
            "_kind": "ours", "id": f"ours_b{blur}_crf28",
            "crf": 28, "blur": blur, "idle_blur": blur * 2 + 1, "gate": 0.25,
        })
    # Ours: vary CRF at blur=21 (rate sweep at fixed perceptual aggressiveness)
    for crf in [22, 28, 34]:
        rows.append({
            "_kind": "ours", "id": f"ours_b21_crf{crf}",
            "crf": crf, "blur": 21, "idle_blur": 51, "gate": 0.25,
        })
    return rows


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", default="results/ablation")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=999)
    args = p.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "rows.jsonl"

    configs = get_configs()
    end = min(args.end, len(configs))
    print(f"Running configs [{args.start}:{end}] of {len(configs)} total")

    for idx in range(args.start, end):
        c = configs[idx]
        if c["_kind"] == "baseline":
            row = run_baseline(args.input, out_dir, crf=c["crf"])
        else:
            row = run_ours(args.input, out_dir, {k: v for k, v in c.items() if not k.startswith("_")})
        row["index"] = idx
        with log_path.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        print(f"  [{idx}] {row['id']}: bytes={row['out_bytes']}  "
              f"PSNR={row['psnr_mean']:.2f}  SSIM={row['ssim_mean']:.3f}  "
              f"trig={row.get('trigger_ratio')}")


if __name__ == "__main__":
    main()

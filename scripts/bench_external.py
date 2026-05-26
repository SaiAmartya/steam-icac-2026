"""
Run the full bg/fg codec benchmark on an arbitrary mp4 file.

Use this when you want to test on footage that isn't in `data/real/` — e.g.
a downloaded VIRAT clip, a Pexels stock 1080p surveillance video, or your
own webcam capture.

What it produces (under results/bench_external/<input_stem>/):
  - source.mp4                         (a copy of your input for reference)
  - baseline_crf<NN>.mp4              (uniform H.265 at each CRF)
  - ours_bgfg_crf<NN>.mp4             (our codec at each CRF)
  - ours_bgfg_crf<NN>_bg.png          (background reference for inspection)
  - comparison_crf<NN>.mp4            (4-panel side-by-side video)
  - bench.json                        (metrics: bytes, PSNR, SSIM, sal-PSNR)
  - bench.md                          (human-readable summary)

Usage:
    python scripts/bench_external.py --input ~/Downloads/virat_clip.mp4
    python scripts/bench_external.py --input /path/to/cctv.mp4 --crfs 22 28
    python scripts/bench_external.py --input ./capture.mp4 --saliency yolo+spectral --show
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
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

OUT_ROOT = ROOT / "results/bench_external"


def kb(p: Path) -> float:
    return p.stat().st_size / 1024 if p.exists() else 0.0


def sample_metrics(ref_path: Path, dist_path: Path, sal_est: SaliencyEstimator,
                   every: int = 15) -> dict:
    cap_r = cv2.VideoCapture(str(ref_path))
    cap_d = cv2.VideoCapture(str(dist_path))
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
            sal = sal_est.predict(fr)
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
        "n_frames_evaluated": len(psnrs),
    }


def build_4panel_comparison(orig_path: Path, base_path: Path, bgfg_path: Path,
                            crf: int, out_path: Path, sal_est: SaliencyEstimator) -> None:
    """4-panel: original | saliency | baseline | ours_bgfg, encoded as H.264 mp4."""
    caps = {n: cv2.VideoCapture(str(p)) for n, p in
            [("original", orig_path), ("baseline", base_path), ("bgfg", bgfg_path)]}
    fps = caps["original"].get(cv2.CAP_PROP_FPS) or 30.0
    w = int(caps["original"].get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(caps["original"].get(cv2.CAP_PROP_FRAME_HEIGHT))
    base_kb = kb(base_path); bgfg_kb = kb(bgfg_path)
    pct = (1 - bgfg_kb / base_kb) * 100 if base_kb > 0 else 0

    def label(frame, text, sub=""):
        out = frame.copy()
        strip_h = max(28, h // 14)
        cv2.rectangle(out, (0, 0), (w, strip_h), (0, 0, 0), -1)
        cv2.putText(out, text, (10, int(strip_h * 0.68)),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.5, h / 700),
                    (255, 255, 255), 1, cv2.LINE_AA)
        if sub:
            cv2.putText(out, sub, (10, strip_h + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.4, h / 900),
                        (200, 200, 200), 1, cv2.LINE_AA)
        return out

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w*4}x{h}", "-r", f"{fps}",
        "-i", "-",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    n = 0
    while True:
        frames = {}
        for name, cap in caps.items():
            ok, f = cap.read()
            if not ok:
                frames = None
                break
            if (f.shape[1], f.shape[0]) != (w, h):
                f = cv2.resize(f, (w, h))
            frames[name] = f
        if frames is None:
            break

        sal_map = sal_est.predict(frames["original"])
        sal_u8 = (np.clip(sal_map, 0, 1) * 255).astype(np.uint8)
        heat = cv2.applyColorMap(sal_u8, cv2.COLORMAP_JET)
        sal_overlay = cv2.addWeighted(frames["original"], 0.55, heat, 0.45, 0.0)

        row = np.hstack([
            label(frames["original"], "original",          f"{kb(orig_path):.0f} KB on disk"),
            label(sal_overlay,         "saliency overlay", f"backend = {sal_est.backend}"),
            label(frames["baseline"],  f"baseline H.265 (CRF {crf})", f"{base_kb:.0f} KB"),
            label(frames["bgfg"],      f"ours_bgfg (CRF {crf})",       f"{bgfg_kb:.0f} KB  ({pct:+.0f}%)"),
        ])
        proc.stdin.write(row.tobytes())
        n += 1

    proc.stdin.close()
    proc.wait()
    for cap in caps.values():
        cap.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path,
                        help="Path to any mp4 (VIRAT, Pexels, webcam capture, etc.)")
    parser.add_argument("--crfs", type=int, nargs="*", default=[22, 28, 34])
    parser.add_argument("--saliency", default="yolo+spectral",
                        choices=["spectral", "finegrained", "yolo", "yolo+spectral"])
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory (default: results/bench_external/<input_stem>/)")
    parser.add_argument("--skip-comparison-video", action="store_true",
                        help="Skip the 4-panel comparison mp4 (faster)")
    parser.add_argument("--show", action="store_true",
                        help="Open the result folder when done")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    out_dir = args.out or (OUT_ROOT / args.input.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_copy = out_dir / "source.mp4"
    if not source_copy.exists() or source_copy.stat().st_size != args.input.stat().st_size:
        shutil.copy(args.input, source_copy)

    # Probe
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / fps if fps else 0
    cap.release()

    print(f"\n=== bench_external — {args.input.name} ===")
    print(f"  resolution  : {w} x {h}")
    print(f"  duration    : {duration:.1f}s ({n_frames} frames @ {fps:.1f}fps)")
    print(f"  output dir  : {out_dir}")
    print(f"  CRFs        : {args.crfs}")
    print(f"  saliency    : {args.saliency}\n")

    sal_est = SaliencyEstimator(backend=args.saliency)
    results = {
        "input": str(args.input.resolve()),
        "resolution": [w, h], "fps": fps, "duration_s": duration, "n_frames": n_frames,
        "saliency_backend": args.saliency,
        "raw_input_bytes": args.input.stat().st_size,
        "per_crf": [],
    }

    for crf in args.crfs:
        print(f"  --- CRF {crf} ---")
        base_path = out_dir / f"baseline_crf{crf}.mp4"
        bgfg_path = out_dir / f"ours_bgfg_crf{crf}.mp4"
        cmp_path  = out_dir / f"comparison_crf{crf}.mp4"

        t0 = time.time()
        encode_uniform(str(args.input), str(base_path), crf=crf)
        t_base = time.time() - t0
        print(f"    baseline H.265 : {kb(base_path):7.1f} KB  ({t_base:.1f}s)")

        t0 = time.time()
        cfg = BgFgConfig(saliency_backend=args.saliency, crf=crf)
        BgFgCodec(cfg).encode(str(args.input), str(bgfg_path))
        t_bgfg = time.time() - t0
        print(f"    ours_bgfg      : {kb(bgfg_path):7.1f} KB  ({t_bgfg:.1f}s)")

        saved = (1 - kb(bgfg_path)/kb(base_path)) * 100 if kb(base_path) > 0 else 0
        print(f"    savings        : {saved:+.1f}%")

        # Metrics
        print("    computing PSNR/SSIM/sal-PSNR...")
        base_metrics = sample_metrics(args.input, base_path, sal_est, every=15)
        bgfg_metrics = sample_metrics(args.input, bgfg_path, sal_est, every=15)
        print(f"    baseline sal-PSNR: {base_metrics['sal_psnr_mean']:.2f} dB")
        print(f"    bgfg     sal-PSNR: {bgfg_metrics['sal_psnr_mean']:.2f} dB")

        if not args.skip_comparison_video:
            print("    rendering 4-panel comparison video...")
            build_4panel_comparison(args.input, base_path, bgfg_path, crf, cmp_path, sal_est)
            print(f"    comparison     : {cmp_path}")

        results["per_crf"].append({
            "crf": crf,
            "baseline_bytes": base_path.stat().st_size,
            "bgfg_bytes":     bgfg_path.stat().st_size,
            "saved_pct":      saved,
            "baseline":       base_metrics,
            "bgfg":           bgfg_metrics,
            "encode_baseline_s": t_base,
            "encode_bgfg_s":    t_bgfg,
        })
        print()

    # Write summary files
    (out_dir / "bench.json").write_text(json.dumps(results, indent=2))
    md = [f"# bench_external — {args.input.name}", "",
          f"- resolution: {w}x{h}",
          f"- duration: {duration:.1f}s ({n_frames} frames @ {fps:.1f}fps)",
          f"- saliency backend: {args.saliency}",
          "",
          "| CRF | baseline KB | bgfg KB | saved | baseline sal-PSNR | bgfg sal-PSNR |",
          "|---|---|---|---|---|---|"]
    for r in results["per_crf"]:
        md.append(
            f"| {r['crf']} | {r['baseline_bytes']/1024:.1f} | "
            f"{r['bgfg_bytes']/1024:.1f} | {r['saved_pct']:+.1f}% | "
            f"{r['baseline']['sal_psnr_mean']:.2f} | "
            f"{r['bgfg']['sal_psnr_mean']:.2f} |"
        )
    (out_dir / "bench.md").write_text("\n".join(md))

    print(f"\n=== done ===")
    print(f"  summary: {out_dir / 'bench.md'}")
    print(f"  json:    {out_dir / 'bench.json'}")
    if args.show:
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", str(out_dir)], check=False)
        except Exception:
            pass


if __name__ == "__main__":
    main()

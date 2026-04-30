"""
CLI entry point.

Usage:
  python scripts/run_pipeline.py --input data/test.mp4 [--out results/]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure src/ is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import PipelineConfig, run_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--out", default="results", help="Output directory")
    parser.add_argument("--crf", type=int, default=28)
    parser.add_argument("--codec", default="libx265")
    parser.add_argument("--blur", type=int, default=21, help="Gaussian blur strength (odd)")
    parser.add_argument("--idle-blur", type=int, default=51)
    parser.add_argument("--gate", type=float, default=0.35, help="Gate threshold 0-1")
    parser.add_argument("--person", action="store_true", help="Enable YOLO person detection")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cfg = PipelineConfig(
        gate_threshold=args.gate,
        gate_enable_person=args.person,
        crf=args.crf,
        codec=args.codec,
        blur_strength=args.blur,
        idle_blur_strength=args.idle_blur,
    )
    result = run_pipeline(args.input, args.out, cfg=cfg)

    print("\n=== SUMMARY ===")
    print(f"input      : {result['input']}")
    print(f"frames     : {result['frames']} @ {result['fps']:.1f} fps")
    print(f"gate hits  : {result['events_triggered']}")
    print("\n-- Ours (saliency-aware) --")
    for k in ("path", "psnr_mean", "ssim_mean", "dist_bytes", "ratio"):
        print(f"  {k}: {result['ours'][k]}")
    print("\n-- Baseline (uniform CRF) --")
    for k in ("path", "psnr_mean", "ssim_mean", "dist_bytes", "ratio"):
        print(f"  {k}: {result['baseline'][k]}")


if __name__ == "__main__":
    main()

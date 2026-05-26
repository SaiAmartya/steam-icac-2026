"""
Render a saliency-overlay video for a single clip.

Useful for demos where you want to *show how saliency moves with the action*
rather than just one static frame from the catalog.

Two modes:
  default       — single panel: heatmap blended over the frame
  --side-by-side — two panels: original | overlay

Usage:
    python scripts/saliency_video.py --clip clip_14
    python scripts/saliency_video.py --clip clip_02 --saliency yolo --side-by-side
    python scripts/saliency_video.py --clip clip_04 --out /tmp/sal_demo.mp4 --show
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.saliency import SaliencyEstimator, TemporalSmoother

DATA = ROOT / "data/real"
DEFAULT_OUT_DIR = ROOT / "results/saliency_videos"


def overlay(frame_bgr: np.ndarray, sal: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    sal_u8 = (np.clip(sal, 0, 1) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(sal_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame_bgr, 1.0 - alpha, heat, alpha, 0.0)


def label(frame_bgr: np.ndarray, text: str) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    out = frame_bgr.copy()
    cv2.rectangle(out, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", required=True, help="e.g. clip_14, or a path to any .mp4")
    parser.add_argument("--saliency", default="yolo+spectral",
                        choices=["spectral", "finegrained", "yolo", "yolo+spectral"])
    parser.add_argument("--smooth", type=int, default=5,
                        help="Temporal smoothing window for saliency (frames)")
    parser.add_argument("--side-by-side", action="store_true",
                        help="Output original | overlay (double width)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    inp = Path(args.clip) if args.clip.endswith(".mp4") else (DATA / f"{args.clip}.mp4")
    if not inp.exists():
        raise SystemExit(f"Missing clip: {inp}")

    args.out = args.out or (DEFAULT_OUT_DIR / f"{inp.stem}_saliency_{args.saliency.replace('+','_')}.mp4")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    estimator = SaliencyEstimator(backend=args.saliency)
    smoother = TemporalSmoother(window=args.smooth)

    cap = cv2.VideoCapture(str(inp))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {inp}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = w * 2 if args.side_by_side else w

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{out_w}x{h}", "-r", f"{fps}",
        "-i", "-",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(args.out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdin is not None

    print(f"  rendering {inp.stem} → {args.out} (backend={args.saliency})")
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        sal_raw = estimator.predict(frame)
        sal = smoother.smooth(sal_raw)
        ov = overlay(frame, sal)
        if args.side_by_side:
            row = np.hstack([label(frame, "original"), label(ov, f"saliency ({args.saliency})")])
            proc.stdin.write(row.tobytes())
        else:
            proc.stdin.write(label(ov, f"saliency overlay ({args.saliency})").tobytes())
        n += 1
        if n % 30 == 0:
            print(f"  ...{n} frames")
    cap.release()
    proc.stdin.close()
    proc.wait()

    print(f"  done — {n} frames, {args.out.stat().st_size/1024:.0f} KB")
    print(f"  open with: open '{args.out}'")
    if args.show:
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", str(args.out)], check=False)
        except Exception:
            pass


if __name__ == "__main__":
    main()

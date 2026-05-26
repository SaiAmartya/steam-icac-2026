"""
Side-by-side full-video comparison for a single clip.

Produces ONE playable mp4 with up to 4 panels stacked horizontally:
    original | saliency overlay | baseline H.265 | ours_bgfg

Best for live demos and slide reveals — judges can play the file in QuickTime
and watch all four versions in lockstep.

Usage:
    python scripts/compare_clip.py --clip clip_14 --crf 22
    python scripts/compare_clip.py --clip clip_04 --crf 28 --no-saliency
    python scripts/compare_clip.py --clip clip_02 --crf 22 --out /tmp/grab_demo.mp4
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
from src.bg_fg_codec import compute_background_median, BgFgCodec, BgFgConfig
from src.compress import _build_alpha

DATA = ROOT / "data/real"
ENCODED = ROOT / "results/ablation_bgfg/encoded"
DEFAULT_OUT_DIR = ROOT / "results/comparisons"


def label_frame(frame_bgr: np.ndarray, text: str, sub: str = "") -> np.ndarray:
    """Stamp a label across the top of a frame."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    # Black strip top
    strip_h = max(28, h // 14)
    cv2.rectangle(out, (0, 0), (w, strip_h), (0, 0, 0), -1)
    # Label
    cv2.putText(out, text, (10, int(strip_h * 0.68)),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.45, h / 700), (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(out, sub, (10, strip_h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.4, h / 900), (200, 200, 200), 1, cv2.LINE_AA)
    return out


def saliency_overlay_bgr(
    frame_bgr: np.ndarray,
    estimator: SaliencyEstimator,
    motion_helper: BgFgCodec | None = None,
    background: np.ndarray | None = None,
    smoother: TemporalSmoother | None = None,
) -> np.ndarray:
    """Overlay the EXACT mask the codec uses to drive the encoder.

    Applies the same pipeline as src/bg_fg_codec.py:
        sal_raw  = max(yolo+spectral, motion-from-background)
        sal      = TemporalSmoother(window=11).smooth(sal_raw)
        alpha    = sigmoid_mask(sal, threshold=0.3, steepness=12)
        alpha    = max(alpha, mask_floor=0.25)

    Without these steps the overlay would show a noisy single-frame mask
    that doesn't reflect what's actually being encoded.
    """
    sal = estimator.predict(frame_bgr)
    if motion_helper is not None and background is not None:
        motion = motion_helper._motion_mask(frame_bgr, background)
        sal = np.maximum(sal, motion)
    if smoother is not None:
        sal = smoother.smooth(sal)
    # Apply sigmoid mask shaping + floor — match BgFgCodec._blend_frame()
    alpha = _build_alpha(sal, mode="sigmoid", threshold=0.30, steepness=12.0)
    alpha = np.maximum(alpha, 0.25)  # mask_floor
    alpha_u8 = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(alpha_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame_bgr, 0.55, heat, 0.45, 0.0)


def kb(p: Path) -> float:
    return p.stat().st_size / 1024 if p.exists() else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", required=True, help="e.g. clip_14")
    parser.add_argument("--crf", type=int, default=22, choices=[22, 28, 34])
    parser.add_argument("--saliency", default="yolo+spectral",
                        choices=["spectral", "finegrained", "yolo", "yolo+spectral"])
    parser.add_argument("--no-saliency", action="store_true",
                        help="Drop the saliency-overlay panel (3-panel output)")
    parser.add_argument("--no-sigmoid", action="store_true", default=True,
                        help="Drop the ours_sigmoid panel (default: dropped)")
    parser.add_argument("--include-sigmoid", dest="no_sigmoid", action="store_false",
                        help="Include the ours_sigmoid panel (off by default)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output mp4 path (default: results/comparisons/<clip>_crf<N>.mp4)")
    parser.add_argument("--show", action="store_true",
                        help="Open the result when done")
    args = parser.parse_args()

    orig_path = DATA / f"{args.clip}.mp4"
    base_path = ENCODED / f"baseline_crf{args.crf}_{args.clip}.mp4"
    sig_path = ENCODED / f"ours_sigmoid_crf{args.crf}_{args.clip}.mp4"
    bgfg_path = ENCODED / f"ours_bgfg_crf{args.crf}_{args.clip}.mp4"

    for p in (orig_path, base_path, bgfg_path):
        if not p.exists():
            raise SystemExit(f"Missing input: {p}\n"
                             f"Did you run scripts/ablation_bgfg.py first?")

    args.out = args.out or (DEFAULT_OUT_DIR / f"{args.clip}_crf{args.crf}.mp4")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    sal_est = SaliencyEstimator(backend=args.saliency) if not args.no_saliency else None

    # For the overlay panel, build the same background + motion helper +
    # temporal smoother the codec uses, so the overlay reflects EXACTLY
    # what's driving the encoded output (not a noisy single-frame view).
    motion_helper = None
    background = None
    overlay_smoother = None
    if not args.no_saliency:
        background = compute_background_median(str(orig_path), n_samples=30)
        motion_helper = BgFgCodec(BgFgConfig(saliency_backend=args.saliency))
        overlay_smoother = TemporalSmoother(window=11)

    # Open every video
    caps = {"original": cv2.VideoCapture(str(orig_path)),
            "baseline": cv2.VideoCapture(str(base_path)),
            "bgfg":     cv2.VideoCapture(str(bgfg_path))}
    if not args.no_sigmoid and sig_path.exists():
        caps["sigmoid"] = cv2.VideoCapture(str(sig_path))

    # All inputs are at the source resolution + fps
    fps = caps["original"].get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(caps["original"].get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(caps["original"].get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Build the output panel order (preserve sensible left-to-right order)
    order = ["original"]
    if not args.no_saliency:
        order.append("saliency")
    order.append("baseline")
    if "sigmoid" in caps:
        order.append("sigmoid")
    order.append("bgfg")

    panel_w, panel_h = src_w, src_h
    out_w = panel_w * len(order)
    out_h = panel_h

    # Labels
    base_kb = kb(base_path); bgfg_kb = kb(bgfg_path); sig_kb = kb(sig_path)
    pct = lambda k: f"{(1 - k/base_kb)*100:+.0f}%" if base_kb > 0 else "?"
    labels = {
        "original":  ("original",                  f"{kb(orig_path):.0f} KB on disk"),
        "saliency":  ("saliency overlay",          f"{args.saliency} + motion"),
        "baseline":  (f"baseline H.265 (CRF {args.crf})", f"{base_kb:.0f} KB"),
        "sigmoid":   (f"ours_sigmoid (CRF {args.crf})",  f"{sig_kb:.0f} KB  ({pct(sig_kb)})"),
        "bgfg":      (f"ours_bgfg (CRF {args.crf})",     f"{bgfg_kb:.0f} KB  ({pct(bgfg_kb)})"),
    }

    # Pipe rawvideo to ffmpeg for H.265 encoding of the output
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{out_w}x{out_h}", "-r", f"{fps}",
        "-i", "-",
        "-c:v", "libx264",          # x264 for broadest player compatibility
        "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(args.out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdin is not None

    print(f"  building {len(order)}-panel side-by-side: {' | '.join(order)}")
    print(f"  output: {args.out}")

    frame_idx = 0
    try:
        while True:
            frames = {}
            for name, cap in caps.items():
                ok, f = cap.read()
                if not ok:
                    frames = None
                    break
                if (f.shape[1], f.shape[0]) != (panel_w, panel_h):
                    f = cv2.resize(f, (panel_w, panel_h))
                frames[name] = f
            if frames is None:
                break

            if not args.no_saliency:
                frames["saliency"] = saliency_overlay_bgr(
                    frames["original"], sal_est, motion_helper, background,
                    smoother=overlay_smoother,
                )

            # Compose row left-to-right
            row = []
            for name in order:
                title, sub = labels[name]
                row.append(label_frame(frames[name], title, sub))
            composed = np.hstack(row)
            proc.stdin.write(composed.tobytes())
            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"  ...{frame_idx} frames")
    finally:
        proc.stdin.close()
        proc.wait()
        for cap in caps.values():
            cap.release()

    print(f"\n  done — {frame_idx} frames, {kb(args.out):.0f} KB")
    print(f"  open with: open '{args.out}'")
    if args.show:
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", str(args.out)], check=False)
            elif sys.platform.startswith("linux"):
                subprocess.run(["xdg-open", str(args.out)], check=False)
        except Exception:
            pass


if __name__ == "__main__":
    main()

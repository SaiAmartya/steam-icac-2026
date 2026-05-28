"""
Side-by-side full-video comparison for a single clip.

Two output modes:

  * default      — single row, panels stacked horizontally:
                     original [ | saliency overlay ] | baseline H.265 | ours_bgfg

  * --internals  — 2-row grid (top 2 panels centered, bottom 3 panels full-width)
                   that tells the codec's whole story in a single frame:
                     Top: [black] | original | baseline H.265 | [black]   ← INPUT vs DUMB BASELINE
                     Bot: saliency mask | static background ref | ours_bgfg ← HOW WE DO BETTER
                   Use `--panel-scale 0.5` to keep the final mp4 at 1080p.

The bottom row is generated on the fly using the EXACT same code path as
`BgFgCodec` — the saliency overlay shows the mask the codec actually sees
per frame, and the background panel shows the temporal-median reference
image (computed once per clip from 30 evenly-spaced samples — it does not
change frame-to-frame; that constancy is exactly why H.265's inter-frame
prediction collapses the non-salient regions to near-zero bytes).

Usage:
    python scripts/compare_clip.py --clip clip_14 --crf 22
    python scripts/compare_clip.py --clip clip_04 --crf 28 --no-saliency
    python scripts/compare_clip.py --clip clip_02 --crf 22 --out /tmp/grab.mp4
    python scripts/compare_clip.py --clip clip_virat --crf 22 --internals --panel-scale 0.5
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
from src.bg_fg_codec import (
    compute_background_median,
    build_rolling_backgrounds,
    bg_for_frame,
    BgFgCodec,
    BgFgConfig,
)
from src.compress import _build_alpha

DATA = ROOT / "data/real"
ENCODED = ROOT / "results/ablation_bgfg/encoded"
DEFAULT_OUT_DIR = ROOT / "results/comparisons"


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------

def label_frame(frame_bgr: np.ndarray, text: str, sub: str = "") -> np.ndarray:
    """Stamp a label across the top of a frame."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    strip_h = max(28, h // 14)
    cv2.rectangle(out, (0, 0), (w, strip_h), (0, 0, 0), -1)
    cv2.putText(out, text, (10, int(strip_h * 0.68)),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.45, h / 700), (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(out, sub, (10, strip_h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.4, h / 900), (200, 200, 200), 1, cv2.LINE_AA)
    return out


def kb(p: Path) -> float:
    return p.stat().st_size / 1024 if p.exists() else 0.0


def _saliency_alpha(
    frame_bgr: np.ndarray,
    estimator: SaliencyEstimator,
    motion_helper: BgFgCodec,
    background: np.ndarray,
    smoother: TemporalSmoother,
) -> np.ndarray:
    """Compute the alpha mask BgFgCodec uses to decide keep-vs-replace per pixel.

    Mirrors the pipeline in `src/bg_fg_codec.py`:
        sal_raw  = max(yolo+spectral, motion-from-background)
        sal      = TemporalSmoother(window=11).smooth(sal_raw)
        alpha    = sigmoid_mask(sal, threshold=0.30, steepness=12.0)
        alpha    = max(alpha, mask_floor=0.25)

    Returns: HxW float32 in [0,1]. 1 = keep frame pixel, 0 = fall back to background.
    """
    sal = estimator.predict(frame_bgr)
    motion = motion_helper._motion_mask(frame_bgr, background)
    sal = np.maximum(sal, motion)
    sal = smoother.smooth(sal)
    alpha = _build_alpha(sal, mode="sigmoid", threshold=0.30, steepness=12.0)
    return np.maximum(alpha, 0.25)


def saliency_overlay_bgr(frame_bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Overlay the JET-colormapped mask on the original frame for visualisation."""
    alpha_u8 = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(alpha_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame_bgr, 0.55, heat, 0.45, 0.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", required=True, help="e.g. clip_14")
    parser.add_argument("--crf", type=int, default=22, choices=[22, 28, 34])
    parser.add_argument("--saliency", default="yolo+spectral",
                        choices=["spectral", "finegrained", "yolo", "yolo+spectral"])
    parser.add_argument("--no-saliency", action="store_true",
                        help="(single-row mode only) drop the saliency-overlay panel; "
                             "gives a clean 3-panel original | baseline | ours_bgfg.")
    parser.add_argument("--internals", action="store_true",
                        help="2-row grid exposing codec internals. "
                             "Top: [black] | original | baseline | [black] (2 panels centered). "
                             "Bot: saliency mask | static background | ours_bgfg (3 panels full-width). "
                             "Combine with --panel-scale 0.5 to keep the output at 1080p.")
    parser.add_argument("--panel-scale", type=float, default=1.0,
                        help="Per-panel downscale factor (default 1.0 = source resolution). "
                             "Recommended 0.5 when --internals is on so 1080p source -> 540p "
                             "panels -> a final 2880x1080 grid that plays smoothly on a Pi.")
    parser.add_argument("--bg-mode", default="static", choices=["static", "rolling"],
                        help="Background source for the saliency overlay + bottom-row "
                             "background panel. Match whatever the encoder ran with.")
    parser.add_argument("--bg-recal-interval", type=float, default=4.0,
                        help="Rolling-mode recalibration interval in seconds.")
    parser.add_argument("--bg-window", type=float, default=6.0,
                        help="Rolling-mode median window in seconds.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output mp4 path (default: results/comparisons/<clip>_crf<N>.mp4)")
    parser.add_argument("--show", action="store_true",
                        help="Open the result when done")
    args = parser.parse_args()

    if args.internals and args.no_saliency:
        raise SystemExit("--internals already shows a saliency panel; "
                         "--no-saliency is for single-row mode only.")
    if args.panel_scale <= 0 or args.panel_scale > 2.0:
        raise SystemExit("--panel-scale must be in (0, 2.0].")

    orig_path = DATA / f"{args.clip}.mp4"
    base_path = ENCODED / f"baseline_crf{args.crf}_{args.clip}.mp4"
    bgfg_path = ENCODED / f"ours_bgfg_crf{args.crf}_{args.clip}.mp4"
    for p in (orig_path, base_path, bgfg_path):
        if not p.exists():
            raise SystemExit(f"Missing input: {p}\n"
                             f"Did you run scripts/ablation_bgfg.py first?")

    args.out = args.out or (DEFAULT_OUT_DIR / f"{args.clip}_crf{args.crf}.mp4")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # --- Saliency / background setup ---------------------------------------
    need_sal_pipeline = args.internals or (not args.no_saliency)
    sal_est = None
    motion_helper = None
    bg_source = None     # either an ndarray (static) or a list-of-segments (rolling)
    overlay_smoother = None
    if need_sal_pipeline:
        sal_est = SaliencyEstimator(backend=args.saliency)
        if args.bg_mode == "rolling":
            print(f"  building rolling background schedule "
                  f"(recal every {args.bg_recal_interval:.1f}s, "
                  f"±{args.bg_window/2:.1f}s window)...")
            bg_source = build_rolling_backgrounds(
                str(orig_path),
                recalibration_interval_s=args.bg_recal_interval,
                rolling_window_s=args.bg_window,
            )
            print(f"  built {len(bg_source)} rolling backgrounds")
        else:
            print(f"  computing static background reference (30-sample temporal median)...")
            bg_source = compute_background_median(str(orig_path), n_samples=30)
        motion_helper = BgFgCodec(BgFgConfig(saliency_backend=args.saliency))
        overlay_smoother = TemporalSmoother(window=11)

    # --- Open source videos ------------------------------------------------
    caps = {"original": cv2.VideoCapture(str(orig_path)),
            "baseline": cv2.VideoCapture(str(base_path)),
            "bgfg":     cv2.VideoCapture(str(bgfg_path))}
    fps = caps["original"].get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(caps["original"].get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(caps["original"].get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- Decide panel layout ----------------------------------------------
    if args.internals:
        # Top: black | original | baseline | black (2 centered panels,
        # 0.5 panel-width borders on each side to match the 3-panel bottom row).
        # Bottom: saliency mask | static background | ours_bgfg (full width).
        top_order = ["original", "baseline"]
        bot_order = ["saliency_mask", "background", "bgfg"]
        n_cols = 3        # grid width is dictated by the bottom row
        n_rows = 2
    else:
        top_order = ["original"]
        if not args.no_saliency:
            top_order.append("saliency_mask")
        top_order += ["baseline", "bgfg"]
        bot_order = []
        n_cols = len(top_order)
        n_rows = 1

    panel_w = max(2, int(src_w * args.panel_scale)) & ~1   # force even
    panel_h = max(2, int(src_h * args.panel_scale)) & ~1
    out_w = panel_w * n_cols
    out_h = panel_h * n_rows

    # Black border for the centered top row (internals mode only). Width is
    # half a panel on each side so total top width = panel + 2*panel = 3*panel.
    border_w = panel_w // 2
    black_border = np.zeros((panel_h, border_w, 3), dtype=np.uint8) if args.internals else None

    # --- Labels (sizes in MB; % shows compression as a negative number) ---
    base_kb = kb(base_path); bgfg_kb = kb(bgfg_path)
    base_mb = base_kb / 1024
    bgfg_mb = bgfg_kb / 1024
    orig_mb = kb(orig_path) / 1024
    # k/base - 1 → negative when ours is smaller → "-81%" instead of "+81%"
    pct = lambda k: f"{(k/base_kb - 1)*100:+.0f}%" if base_kb > 0 else "?"
    if args.bg_mode == "rolling":
        bg_title = "rolling background reference"
        bg_sub = (f"recal every {args.bg_recal_interval:.0f}s "
                  f"({len(bg_source)} segments, ±{args.bg_window/2:.0f}s window)")
    else:
        bg_title = "static background reference"
        bg_sub = "temporal median of 30 samples (clip-wide)"
    labels = {
        "original":      ("original",                          f"{orig_mb:.1f} MB on disk"),
        "baseline":      (f"baseline H.265 (CRF {args.crf})",  f"{base_mb:.1f} MB"),
        "bgfg":          (f"ours_bgfg (CRF {args.crf})",       f"{bgfg_mb:.1f} MB  ({pct(bgfg_kb)})"),
        "saliency_mask": ("saliency mask",                     f"{args.saliency} + motion fusion"),
        "background":    (bg_title,                            bg_sub),
    }

    # Pre-render the background panel ONLY in static mode (one bg for the
    # whole clip). In rolling mode we re-label per frame so the bottom-right
    # background panel actually changes when a new segment begins.
    bg_panel_cached = None
    if args.internals and isinstance(bg_source, np.ndarray):
        bg_resized = cv2.resize(bg_source, (panel_w, panel_h)) \
            if (panel_w, panel_h) != bg_source.shape[1::-1] else bg_source
        bg_panel_cached = label_frame(bg_resized, *labels["background"])

    # --- ffmpeg sink -------------------------------------------------------
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

    layout_desc = (" | ".join(top_order)
                   + ("   //   " + " | ".join(bot_order) if bot_order else ""))
    print(f"  building {n_rows}x{n_cols} grid: {layout_desc}")
    print(f"  output dims: {out_w}x{out_h} @ {fps:.2f} fps")
    print(f"  output file: {args.out}")

    def panelize(name: str, raw: np.ndarray) -> np.ndarray:
        """Resize a raw BGR frame to (panel_w, panel_h) and stamp its label."""
        if (raw.shape[1], raw.shape[0]) != (panel_w, panel_h):
            raw = cv2.resize(raw, (panel_w, panel_h))
        title, sub = labels[name]
        return label_frame(raw, title, sub)

    frame_idx = 0
    try:
        while True:
            # Read the three source streams in lockstep.
            raws = {}
            done = False
            for name, cap in caps.items():
                ok, f = cap.read()
                if not ok:
                    done = True
                    break
                raws[name] = f
            if done:
                break

            # Build the codec-internals panels (if requested).
            # The per-frame background — single static image or current
            # rolling segment — drives both the saliency overlay's motion
            # mask AND the bottom-right background panel.
            current_bg = bg_for_frame(bg_source, frame_idx) if need_sal_pipeline else None

            if args.internals:
                orig = raws["original"]
                alpha = _saliency_alpha(orig, sal_est, motion_helper, current_bg, overlay_smoother)
                raws["saliency_mask"] = saliency_overlay_bgr(orig, alpha)
                # Rolling: the bg panel changes per segment, so re-label per frame.
                # Static: use the cached one.
                if bg_panel_cached is not None:
                    raws["__bg_panel_prerendered"] = bg_panel_cached
                else:
                    bg_resized = cv2.resize(current_bg, (panel_w, panel_h)) \
                        if (panel_w, panel_h) != current_bg.shape[1::-1] else current_bg
                    raws["background"] = bg_resized
            elif not args.no_saliency:
                orig = raws["original"]
                alpha = _saliency_alpha(orig, sal_est, motion_helper, current_bg, overlay_smoother)
                raws["saliency_mask"] = saliency_overlay_bgr(orig, alpha)

            # Compose the top row.
            top_panels = [panelize(n, raws[n]) for n in top_order]
            if args.internals:
                # Centre the 2 panels with half-panel black borders on each side
                # so the top row width matches the 3-panel bottom row exactly.
                top_row = np.hstack([black_border, *top_panels, black_border])
            else:
                top_row = np.hstack(top_panels)

            # Compose the bottom row (internals only).
            if bot_order:
                bot_panels = []
                for n in bot_order:
                    if n == "background" and "__bg_panel_prerendered" in raws:
                        bot_panels.append(raws["__bg_panel_prerendered"])
                    else:
                        bot_panels.append(panelize(n, raws[n]))
                bot_row = np.hstack(bot_panels)
                composed = np.vstack([top_row, bot_row])
            else:
                composed = top_row

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

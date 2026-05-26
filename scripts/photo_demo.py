"""
Webcam snap → saliency-aware vs baseline H.265 comparison.

Captures a short burst from the laptop camera, runs it through the project
pipeline (ours sigmoid vs uniform H.265 baseline at the same CRF), and renders
the 4-panel comparison PNG: original / saliency overlay / baseline / ours.

Usage:
    python scripts/photo_demo.py
    python scripts/photo_demo.py --duration 3 --crf 28 --camera 0 --show
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import PipelineConfig, run_pipeline  # noqa: E402
from src.bg_fg_codec import BgFgCodec, BgFgConfig  # noqa: E402
from src.compress import encode_uniform  # noqa: E402
from src.metrics import video_metrics, saliency_weighted_psnr  # noqa: E402
from src.saliency import SaliencyEstimator  # noqa: E402

OUT_DIR = ROOT / "results" / "photo_demo"


def countdown(seconds: int = 3) -> None:
    for i in range(seconds, 0, -1):
        print(f"  capturing in {i}...", end="\r", flush=True)
        time.sleep(1)
    print("  CAPTURING NOW          ")


def capture_burst(camera_index: int, duration_s: float, out_path: Path) -> tuple[int, int, float, int]:
    """Record `duration_s` seconds from the webcam to an mp4. Returns (w, h, fps, n_frames)."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {camera_index}")

    # Try to warm up auto-exposure / white balance with a few throwaway reads
    for _ in range(15):
        cap.read()

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_frames = int(round(fps * duration_s))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    print(f"  recording {duration_s:.1f}s @ {w}x{h} {fps:.1f}fps → {target_frames} frames")
    countdown(3)

    n = 0
    while n < target_frames:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        n += 1

    cap.release()
    writer.release()
    print(f"  wrote {n} frames → {out_path.name}")
    return w, h, fps, n


def get_middle_frame(path: Path) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 2))
    ok, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None


def saliency_overlay(rgb: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    sal_obj = cv2.saliency.StaticSaliencySpectralResidual_create()
    ok, sal = sal_obj.computeSaliency(bgr)
    if not ok:
        return rgb
    sal = (sal * 255).clip(0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(sal, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(bgr, 0.55, heat, 0.45, 0.0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def kb(path: Path) -> float:
    return path.stat().st_size / 1024 if path.exists() else 0.0


def render_panel(
    original_path: Path,
    baseline_path: Path,
    sigmoid_path: Path | None,
    bgfg_path: Path | None,
    crf: int,
    out_png: Path,
    subtitle_extra: str = "",
) -> None:
    orig = get_middle_frame(original_path)
    base = get_middle_frame(baseline_path)
    if orig is None or base is None:
        raise SystemExit("Could not read middle frame from one of the videos")

    sal = saliency_overlay(orig)
    base_kb = kb(baseline_path)

    panels: list[tuple[np.ndarray, str]] = [
        (orig, "original (your webcam)"),
        (sal, "saliency overlay"),
        (base, f"baseline H.265 (CRF {crf})\n{base_kb:.0f} KB"),
    ]
    if sigmoid_path is not None and sigmoid_path.exists():
        s = get_middle_frame(sigmoid_path)
        if s is not None:
            sk = kb(sigmoid_path)
            pct = (1 - sk / base_kb) * 100 if base_kb > 0 else 0
            panels.append((s, f"ours_sigmoid (CRF {crf})\n{sk:.0f} KB  ({pct:+.0f}%)"))
    if bgfg_path is not None and bgfg_path.exists():
        b = get_middle_frame(bgfg_path)
        if b is not None:
            bk = kb(bgfg_path)
            pct = (1 - bk / base_kb) * 100 if base_kb > 0 else 0
            panels.append((b, f"ours_bgfg (CRF {crf})\n{bk:.0f} KB  ({pct:+.0f}%)"))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(n * 3.85, 3.7))
    if n == 1:
        axes = [axes]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    sub = f"Webcam snap — middle frame, CRF {crf}"
    if subtitle_extra:
        sub += "   |   " + subtitle_extra
    fig.suptitle(sub, fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def open_file(path: Path) -> None:
    """Best-effort: open the file with the OS default viewer."""
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        elif sys.platform.startswith("linux"):
            subprocess.run(["xdg-open", str(path)], check=False)
        elif sys.platform.startswith("win"):
            subprocess.run(["start", "", str(path)], check=False, shell=True)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--duration", type=float, default=2.0, help="Capture seconds (default 2.0)")
    parser.add_argument("--crf", type=int, default=28, help="CRF for both encoders (default 28)")
    parser.add_argument("--codec", choices=["sigmoid", "bgfg", "both"], default="both",
                        help="Which 'ours' encoder(s) to compare against baseline")
    parser.add_argument("--saliency", default="yolo+spectral",
                        choices=["spectral", "finegrained", "yolo", "yolo+spectral"],
                        help="Saliency backend (yolo+spectral recommended for surveillance)")
    parser.add_argument("--gate", type=float, default=0.0,
                        help="Gate threshold for sigmoid pipeline; 0 = always-on")
    parser.add_argument("--blur", type=int, default=21)
    parser.add_argument("--keep-tmp", action="store_true", help="Keep the captured source mp4")
    parser.add_argument("--show", action="store_true", help="Open the result PNG when done")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="photo_demo_"))
    src_path = tmp_dir / "capture.mp4"

    print("=== STEAM IC webcam → compression demo ===")
    capture_burst(args.camera, args.duration, src_path)

    base_path = OUT_DIR / "baseline_uniform.mp4"
    sig_path = OUT_DIR / "ours_saliency.mp4"
    bgfg_path = OUT_DIR / "ours_bgfg.mp4"
    out_png = OUT_DIR / "comparison.png"

    # Always make the baseline (uniform H.265 at same CRF)
    print("\n  encoding baseline H.265...")
    encode_uniform(str(src_path), str(base_path), crf=args.crf)

    sigmoid_metrics = None
    bgfg_metrics = None

    if args.codec in ("sigmoid", "both"):
        print("  running sigmoid pipeline...")
        cfg = PipelineConfig(
            gate_threshold=args.gate,
            saliency_backend=args.saliency,
            crf=args.crf,
            baseline_crf=args.crf,
            blur_strength=args.blur,
        )
        sigmoid_metrics = run_pipeline(str(src_path), str(OUT_DIR), cfg=cfg)
        # run_pipeline writes ours_saliency.mp4 + baseline_uniform.mp4 — we keep our baseline.

    if args.codec in ("bgfg", "both"):
        print("  running bg/fg codec...")
        cfg = BgFgConfig(
            saliency_backend=args.saliency,
            crf=args.crf,
            mask_mode="sigmoid",
            mask_threshold=0.30,
            mask_steepness=12.0,
            smooth_window=7,
        )
        bgfg_stats = BgFgCodec(cfg).encode(str(src_path), str(bgfg_path))
        bgfg_metrics = {"stream_bytes": bgfg_stats["bytes"]}

    # Compute sal-PSNR for the honest comparison
    sal_est = SaliencyEstimator(backend=args.saliency)
    extras = []
    base_psnr_full = video_metrics(str(src_path), str(base_path), every=5).get("psnr_mean")
    if base_psnr_full is not None:
        extras.append(f"baseline PSNR {base_psnr_full:.1f} dB")
    if args.codec in ("bgfg", "both") and bgfg_path.exists():
        bgfg_psnr_full = video_metrics(str(src_path), str(bgfg_path), every=5).get("psnr_mean")
        extras.append(f"bgfg PSNR {bgfg_psnr_full:.1f} dB")
    subtitle_extra = " · ".join(extras)

    print("\n  rendering comparison panel...")
    render_panel(
        src_path, base_path,
        sig_path if args.codec in ("sigmoid", "both") else None,
        bgfg_path if args.codec in ("bgfg", "both") else None,
        args.crf, out_png, subtitle_extra=subtitle_extra,
    )

    if args.keep_tmp:
        shutil.copy(src_path, OUT_DIR / "capture.mp4")

    # Summary
    print("\n=== RESULT ===")
    print(f"  original (capture) : {kb(src_path):7.1f} KB")
    print(f"  baseline H.265     : {kb(base_path):7.1f} KB")
    if args.codec in ("sigmoid", "both") and sig_path.exists():
        sk = kb(sig_path); pct = (1 - sk/kb(base_path))*100 if kb(base_path) > 0 else 0
        print(f"  ours_sigmoid       : {sk:7.1f} KB   ({pct:+.0f}% vs baseline)")
    if args.codec in ("bgfg", "both") and bgfg_path.exists():
        bk = kb(bgfg_path); pct = (1 - bk/kb(base_path))*100 if kb(base_path) > 0 else 0
        print(f"  ours_bgfg          : {bk:7.1f} KB   ({pct:+.0f}% vs baseline)")
    print(f"\n  comparison         : {out_png}")

    if args.show:
        open_file(out_png)


if __name__ == "__main__":
    main()

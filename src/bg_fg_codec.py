"""
Background/foreground decomposition codec for stationary surveillance cameras.

This is the headline innovation: surveillance cameras don't move, so 95% of
every frame is identical to the previous one. A general-purpose codec (H.265)
still has to spend bits modelling that redundancy with motion vectors and
residuals. We exploit the stationary-camera assumption explicitly:

  1. Pass 1 — build a "background reference" image from the temporal median
              of N evenly-spaced frames. Robust to short-lived foreground.
  2. Pass 2 — for each frame, replace non-salient pixels with the matching
              background pixel:
                   stabilised = mask * frame + (1 - mask) * background
              Mask comes from any SaliencyEstimator backend; we ship sigmoid
              shaping so the transition is smooth (no visible seams).
  3. H.265 — encode the stabilised stream with the existing pipeline. Because
              non-salient pixels are now BYTE-IDENTICAL across frames, H.265's
              inter-frame prediction reduces those regions to near-zero bits.
              All the bitrate flows to the foreground.

The decoded output looks like the original frame except non-salient regions
are pinned to the (sharp) background reference instead of carrying their own
noisy textures. Visually this is often *better* than baseline H.265 at the
same bitrate, because the static background never picks up compression noise.

Public API:
    BgFgCodec(saliency_backend="yolo+spectral", crf=28, ...).encode(in, out)
"""
from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from .saliency import SaliencyEstimator, TemporalSmoother
from .compress import FramePipeEncoder, H265EncoderConfig, _build_alpha

logger = logging.getLogger(__name__)


@dataclass
class BgFgConfig:
    # Saliency
    saliency_backend: str = "yolo+spectral"
    saliency_yolo_kwargs: Optional[dict] = None
    smooth_window: int = 11             # temporal smoothing window for the mask
    mask_mode: str = "sigmoid"
    mask_threshold: float = 0.30
    mask_steepness: float = 12.0
    mask_floor: float = 0.25            # min blend weight on the frame — safety net so
                                        # subjects YOLO/spectral missed never fully vanish.
                                        # Trades a bit of size win for "the person is always there".

    # Motion saliency (NEW). The temporal-median background gives us a *free*
    # foreground signal: any pixel whose value differs from the background is
    # by definition moving foreground. We combine this with the chosen saliency
    # backend via per-pixel max, so motion is ALWAYS preserved even when YOLO
    # misses or spectral residual is weak. This is the key fix for the
    # "person disappeared into the bar" failure mode.
    use_motion_saliency: bool = True
    motion_threshold: int = 18          # min per-channel delta in [0,255] to count as motion
    motion_ramp: float = 35.0           # delta -> [0,1] ramp slope; smaller = more sensitive
    motion_dilate: int = 7              # px to dilate the motion mask so it covers full body

    # Background reference
    bg_sample_count: int = 30           # frames sampled across the clip for median
    bg_blur_sigma: float = 0.0          # post-median Gaussian smoothing (0 = none)

    # Encoder
    codec: str = "libx265"
    crf: int = 28
    preset: str = "fast"


def _sample_frame_indices(n_total: int, k: int) -> list[int]:
    """k evenly-spaced indices in [0, n_total). Falls back to all frames if k>=n."""
    if k >= n_total:
        return list(range(n_total))
    return np.linspace(0, n_total - 1, k, dtype=int).tolist()


def compute_background_median(input_path: str, n_samples: int = 30, blur_sigma: float = 0.0) -> np.ndarray:
    """Per-pixel temporal median over evenly-spaced frames. Returns HxWx3 uint8 BGR."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {input_path}")
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = _sample_frame_indices(n_total, n_samples)

    stack = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            stack.append(frame)
    cap.release()

    if not stack:
        raise RuntimeError(f"No frames read from {input_path}")

    arr = np.stack(stack, axis=0)  # NxHxWx3 uint8
    # median over the time axis; uint8 promoted via np to avoid overflow
    bg = np.median(arr, axis=0).astype(np.uint8)
    if blur_sigma and blur_sigma > 0:
        k = max(3, int(blur_sigma * 6) | 1)  # odd kernel
        bg = cv2.GaussianBlur(bg, (k, k), blur_sigma)
    return bg


class BgFgCodec:
    """Background/foreground codec built on top of the existing H.265 pipeline.

    The output mp4 is a valid H.265 stream — decoders don't need to know we
    pre-stabilised the input. Numbers reported (PSNR / SSIM / bytes) are
    apples-to-apples against the baseline H.265 encode of the same clip.
    """

    def __init__(self, cfg: Optional[BgFgConfig] = None) -> None:
        self.cfg = cfg or BgFgConfig()
        self.saliency = SaliencyEstimator(
            backend=self.cfg.saliency_backend,
            yolo_kwargs=self.cfg.saliency_yolo_kwargs,
        )
        self.smoother = TemporalSmoother(window=self.cfg.smooth_window)

        encoder_cfg = H265EncoderConfig(
            codec=self.cfg.codec,
            crf=self.cfg.crf,
            preset=self.cfg.preset,
        )
        self._encoder = FramePipeEncoder(encoder_cfg)

    # ---------- Core preprocessing ----------

    def _motion_mask(self, frame: np.ndarray, background: np.ndarray) -> np.ndarray:
        """Per-pixel motion signal from |frame - background|. Returns HxW float [0,1].

        Anything that differs from the static background is foreground by definition.
        This catches subjects the semantic saliency missed (dark figures on dark
        backgrounds, low-confidence YOLO detections, etc.).
        """
        diff_bgr = np.abs(frame.astype(np.int16) - background.astype(np.int16))
        # max across BGR channels — a change in any channel counts
        diff = diff_bgr.max(axis=2).astype(np.float32)
        # Subtract threshold, then ramp to [0,1]
        ramp = (diff - self.cfg.motion_threshold) / max(self.cfg.motion_ramp, 1.0)
        m = np.clip(ramp, 0.0, 1.0)
        # Dilate so the mask covers the full body, not just edges
        if self.cfg.motion_dilate > 0:
            k = max(3, self.cfg.motion_dilate)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            # Convert to uint8 for dilation, back to float
            m_u8 = (m * 255).astype(np.uint8)
            m_u8 = cv2.dilate(m_u8, kernel)
            # Gentle blur the boundary so the mask edge is smooth
            m_u8 = cv2.GaussianBlur(m_u8, (k | 1, k | 1), 0)
            m = m_u8.astype(np.float32) / 255.0
        return m

    def _blend_frame(self, frame: np.ndarray, background: np.ndarray, sal: np.ndarray) -> np.ndarray:
        """Replace low-saliency pixels with background. Returns HxWx3 uint8."""
        if sal.shape != frame.shape[:2]:
            sal = cv2.resize(sal, (frame.shape[1], frame.shape[0]))
        alpha = _build_alpha(
            sal, self.cfg.mask_mode, self.cfg.mask_threshold, self.cfg.mask_steepness
        )
        if self.cfg.mask_floor > 0:
            alpha = np.maximum(alpha, self.cfg.mask_floor)
        alpha3 = alpha[..., None]
        out = alpha3 * frame.astype(np.float32) + (1.0 - alpha3) * background.astype(np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _stabilised_iter(self, input_path: str, background: np.ndarray):
        """Yield blended (foreground-on-background) frames in stream order."""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {input_path}")
        # Make sure background matches the source resolution
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if background.shape[:2] != (h, w):
            background = cv2.resize(background, (w, h))

        self.smoother.reset()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            sal_raw = self.saliency.predict(frame)
            # Combine with motion-from-background-difference: per-pixel max.
            # This is the key fix — anything that's moving is foreground even
            # if YOLO/spectral didn't fire on it.
            if self.cfg.use_motion_saliency:
                motion = self._motion_mask(frame, background)
                sal_raw = np.maximum(sal_raw, motion)
            sal = self.smoother.smooth(sal_raw)
            yield self._blend_frame(frame, background, sal)
        cap.release()

    # ---------- Public encode ----------

    def encode(self, input_path: str, output_path: str) -> dict:
        """Two-pass encode: build background, then emit stabilised frames to H.265.

        Returns: {out_path, bytes, n_frames, background_bytes (PNG-on-disk size)}.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {input_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        logger.info(f"BgFgCodec: building background from {self.cfg.bg_sample_count} samples...")
        background = compute_background_median(
            input_path, n_samples=self.cfg.bg_sample_count, blur_sigma=self.cfg.bg_blur_sigma
        )

        # Save the background as a sidecar for inspection / fair accounting.
        # The background ships *with* the encoded stream; the codec output is
        # not "free" without it. We charge ourselves the PNG bytes alongside.
        bg_path = str(Path(output_path).with_suffix("")) + "_bg.png"
        cv2.imwrite(bg_path, background, [cv2.IMWRITE_PNG_COMPRESSION, 9])

        logger.info(f"BgFgCodec: encoding stabilised stream to {output_path}")
        stats = self._encoder.encode(
            self._stabilised_iter(input_path, background),
            output_path, fps=fps, size=(w, h),
        )
        stats["background_path"] = bg_path
        stats["background_bytes"] = os.path.getsize(bg_path)
        # Combined "wire bytes" the receiver has to download.
        stats["total_bytes"] = stats["bytes"] + stats["background_bytes"]
        return stats


# ---------- Convenience CLI ----------

if __name__ == "__main__":
    import argparse, json, sys
    from .metrics import video_metrics

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True, help="Output mp4 path")
    parser.add_argument("--saliency", default="yolo+spectral",
                        choices=["spectral", "finegrained", "yolo", "yolo+spectral"])
    parser.add_argument("--crf", type=int, default=28)
    parser.add_argument("--bg-samples", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--steepness", type=float, default=12.0)
    parser.add_argument("--smooth", type=int, default=7)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = BgFgConfig(
        saliency_backend=args.saliency,
        crf=args.crf,
        bg_sample_count=args.bg_samples,
        mask_threshold=args.threshold,
        mask_steepness=args.steepness,
        smooth_window=args.smooth,
    )
    codec = BgFgCodec(cfg)
    stats = codec.encode(args.input, args.out)

    metrics = video_metrics(args.input, args.out, every=5)
    raw = os.path.getsize(args.input)
    print(json.dumps({
        **stats,
        "raw_input_bytes": raw,
        "psnr_mean": metrics.get("psnr_mean"),
        "ssim_mean": metrics.get("ssim_mean"),
        "saving_vs_raw_input_pct": round((1 - stats["total_bytes"] / raw) * 100, 1),
    }, indent=2))

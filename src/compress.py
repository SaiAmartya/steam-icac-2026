"""
Saliency-aware perceptual compression.

Implements three tiers of increasing fidelity (per PLAN.md §5.2):
  Tier C (default, Day 1): pre-blur non-salient regions using a saliency mask,
                            then encode uniformly with ffmpeg / libx265.
  Tier B (Day 3): per-frame average-QP modulation via ffmpeg qp-file.
  Tier A (stretch): real per-block QP via libx265 ROI.

Public API:
  SaliencyCompressor(tier="C", crf=28, codec="libx265", blur_strength=21, qp_delta=6)
  .process_frame(frame_bgr, sal_map) -> np.ndarray
  .encode(frames_iter, out_path, fps, size) -> dict   # returns stats
  encode_uniform(input_path, out_path, crf, codec)     # baseline encode for comparison
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np


@dataclass
class CompressorConfig:
    tier: str = "C"
    codec: str = "libx265"        # libx265 or libx264
    crf: int = 28                 # baseline quality (higher = smaller)
    blur_strength: int = 21       # Gaussian kernel size for Tier C
    qp_delta: int = 6             # for Tier B/A only
    preset: str = "fast"
    # Saliency-mask shaping (added 2026-04-30 after empirical evidence that the
    # default soft alpha-blend was destroying detail across the whole frame):
    #   "alpha"   — original soft alpha = saliency, blends gradually (legacy)
    #   "sigmoid" — steep sigmoid around mask_threshold; smooth at boundary,
    #               but salient interior is fully sharp; non-salient is fully blurred
    #   "binary"  — hard threshold (sharp where sal > threshold, else blurred)
    mask_mode: str = "alpha"
    mask_threshold: float = 0.4
    mask_steepness: float = 12.0  # only used by 'sigmoid'


def _build_alpha(sal: np.ndarray, mode: str, threshold: float, steepness: float) -> np.ndarray:
    """Convert a continuous saliency map into a per-pixel sharp/blur weight.
    Output: HxW float32 in [0,1]. 1 = fully sharp, 0 = fully blurred."""
    if mode == "alpha":
        return np.clip(sal, 0.0, 1.0).astype(np.float32)
    if mode == "binary":
        return (sal >= threshold).astype(np.float32)
    if mode == "sigmoid":
        # Sharp logistic transition centred at threshold. Saturates near 0/1
        # so salient interior is fully sharp and non-salient interior fully blurred,
        # but the boundary is smooth (no hard edges that would create artefacts).
        return 1.0 / (1.0 + np.exp(-steepness * (sal - threshold))).astype(np.float32)
    raise ValueError(f"Unknown mask_mode: {mode!r}")


def _tier_c_blend(frame_bgr: np.ndarray, sal: np.ndarray, blur_k: int,
                  mask_mode: str = "alpha", mask_threshold: float = 0.4,
                  mask_steepness: float = 12.0) -> np.ndarray:
    """Blend blurred and sharp frame using a saliency-derived alpha mask.

    sal: HxW float32 in [0,1].  Higher saliency -> more of the sharp frame.
    """
    if blur_k % 2 == 0:
        blur_k += 1
    blurred = cv2.GaussianBlur(frame_bgr, (blur_k, blur_k), 0)
    if sal.shape != frame_bgr.shape[:2]:
        sal = cv2.resize(sal, (frame_bgr.shape[1], frame_bgr.shape[0]))
    alpha = _build_alpha(sal, mask_mode, mask_threshold, mask_steepness)[..., None]
    out = alpha * frame_bgr.astype(np.float32) + (1.0 - alpha) * blurred.astype(np.float32)
    return out.astype(np.uint8)


class SaliencyCompressor:
    """Saliency-aware compression using pre-blur + uniform-QP encode (Tier C)."""

    def __init__(self, cfg: Optional[CompressorConfig] = None) -> None:
        self.cfg = cfg or CompressorConfig()
        if self.cfg.tier != "C":
            raise NotImplementedError(
                f"Tier {self.cfg.tier!r} not implemented yet (Day 3 work). Use 'C'."
            )

    def process_frame(self, frame_bgr: np.ndarray, sal: np.ndarray) -> np.ndarray:
        return _tier_c_blend(
            frame_bgr, sal, blur_k=self.cfg.blur_strength,
            mask_mode=self.cfg.mask_mode,
            mask_threshold=self.cfg.mask_threshold,
            mask_steepness=self.cfg.mask_steepness,
        )

    def encode(
        self,
        frames_iter: Iterable[np.ndarray],
        out_path: str,
        fps: float,
        size: tuple[int, int],
    ) -> dict:
        """Encode a stream of pre-processed BGR frames to out_path via ffmpeg.

        Pipes rawvideo to ffmpeg using libx265/libx264 at the configured CRF.
        """
        w, h = size
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}",
            "-r", f"{fps}",
            "-i", "-",
            "-c:v", self.cfg.codec,
            "-preset", self.cfg.preset,
            "-crf", f"{self.cfg.crf}",
            "-pix_fmt", "yuv420p",
            out_path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        assert proc.stdin is not None

        n = 0
        try:
            for f in frames_iter:
                if f.shape[1] != w or f.shape[0] != h:
                    f = cv2.resize(f, (w, h))
                proc.stdin.write(f.tobytes())
                n += 1
        finally:
            proc.stdin.close()
            proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed with code {proc.returncode}")

        return {
            "out_path": out_path,
            "n_frames": n,
            "bytes": os.path.getsize(out_path) if os.path.exists(out_path) else 0,
        }


def encode_uniform(
    input_path: str,
    out_path: str,
    crf: int = 28,
    codec: str = "libx265",
    preset: str = "fast",
) -> dict:
    """Baseline: re-encode an existing video uniformly, no saliency awareness."""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", codec, "-preset", preset, "-crf", f"{crf}",
        "-pix_fmt", "yuv420p", out_path,
    ]
    subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
    return {"out_path": out_path, "bytes": os.path.getsize(out_path)}


if __name__ == "__main__":
    # Smoke test: make 30 noisy frames, fake a saliency map, run Tier C compression to /tmp.
    import itertools

    rng = np.random.default_rng(0)
    frames = [(rng.random((240, 320, 3)) * 255).astype(np.uint8) for _ in range(30)]
    # fake saliency: center-weighted gaussian
    yy, xx = np.mgrid[0:240, 0:320]
    cy, cx = 120, 160
    sal = np.exp(-((yy - cy) ** 2 / (2 * 60 ** 2) + (xx - cx) ** 2 / (2 * 60 ** 2))).astype(np.float32)

    comp = SaliencyCompressor()
    processed = (comp.process_frame(f, sal) for f in frames)
    out = "/tmp/compress_smoke.mp4"
    stats = comp.encode(processed, out, fps=10.0, size=(320, 240))
    print("compress.py smoke:", stats)

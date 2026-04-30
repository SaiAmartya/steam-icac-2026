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


def _tier_c_blend(frame_bgr: np.ndarray, sal: np.ndarray, blur_k: int) -> np.ndarray:
    """Blend blurred and sharp frame using saliency as alpha.

    sal: HxW float32 in [0,1].  Higher saliency -> more of the sharp frame.
    """
    if blur_k % 2 == 0:
        blur_k += 1
    blurred = cv2.GaussianBlur(frame_bgr, (blur_k, blur_k), 0)
    # 3-channel alpha
    if sal.shape != frame_bgr.shape[:2]:
        sal = cv2.resize(sal, (frame_bgr.shape[1], frame_bgr.shape[0]))
    alpha = np.clip(sal, 0.0, 1.0).astype(np.float32)[..., None]
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
        return _tier_c_blend(frame_bgr, sal, blur_k=self.cfg.blur_strength)

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

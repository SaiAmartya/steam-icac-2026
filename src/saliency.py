"""
Per-frame saliency estimation.

Two backends:
  - "spectral" : OpenCV StaticSaliencySpectralResidual (Hou & Zhang 2007).
                 Classical, no training, runs at 100+ fps on a laptop CPU.
                 Perfect for a Day-1 prototype.
  - "model"    : hook for a pretrained CNN (TASED-Net / MobileNet-based).
                 Stub for now; raises NotImplementedError until Day 2-3.

Public API:
  SaliencyEstimator(backend="spectral")
  .predict(frame_bgr: np.ndarray) -> np.ndarray          # HxW float32 in [0,1]
  .predict_batch(frames) -> np.ndarray                    # NxHxW float32

Temporal smoothing (optional) is provided by TemporalSmoother — used by the
pipeline to avoid flickering QP maps between adjacent frames.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SaliencyEstimator:
    def __init__(self, backend: str = "spectral") -> None:
        self.backend = backend
        if backend == "spectral":
            try:
                self._impl = cv2.saliency.StaticSaliencySpectralResidual_create()
            except AttributeError as e:
                raise RuntimeError(
                    "cv2.saliency not available. Install opencv-contrib-python."
                ) from e
        elif backend == "finegrained":
            self._impl = cv2.saliency.StaticSaliencyFineGrained_create()
        elif backend == "model":
            # Day 2-3 hook: load TASED-Net or a MobileNet-based saliency CNN
            raise NotImplementedError(
                "Model-based saliency backend is not wired up yet. "
                "Use backend='spectral' for now."
            )
        else:
            raise ValueError(f"Unknown backend: {backend!r}")

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return HxW saliency map in [0,1] float32. Matches frame's HxW."""
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 BGR frame, got shape {frame_bgr.shape}")

        success, sal = self._impl.computeSaliency(frame_bgr)
        if not success:
            # Fallback: flat map, treat everything equally
            logger.warning("Saliency compute failed; returning flat map.")
            return np.full(frame_bgr.shape[:2], 0.5, dtype=np.float32)

        sal = sal.astype(np.float32)
        # Normalise per-frame to [0,1]
        mn, mx = float(sal.min()), float(sal.max())
        if mx - mn < 1e-8:
            return np.full_like(sal, 0.5, dtype=np.float32)
        return (sal - mn) / (mx - mn)

    def predict_batch(self, frames: np.ndarray) -> np.ndarray:
        return np.stack([self.predict(f) for f in frames], axis=0)


class TemporalSmoother:
    """Moving-average smoother for saliency maps, to suppress flicker."""

    def __init__(self, window: int = 5) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self._buf: Deque[np.ndarray] = deque(maxlen=window)

    def reset(self) -> None:
        self._buf.clear()

    def smooth(self, sal: np.ndarray) -> np.ndarray:
        self._buf.append(sal.astype(np.float32))
        return np.mean(np.stack(list(self._buf), axis=0), axis=0)


def saliency_to_qp_map(
    sal: np.ndarray,
    qp_baseline: int = 28,
    qp_delta: int = 6,
    block_size: int = 16,
) -> np.ndarray:
    """Convert saliency map (HxW, [0,1]) to a per-block QP map suitable for ROI encoding.

    Low saliency -> higher QP (coarser); high saliency -> lower QP (crisper).
    Returns an int32 grid of shape (H//block_size, W//block_size).
    """
    h, w = sal.shape
    bh, bw = h // block_size, w // block_size
    sal_small = cv2.resize(sal, (bw, bh), interpolation=cv2.INTER_AREA)
    qp = qp_baseline - qp_delta * sal_small  # high saliency -> low QP
    return np.clip(np.round(qp), 18, 40).astype(np.int32)


if __name__ == "__main__":
    # Smoke test
    logging.basicConfig(level=logging.INFO)
    est = SaliencyEstimator(backend="spectral")
    smoother = TemporalSmoother(window=5)

    for i in range(5):
        frame = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
        sal = est.predict(frame)
        sal_s = smoother.smooth(sal)
        qp = saliency_to_qp_map(sal_s)
        print(f"frame {i}: sal range=[{sal.min():.3f}, {sal.max():.3f}] "
              f"qp range=[{qp.min()}, {qp.max()}] qp_shape={qp.shape}")
    print("saliency.py smoke test passed.")

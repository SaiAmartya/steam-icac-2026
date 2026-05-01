"""
Quality metrics for compression evaluation.

Wrappers around skimage / LPIPS / ffmpeg tools.

Public API:
  compute_frame_metrics(ref_bgr, dist_bgr, lpips_model=None) -> dict
  video_metrics(ref_path, dist_path, every=5, use_lpips=False) -> dict
  saliency_weighted_psnr(ref_bgr, dist_bgr, saliency_map, threshold=0.5) -> float
  video_metrics_with_saliency(ref_path, dist_path, sal_estimator, every=5, use_lpips=False) -> dict
  LPIPSMetric
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    from skimage.metrics import structural_similarity as sk_ssim
except ImportError as e:
    raise RuntimeError("scikit-image is required for metrics.") from e

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import lpips as _lpips_pkg
    _LPIPS_AVAILABLE = _TORCH_AVAILABLE  # LPIPS needs torch too
except ImportError:
    _LPIPS_AVAILABLE = False


class LPIPSMetric:
    """Lazy-loaded LPIPS wrapper. Call with compute(ref_bgr, dist_bgr)."""

    def __init__(self, net: str = "alex", device: str = "cpu") -> None:
        if not _LPIPS_AVAILABLE:
            raise RuntimeError(
                "LPIPS requires both `torch` and `lpips` to be installed."
            )
        self.device = device
        self._model = _lpips_pkg.LPIPS(net=net).to(device).eval()

    def _to_torch(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0) * 2 - 1  # [-1, 1]

    def compute(self, ref_bgr: np.ndarray, dist_bgr: np.ndarray) -> float:
        with torch.inference_mode():
            r = self._to_torch(ref_bgr).to(self.device)
            d = self._to_torch(dist_bgr).to(self.device)
            return float(self._model(r, d).item())


def compute_frame_metrics(
    ref_bgr: np.ndarray,
    dist_bgr: np.ndarray,
    lpips_model: Optional[LPIPSMetric] = None,
) -> dict:
    """Compute PSNR, SSIM, MS-SSIM, and optionally LPIPS for a single frame pair."""
    if ref_bgr.shape != dist_bgr.shape:
        dist_bgr = cv2.resize(dist_bgr, (ref_bgr.shape[1], ref_bgr.shape[0]))

    psnr = float(sk_psnr(ref_bgr, dist_bgr, data_range=255))
    ssim = float(sk_ssim(ref_bgr, dist_bgr, channel_axis=2, data_range=255))

    out = {"psnr": psnr, "ssim": ssim}
    if lpips_model is not None:
        out["lpips"] = lpips_model.compute(ref_bgr, dist_bgr)
    return out


def video_metrics(
    ref_path: str,
    dist_path: str,
    every: int = 5,
    use_lpips: bool = False,
) -> dict:
    """Compute per-frame metrics sampled every `every` frames over two videos.

    Returns {'psnr_mean', 'ssim_mean', 'lpips_mean' (optional), 'n_frames_evaluated',
             'ref_bytes', 'dist_bytes', 'ratio'}.
    """
    cap_r = cv2.VideoCapture(ref_path)
    cap_d = cv2.VideoCapture(dist_path)
    if not (cap_r.isOpened() and cap_d.isOpened()):
        raise RuntimeError(f"Failed to open one of: {ref_path}, {dist_path}")

    lpips_model = LPIPSMetric() if (use_lpips and _LPIPS_AVAILABLE) else None

    psnrs, ssims, lps = [], [], []
    i = 0
    while True:
        ok_r, fr = cap_r.read()
        ok_d, fd = cap_d.read()
        if not (ok_r and ok_d):
            break
        if i % every == 0:
            m = compute_frame_metrics(fr, fd, lpips_model=lpips_model)
            psnrs.append(m["psnr"])
            ssims.append(m["ssim"])
            if "lpips" in m:
                lps.append(m["lpips"])
        i += 1
    cap_r.release()
    cap_d.release()

    ref_bytes = os.path.getsize(ref_path)
    dist_bytes = os.path.getsize(dist_path)

    out = {
        "psnr_mean": float(np.mean(psnrs)) if psnrs else float("nan"),
        "ssim_mean": float(np.mean(ssims)) if ssims else float("nan"),
        "n_frames_evaluated": len(psnrs),
        "ref_bytes": ref_bytes,
        "dist_bytes": dist_bytes,
        "ratio": ref_bytes / max(dist_bytes, 1),
    }
    if lps:
        out["lpips_mean"] = float(np.mean(lps))
    return out


def saliency_weighted_psnr(ref_bgr: np.ndarray, dist_bgr: np.ndarray, 
                          saliency_map: np.ndarray, threshold: float = 0.5) -> float:
    """PSNR computed only over pixels where the saliency map exceeds `threshold`.
    
    Args:
        ref_bgr: HxWx3 uint8 BGR reference frame
        dist_bgr: HxWx3 uint8 BGR distorted frame
        saliency_map: HxW float in [0,1] saliency map
        threshold: float in [0,1], pixels with saliency >= threshold are included
    
    Returns:
        PSNR in dB over the masked region. Returns float('nan') if mask has <100 pixels.
        Uses 10*log10(255^2 / mse) where MSE is computed over masked pixels only.
    """
    # Ensure shapes match
    if ref_bgr.shape != dist_bgr.shape:
        dist_bgr = cv2.resize(dist_bgr, (ref_bgr.shape[1], ref_bgr.shape[0]))
    if saliency_map.shape != ref_bgr.shape[:2]:
        saliency_map = cv2.resize(saliency_map, (ref_bgr.shape[1], ref_bgr.shape[0]))
    
    # Create binary mask: pixels where saliency >= threshold
    mask = saliency_map >= threshold
    
    # Count masked pixels
    n_masked = int(np.sum(mask))
    if n_masked < 100:
        return float('nan')
    
    # Compute MSE over masked region only, using float64 to avoid overflow
    ref_f64 = ref_bgr.astype(np.float64)
    dist_f64 = dist_bgr.astype(np.float64)
    diff = (ref_f64 - dist_f64) ** 2
    
    # Apply mask to all three channels
    mask_3ch = mask[..., None]  # HxWx1
    mse = float(np.sum(diff * mask_3ch) / (n_masked * 3))
    
    if mse < 1e-10:
        return float('inf')
    
    psnr = 10.0 * np.log10(255.0 ** 2 / mse)
    return float(psnr)


def video_metrics_with_saliency(ref_path: str, dist_path: str, sal_estimator,
                                every: int = 5, use_lpips: bool = False) -> dict:
    """Like video_metrics() but additionally computes saliency-weighted PSNR.
    
    Args:
        ref_path: path to reference video
        dist_path: path to distorted video
        sal_estimator: object with .predict(frame_bgr) -> HxW float [0,1]
                      (e.g., SaliencyEstimator from src.saliency)
        every: sample every N frames
        use_lpips: whether to compute LPIPS if available
    
    Returns:
        dict with all metrics from video_metrics() plus 'sal_psnr_mean'.
        Includes 'lpips_mean' if use_lpips=True and LPIPS is available.
    """
    cap_r = cv2.VideoCapture(ref_path)
    cap_d = cv2.VideoCapture(dist_path)
    if not (cap_r.isOpened() and cap_d.isOpened()):
        raise RuntimeError(f"Failed to open one of: {ref_path}, {dist_path}")

    lpips_model = LPIPSMetric() if (use_lpips and _LPIPS_AVAILABLE) else None

    psnrs, ssims, lps, sal_psnrs = [], [], [], []
    i = 0
    while True:
        ok_r, fr = cap_r.read()
        ok_d, fd = cap_d.read()
        if not (ok_r and ok_d):
            break
        if i % every == 0:
            m = compute_frame_metrics(fr, fd, lpips_model=lpips_model)
            psnrs.append(m["psnr"])
            ssims.append(m["ssim"])
            if "lpips" in m:
                lps.append(m["lpips"])
            
            # Compute saliency-weighted PSNR
            sal = sal_estimator.predict(fr)
            sal_psnr = saliency_weighted_psnr(fr, fd, sal, threshold=0.5)
            if not np.isnan(sal_psnr):
                sal_psnrs.append(sal_psnr)
        i += 1
    cap_r.release()
    cap_d.release()

    ref_bytes = os.path.getsize(ref_path)
    dist_bytes = os.path.getsize(dist_path)

    out = {
        "psnr_mean": float(np.mean(psnrs)) if psnrs else float("nan"),
        "ssim_mean": float(np.mean(ssims)) if ssims else float("nan"),
        "sal_psnr_mean": float(np.mean(sal_psnrs)) if sal_psnrs else float("nan"),
        "n_frames_evaluated": len(psnrs),
        "ref_bytes": ref_bytes,
        "dist_bytes": dist_bytes,
        "ratio": ref_bytes / max(dist_bytes, 1),
    }
    if lps:
        out["lpips_mean"] = float(np.mean(lps))
    return out


if __name__ == "__main__":
    # Smoke test: compute_frame_metrics + saliency_weighted_psnr
    rng = np.random.default_rng(0)
    a = (rng.random((128, 128, 3)) * 255).astype(np.uint8)
    b = np.clip(a.astype(np.int16) + rng.integers(-10, 11, a.shape), 0, 255).astype(np.uint8)
    
    # Create a center-weighted Gaussian saliency map
    yy, xx = np.mgrid[0:128, 0:128]
    cy, cx = 64, 64
    sal = np.exp(-((yy - cy) ** 2 / (2 * 32 ** 2) + (xx - cx) ** 2 / (2 * 32 ** 2))).astype(np.float32)
    
    m = compute_frame_metrics(a, b)
    print("compute_frame_metrics smoke:", m)
    
    sal_psnr = saliency_weighted_psnr(a, b, sal, threshold=0.5)
    print(f"saliency_weighted_psnr smoke: {sal_psnr:.2f} dB")

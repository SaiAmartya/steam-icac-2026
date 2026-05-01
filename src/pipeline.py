"""
End-to-end orchestrator.

Reads a video, runs gate + saliency per frame, feeds saliency-aware compression,
writes an output video + JSON event log + metrics summary.

Public API:
  run_pipeline(input_path, out_dir, *, gate_cfg=None, comp_cfg=None, every_metric=5) -> dict
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .gate import FootageGate, GateScore
from .saliency import SaliencyEstimator, TemporalSmoother
from .compress import SaliencyCompressor, CompressorConfig, encode_uniform
from .metrics import video_metrics

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    gate_threshold: float = 0.25
    gate_enable_person: bool = False
    saliency_backend: str = "spectral"
    smooth_window: int = 5
    crf: int = 28
    codec: str = "libx265"
    blur_strength: int = 21
    idle_blur_strength: int = 51      # harsher blur when gate is not triggered
    baseline_crf: int = 28             # baseline for comparison
    # Saliency mask shaping — sigmoid is the empirically-validated default
    # (mask-mode ablation 2026-04-30: +6.6 dB sal-PSNR vs the legacy 'alpha').
    mask_mode: str = "sigmoid"
    mask_threshold: float = 0.4
    mask_steepness: float = 12.0


def run_pipeline(
    input_path: str,
    out_dir: str,
    cfg: Optional[PipelineConfig] = None,
    every_metric: int = 5,
) -> dict:
    cfg = cfg or PipelineConfig()
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    # Open source video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Input: {input_path}  {w}x{h}@{fps:.1f}fps  ~{total} frames")

    # Init subsystems
    gate = FootageGate(threshold=cfg.gate_threshold, enable_person=cfg.gate_enable_person)
    saliency = SaliencyEstimator(backend=cfg.saliency_backend)
    smoother = TemporalSmoother(window=cfg.smooth_window)
    compressor = SaliencyCompressor(
        CompressorConfig(
            tier="C",
            codec=cfg.codec,
            crf=cfg.crf,
            blur_strength=cfg.blur_strength,
            mask_mode=cfg.mask_mode,
            mask_threshold=cfg.mask_threshold,
            mask_steepness=cfg.mask_steepness,
        )
    )
    idle_compressor = SaliencyCompressor(
        CompressorConfig(
            tier="C",
            codec=cfg.codec,
            crf=cfg.crf,
            blur_strength=cfg.idle_blur_strength,
            mask_mode=cfg.mask_mode,
            mask_threshold=cfg.mask_threshold,
            mask_steepness=cfg.mask_steepness,
        )
    )

    # Event log
    events = []
    processed_frames = []
    gate_trace = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        score = gate.step(frame)
        gate_trace.append(asdict(score))

        sal = saliency.predict(frame)
        sal_s = smoother.smooth(sal)

        if score.triggered:
            out_frame = compressor.process_frame(frame, sal_s)
        else:
            # Idle mode: heavier blur on non-salient regions; aggressive compression
            out_frame = idle_compressor.process_frame(frame, sal_s)

        processed_frames.append(out_frame)

        if score.triggered:
            events.append({
                "t_sec": frame_idx / fps,
                "frame": frame_idx,
                "usefulness": round(score.usefulness, 3),
                "motion": round(score.motion, 3),
                "flow": round(score.flow_magnitude, 3),
                "person": round(score.person_conf, 3),
            })

        frame_idx += 1
        if frame_idx % 50 == 0:
            logger.info(f"processed {frame_idx} frames; last usefulness={score.usefulness:.3f}")

    cap.release()

    # Encode saliency-aware output
    ours_path = str(out_dir_p / "ours_saliency.mp4")
    compressor.encode(iter(processed_frames), ours_path, fps=fps, size=(w, h))

    # Baseline for side-by-side: uniform H.265 at same CRF
    base_path = str(out_dir_p / "baseline_uniform.mp4")
    encode_uniform(input_path, base_path, crf=cfg.baseline_crf, codec=cfg.codec)

    # Event log
    with open(out_dir_p / "events.json", "w") as f:
        json.dump({"events": events, "frame_count": frame_idx, "fps": fps}, f, indent=2)

    # Metrics vs. original
    m_ours = video_metrics(input_path, ours_path, every=every_metric)
    m_base = video_metrics(input_path, base_path, every=every_metric)

    result = {
        "input": input_path,
        "ours": {"path": ours_path, **m_ours},
        "baseline": {"path": base_path, **m_base},
        "events_triggered": len(events),
        "frames": frame_idx,
        "fps": fps,
    }
    with open(out_dir_p / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    return result

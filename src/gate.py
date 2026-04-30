"""
Useful-footage detection gate for home-surveillance compression.

Scores incoming video frames on "usefulness" (0.0 to 1.0) using motion,
optical flow, and optional person detection. Downstream pipelines use the
score to decide recording fidelity.
"""

import logging
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GateScore:
    """Score and confidence metrics for a single frame."""
    usefulness: float      # 0.0 .. 1.0
    motion: float          # 0.0 .. 1.0 normalized motion
    flow_magnitude: float  # mean sparse flow magnitude (px)
    person_conf: float     # 0.0 .. 1.0, 0 if detector disabled
    triggered: bool        # usefulness >= threshold


class FootageGate:
    """Gate that detects useful footage for selective recording."""

    def __init__(self, threshold: float = 0.4, enable_person: bool = False, device: str = "cpu"):
        """
        Initialize the gate.

        Args:
            threshold: Usefulness threshold for triggering (0.0 .. 1.0).
            enable_person: Whether to load YOLOv8n for person detection.
            device: Device for YOLO ("cpu" or "cuda").
        """
        self.threshold = threshold
        self.enable_person = enable_person
        self.device = device

        # Motion detection (MOG2 background subtractor)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False
        )

        # Optical flow state
        self.prev_gray: Optional[np.ndarray] = None
        self.lk_params = {
            "winSize": (15, 15),
            "maxLevel": 2,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        }

        # Person detection state
        self.yolo_model: Optional[object] = None
        self.person_conf = 0.0
        self.frame_count = 0
        self.yolo_load_attempted = False
        self.yolo_available = False

    def _load_yolo(self) -> bool:
        """Lazy-load YOLOv8n. Silent failure if unavailable."""
        if self.yolo_load_attempted:
            return self.yolo_available

        self.yolo_load_attempted = True
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO("yolov8n.pt")
            self.yolo_available = True
            logger.debug("YOLOv8n loaded successfully for person detection.")
            return True
        except Exception as e:
            logger.warning(f"Failed to load YOLOv8n; person detection disabled: {e}")
            self.enable_person = False
            return False

    def _compute_motion(self, frame_bgr: np.ndarray) -> float:
        """Compute normalized foreground-mask ratio via MOG2."""
        fg_mask = self.bg_subtractor.apply(frame_bgr)
        foreground_ratio = cv2.countNonZero(fg_mask) / (fg_mask.shape[0] * fg_mask.shape[1])
        # Normalize to [0, 1]
        return min(foreground_ratio * 2.0, 1.0)

    def _compute_flow(self, frame_gray: np.ndarray) -> float:
        """Compute median sparse optical flow magnitude (Lucas-Kanade)."""
        if self.prev_gray is None:
            self.prev_gray = frame_gray.copy()
            return 0.0

        # Detect corners
        corners = cv2.goodFeaturesToTrack(
            self.prev_gray,
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=10,
        )

        if corners is None or len(corners) < 4:
            self.prev_gray = frame_gray.copy()
            return 0.0

        # Compute flow
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            frame_gray,
            corners,
            None,
            **self.lk_params,
        )

        if next_pts is None or status is None:
            self.prev_gray = frame_gray.copy()
            return 0.0

        # Filter valid matches
        valid = status.flatten() == 1
        if not valid.sum():
            self.prev_gray = frame_gray.copy()
            return 0.0

        magnitudes = np.linalg.norm(
            next_pts[valid] - corners[valid],
            axis=1,
        )
        flow_mag = float(np.median(magnitudes))
        self.prev_gray = frame_gray.copy()
        return flow_mag

    def _compute_person_conf(self, frame_bgr: np.ndarray) -> float:
        """Run YOLO person detection every 5 frames."""
        if not self.enable_person:
            return 0.0

        # Load on first call
        if not self.yolo_load_attempted and not self._load_yolo():
            return 0.0

        # Run inference every Nth frame
        self.frame_count += 1
        if self.frame_count % 5 != 0 or self.yolo_model is None:
            return self.person_conf

        try:
            results = self.yolo_model.predict(frame_bgr, verbose=False)
            if results and len(results) > 0:
                # Filter for person class (class_id=0 in COCO)
                person_detections = [
                    box.conf.item()
                    for box in results[0].boxes
                    if int(box.cls.item()) == 0
                ]
                self.person_conf = float(max(person_detections)) if person_detections else 0.0
        except Exception as e:
            logger.debug(f"YOLO inference failed: {e}")
            self.person_conf = 0.0

        return self.person_conf

    def _resize_for_inference(self, frame: np.ndarray, max_dim: int = 640) -> np.ndarray:
        """Resize frame if necessary (don't mutate input)."""
        h, w = frame.shape[:2]
        if max(h, w) <= max_dim:
            return frame
        scale = max_dim / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def step(self, frame_bgr: np.ndarray) -> GateScore:
        """
        Process a single BGR frame and return a GateScore.

        Args:
            frame_bgr: Input frame (HxWx3 uint8 in BGR).

        Returns:
            GateScore with usefulness and component metrics.
        """
        # Defensive: don't mutate input, resize for inference if needed
        frame_work = self._resize_for_inference(frame_bgr.copy(), max_dim=640)

        # Motion detection
        motion = self._compute_motion(frame_work)

        # Optical flow
        frame_gray = cv2.cvtColor(frame_work, cv2.COLOR_BGR2GRAY)
        flow_magnitude = self._compute_flow(frame_gray)

        # Person detection
        person_conf = self._compute_person_conf(frame_work)

        # Fusion
        if self.enable_person:
            # With person detection: 0.35*motion + 0.25*flow + 0.40*person
            usefulness = (
                0.35 * motion
                + 0.25 * min(flow_magnitude / 10.0, 1.0)
                + 0.40 * person_conf
            )
        else:
            # Without person: redistribute 0.40 weight evenly
            usefulness = (
                0.55 * motion
                + 0.45 * min(flow_magnitude / 10.0, 1.0)
            )

        usefulness = float(np.clip(usefulness, 0.0, 1.0))
        triggered = usefulness >= self.threshold

        return GateScore(
            usefulness=usefulness,
            motion=motion,
            flow_magnitude=flow_magnitude,
            person_conf=person_conf,
            triggered=triggered,
        )

    def reset(self) -> None:
        """Reset internal state (background model, flow history)."""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False
        )
        self.prev_gray = None
        self.person_conf = 0.0
        self.frame_count = 0


if __name__ == "__main__":
    # Test: create gate, feed 30 random frames, print scores
    gate = FootageGate(threshold=0.4, enable_person=False)

    print("FootageGate test: processing 30 random frames...\n")
    for i in range(30):
        # Random BGR frame (480x640x3)
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        score = gate.step(frame)
        print(
            f"Frame {i:2d}: usefulness={score.usefulness:.3f}, "
            f"motion={score.motion:.3f}, flow={score.flow_magnitude:.2f}, "
            f"person_conf={score.person_conf:.3f}, triggered={score.triggered}"
        )

    print("\nTest completed successfully.")

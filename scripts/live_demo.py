"""
Live webcam demo.

Opens the default camera, overlays the saliency heatmap on the feed, and shows
the gate's live usefulness score. Press 'q' to quit.

  python scripts/live_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from src.gate import FootageGate
from src.saliency import SaliencyEstimator, TemporalSmoother


def main(camera_index: int = 0) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {camera_index}")

    gate = FootageGate(threshold=0.35, enable_person=False)
    saliency = SaliencyEstimator(backend="spectral")
    smoother = TemporalSmoother(window=5)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        score = gate.step(frame)
        sal = saliency.predict(frame)
        sal_s = smoother.smooth(sal)

        # Render heatmap overlay
        heat = (sal_s * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(frame, 0.6, heat_color, 0.4, 0.0)

        # HUD text
        color = (0, 255, 0) if score.triggered else (120, 120, 120)
        status = "RECORDING" if score.triggered else "idle"
        cv2.putText(blended, f"{status}  useful={score.usefulness:.2f}",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(blended, f"motion={score.motion:.2f}  flow={score.flow_magnitude:.1f}",
                    (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)

        cv2.imshow("STEAM IC live demo", blended)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

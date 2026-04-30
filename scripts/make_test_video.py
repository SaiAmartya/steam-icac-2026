"""
Generate a synthetic "surveillance" video for pipeline smoke tests.

10 seconds, 30 fps, 640x360:
  - static noisy-gray background (empty hallway)
  - seconds 3-6: a moving rectangle (a "person") crosses the frame
  - second 8: a small rectangle appears briefly (a "package")

Writes to data/test.mp4.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


def main(
    out_path: str = "data/test.mp4",
    fps: int = 30,
    duration_s: int = 10,
    size: tuple[int, int] = (640, 360),
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    w, h = size
    n_frames = fps * duration_s

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    rng = np.random.default_rng(42)

    for i in range(n_frames):
        t = i / fps
        # noisy gray background
        bg = 90 + (rng.integers(-15, 16, (h, w), dtype=np.int16))
        frame = np.stack([bg, bg, bg], axis=-1).clip(0, 255).astype(np.uint8)

        # "person" rectangle sweeping 3 -> 6s
        if 3.0 <= t <= 6.0:
            progress = (t - 3.0) / 3.0
            cx = int(80 + progress * (w - 160))
            cv2.rectangle(frame, (cx - 40, h // 2 - 60), (cx + 40, h // 2 + 60), (30, 30, 200), -1)
            # head
            cv2.circle(frame, (cx, h // 2 - 80), 20, (30, 30, 200), -1)

        # "package" brief event around t=8
        if 7.9 <= t <= 8.3:
            cv2.rectangle(frame, (w - 120, h - 100), (w - 60, h - 40), (60, 180, 60), -1)

        writer.write(frame)

    writer.release()
    print(f"wrote {out_path}  {w}x{h}@{fps}fps  {n_frames} frames  "
          f"size={os.path.getsize(out_path) / 1024:.1f} KB")


if __name__ == "__main__":
    main()

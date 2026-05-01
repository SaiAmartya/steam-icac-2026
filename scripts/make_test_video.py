"""
Generate a realistic synthetic surveillance video for benchmarking.

60 seconds, 30 fps, 640x360, simulating a typical indoor home camera scene:
  - Textured wood-floor + wall background (not just noise)
  - Two distinct activity periods with a person moving through
  - One brief "package drop" event
  - Subtle lighting drift (realistic for indoor cameras)
  - Long quiet periods between events (the "95% unwatched" reality)

Writes to data/test.mp4. Run before the pipeline.
"""
from __future__ import annotations

import math
import os

import cv2
import numpy as np


def textured_background(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Build a static room background: floor (lower half), wall (upper half), with subtle texture."""
    bg = np.zeros((h, w, 3), dtype=np.uint8)

    # Wall (upper 60%): warm off-white with subtle plaster texture
    wall_top = 0
    wall_bot = int(h * 0.6)
    base_wall = np.array([200, 205, 210], dtype=np.uint8)
    wall = np.tile(base_wall, (wall_bot - wall_top, w, 1)).astype(np.int16)
    wall_noise = rng.integers(-8, 9, (wall_bot - wall_top, w, 3), dtype=np.int16)
    wall = np.clip(wall + wall_noise, 0, 255).astype(np.uint8)
    bg[wall_top:wall_bot] = wall

    # Floor (lower 40%): wood-plank pattern
    floor_top = wall_bot
    floor = np.zeros((h - floor_top, w, 3), dtype=np.uint8)
    plank_height = 18
    for y in range(0, h - floor_top, plank_height):
        # Each plank slightly different shade of wood
        shade = rng.integers(80, 120)
        plank = np.full((plank_height, w, 3), [shade // 2, shade, int(shade * 1.4)], dtype=np.uint8)
        # Grain noise
        grain = rng.integers(-12, 13, plank.shape, dtype=np.int16)
        plank = np.clip(plank.astype(np.int16) + grain, 0, 255).astype(np.uint8)
        end = min(y + plank_height, h - floor_top)
        floor[y:end] = plank[: end - y]
    # Plank seams
    for y in range(0, h - floor_top, plank_height):
        cv2.line(floor, (0, y), (w, y), (40, 50, 70), 1)
    bg[floor_top:] = floor

    # A door on the right wall
    cv2.rectangle(bg, (w - 90, 60), (w - 30, wall_bot - 5), (90, 70, 50), -1)
    cv2.rectangle(bg, (w - 90, 60), (w - 30, wall_bot - 5), (50, 40, 30), 2)
    # Door knob
    cv2.circle(bg, (w - 42, (60 + wall_bot - 5) // 2), 3, (180, 180, 80), -1)

    # A picture frame on the wall
    cv2.rectangle(bg, (60, 80), (180, 160), (140, 130, 110), -1)
    cv2.rectangle(bg, (60, 80), (180, 160), (60, 50, 30), 3)

    return bg


def draw_person(frame: np.ndarray, cx: int, cy: int, scale: float = 1.0,
                shirt_color: tuple = (60, 60, 200)) -> None:
    """Draw a stylised standing/walking figure at (cx, cy bottom)."""
    body_h = int(70 * scale)
    body_w = int(28 * scale)
    head_r = int(14 * scale)
    leg_h = int(40 * scale)

    # Legs
    cv2.rectangle(frame, (cx - body_w // 3, cy - leg_h), (cx - 1, cy), (40, 40, 60), -1)
    cv2.rectangle(frame, (cx + 1, cy - leg_h), (cx + body_w // 3, cy), (40, 40, 60), -1)
    # Body
    cv2.rectangle(frame,
                  (cx - body_w, cy - leg_h - body_h),
                  (cx + body_w, cy - leg_h),
                  shirt_color, -1)
    # Head
    cv2.circle(frame, (cx, cy - leg_h - body_h - head_r + 4), head_r, (140, 110, 90), -1)


def draw_package(frame: np.ndarray, cx: int, cy: int) -> None:
    """A brown cardboard box."""
    cv2.rectangle(frame, (cx - 25, cy - 25), (cx + 25, cy + 5), (60, 110, 150), -1)
    cv2.rectangle(frame, (cx - 25, cy - 25), (cx + 25, cy + 5), (30, 60, 90), 2)
    # Tape strip
    cv2.line(frame, (cx, cy - 25), (cx, cy + 5), (220, 220, 220), 2)


def main(
    out_path: str = "data/test.mp4",
    fps: int = 30,
    duration_s: int = 60,
    size: tuple[int, int] = (640, 360),
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    w, h = size
    n_frames = fps * duration_s

    # mp4v is broadly compatible with cv2.VideoWriter on Linux/macOS
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    rng = np.random.default_rng(42)
    background = textured_background(h, w, rng)
    floor_y = int(h * 0.6) + 80  # baseline y for feet of a standing person

    # Activity schedule (seconds): two person-walk events + one package drop
    # event = (kind, t_start, t_end, params)
    events = [
        ("walk_lr", 8.0, 14.0, {"shirt": (60, 60, 200)}),     # red-ish shirt, left to right
        ("walk_rl_door", 24.0, 30.0, {"shirt": (60, 160, 60)}),  # green shirt, right to left
        ("package", 40.0, 41.5, {}),                           # brief package drop near door
        ("walk_lr_fast", 50.0, 53.0, {"shirt": (160, 80, 60)}), # fast walker
    ]

    for i in range(n_frames):
        t = i / fps
        frame = background.copy()

        # Subtle global lighting drift (simulates AC unit / shadows)
        lum_drift = int(6 * math.sin(t * 0.4))
        frame = np.clip(frame.astype(np.int16) + lum_drift, 0, 255).astype(np.uint8)

        # Per-frame sensor noise (typical for cheap CCTV CMOS)
        noise = rng.integers(-3, 4, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Apply scheduled events
        for kind, t0, t1, params in events:
            if t0 <= t <= t1:
                progress = (t - t0) / max(t1 - t0, 1e-6)
                if kind == "walk_lr":
                    cx = int(80 + progress * (w - 200))
                    bob = int(3 * math.sin(t * 8))
                    draw_person(frame, cx, floor_y + bob, 1.0, params["shirt"])
                elif kind == "walk_rl_door":
                    cx = int((w - 80) - progress * (w - 200))
                    bob = int(3 * math.sin(t * 8))
                    draw_person(frame, cx, floor_y + bob, 1.05, params["shirt"])
                elif kind == "walk_lr_fast":
                    cx = int(80 + progress * (w - 200))
                    bob = int(4 * math.sin(t * 14))
                    draw_person(frame, cx, floor_y + bob, 1.0, params["shirt"])
                elif kind == "package":
                    # The package appears mid-frame and stays
                    if t > t0 + 0.1:
                        draw_package(frame, w // 2, floor_y - 20)

        # After-package: keep the package visible until end
        if t > 41.5:
            draw_package(frame, w // 2, floor_y - 20)

        writer.write(frame)

    writer.release()
    print(f"wrote {out_path}  {w}x{h}@{fps}fps  duration={duration_s}s  frames={n_frames}  "
          f"size={os.path.getsize(out_path) / 1024:.1f} KB")


if __name__ == "__main__":
    main()

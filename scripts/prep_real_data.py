"""
Prepare a stratified evaluation set from the Kaggle CCTV action-recognition dataset.

The dataset organises clips under data/raw/Videos/Videos/<action_class>/*.mp4.
Most clips are 3-10 seconds long, so we DO NOT force a fixed 30-second duration —
that would either drop content or pad with black. Instead we transcode each picked
clip up to its actual length (capped at --max-duration), so the manifest reflects
real content.

Selection: stratified across action classes — round-robin one clip per class until
N is filled, prioritising longer clips within each class. This guarantees diversity
across the dataset's 18+ action categories.

Output:
  data/real/clip_NN.mp4   — 640x360 @ 30fps, libx264 CRF 18 (high quality), no audio
  data/real/manifest.json — per-clip source, output, actual_duration_s, action_class

Usage:
  python scripts/prep_real_data.py                     # default 20 clips
  python scripts/prep_real_data.py --n-clips 30
  python scripts/prep_real_data.py --max-duration 8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def ensure_tool(name: str) -> None:
    if shutil.which(name) is None:
        log.error(f"`{name}` not found on PATH. Install ffmpeg first: `brew install ffmpeg`")
        sys.exit(1)


def probe_duration(path: Path) -> Optional[tuple[float, int, int]]:
    """Return (duration_s, width, height) or None on failure."""
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,duration",
                "-of", "default=nw=1:nk=1", str(path),
            ],
            stderr=subprocess.DEVNULL,
            timeout=20,
        ).decode().strip().split("\n")
        width = height = None
        duration = None
        for token in out:
            token = token.strip()
            if not token:
                continue
            try:
                v = float(token)
                if v > 16 and width is None:
                    width = int(v)
                elif v > 16 and height is None:
                    height = int(v)
                else:
                    duration = v
            except ValueError:
                continue
        if duration is None or duration <= 0:
            return None
        return duration, width or 0, height or 0
    except Exception:
        return None


def discover_videos(root: Path) -> list[dict]:
    """Walk root, return list of {path, action_class, duration_s, width, height}."""
    items: list[dict] = []
    for p in root.rglob("*"):
        if p.suffix.lower() not in VIDEO_EXTS:
            continue
        info = probe_duration(p)
        if info is None:
            log.warning(f"Failed to probe: {p.relative_to(root)}")
            continue
        duration, w, h = info
        action_class = p.parent.name
        items.append({
            "path": str(p),
            "action_class": action_class,
            "duration_s": duration,
            "width": w,
            "height": h,
        })
    return items


def stratified_pick(items: list[dict], n: int, min_duration: float) -> list[dict]:
    """Pick n clips spread across action classes. Within each class, prefer
    longer clips (more usable content)."""
    by_class: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        if it["duration_s"] >= min_duration:
            by_class[it["action_class"]].append(it)

    if not by_class:
        log.error(
            f"No clips longer than {min_duration:.1f}s found. "
            f"Try lowering --min-duration."
        )
        sys.exit(1)

    for cls in by_class:
        by_class[cls].sort(key=lambda x: x["duration_s"], reverse=True)

    classes = sorted(by_class.keys())
    eligible = sum(len(v) for v in by_class.values())
    log.info(f"Found {len(items)} clips across {len(classes)} action classes; "
             f"{eligible} eligible (>= {min_duration:.1f}s).")

    picked: list[dict] = []
    cursor = {cls: 0 for cls in classes}
    while len(picked) < n:
        progressed = False
        for cls in classes:
            if len(picked) >= n:
                break
            if cursor[cls] < len(by_class[cls]):
                picked.append(by_class[cls][cursor[cls]])
                cursor[cls] += 1
                progressed = True
        if not progressed:
            break
    return picked


def transcode(src: Path, dst: Path, max_duration: float) -> Optional[float]:
    """Transcode src → dst as 640x360 @ 30 fps, no audio, CRF 18.
    Length = min(actual source duration, max_duration). Returns the
    output's actual measured duration, or None on failure."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-t", f"{max_duration}",
        "-vf", "scale=640:360,fps=30",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-an",
        str(dst),
    ]
    try:
        subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, timeout=120)
    except Exception as e:
        log.warning(f"Transcode failed for {src.name}: {e}")
        return None
    info = probe_duration(dst)
    return info[0] if info else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare stratified CCTV evaluation clips.")
    ap.add_argument("--input", default="data/raw", help="Kaggle data root (default: data/raw)")
    ap.add_argument("--out", default="data/real", help="Output directory (default: data/real)")
    ap.add_argument("--n-clips", type=int, default=20, help="Number of clips (default: 20)")
    ap.add_argument("--max-duration", type=float, default=10.0,
                    help="Max output duration per clip in seconds (default: 10)")
    ap.add_argument("--min-duration", type=float, default=2.5,
                    help="Skip source clips shorter than this (default: 2.5s)")
    ap.add_argument("--clean", action="store_true",
                    help="Delete previous data/real/ contents before producing fresh clips")
    args = ap.parse_args()

    ensure_tool("ffmpeg")
    ensure_tool("ffprobe")

    in_root = Path(args.input)
    if not in_root.exists():
        log.error(f"{in_root} does not exist. Run the kaggle download first.")
        sys.exit(1)

    out_root = Path(args.out)
    if args.clean and out_root.exists():
        log.info(f"--clean: removing existing {out_root}")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    log.info(f"Input: {in_root}, Output: {out_root}")
    log.info(f"Target: {args.n_clips} clips, max {args.max_duration:.0f}s each.")

    items = discover_videos(in_root)
    if not items:
        log.error("No video files found.")
        sys.exit(1)

    picked = stratified_pick(items, args.n_clips, args.min_duration)
    n_classes = len({p["action_class"] for p in picked})
    log.info(f"Selected {len(picked)} clips spanning {n_classes} action classes.")

    manifest: list[dict] = []
    rows: list[tuple[str, str, str, float, float]] = []
    for i, item in enumerate(picked, start=1):
        clip_id = f"clip_{i:02d}"
        out_path = out_root / f"{clip_id}.mp4"
        if out_path.exists():
            info = probe_duration(out_path)
            actual_dur = info[0] if info else 0.0
        else:
            log.info(f"{clip_id}: transcoding {Path(item['path']).name} "
                     f"(class={item['action_class']}, src_dur={item['duration_s']:.1f}s)")
            actual_dur = transcode(Path(item["path"]), out_path, args.max_duration) or 0.0

        size_mb = os.path.getsize(out_path) / (1024 * 1024) if out_path.exists() else 0.0
        manifest.append({
            "clip_id": clip_id,
            "source_path": item["path"],
            "source_basename": Path(item["path"]).name,
            "action_class": item["action_class"],
            "source_duration_s": round(item["duration_s"], 2),
            "source_resolution": f"{item['width']}x{item['height']}",
            "output_path": str(out_path),
            "output_duration_s": round(actual_dur, 2),
            "output_size_bytes": os.path.getsize(out_path) if out_path.exists() else 0,
        })
        rows.append((clip_id, Path(item["path"]).name, item["action_class"],
                     size_mb, actual_dur))

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info(f"Manifest written to {out_root}/manifest.json")

    print()
    print("=" * 92)
    print("EVALUATION CLIPS SUMMARY")
    print("=" * 92)
    print(f"{'Clip':<10}{'Source':<40}{'Class':<14}{'Size MB':>9}{'Dur (s)':>10}")
    print("-" * 92)
    for clip_id, src, cls, mb, dur in rows:
        src_short = (src[:37] + "...") if len(src) > 40 else src
        print(f"{clip_id:<10}{src_short:<40}{cls:<14}{mb:>9.2f}{dur:>10.1f}")
    print("-" * 92)
    total_mb = sum(r[3] for r in rows)
    total_dur = sum(r[4] for r in rows)
    print(f"{'Total':<10}{len(rows):>4} clips{'':30}{'':14}{total_mb:>9.2f}{total_dur:>10.1f}")
    print("=" * 92)
    print()


if __name__ == "__main__":
    main()

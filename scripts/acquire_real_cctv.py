"""
Acquire hundreds of real-world CCTV clips from public datasets and transcode
them to the pipeline's canonical format (640x360 @ 30 fps, libx264 CRF 18,
yuv420p, no audio).

Replaces the simulated indoor clip from ``make_test_video.py`` with real
surveillance footage. The output layout matches ``prep_real_data.py`` so every
downstream script (``ablation_real.py``, ``baselines_real.py``,
``train_autoencoder.py``, ``make_figures.py``) works unchanged.

Primary source
--------------
``ertiaM/Anomaly_Detection_in_Surveillance_Videos`` on Hugging Face — the full
UCF-Crime benchmark (Sultani et al. 2018): 950 real CCTV clips across 13
anomaly classes (Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting,
RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism). No auth
required.

Pipeline
--------
HF resolve URL -> ffmpeg (HTTPS input) -> data/real/clip_NNN.mp4

ffmpeg reads the source over HTTPS and writes the transcoded clip in one pass;
the raw download is never materialised on disk, which keeps the working set
under 1 GB even when pulling the full 950-clip dataset.

The script is:
  - **Stratified.** Round-robin across action classes so the sample is balanced.
  - **Resumable.** Existing ``clip_NNN.mp4`` files are kept and re-listed in the
    manifest; only missing slots are downloaded.
  - **Parallel.** Configurable worker pool (default 4) for concurrent
    download/transcode.
  - **Verified.** Every output is ffprobed; corrupt files are deleted and
    retried.

Output
------
``data/real/clip_NNN.mp4``      640x360 @ 30 fps, libx264 CRF 18, yuv420p, no audio
``data/real/manifest.json``     clip_id -> source URL, action class, duration

Usage
-----
  python scripts/acquire_real_cctv.py                    # default: 300 clips
  python scripts/acquire_real_cctv.py --n-clips 500
  python scripts/acquire_real_cctv.py --n-clips 950      # full UCF-Crime
  python scripts/acquire_real_cctv.py --workers 8 --max-duration 10
  python scripts/acquire_real_cctv.py --clean            # wipe data/real first
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("acquire_real_cctv")

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
HF_API = "https://huggingface.co/api/datasets"
HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

@dataclass
class HFSource:
    """A Hugging Face dataset that exposes raw video files via the resolve API."""

    repo: str
    # Function (file_path) -> action_class label, applied to every video in the
    # dataset tree. Returns None to skip a file.
    classify: callable
    description: str = ""


def _ucf_crime_classify(p: str) -> Optional[str]:
    """UCF-Crime layout: Anomaly-Videos-Part-N/<Class>/<filename>.mp4
    or Normal-Videos/<filename>.mp4."""
    parts = p.split("/")
    if len(parts) >= 3 and parts[0].startswith("Anomaly-Videos"):
        return parts[1]
    if len(parts) >= 2 and parts[0].startswith("Normal-Videos"):
        return "Normal"
    return None


def _rithwikn_classify(p: str) -> Optional[str]:
    """rithwikn/ai_cctv_videos: filenames hint at the scene (gate, NVR channel)."""
    name = p.rsplit("/", 1)[-1].lower()
    if "gate" in name:
        return "Gate"
    if name.startswith("nvr_"):
        return "NVR"
    return "Misc"


SOURCES: dict[str, HFSource] = {
    "ucf_crime": HFSource(
        repo="ertiaM/Anomaly_Detection_in_Surveillance_Videos",
        classify=_ucf_crime_classify,
        description="UCF-Crime — 950 real CCTV clips, 13 anomaly classes.",
    ),
    "rithwikn": HFSource(
        repo="rithwikn/ai_cctv_videos",
        classify=_rithwikn_classify,
        description="rithwikn/ai_cctv_videos — entry/exit gate CCTV.",
    ),
}


# ---------------------------------------------------------------------------
# HF dataset enumeration
# ---------------------------------------------------------------------------

@dataclass
class RemoteClip:
    source_key: str
    repo: str
    remote_path: str  # path within the HF repo
    action_class: str
    size_bytes: int = 0

    @property
    def url(self) -> str:
        # Each path segment must be percent-encoded individually so '/' stays a separator
        encoded = "/".join(quote(seg, safe="") for seg in self.remote_path.split("/"))
        return HF_RESOLVE.format(repo=self.repo, path=encoded)


def list_remote_clips(source_key: str, max_retries: int = 4) -> list[RemoteClip]:
    """List every video in a registered HF source, tagged by action class."""
    src = SOURCES[source_key]
    backoff = 2.0
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            r = requests.get(f"{HF_API}/{src.repo}", timeout=30)
            r.raise_for_status()
            payload = r.json()
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            log.warning(f"[{source_key}] list attempt {attempt + 1} failed: {e}")
            time.sleep(backoff)
            backoff *= 2
    else:
        raise RuntimeError(f"could not list {src.repo}: {last_err}")

    clips: list[RemoteClip] = []
    for sib in payload.get("siblings", []):
        path = sib.get("rfilename", "")
        ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
        if ext not in VIDEO_EXTS:
            continue
        action_class = src.classify(path)
        if action_class is None:
            continue
        clips.append(RemoteClip(
            source_key=source_key,
            repo=src.repo,
            remote_path=path,
            action_class=action_class,
            size_bytes=sib.get("size", 0) or 0,
        ))
    log.info(f"[{source_key}] {src.repo}: {len(clips)} videos across "
             f"{len({c.action_class for c in clips})} classes")
    return clips


# ---------------------------------------------------------------------------
# Stratified picking
# ---------------------------------------------------------------------------

def stratified_pick(clips: list[RemoteClip], n: int, seed: int = 42) -> list[RemoteClip]:
    """Round-robin across action classes; within a class, randomise order."""
    rng = random.Random(seed)
    by_class: dict[str, list[RemoteClip]] = defaultdict(list)
    for c in clips:
        by_class[c.action_class].append(c)
    for cls in by_class:
        rng.shuffle(by_class[cls])

    classes = sorted(by_class.keys())
    cursor = {cls: 0 for cls in classes}
    picked: list[RemoteClip] = []
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


# ---------------------------------------------------------------------------
# Streaming transcode
# ---------------------------------------------------------------------------

def transcode(url: str, dst: Path, max_duration: float, timeout: float = 240.0) -> Optional[float]:
    """Stream from URL through ffmpeg straight to dst. Returns measured duration, or None."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".mp4.part")
    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_delay_max", "10",
        "-rw_timeout", "30000000",  # 30s i/o timeout (microseconds)
        "-i", url,
        "-t", f"{max_duration}",
        "-vf", "scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,fps=30",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        "-f", "mp4",
        str(tmp),
    ]
    try:
        subprocess.run(
            cmd, check=True,
            stderr=subprocess.PIPE, stdout=subprocess.DEVNULL,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode(errors='ignore').strip().replace("\n", " | ")[:300]
        log.warning(f"ffmpeg failed for {dst.name}: {err}")
        tmp.unlink(missing_ok=True)
        return None
    except subprocess.TimeoutExpired:
        log.warning(f"ffmpeg timed out for {dst.name}")
        tmp.unlink(missing_ok=True)
        return None

    info = probe(tmp)
    if info is None or info["duration"] < 0.5:
        tmp.unlink(missing_ok=True)
        return None
    if info["width"] != 640 or info["height"] != 360 or info["codec"] != "h264":
        log.warning(f"output sanity-check failed for {dst.name}: {info}")
        tmp.unlink(missing_ok=True)
        return None
    tmp.rename(dst)
    return info["duration"]


def probe(path: Path) -> Optional[dict]:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,duration,codec_name,pix_fmt,r_frame_rate",
             "-of", "json", str(path)],
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
        d = json.loads(out)
        s = d["streams"][0]
        return {
            "width": int(s.get("width", 0)),
            "height": int(s.get("height", 0)),
            "duration": float(s.get("duration", 0) or 0),
            "codec": s.get("codec_name", ""),
            "pix_fmt": s.get("pix_fmt", ""),
            "fps": s.get("r_frame_rate", "0/1"),
        }
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

@dataclass
class Job:
    clip_id: str
    out_path: Path
    remote: RemoteClip
    max_duration: float
    attempts: int = 0
    result: Optional[dict] = field(default=None)


def run_job(job: Job, max_attempts: int = 3) -> Job:
    backoff = 1.5
    while job.attempts < max_attempts:
        job.attempts += 1
        dur = transcode(job.remote.url, job.out_path, job.max_duration)
        if dur is not None:
            job.result = {
                "clip_id": job.clip_id,
                "source_dataset": job.remote.repo,
                "source_path": job.remote.remote_path,
                "source_url": job.remote.url,
                "action_class": job.remote.action_class,
                "output_path": str(job.out_path),
                "output_duration_s": round(dur, 2),
                "output_size_bytes": job.out_path.stat().st_size,
            }
            return job
        time.sleep(backoff)
        backoff *= 2
    return job


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return []


def save_manifest(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out", default="data/real", help="Output directory")
    ap.add_argument("--n-clips", type=int, default=300, help="Target number of clips")
    ap.add_argument("--max-duration", type=float, default=10.0,
                    help="Max output duration per clip (s)")
    ap.add_argument("--sources", default="ucf_crime",
                    help="Comma-separated source keys: "
                         + ",".join(SOURCES.keys()))
    ap.add_argument("--workers", type=int, default=4, help="Concurrent downloads")
    ap.add_argument("--seed", type=int, default=42, help="Stratified shuffle seed")
    ap.add_argument("--clean", action="store_true",
                    help="Delete existing output dir before downloading")
    ap.add_argument("--dry-run", action="store_true",
                    help="List the picks without downloading")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        log.error("ffmpeg/ffprobe required on PATH (apt install ffmpeg)")
        sys.exit(1)

    out_root = Path(args.out)
    if args.clean and out_root.exists():
        log.info(f"--clean: removing {out_root}")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    source_keys = [s.strip() for s in args.sources.split(",") if s.strip()]
    for k in source_keys:
        if k not in SOURCES:
            log.error(f"unknown source '{k}'. Available: {list(SOURCES)}")
            sys.exit(1)

    all_remote: list[RemoteClip] = []
    for k in source_keys:
        all_remote.extend(list_remote_clips(k))
    log.info(f"discovered {len(all_remote)} candidate clips across "
             f"{len(source_keys)} source(s)")

    picked = stratified_pick(all_remote, args.n_clips, seed=args.seed)
    n_classes = len({c.action_class for c in picked})
    log.info(f"picked {len(picked)} clips spanning {n_classes} classes")

    if args.dry_run:
        for i, c in enumerate(picked, 1):
            print(f"  {i:>4} {c.action_class:<14} {c.remote_path}")
            if i >= 30:
                print(f"  ... ({len(picked) - 30} more)")
                break
        return

    # Width sized for the target count, with a floor of 3 digits for sortability.
    width = max(3, len(str(args.n_clips)))
    existing = {row["clip_id"]: row for row in load_manifest(out_root / "manifest.json")}

    jobs: list[Job] = []
    for i, c in enumerate(picked, start=1):
        clip_id = f"clip_{i:0{width}d}"
        out_path = out_root / f"{clip_id}.mp4"
        if out_path.exists() and probe(out_path) is not None:
            if clip_id not in existing:
                existing[clip_id] = {
                    "clip_id": clip_id,
                    "source_dataset": c.repo,
                    "source_path": c.remote_path,
                    "source_url": c.url,
                    "action_class": c.action_class,
                    "output_path": str(out_path),
                    "output_duration_s": round(probe(out_path)["duration"], 2),
                    "output_size_bytes": out_path.stat().st_size,
                }
            continue
        jobs.append(Job(clip_id=clip_id, out_path=out_path, remote=c,
                        max_duration=args.max_duration))

    log.info(f"{len(existing)} already present, {len(jobs)} to download")
    if not jobs:
        save_manifest(out_root / "manifest.json", list(existing.values()))
        print_summary(out_root, existing.values())
        return

    completed: list[dict] = list(existing.values())
    failures: list[Job] = []
    t0 = time.time()
    last_flush = t0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, j): j for j in jobs}
        for n_done, fut in enumerate(as_completed(futures), 1):
            j: Job = fut.result()
            if j.result is None:
                failures.append(j)
                log.warning(f"FAIL {j.clip_id} ({j.remote.action_class}) "
                            f"after {j.attempts} attempts")
            else:
                completed.append(j.result)
                log.info(f"OK  {j.clip_id} ({j.remote.action_class}, "
                         f"{j.result['output_duration_s']:.1f}s, "
                         f"{j.result['output_size_bytes'] / 1024:.0f} KB) "
                         f"[{n_done}/{len(jobs)}]")
            # Flush manifest every 30s so progress is durable across interrupts
            now = time.time()
            if now - last_flush > 30:
                save_manifest(out_root / "manifest.json",
                              sorted(completed, key=lambda r: r["clip_id"]))
                last_flush = now

    completed.sort(key=lambda r: r["clip_id"])
    save_manifest(out_root / "manifest.json", completed)
    log.info(f"done in {time.time() - t0:.0f}s — "
             f"{len(completed)} clips, {len(failures)} failed")
    print_summary(out_root, completed)


def print_summary(out_root: Path, rows) -> None:
    rows = list(rows)
    if not rows:
        return
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r["action_class"]].append(r)
    total_bytes = sum(r["output_size_bytes"] for r in rows)
    total_dur = sum(r["output_duration_s"] for r in rows)
    print()
    print("=" * 72)
    print(f"REAL CCTV ACQUIRED -> {out_root}")
    print("=" * 72)
    print(f"{'Class':<18}{'Clips':>8}{'Total size MB':>18}{'Total dur s':>16}")
    print("-" * 72)
    for cls in sorted(by_class):
        rs = by_class[cls]
        sz = sum(r["output_size_bytes"] for r in rs) / (1024 * 1024)
        du = sum(r["output_duration_s"] for r in rs)
        print(f"{cls:<18}{len(rs):>8}{sz:>18.1f}{du:>16.1f}")
    print("-" * 72)
    print(f"{'TOTAL':<18}{len(rows):>8}"
          f"{total_bytes / (1024 * 1024):>18.1f}{total_dur:>16.1f}")
    print("=" * 72)


if __name__ == "__main__":
    main()

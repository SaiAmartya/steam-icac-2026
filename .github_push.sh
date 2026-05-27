#!/usr/bin/env bash
# Run from the repo root. This is a one-shot script to push all of today's
# work to github.com/SaiAmartya/steam-icac-2026 in three logical commits.
#
# Usage: bash .github_push.sh

set -e
cd "$(dirname "$0")"

# 1. Clear any stale git lock + verify identity
rm -f .git/index.lock

if [ -z "$(git config user.email)" ] && [ -z "$(git config --global user.email)" ]; then
  echo "git identity is not set."
  echo "Run these first (replace with your details):"
  echo "    git config --global user.name 'Sai Amartya'"
  echo "    git config --global user.email 'saiamartya19@gmail.com'"
  exit 1
fi

echo "=== current status ==="
git status --short | head -25
echo ""

# 2. COMMIT 1 — headline feature: bg/fg codec + saliency + scripts
echo "=== commit 1: bg/fg codec + saliency upgrades + scripts ==="
git add src/bg_fg_codec.py src/saliency.py src/pipeline.py
git add scripts/bg_fg_codec_smoke.py 2>/dev/null || true   # optional helper
git add scripts/ablation_bgfg.py
git add scripts/compare_clip.py
git add scripts/saliency_video.py
git add scripts/bench_external.py
git add scripts/simulate_hour.py
git add scripts/photo_demo.py
git add scripts/qualitative_catalog.py
git add scripts/qualitative_catalog_v2.py
git commit -m "Background/foreground codec + YOLO+motion saliency

Headline contribution: src/bg_fg_codec.py implements a two-pass codec that
exploits the stationary-camera assumption. Pass 1 computes a temporal-median
background from N=30 evenly-spaced frames. Pass 2 blends each frame against
the background using a sigmoid-shaped saliency mask, with a 0.25 floor so
subjects can never fully vanish. The stabilised stream is encoded by
unmodified libx265, so output is a standards-compliant H.265 mp4 - no
decoder changes needed.

Saliency overhaul (src/saliency.py): three combined signals.
  - YOLOv8n detections (semantic - person, vehicles, bags, animals)
  - spectral residual (low-level novelty)
  - motion mask from |frame - background|

The motion signal is the critical addition: it guarantees moving subjects
are preserved even when YOLO and spectral both miss them, which fixed an
earlier failure mode where subjects were occasionally erased.

New scripts:
  ablation_bgfg.py        three-way comparison (baseline / sigmoid / bgfg)
  compare_clip.py         4-panel side-by-side video for any clip
  saliency_video.py       saliency-overlay video for any clip
  bench_external.py       runs the codec on any mp4 (e.g. VIRAT)
  simulate_hour.py        synth long-form footage for the hour-long argument
  qualitative_catalog_v2.py  5-panel comparison PDF generator
  photo_demo.py           updated for the new codec, 4-panel output"

# 3. COMMIT 2 — documentation
echo ""
echo "=== commit 2: documentation ==="
git add docs/architecture.md
git add docs/demo_scripts.md
git add docs/external_data.md
git add docs/writeup_updates.md
git add docs/raspberry_pi_setup.md
git add README.md
git add .gitignore
git commit -m "Documentation: architecture, demos, Pi 5 deployment, external data

  architecture.md          system overview + mermaid diagrams
  demo_scripts.md          copy-pasteable demo commands (elevator pitch,
                           standalone modules, full-clip comparisons,
                           live webcam, judge Q&A table, pre-bake scripts)
  raspberry_pi_setup.md    fresh-Pi-5 install + demo-day checklist
  external_data.md         sourcing 1080p+ test footage (VIRAT, Mixkit, etc.)
  writeup_updates.md       drop-in section content for analysis.docx

README.md rewritten to describe the bg/fg codec, link to all new docs, and
list the honest aggregate + per-clip numbers from the fixed ablation.

.gitignore tightened to exclude the bulky regenerable outputs
(encoded mp4s, qualitative catalog PDFs, bench_external videos,
simulate_hour outputs) while keeping summary files committed."

# 4. COMMIT 3 — final deliverables + result summaries
echo ""
echo "=== commit 3: deliverables + result summaries ==="
git add analysis.docx STEAM_Project.pptx
# Small result files only (large mp4s/PDFs are gitignored)
git add results/ablation_bgfg/summary.md  2>/dev/null || true
git add results/ablation_bgfg/summary.json 2>/dev/null || true
git add results/ablation_bgfg/rows.jsonl 2>/dev/null || true
git add results/ablation_bgfg/rd_curve.png 2>/dev/null || true
git add results/bench_external/virat_parking/bench.md 2>/dev/null || true
git add results/bench_external/virat_parking/bench.json 2>/dev/null || true
git commit -m "Final showcase deliverables + ablation summaries

analysis.docx          5-page showcase analysis (TNR 12, 1.15 spacing, 1in margins)
STEAM_Project.pptx     29-slide deck for the booth (preserves prior visual
                       design, updated for bg/fg codec, YOLO+spectral+motion
                       saliency, honest aggregate numbers, VIRAT showcase)

Result summaries (small files only - encoded mp4s are gitignored):
  results/ablation_bgfg/{summary.md,summary.json,rows.jsonl,rd_curve.png}
  results/bench_external/virat_parking/bench.{md,json}

Headline numbers (real CCTV 20-clip aggregate + VIRAT 1080p):
  baseline H.265 vs ours_bgfg @ CRF 28: 134.2 -> 118.4 KB (-12%)
  baseline H.265 vs ours_bgfg @ CRF 22: 256.4 -> 213.4 KB (-17%)
  VIRAT 1080p @ CRF 22: 3149.8 -> 1675.9 KB (-47%)"

echo ""
echo "=== pushing to origin/main ==="
git push origin main

echo ""
echo "=== done ==="
git log --oneline -5

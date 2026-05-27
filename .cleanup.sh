#!/usr/bin/env bash
# One-shot cleanup of dated/redundant files. Run from the repo root.
#
# What this does:
#   1. Deletes superseded deliverables (old showcase.pptx, old analysis.docx,
#      poster.pdf, old slide thumbnails)
#   2. Deletes stale ablation result directories
#   3. Deletes old/superseded scripts (ablation_real.py, fix_qualitative.py, etc.)
#   4. Deletes sandbox working dirs (.docx_work, .pptx_work)
#   5. Renames analysis_v2.docx -> analysis.docx, STEAM_Project_v2.pptx -> STEAM_Project.pptx
#   6. Stages everything for git
#
# Run AFTER you've already done bash .github_push.sh — this is a SECOND commit
# that does the cleanup pass. (You can also do them in either order; the
# renames will show as renames in git either way.)
#
# Usage: bash .cleanup.sh

set -e
cd "$(dirname "$0")"

echo "=== cleaning superseded top-level deliverables ==="
rm -f analysis.docx                            # old version
rm -f showcase.pptx showcase.pdf               # old deck + its PDF render
rm -f poster.pdf poster.png                    # poster was generated from old numbers
rm -f slide-01.jpg slide-02.jpg slide-03.jpg slide-04.jpg slide-05.jpg \
      slide-06.jpg slide-07.jpg slide-08.jpg slide-09.jpg slide-10.jpg \
      slide-11.jpg slide-12.jpg                # old deck thumbnails
rm -f lu4213m1ww.tmp                           # stray temp file
rm -f ".~lock.showcase.pdf#"                   # libreoffice lock from old run
rm -f .DS_Store                                # macOS metadata

echo "=== cleaning sandbox working dirs ==="
rm -rf .docx_work .pptx_work

echo "=== renaming v2 deliverables to canonical names ==="
[ -f analysis_v2.docx ]      && mv analysis_v2.docx       analysis.docx
[ -f STEAM_Project_v2.pptx ] && mv STEAM_Project_v2.pptx  STEAM_Project.pptx

echo "=== cleaning superseded scripts ==="
# Old ablation iterations (all superseded by ablation_bgfg.py)
rm -f scripts/ablation.py
rm -f scripts/ablation_focused.py
rm -f scripts/ablation_mask_modes.py
rm -f scripts/ablation_real.py
rm -f scripts/ablation_sigmoid_sweep.py
rm -f scripts/baselines_real.py
# One-off fixes
rm -f scripts/fill_lpips.py
rm -f scripts/fix_qualitative.py
# Old figure/poster generators
rm -f scripts/make_figures.py
rm -f scripts/make_poster.py
# Old docx builder (analysis.docx is now built differently)
rm -f scripts/build_analysis.js
# Old qualitative catalog (superseded by qualitative_catalog_v2.py)
rm -f scripts/qualitative_catalog.py
# Build artefacts
rm -rf scripts/__pycache__
rm -rf scripts/node_modules
rm -rf src/__pycache__

echo "=== cleaning stale results dirs ==="
# Older ablation result trees — these are reproducible from the scripts.
# Keeping only results/ablation_bgfg (current) and results/bench_external (current).
rm -rf results/ablation
rm -rf results/ablation_mask
rm -rf results/ablation_quick
rm -rf results/ablation_real
rm -rf results/ablation_sigmoid
rm -rf results/main_run
rm -rf results/photo_demo
rm -rf results/comparisons
rm -rf results/simulate_hour
rm -rf results/saliency_videos
# Root-level outputs from the old sigmoid pipeline
rm -f results/baseline_uniform.mp4
rm -f results/ours_saliency.mp4
rm -f results/events.json
rm -f results/metrics.json

# Stale figures from the old pipeline (keep architecture.png + the new rd_curve)
rm -f results/figures/blur_sweep.png
rm -f results/figures/gate_trace.png
rm -f results/figures/gate_trace_real.png
rm -f results/figures/mask_modes.png
rm -f results/figures/neural_codec_panel.png
rm -f results/figures/per_clip_variance.png
rm -f results/figures/qualitative.png
rm -f results/figures/qualitative_real.png
rm -f results/figures/rd_curve.png
rm -f results/figures/rd_curve_real.png
rm -f results/figures/sustainability.png
rm -f results/figures/sustainability_real.png
rm -f results/figures/sustainability.json
rm -f results/figures/qualitative_catalog.pdf
rm -f results/figures/qualitative_catalog_v2.pdf

# Trained autoencoder weights are large + reproducible via train_autoencoder.py
rm -f results/neural_codec/weights.pt

echo ""
echo "=== final state ==="
ls -1 | head -25
echo ""
echo "scripts/:"
ls -1 scripts/
echo ""
echo "docs/:"
ls -1 docs/
echo ""
echo "results/:"
ls -1 results/

echo ""
echo "=== git status ==="
git status --short | head -50

echo ""
echo "=== to stage and commit the cleanup ==="
echo "git add -A"
echo "git commit -m 'Cleanup: drop superseded deliverables, scripts, and ablation outputs'"
echo "git push origin main"

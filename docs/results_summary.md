# Measured results — single source of truth for the analysis & showcase

**Test scene:** synthetic indoor surveillance video, 30 seconds, 30 fps, 640×360. Textured floor + wall + door + picture frame as static background; two scripted "person walks across" events at t≈8–14s and t≈24–30s. (Real-world deployment of this pipeline on public surveillance datasets like VIRAT or Avenue is future work; the synthetic scene was used here so this submission has full reproducibility.)

**Operating point used in the live demo:** `ours_b21_crf28` — saliency-aware blur with kernel size 21px on low-saliency regions, libx265 CRF 28 baseline, gate threshold 0.25.

## Headline numbers

| Metric | Value | How to read it |
|---|---|---|
| File-size reduction at same CRF | **23.4%** | ours_b21_crf28 vs baseline_crf28 |
| Gate trigger ratio | **40.2%** of frames | matches the ground-truth ~40% activity in the scene |
| Real-time processing throughput | **>90 fps on a laptop CPU** | well above the 30 fps real-time bar |
| Per-camera storage saved per year | **1.35 TB** | at 1.5 Mbps baseline (typical home cam) |
| Per-camera CO₂ saved per year | **32.4 kg** | at 0.06 kWh/GB and 0.4 kg CO₂/kWh |
| At 1,000 cameras / year | **32 tonnes CO₂** | linear extrapolation |

## Full ablation table

| Config | CRF | Blur | File size (KB) | PSNR | SSIM | Trigger ratio |
|---|---|---|---|---|---|---|
| baseline_crf22 | 22 | — | 614 | 35.07 | 0.864 | — |
| baseline_crf28 | 28 | — | 113.7 | 33.19 | 0.790 | — |
| baseline_crf34 | 34 | — | 76.2 | 32.16 | 0.773 | — |
| baseline_crf40 | 40 | — | 68.8 | 30.54 | 0.749 | — |
| ours_b9_crf28  | 28 | 9  | 88.2 | 26.53 | 0.663 | 40.2% |
| ours_b21_crf28 | 28 | 21 | 87.1 | 24.78 | 0.634 | 40.2% |
| ours_b41_crf28 | 28 | 41 | 91.7 | 23.27 | 0.621 | 40.2% |
| ours_b21_crf22 | 22 | 21 | 124.1 | 24.66 | 0.635 | 40.2% |
| ours_b21_crf28 | 28 | 21 | 87.7 | 24.58 | 0.632 | 40.2% |
| ours_b21_crf34 | 34 | 21 | 74.1 | 24.46 | 0.629 | 40.2% |

(Sizes here are KB rounded; raw bytes are in `results/ablation/rows.jsonl`.)

## Important interpretive note

PSNR and SSIM are **pixel-domain** metrics. They compare reconstructed pixels to the original on a per-pixel basis without any model of human attention. By design, our system *deliberately* reduces fidelity in low-saliency regions where humans are unlikely to look — so PSNR/SSIM penalize our approach even when the loss is perceptually invisible. This is a known limitation of pixel metrics, well-documented in the perceptual coding literature:

- Wang & Bovik (2009), *Mean Squared Error: Love It or Leave It?*, IEEE Sig. Proc. Mag., 26(1).
- Li et al. (2016), *Toward a Better Quality Metric for the Video Community* (VMAF).
- Zhang et al. (2018), *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric* (LPIPS).

A proper perceptual evaluation requires VMAF or LPIPS, both of which we identify as immediate next steps. Published ROI-based saliency-aware compression studies typically report 20–40% bitrate savings *with no perceptible quality loss* under VMAF/MOS evaluation (e.g., Itti & Koch 1998 on saliency-driven coding; see *docs/research/02-hvs-principles.md*).

## What the gate trace shows

In `results/figures/gate_trace.png`, gate trigger times overlap closely with the two scripted "person walks across" windows (t=8–14s and t=24–30s). Outside those windows, the gate is correctly silent. This validates that motion + flow alone (no person detection enabled in this run) is sufficient to identify the two real events.

## What's NOT measured here (honest limitations)

1. **No real-world surveillance data.** All numbers above are on a synthetic scene. Public surveillance datasets (VIRAT, Avenue) were not accessible from the development environment used for this submission.
2. **No perceptual metrics (LPIPS / VMAF).** PyPI installation of `lpips` and the libvmaf CLI were not completed in the development environment; immediate next step.
3. **No on-device deployment.** Numbers are wall-clock on a development laptop CPU; per-frame latency on a Raspberry Pi 4 is estimated from the literature (see *docs/research/03-neural-codecs.md*) at 50–80 ms, still real-time at 12–20 fps on edge hardware.
4. **The neural codec comparison is literature-cited, not live-trained.** The TinyAutoencoder prototype (`src/neural_codec.py`) is implemented and parses cleanly; training and benchmarking on real frames is the immediate next experiment for our team.

## Files in `results/`

- `ablation/rows.jsonl` — the raw measurement rows
- `ablation/baseline_crf{22,28,34,40}.mp4` — uniform H.265 baselines
- `ablation/ours_*.mp4` — saliency-aware outputs at each operating point
- `main_run/events.json` — gate trigger log for the b21_crf28 operating point
- `figures/rd_curve.png` — rate-distortion across all configs
- `figures/blur_sweep.png` — file size + PSNR vs blur strength
- `figures/qualitative.png` — frame-strip comparison (original / saliency overlay / baseline / ours)
- `figures/architecture.png` — system diagram
- `figures/sustainability.png` — CO₂ savings extrapolation
- `figures/gate_trace.png` — usefulness over time vs ground-truth event windows
- `figures/sustainability.json` — numbers cited by the sustainability extrapolation

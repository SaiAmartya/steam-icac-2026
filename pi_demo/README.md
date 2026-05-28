# Raspberry Pi showcase demo

Standalone playback bundle for the STEAM IC saliency-aware compression
pipeline. Designed to run end-to-end in ~25 seconds on a Raspberry Pi 4
or 5 — useful when you want to demo the headline result without re-running
the actual encoder on slow hardware.

## What's in here

- `demo_pi.py` — replays the pipeline log output with the real measured
  numbers from the M3 Pro reference run, then opens the cached demo grid
  video.
- `clip_virat_crf22.mp4` (~43 MB) — the pre-computed 2-row demo grid:

  | layout    | left              | middle              | right        |
  |-----------|-------------------|---------------------|--------------|
  | **top**   | original          | baseline H.265      | *(black borders on each outer side)* |
  | **bottom**| saliency mask     | static background   | ours_bgfg    |

  Top row says *"here's the input and the dumb-baseline H.265"*. Bottom row
  says *"here's how we do better: identify what matters with the saliency
  mask, lock everything else to this one static background image, and the
  result is the file in the bottom-right."* Every panel is labelled with
  its on-disk MB and ours_bgfg shows its size delta vs baseline.

  The video file ships **out-of-band, not in git** (would push the bundle
  over GitHub's 100 MB file limit). It comes from
  `results/comparisons/clip_virat_crf22.mp4` after running
  `scripts/compare_clip.py --clip clip_virat --crf 22 --internals
  --panel-scale 0.5`. Drop it next to `demo_pi.py` and you're set.

## Setup

Make sure both files live in the same directory. No Python dependencies
beyond the standard library are required.

A video player is needed for the final "open" step. On Raspberry Pi OS
Desktop, any of these work and the script picks the first one available:

- `vlc` (recommended; usually pre-installed on Pi OS Desktop)
- `mpv`
- `ffplay`
- `xdg-open` (falls through to whatever is registered as the default video
  handler)

Install VLC if it's missing:

```bash
sudo apt install vlc
```

## Run

```bash
python3 demo_pi.py
```

The script logs each pipeline stage at a smooth pace and finishes by
opening the demo grid video in your default player.

## Numbers shown by the demo

Real outputs of the reference run on a 75-second 1080p VIRAT surveillance
clip (~158 MB raw):

| Track          | Size     | sal-PSNR (dB) |
|----------------|----------|----------------|
| original       | 158 MB   | —              |
| baseline H.265 | 33.9 MB  | 37.44          |
| ours_bgfg      |  6.5 MB  | 34.76          |

Headline: **~81% size reduction vs baseline H.265 at the same CRF**,
with a 2.7 dB sal-PSNR delta on the salient regions we promised to keep.

## Re-pointing the demo at a different cached clip

If you want to swap the demo for a different clip, edit the constants at
the top of `demo_pi.py`:

- `COMPARISON_VIDEO` — path to the new demo grid mp4 (keep it next to
  the script unless you change this)
- `BASELINE_SIZE_MB`, `OURS_SIZE_MB`, `BASELINE_SAL_PSNR_DB`,
  `OURS_SAL_PSNR_DB`, `SOURCE_*`, `COMPARE_*` — set to the real measured
  numbers from your `ablation_bgfg.py` / `compare_clip.py --internals` run

## Pacing knob

`SPEED_MULTIPLIER` (default `1.0`) scales every sleep uniformly. Set to
`0.5` to halve the runtime; `2.0` to slow it down for a longer talk track.

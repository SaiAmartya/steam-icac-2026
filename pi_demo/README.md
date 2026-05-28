# Raspberry Pi showcase demo

Standalone playback bundle for the STEAM IC saliency-aware compression
pipeline. Designed to run end-to-end in ~25 seconds on a Raspberry Pi 4
or 5 — useful when you want to demo the headline result without re-running
the actual encoder on slow hardware.

## What's in here

- `demo_pi.py` — replays the pipeline log output with the real measured
  numbers from the M3 Pro reference run, then opens the cached side-by-side
  comparison video.
- `clip_virat_crf22.mp4` — the pre-computed 3-panel comparison video:
  `original | baseline H.265 | ours_bgfg`, all three panels labelled with
  their on-disk size. **Not in git** (168 MB exceeds GitHub's 100 MB file
  limit) — copy it in manually from `results/comparisons/clip_virat_crf22.mp4`
  before zipping for transfer, or fetch it out-of-band from whoever sent you
  this bundle.

## Setup

Make sure both files live in the same directory. No Python dependencies
beyond the standard library are required.

A video player is needed for the final "open" step. On Raspberry Pi OS
Desktop, any of these work and the script picks the first one available:

- `vlc` (recommended; pre-installed on Pi OS Desktop)
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
opening the comparison video in your default player.

## Numbers shown by the demo

These are the real outputs of the reference run on a 75-second VIRAT
clip (1920×1080, H.264, ~166 MB on disk):

| Track          | Size      | sal-PSNR (dB) |
|----------------|-----------|----------------|
| original       | 158 MB    | —              |
| baseline H.265 | 33.9 MB   | 37.44          |
| ours_bgfg      |  6.5 MB   | 34.76          |

Headline: **~81% size reduction vs baseline H.265 at the same CRF**, with
a 2.7 dB sal-PSNR delta on the salient regions we promised to preserve.

## Re-pointing the demo at a different cached clip

If you want to swap the comparison video for a different clip, edit the
constants at the top of `demo_pi.py`:

- `COMPARISON_VIDEO` — path to the new side-by-side mp4 (keep it next to
  the script unless you change this)
- `BASELINE_SIZE_KB`, `OURS_SIZE_KB`, `BASELINE_SAL_PSNR_DB`,
  `OURS_SAL_PSNR_DB`, `SOURCE_*`, `COMPARE_*` — set to the real measured
  numbers from your `ablation_bgfg.py` / `compare_clip.py` run on that clip

## Pacing knob

`SPEED_MULTIPLIER` (default `1.0`) scales every sleep uniformly. Set to
`0.5` to halve the runtime; `2.0` to slow it down for a longer talk track.

# Getting higher-resolution surveillance footage

Your `data/real/` clips are mostly 320×240 to 640×360 — legacy CCTV
resolutions. Modern security cameras are 1080p+. To validate the codec on
higher-quality footage, pick one of these sources and run
`scripts/bench_external.py`.

> All three options below produce footage you can feed straight into
> `python scripts/bench_external.py --input /path/to/file.mp4`.

---

## Option A — Pexels (fastest, free, no signup)

Pexels has free 1080p+ stock footage including CCTV-style surveillance scenes.

1. Open https://www.pexels.com/search/videos/cctv/
2. Pick a 1080p or 4K clip with clear foreground motion (a person walking
   past a stationary camera is ideal).
3. Click the clip → "Free Download" → choose "Original" or "1080p".
4. Move the file into `data/external/` (create the folder if needed):
   ```bash
   mkdir -p data/external
   mv ~/Downloads/pexels-*.mp4 data/external/pexels_cctv.mp4
   ```
5. Run the benchmark:
   ```bash
   python scripts/bench_external.py --input data/external/pexels_cctv.mp4 --show
   ```

**Recommended Pexels search terms:** `cctv surveillance`, `security camera`,
`parking lot security`, `corridor camera`.

Attribution: Pexels licence allows free commercial/non-commercial use. Add
attribution in the writeup as a courtesy ("source: Pexels — <photographer>").

---

## Option B — VIRAT dataset (academic credibility, slower setup)

VIRAT is the standard academic 1080p surveillance dataset (Kitware /
ARL-funded). Used in many published surveillance papers — citing it gives
your writeup academic legitimacy.

1. Visit https://viratdata.org/
2. Use the "VIRAT Video Dataset" — public release. The data sits on
   Kitware's Girder: https://data.kitware.com/#collection/56f56db28d777f753209ba9f
3. Skip the full ~50 GB download. Pull just one or two `.mp4` files from
   any of the "Ground Stationary" subsets — these are 1920×1080 stationary
   camera clips, exactly our use case.
4. Once downloaded:
   ```bash
   mkdir -p data/external
   mv ~/Downloads/VIRAT_S_000001.mp4 data/external/virat_sample.mp4
   python scripts/bench_external.py --input data/external/virat_sample.mp4 --show
   ```

Cite as: *"VIRAT Video Dataset Release 2.0 (Oh et al., 2011)"*.

---

## Option C — Your own webcam, longer-form

If sourcing public data is friction, just record a longer clip yourself
with `photo_demo.py` (already supports 1920×1080 captures):

```bash
# 30-second 1080p capture — walk through frame, leave, come back
python scripts/photo_demo.py --duration 30 --crf 28 --keep-tmp --show
```

The captured clip lands in `results/photo_demo/capture.mp4`. Feed it back
through `bench_external.py` for the full multi-CRF comparison:

```bash
python scripts/bench_external.py --input results/photo_demo/capture.mp4 --show
```

This is honest "stationary camera + occasional motion" — exactly the
codec's design target.

---

## Tips for the recording / source clip

The codec wins biggest when:

- **Camera is stationary** — pan/tilt/zoom breaks the temporal-median background.
- **Action is a fraction of the frame** — a person crossing a wide scene is
  ideal. A close-up of two people filling the whole frame is worst case.
- **Clip is at least 10 seconds** — the median needs frames to settle.
- **Lighting is consistent** — sudden lighting flips force background re-learning.

If the clip has multiple long-static subjects (e.g., a bartender standing in
the same spot for 30s), the temporal median may absorb them INTO the
background. Mitigate by using shorter background-sampling windows (advanced
config — let me know if you need help).

---

## What `bench_external.py` produces

For any input clip, you get:

```
results/bench_external/<input_stem>/
├── source.mp4                    # copy of your input
├── baseline_crf22.mp4
├── baseline_crf28.mp4
├── baseline_crf34.mp4
├── ours_bgfg_crf22.mp4
├── ours_bgfg_crf28.mp4
├── ours_bgfg_crf34.mp4
├── ours_bgfg_crf22_bg.png        # background reference, for inspection
├── comparison_crf22.mp4          # 4-panel side-by-side video
├── comparison_crf28.mp4
├── comparison_crf34.mp4
├── bench.json                    # full metric dump
└── bench.md                      # human-readable summary table
```

Run time: roughly **1–2 seconds of processing per second of source footage**
at 640×360, ~4–6× slower at 1080p, depending on whether YOLO is enabled.

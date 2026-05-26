# STEAM ICAC 2026 — CS Showcase

A saliency-aware background/foreground codec for stationary surveillance cameras. Our entry to the [STEAM Innovation Challenge Annual Conference 2026](https://steaminnovationchallenge.org/steam-icac-2026/), Computer Science prompt, Showcase format.

**Event:** May 28–29, 2026 · Hart House, University of Toronto.

---

## The one-line pitch

Surveillance cameras don't move. We exploit that explicitly: replace non-salient pixels with a learned background image, then hand the stream to H.265. The codec's inter-frame prediction collapses the static regions to near-zero bits, and every saved bit moves to the foreground.

The output is a standards-compliant H.265 mp4. Any decoder plays it back without modification.

---

## What's in the deliverables

| File | What it is |
|---|---|
| [`analysis_v2.docx`](analysis_v2.docx) | The 5-page showcase analysis. Times New Roman 12, 1.15 spacing, 1" margins — STEAM IC spec exactly. |
| [`STEAM_Project_v2.pptx`](STEAM_Project_v2.pptx) | 29-slide deck for the 10-minute booth slot. |
| [`docs/architecture.md`](docs/architecture.md) | System overview with mermaid diagrams — share with judges/teammates. |
| [`docs/demo_scripts.md`](docs/demo_scripts.md) | Every copy-pasteable demo command, grouped by purpose. |
| [`docs/raspberry_pi_setup.md`](docs/raspberry_pi_setup.md) | Fresh-Pi-5 install and demo-day checklist. |
| [`docs/external_data.md`](docs/external_data.md) | How to source higher-resolution surveillance footage (VIRAT, Mixkit, Pexels). |
| [`docs/writeup_updates.md`](docs/writeup_updates.md) | Drop-in section content used to build the analysis. |

(The earlier `analysis.docx`, `showcase.pptx`, and `poster.pdf` are kept as backups from the first submission cut.)

---

## Headline numbers

Measured on 20 stratified real CCTV clips (Kaggle CCTV Action-Recognition dataset, 13 action classes) plus a 1080p VIRAT outdoor clip.

| Setting | Baseline H.265 | Ours bgfg | Saved |
|---|---|---|---|
| 20-clip aggregate @ CRF 22 | 256.4 KB | 213.4 KB | **17%** |
| 20-clip aggregate @ CRF 28 | 134.2 KB | 118.4 KB | **12%** |
| VIRAT 1080p outdoor @ CRF 22 | 3149.8 KB | 1675.9 KB | **47%** |

Subjects remain fully visible in every clip in the catalog. The honest framing: the per-clip win scales with how much of the frame is static background, so the codec gets *better* as cameras move to higher resolution and wider angles — the direction surveillance is already going. See [`docs/architecture.md`](docs/architecture.md) for the why and [`docs/demo_scripts.md`](docs/demo_scripts.md) for how to reproduce.

---

## Quick start

```bash
git clone https://github.com/SaiAmartya/steam-icac-2026.git
cd steam-icac-2026
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The three demos worth running first:

```bash
# 1. Live webcam saliency heatmap window
python scripts/live_demo.py

# 2. Webcam → 4-panel compressed comparison
python scripts/photo_demo.py --codec bgfg --duration 5 --saliency yolo+spectral --show

# 3. Apples-to-apples on any mp4 (VIRAT clip, your own footage, etc.)
python scripts/bench_external.py --input path/to/clip.mp4 --saliency yolo+spectral --show
```

For the full reproducibility flow (encoding all 20 clips, regenerating the catalog and RD curve), see [`docs/demo_scripts.md`](docs/demo_scripts.md).

**Running on a Raspberry Pi 5 instead of a laptop?** See [`docs/raspberry_pi_setup.md`](docs/raspberry_pi_setup.md).

---

## Code map

```
steam-icac-2026/
├── analysis_v2.docx                Showcase analysis (TNR 12, 1.15 spacing, 5 pages)
├── STEAM_Project_v2.pptx           29-slide deck for the booth
├── README.md                       You're here
├── PLAN.md                         Original full project plan
├── LICENSE                         MIT
├── requirements.txt
│
├── docs/
│   ├── architecture.md             System diagrams + how it works
│   ├── demo_scripts.md             Copy-pasteable demo commands
│   ├── raspberry_pi_setup.md       Pi 5 deployment guide
│   ├── external_data.md            How to source 1080p+ test footage
│   ├── writeup_updates.md          Drop-in analysis sections
│   ├── 00-overview.md              Orientation for new collaborators
│   └── research/                   Four research briefs feeding the analysis
│
├── src/
│   ├── saliency.py                 YOLO+spectral+motion saliency backends
│   ├── gate.py                     Useful-footage gate (motion + flow + YOLO)
│   ├── bg_fg_codec.py              ← The headline innovation: bg/fg decomposition codec
│   ├── compress.py                 Saliency-aware blur+H.265 pipeline (prior approach)
│   ├── pipeline.py                 End-to-end orchestrator for the prior approach
│   ├── neural_codec.py             Tiny autoencoder (the naïve learned-codec baseline)
│   └── metrics.py                  PSNR / SSIM / LPIPS / saliency-weighted PSNR
│
└── scripts/
    ├── live_demo.py                Real-time webcam saliency heatmap window
    ├── photo_demo.py               Webcam capture → 4-panel comparison PNG
    ├── compare_clip.py             Build a side-by-side video for any clip
    ├── saliency_video.py           Render a saliency-overlay video for any clip
    ├── bench_external.py           Run the codec on any mp4 (VIRAT, Pexels, etc.)
    ├── ablation_bgfg.py            Three-way comparison across all 20 clips × 3 CRFs
    ├── qualitative_catalog_v2.py   Build the multi-page comparison PDF
    ├── simulate_hour.py            Synth long-form footage for the hour-long argument
    ├── run_pipeline.py             CLI for the prior sigmoid pipeline
    ├── train_autoencoder.py        Trains the neural-codec baseline
    ├── make_test_video.py          Synthetic test clip generator
    └── (older ablation scripts: ablation_real.py, ablation_mask_modes.py, etc.)
```

Every module exposes a small public API documented in its docstring. `bg_fg_codec.py` is the integration surface for the headline codec; `pipeline.py` is the older sigmoid-blur path we keep as the prior baseline.

---

## How the codec works (30-second version)

1. **Sample 30 evenly-spaced frames** from the clip. Take the per-pixel temporal median — that's the background image. Short-lived foreground (people walking past) is rejected by the median.
2. **For each frame**, compute a saliency map from three combined signals: YOLOv8n detections (semantic — people, vehicles, bags, animals), spectral residual (low-level novelty), and motion against the background image. Temporal-smooth across an 11-frame window.
3. **Blend each frame against the background** using a sigmoid-shaped saliency mask, with a 0.25 floor so the original frame is never *fully* replaced. The blended stream goes to an unmodified `libx265` at the target CRF.
4. **The output is a normal H.265 mp4.** Non-salient regions are byte-identical across consecutive frames, so the codec's inter-frame prediction emits skip flags rather than residuals. All the bits move to the foreground.

See [`docs/architecture.md`](docs/architecture.md) for the diagrams.

---

## Honest limitations

The codec assumes a **stationary camera**. Pan/tilt/zoom footage breaks the temporal-median background model, and handheld VIRAT clips lose roughly half the size advantage. When a subject **fills most of the frame** (close-ups, dense two-person scenes), there's barely any background to replace and savings shrink toward zero. When a subject **stands still for >50% of the clip duration**, the median absorbs them into the background — the codec then "sees" them as part of the scene.

These are scope restrictions, not bugs. Most home surveillance deployments are stationary cameras watching mostly-empty spaces, which is exactly where the codec wins. The deck and the analysis call out all three failure modes explicitly.

We also report **saliency-weighted PSNR** (sal-PSNR), which only counts error inside regions our saliency map said matter. Plain PSNR penalises us for intentionally replacing background pixels — which is the *contract* of the codec, not a bug. See §7 of `analysis_v2.docx` for the metric defense.

---

## Conventions

- **Python 3.10+**, type hints, docstrings, no prints in library code (use `logging`).
- **No data committed.** `data/` and most of `results/` are gitignored — see `.gitignore` for exactly what's kept.
- **No model weights committed.** YOLOv8n auto-downloads on first call (~6 MB).
- **Imports.** Modules under `src/` use relative imports (`from .gate import ...`). Scripts under `scripts/` add the repo root to `sys.path` and use absolute imports.
- **Smart quotes are ASCII** in source code, fancy in writeups.

---

## License

[MIT](LICENSE) — feel free to learn from this, fork it, or build on it.

---

## Inspiration

Sebastian Lague, *AI Compression is 300× Better (but we don't use it)* — https://www.youtube.com/watch?v=i6l3535vRjA. The video's framing — neural codecs are dramatically better in theory but impractical at the edge — is the hinge of the project, and the reason our trained autoencoder is included as the *naïve* learned-codec baseline rather than the actual contribution.

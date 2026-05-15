# STEAM ICAC 2026 — CS Showcase

A perceptually-guided, sensor-gated compression pipeline for home-surveillance video. Our entry to the [STEAM Innovation Challenge Annual Conference 2026](https://steaminnovationchallenge.org/steam-icac-2026/), Computer Science prompt, Showcase format.

**Event:** May 28–29, 2026 · Hart House, University of Toronto.
**Submission deadline:** May 1, 2026.

## What to submit / bring

| When | What | Path |
|---|---|---|
| **By May 1, 2026** | Upload **`analysis.docx`** to the STEAM IC dropbox. That is the analysis — it is already formatted to the competition's spec (TNR 12, 1.15 spacing, 1" margins, 5 pages exactly). | [`analysis.docx`](analysis.docx) |
| **On May 28–29 (booth setup)** | Bring the laptop running the pipeline, a printed copy of the analysis, the poster, and have the deck loaded. | [`showcase.pptx`](showcase.pptx) · [`poster.pdf`](poster.pdf) |
| **During the 10-min slot** | Open the deck, run `python scripts/live_demo.py` for the live saliency overlay, and walk through Slides 7–11 with the booth poster taped behind the laptop. | — |

---

## What we're building

Three subsystems answering the three sub-questions of the CS prompt:

1. **Useful-footage gate** — multi-signal sensor fusion (motion + optical flow + on-device person detection + audio events) decides per frame whether something is "useful" enough to record at full fidelity.
2. **Saliency-aware perceptual compression** — a saliency map drives where bits go; non-salient regions are aggressively compressed, salient regions stay crisp.
3. **Neural-codec side-by-side** — a tiny conv autoencoder runs alongside the classical pipeline so we can directly answer the "AI compression is 300× better — why don't we use it?" question with our own numbers.

The whole thing runs on a laptop with a built-in webcam — no special hardware needed for the showcase booth.

---

## For new teammates: where to start

1. **Read [`docs/00-overview.md`](docs/00-overview.md)** — 3-minute orientation: competition context, what we're building, deadlines, rubric.
2. **Read [`PLAN.md`](PLAN.md)** — full project plan: architecture, scientific foundations, day-by-day timeline, rubric mapping, risks. ~25 minutes; this is the source of truth.
3. **Skim [`docs/research/`](docs/research/)** — four research briefs (saliency compression, HVS principles, neural codecs, event detection) gathered during planning. Lean on these when writing the analysis.
4. **Run the pipeline locally** — see "Quick start" below.
5. **Open a thread on what to pick up** — see "Where the help is needed" further down.

---

## Quick start

```bash
git clone https://github.com/SaiAmartya/steam-icac-2026.git
cd "steam-icac-2026"
```

Option 1: Use setup.sh file (macOS only)
```bash 
chmod +x setup.sh
./setup.sh                          # installs ffmpeg via Homebrew + creates .venv + pip install
source .venv/bin/activate
```

Alternative: Use `just setup` (cross platform)

```bash
just setup
```
You need to install [just](https://github.com/casey/just) for this to work

# 1a) Pull hundreds of real CCTV clips (stratified across 13 UCF-Crime classes)
```bash
python scripts/acquire_real_cctv.py --n-clips 400          # ~3 min, ~180 MB
# Output: data/real/clip_NNN.mp4 + data/real/manifest.json
```

The acquisition streams the public UCF-Crime mirror on Hugging Face straight through ffmpeg and writes 640×360 @ 30 fps libx264 CRF 18 clips — the same format the rest of the pipeline expects. Use `--n-clips 950` to pull the full benchmark, `--max-duration 30` for longer clips, `--clean` to start over.

# 1b) Or generate a 10-second synthetic clip (no network needed)
```bash
python scripts/make_test_video.py
```

# 2) Run the full pipeline
```bash
python scripts/run_pipeline.py --input data/real/clip_001.mp4 --out results --verbose
```

# 3) Live webcam demo (saliency overlay + gate HUD)

```bash
python scripts/live_demo.py
```

The pipeline writes:
- `results/ours_saliency.mp4` — our saliency-aware output.
- `results/baseline_uniform.mp4` — same-CRF H.265 baseline for comparison.
- `results/events.json` — every frame the gate triggered on.
- `results/metrics.json` — file sizes, PSNR, SSIM, compression ratio.

### Linux

The same commands work; replace the Homebrew step with `apt install ffmpeg`.

### Windows

Untested. Use WSL2 + Ubuntu or expect to figure out ffmpeg / libx265 yourself.

---

## Code map

```
STEAM IC/
├── PLAN.md                      Full project plan — read this first
├── README.md                    You're here
├── LICENSE                      MIT
├── requirements.txt
├── setup.sh                     One-shot macOS setup
│
├── docs/
│   ├── 00-overview.md           Quick orientation for new collaborators
│   └── research/                Four research briefs feeding the analysis
│       ├── 01-saliency-compression.md
│       ├── 02-hvs-principles.md
│       ├── 03-neural-codecs.md
│       └── 04-event-detection.md
│
├── src/
│   ├── gate.py                  Useful-footage detection (motion + flow + optional YOLO)
│   ├── saliency.py              Per-frame saliency map (spectral-residual default)
│   ├── compress.py              Saliency-aware compression (Tier C today; B/A planned)
│   ├── neural_codec.py          Tiny conv autoencoder — the "Option 1" side-by-side
│   ├── metrics.py               PSNR / SSIM / LPIPS wrappers
│   └── pipeline.py              End-to-end orchestrator
│
└── scripts/
    ├── make_test_video.py       Synthetic test clip generator
    ├── run_pipeline.py          CLI entry point
    └── live_demo.py             Webcam demo with live HUD
```

Every module exposes a small public API documented in its docstring; the `pipeline.py` orchestrator is the integration surface.

---

## Status (as of 2026-04-30, 7 days into the build)

**Deliverables ready for the May 1 submission and the May 28–29 booth:**

| File | What it is |
|---|---|
| [`analysis.docx`](analysis.docx) | The 5-page analysis for the dropbox submission. TNR 12, 1.15 spacing, 1" margins. |
| [`showcase.pptx`](showcase.pptx) | The 12-slide deck for the 10-minute live showcase. |
| [`showcase.pdf`](showcase.pdf) | PDF preview of the deck (for fallback / printing). |
| [`poster.pdf`](poster.pdf) | Single-page printable booth poster (A3 landscape). |
| [`poster.png`](poster.png) | PNG version of the poster. |

**Headline measured numbers** (real CCTV, 20 stratified clips across 13 action classes from the public Kaggle CCTV Action-Recognition dataset):

- **+0.44 dB sal-PSNR vs uniform H.265 at iso-bitrate (~45 KB)** — saliency-aware compression matches or beats baseline below the ~50 KB crossover, exactly the regime edge surveillance actually operates in.
- **+6.6 dB sal-PSNR** from the sigmoid-mask methodology over the legacy soft alpha-blend (mask-mode ablation).
- **65.2% gate trigger ratio** on real CCTV — sensor fusion validated across all 13 action classes.
- **76,131-parameter TinyAutoencoder** trained on 1,997 frames sampled from the full 2,288-clip dataset; **1.43 ms encode / 1.35 ms decode** on Apple M3 Pro MPS; 28.95 dB reconstruction PSNR at 6× compression — first-person edge measurement that quantifies the "AI compression is 300× better but we don't use it" claim.
- **>90 fps end-to-end** on a laptop CPU — real-time-capable.
- **23% file-size reduction at fixed CRF**; ~32 kg CO₂ per camera per year, ~32 tonnes annually for a 1,000-camera deployment.

Full results tables live in [`docs/results_summary.md`](docs/results_summary.md) — the single source of truth that the analysis, deck, and poster all read from. Raw rows: `results/ablation_real/`, `results/ablation_sigmoid/`, `results/ablation_mask/`, `results/neural_codec/`.

**Honest limitations** (called out in §8 of the analysis):
- The saliency model is classical (spectral-residual). A learned saliency CNN (TASED-Net) would tighten where the salient-region mask lands.
- Tier-C (pre-blur) is the implemented compression path. Tier-A (true per-block QP via libx265's ROI API) would avoid the LPIPS penalty for non-salient blur.
- LPIPS is whole-frame; a saliency-weighted LPIPS is the natural next perceptual metric.
- No Pi-4 first-person edge measurement yet — autoencoder latency on Apple Silicon (1.43 ms) is the current data point.

**See `PLAN.md` §10 for the original day-by-day timeline.**

---

## Where the help is needed

If you've just landed on the project and want to pick something up, here are the high-leverage things:

| Stream | Skill needed | Effort | What it unblocks |
|---|---|---|---|
| Gate tuning on real footage | OpenCV familiarity | half a day | Right MOG2 / flow thresholds + small hand-labelled set for fusion-weight learning. |
| Compression Tier B upgrade | ffmpeg + Python | 1 day | Per-frame average-QP via `--qp-file`, replacing the Tier C blur fallback. |
| Neural codec training | PyTorch | 1 day | Real numbers for the side-by-side comparison: encode/decode latency, PSNR vs file size on a few hundred real frames. |
| Fractyl3D architecture diagram | drawing chops | half a day | Hits the Analysis & Presentation criterion explicitly. |
| Analysis document drafting | scientific writing | 2 days | The actual deliverable. 5 pages, structure already laid out in `PLAN.md §11`. |
| Showcase rehearsal & poster | presentation chops | post-May 1 | The 7 model points hinge on a clean live demo. |

If you're not sure where to slot in, ping Sai and we'll match you to whatever's the bottleneck that day.

---

## Conventions

- **Python 3.10+**, type hints, docstrings, no prints in library code (use `logging`).
- **No data committed.** `data/` and `results/` are gitignored.
- **No model weights committed.** They're either downloaded by the user or trained locally. Keeps the repo small.
- **Imports.** Modules under `src/` use relative imports (`from .gate import ...`). Scripts under `scripts/` add `src/` to `sys.path` and use absolute imports.
- **PRs.** Small and focused. One logical change per branch. Reference the `PLAN.md` section your change touches in the PR description.

---

## License

[MIT](LICENSE) — feel free to learn from this, fork it, or build on it.

---

## Inspiration

Sebastian Lague, *AI Compression is 300× Better (but we don't use it)* — https://www.youtube.com/watch?v=i6l3535vRjA. The video's framing (neural codecs are dramatically better in theory but impractical at the edge) is the hinge of the project.

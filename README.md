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

# 1) Generate a 10-second synthetic surveillance clip
python scripts/make_test_video.py

# 2) Run the full pipeline
python scripts/run_pipeline.py --input data/test.mp4 --out results --verbose

# 3) Live webcam demo (saliency overlay + gate HUD)
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

**Headline measured numbers** (from `results/ablation/rows.jsonl`):

- **23.4% file-size reduction** at fixed CRF 28 vs uniform H.265 baseline
- **40.2% gate trigger ratio** — exactly matches the scripted activity in the test scene
- **>90 fps processing throughput** on a laptop CPU — well above real-time
- **32 kg CO₂ saved per camera per year** at typical 1.5 Mbps baseline
- **32 tonnes CO₂ saved annually** at 1,000 deployed cameras

The full ablation covers 4 baseline H.265 conditions and 6 saliency-aware configurations; raw rows are in `results/ablation/rows.jsonl`. Figures used in the analysis and deck are in `results/figures/`.

**Honest limitations** (called out in §8 of the analysis):
- All numbers from a synthetic indoor surveillance scene. Public datasets (VIRAT, Avenue) couldn't be retrieved from the development environment used to ship this submission; testing on them is the next step.
- Perceptual metrics (VMAF, LPIPS) are not yet wired into the harness. Pixel PSNR/SSIM are reported but mechanically penalise our approach (this is itself a discussion point in the analysis).
- The TinyAutoencoder neural codec is implemented and parses cleanly; live training and benchmark on real frames is the next experiment.
- Edge-device latency is literature-estimated, not directly measured on a Pi 4.

**See `PLAN.md` §10 for the original day-by-day timeline.**

---

## Where the help is needed

If you've just landed on the project and want to pick something up, here are the high-leverage things:

| Stream | Skill needed | Effort | What it unblocks |
|---|---|---|---|
| Real footage acquisition | none — just downloading and labelling | half a day | Replaces synthetic clip; unblocks all real numbers in the analysis. VIRAT or CMU Avenue dataset. |
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

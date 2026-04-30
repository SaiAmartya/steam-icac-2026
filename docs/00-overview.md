# Project Overview

## The competition
[STEAM ICAC 2026](https://steaminnovationchallenge.org/steam-icac-2026/) — high-school innovation conference, May 28–29, 2026 at Hart House, University of Toronto.

We're entering the **Computer Science prompt** in the **Showcase format**:

> Over 95% of home security camera video surveillance is never viewed, and the majority of footage only wastes storage and energy. How can stored video be compressed according to principles of human perceptual science to reduce storage needs without compromising perceived quality? How can "useful" footage be automatically identified using sensor data from the camera system or nearby devices? What algorithms and edge-processing techniques make this system scalable, privacy-conscious, and precise?

## Our answer
A **perceptually-guided, sensor-gated compression pipeline** that:

1. Uses a multi-signal sensor gate (motion + optical flow + on-device person detection + audio events) to decide when footage is "useful" enough to record at full fidelity.
2. Applies **saliency-aware perceptual compression** to every stored frame — bits go where a human's eyes would actually look, the rest is aggressively compressed.
3. Ships a side-by-side **neural-codec prototype** (a tiny conv autoencoder) so we can directly answer the inspiration video's question — "AI compression is 300× better, but we don't use it" — with real numbers showing why a hybrid is the realistic answer for edge hardware today.

## Where things live in this repo
- [`PLAN.md`](../PLAN.md) — the full project plan: architecture, scientific foundations, timeline, rubric mapping, risks. **Read this first.**
- [`README.md`](../README.md) — quick-start and code map for engineers landing on the repo.
- [`docs/research/`](research/) — research briefs gathered during planning. Useful when you're writing the analysis sections.
- [`src/`](../src/) — implementation.
- [`scripts/`](../scripts/) — CLI entry points (test-video generator, pipeline run, live demo).

## The deadlines that matter
- **2026-05-01** — analysis document (max 5 pages, Times New Roman 12, 1.15 spacing) submitted to the dropbox.
- **2026-05-28 / 29** — showcase event at Hart House.

## How we're scoring (30 pts)
| Criterion | Points | Where we earn them |
|---|---|---|
| Analysis & Presentation | 5 | Fractyl3D diagrams, 10-min showcase, fallback video |
| Model | 7 | Laptop + webcam running the live pipeline + tri-fold poster |
| Technology | 7 | Live saliency overlay, real-time event gate, neural-codec comparison |
| Scientific Application & Innovation | 6 | HVS section + citations + ablation study |
| Sustainability | 5 | Storage/bandwidth/energy savings, quantified |

See `PLAN.md §12` for the detail.

# Research brief: neural video codecs and why they don't ship to the edge

> **Note:** This brief feeds the "why not just AI?" page of the analysis. Numbers below are order-of-magnitude estimates from secondary sources; verify exact figures before printing them in the report.

## Inspiration
Sebastian Lague, *AI Compression is 300x Better (but we don't use it)* — https://www.youtube.com/watch?v=i6l3535vRjA. The video's thesis: autoencoder-based compression can hit dramatically smaller file sizes than H.264/H.265 in theory, but compute cost and lack of hardware acceleration keep it out of production.

## State of the art (2019–2025)

| System | Year | Reported gain vs H.265 | Notes |
|---|---|---|---|
| **DVC** (Lu et al.) | 2019 | ~+0.15–0.5 dB at 0.5–2 Mbps | First end-to-end learned video codec. |
| **HiFiC** (Mentzer et al.) | 2020 | near-imperceptible at 0.1–1 bpp | Image-focused; GAN-based. |
| **ELIC** (He et al.) | 2022 | 15–25% rate-distortion gain | Image, not video, but architecturally relevant. |
| **VCT** (Mentzer et al.) | 2022–2023 | competitive with HEVC | 500–2000 ms/frame on GPU; not edge-deployable. |
| **Recent transformer-based codecs** | 2024–2025 | 5–15% BD-rate over H.265 | Gains collapse below ~256 kbps and on high-motion footage. |

Public codebase: **CompressAI** by InterDigital — https://github.com/InterDigitalInc/CompressAI. Research framework, not a deployment-ready video codec equivalent to libx264/libx265.

## Why pure neural codecs don't deploy at the edge

**Inference latency:** 100–2000 ms per frame depending on architecture. Real-time (33 ms / frame at 30 fps) is the requirement. H.265 software decode achieves ~10 Mbps on a Raspberry Pi 4; full neural codec achieves <1 Mbps.

**Model size and memory:** weights typically 50–300 MB. Pi 4 has ~1 GB RAM; a single codec consumes 5–30% of available memory before the OS, camera driver, or any other application is even loaded.

**Memory bandwidth:** neural codecs require ~8–12 GB/s for batched feature processing and entropy table access. Edge SBCs deliver a fraction of that — bandwidth becomes the bottleneck before compute does.

**Hardware acceleration gap:** every smartphone, security camera, and modern SBC has dedicated H.264/H.265 silicon. **Zero** consumer hardware has a dedicated neural-codec block. So the comparison is "dedicated silicon vs general-purpose CPU" — the dedicated silicon wins by orders of magnitude.

**Training data and overfitting:** learned codecs trained on diverse datasets generalise poorly to narrow domains (e.g., static night-mode security footage). Per-domain retraining is a deployment nightmare.

**Latency tail:** entropy encoding in neural codecs adds 50–500 ms (iterative arithmetic coding). H.265 in real-time mode encodes in <33 ms.

## The middle ground (what we're actually doing)

Hybrid approaches that take some of the AI advantage without paying the full neural-codec cost:

- **Learned pre/post processing** — small CNN denoisers/enhancers around standard H.265.
- **AI-driven rate control** — saliency or attention networks predict spatial importance; H.265 ROI bit allocation. Reported gains 10–20% over fixed-QP H.265.
- **Perceptual loss-guided rate-distortion** — perceptual metrics shape the standard codec's optimisation without changing its architecture.
- **Locality-aware compression** — tiny edge detectors flag people/vehicles; H.265 ROI focuses bits there.

**Industry practice:** Wyze, Eufy, Amcrest, Reolink — all use H.265 + on-device motion detection + selective cloud upload. None of them deploy neural codecs at scale.

## Privacy angle

GDPR / CCPA push toward on-device processing for surveillance video. Raw video transmission is a high-risk operation; on-device compression + selective upload of "useful" clips is the compliant pattern. Neural codecs *could* learn task-specific compression (e.g., compress for person-detection downstream) but the training-data leakage and complexity costs outweigh the benefit vs H.265 + explicit saliency masking.

## Citeable references
- Lu, G. et al. (2019). DVC: an end-to-end deep video compression framework. *CVPR*.
- Ballé, J. et al. (2018). Variational image compression with a scale hyperprior. *ICLR*.
- He, D. et al. (2022). ELIC: efficient learned image compression. *CVPR*.
- Mentzer, F. et al. (2020). High-fidelity generative image compression. *NeurIPS*.
- CompressAI: https://github.com/InterDigitalInc/CompressAI

## How we use this in the analysis
One full page, structured as:
1. **What neural codecs achieve** (the 300× headline + caveat).
2. **Why they don't ship to the edge** (the four numeric reasons above).
3. **Our bridge** — borrow the perceptual-allocation philosophy, run on classical rails, pay milliseconds instead of seconds.

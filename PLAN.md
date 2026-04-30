# STEAM ICAC 2026 — Computer Science Showcase
## Project Plan: A Perceptually-Guided, Sensor-Gated Compression Pipeline for Home Security Cameras

**Author:** Sai Amartya (solo entry)
**Format:** Showcase (CS prompt)
**Analysis due:** May 1, 2026
**Event:** May 28–29, 2026, Hart House, UofT

---

## 1. Executive summary

More than 95% of home-security footage is never watched. It still costs storage, energy, bandwidth, and data-center capacity. The CS prompt asks three connected questions: how to compress video using principles of human perception, how to automatically identify which footage is actually "useful," and what algorithms make such a system scalable, privacy-conscious, and precise on edge hardware.

This project answers all three with one coherent system: a real-time video pipeline that (a) uses a lightweight multi-sensor gate — motion, optical flow, on-device object detection, and audio-event detection — to decide when footage is "useful," and (b) applies *saliency-aware perceptual compression* to every frame so that bits go where a human would actually look. The demo runs entirely in software on a laptop, uses only open-source and pretrained components, and ships with a measurable bitrate saving versus a standard H.265 baseline on real surveillance footage.

A one-page "why not a pure neural codec?" analysis — grounded in Sebastian Lague's "AI Compression is 300× Better" video and the underlying literature — explains why the hybrid approach is the realistic answer for edge deployment today, scoring the Scientific Application & Innovation criterion without having to ship an unrealistic system.

---

## 2. Why this project fits the prompt

The prompt asks three sub-questions. Each has a dedicated subsystem:

| Sub-prompt | Our answer |
|---|---|
| Compress video using principles of human perceptual science | Saliency-aware region-adaptive quantization, layered on H.265 |
| Automatically identify "useful" footage using sensor data | Multi-modal fusion gate: motion + optical flow + object detection + audio events |
| Algorithms & edge-processing techniques that are scalable, privacy-conscious, precise | Classical codec + tiny ML models, all on-device, no raw video leaves the camera |

Because the prompt is phrased as three *questions*, the judges are effectively a rubric: a high-scoring analysis will visibly answer each one. The architecture is designed so each block in the pipeline maps directly to one of them.

---

## 3. The core technical idea

### 3.1 One-sentence pitch
A home-security camera that only records at full fidelity when its sensors agree something interesting is happening, and even then compresses each frame by the rules of human vision rather than uniformly — cutting storage by roughly an order of magnitude over naive recording, at near-zero extra compute cost.

### 3.2 System architecture

```
        ┌──────────────┐
        │ Camera input │  (webcam or public surveillance clips)
        └──────┬───────┘
               │
               ▼
     ┌──────────────────────┐
     │  A. Useful-footage   │  <- MOG2 / optical flow (always on, cheap)
     │     gating layer     │  <- YOLOv8n person/vehicle detector (triggered)
     │                      │  <- YAMNet audio event detector (triggered)
     │                      │  -> usefulness score [0..1]
     └──────────┬───────────┘
                │
       ┌────────┴────────┐
       │                 │
  score ≥ τ            score < τ
       │                 │
       ▼                 ▼
 ┌──────────────┐  ┌────────────────┐
 │ B. Full-     │  │ D. Timelapse / │
 │    fidelity  │  │    sparse KF   │
 │    record    │  │                │
 └──────┬───────┘  └────────┬───────┘
        │                   │
        ▼                   ▼
 ┌─────────────────────────────────┐
 │ C. Saliency-aware perceptual    │
 │    compression core             │
 │   (runs on EVERY stored frame)  │
 │                                 │
 │   tiny saliency CNN ──▶ QP map  │
 │   H.265 ROI encode (libx265)    │
 └─────────────┬───────────────────┘
               │
               ▼
         compressed .mp4
         + event log (JSON)
```

### 3.3 What makes each block scientific, not ad-hoc

- **The gating layer** is a sensor-fusion classifier. Every signal is cheap enough to run continuously on a laptop CPU. A weighted sum + threshold is the baseline; a tiny logistic regression over fusion features is the stretch goal. This is the "useful footage" answer.
- **The compression core** is where the human-visual-system science lives. Saliency is a proxy for gaze. Gaze is where distortion hurts. Bits go there; bits elsewhere are wasted. This is grounded in Itti & Koch 1998 (saliency), Campbell & Robson 1968 (CSF), Mannos & Sakrison 1974 (JND), and the HEVC perceptual-coding literature — real citations, not hand-wavy.
- **The "what didn't we do"** section — pure neural codecs — gets its own page. We summarise the state of the art (DVC, ELIC, VCT), the reasons they don't deploy to edge (compute, memory, no silicon acceleration), and explain that our hybrid *takes the perceptual-weighting philosophy of neural codecs and realises it on classical rails*. That framing directly addresses the inspiration video and gets us serious "Scientific Application & Innovation" credit.

---

## 4. Scientific foundations (the HVS story)

The analysis document should dedicate ~1 page to this. Judges reward real scientific grounding over engineering narrative. The five principles below are each citeable from primary sources:

1. **Foveation.** The human fovea has ~10× the spatial resolution of peripheral retina. Visual acuity falls off predictably with eccentricity. *Wandell 1995 (Foundations of Vision); Geisler & Perry 1998 (foveated multiresolution pyramids).*

2. **Contrast Sensitivity Function (CSF).** The eye is most sensitive to mid spatial frequencies (~4 cpd), less to very low or very high frequencies. JPEG's quantization matrix was literally derived from the CSF. *Campbell & Robson 1968; Watson & Ahumada 2005.*

3. **Just-Noticeable-Difference (JND).** There is a distortion threshold below which error is invisible. Modern codecs tune quantization to stay under it. *Mannos & Sakrison 1974; Yang et al. 2005.*

4. **Spatial & temporal masking.** Distortion hides in high-activity regions (textured areas, edges, motion). HEVC and AV1 use local-activity metrics to modulate QP accordingly. *Winkler 2005; Sullivan et al. 2012 (HEVC overview).*

5. **Saliency & attention.** Attention maps correlate strongly with gaze data (DHF1K, SALICON datasets). Distortion in attended regions is more noticeable. *Itti, Koch & Niebur 1998; Borji & Itti 2013.*

Our compression strategy uses (5) as the spatial *mask* and (3)(4) to decide how aggressively to quantize outside the mask. The evaluation uses HVS-grounded metrics — MS-SSIM, PSNR-HVS-M — rather than raw PSNR.

---

## 5. Subsystem details

### 5.1 Useful-footage gating layer

A four-signal fusion pipeline. All components are open-source and run on laptop CPU:

| Signal | Library / model | Approx. CPU cost | What it's good at |
|---|---|---|---|
| Frame differencing / MOG2 background subtraction | OpenCV `cv2.createBackgroundSubtractorMOG2` | <10 ms / frame at 480p | Any pixel-level change; runs always-on |
| Sparse optical flow (Lucas-Kanade) | OpenCV | ~5–10 ms / frame | Distinguishes real motion from lighting flicker |
| Tiny object detector (person / vehicle / package) | YOLOv8n (Ultralytics), INT8 | ~40–60 ms / frame on laptop CPU | Semantic "is this a person" gate; triggered only by motion |
| Audio event detection | YAMNet (TensorFlow, pretrained on AudioSet) | ~50 ms / 1 s audio window | Glass break, dog bark, doorbell, shouting |

**Fusion rule (baseline):** a weighted sum of normalised confidences with a tunable threshold `τ`. Written as:

```
usefulness = 0.25·motion + 0.25·flow + 0.30·person + 0.20·audio_event
record_full_fidelity = usefulness > τ
```

**Stretch goal:** train a 5-feature logistic regression on a small hand-labeled dataset of ~30 min of home-security clips to learn the weights and threshold rather than hand-tuning them. This is the "scalable, precise" part of the prompt — we're not just using a constant, we're learning the fusion policy.

**Privacy:** all inference happens on-device. The event log contains only semantic labels and timestamps (`{"t": 1716000000, "event": "person", "conf": 0.82}`), never raw frames. This is the industry pattern already used by Wyze, Eufy, Amcrest — we just make it explicit in the analysis.

### 5.2 Saliency-aware perceptual compression core

**Step 1 — saliency inference.** Use a lightweight pretrained saliency model. TASED-Net is the obvious first pick (small, PyTorch, good documentation), with a BASNet fallback if inference is too slow. Both output a 0–255 saliency heatmap per frame. Smooth temporally with a 5-frame moving average to avoid flicker between consecutive frames' QP maps.

**Step 2 — saliency → QP map.** Normalize the saliency map, resize it to match the codec's coding-tree-unit (CTU) grid (16×16 for H.264, 64×64 for H.265), then map:
```
QP_block = QP_baseline − Δ · saliency_block_normalised
```
With `QP_baseline = 28` and `Δ = 6`, high-saliency regions get QP=22 (crisp), low-saliency get QP=34 (aggressive). The range is tunable and will be ablated in the evaluation.

**Step 3 — encode with per-region QP.** Three possible routes, in descending order of fidelity-to-the-idea / ascending order of ease:

- **(a) Real ROI encoding in x265.** libx265 accepts per-block QP offsets via its C API. Harder to wire up, but faithful to the pitch. A Python wrapper like `python-libx265` can bridge it; expect some yak-shaving.
- **(b) Per-frame average QP in x264.** ffmpeg has `--qp-file` for per-frame QP. We modulate average QP based on frame-level saliency mean. Loses spatial granularity but still shows the principle end-to-end. *Recommended day-1 baseline.*
- **(c) Software fallback: saliency-weighted spatial blur + standard encode.** If neither codec integration works in time, pre-blur non-salient regions before a uniform-QP encode. This is strictly worse than true ROI but is bulletproof in under 30 lines of OpenCV.

Plan: build (c) first (end-to-end in a day), then move to (b), then push to (a) only if time allows. The analysis document discusses (a) as the principled approach regardless.

### 5.3 The neural-codec comparison (Option 1 graft)

A single page in the analysis, structured as:

- **What neural codecs achieve.** DVC (Lu et al. 2019), ELIC (He et al. 2022), VCT (2023). Typical gains: ~5–15% BD-rate improvement over HEVC in controlled benchmarks; HiFiC-style GANs achieve near-imperceptible quality at very low bitrates on single images.
- **Why they don't ship to the edge.** Four numeric reasons:
  - Inference latency: 100–2000 ms/frame on a laptop CPU vs ~33 ms for real-time H.265.
  - Model size: 50–300 MB of weights, vs tens of KB for codec tables.
  - Memory bandwidth: neural codecs need several GB/s; edge SBCs have a fraction of that.
  - Zero hardware acceleration: every phone and camera SoC has a dedicated H.264/H.265 block; none have a neural-codec block.
- **Our bridge.** We borrow the *idea* behind neural codecs — bits should be allocated to what humans actually perceive — and implement it on classical rails using a tiny saliency CNN guiding a production codec. We pay milliseconds, not seconds.

This section is what turns a "nice CS project" into one that engages with the research frontier. It's also the section most directly tied to the inspiration video.

---

## 6. Privacy-conscious design — the story we tell

The prompt explicitly asks about privacy. Our answers:

1. **All inference is on-device.** Nothing leaves the camera except optionally the event log.
2. **Selective upload.** Only clips flagged as "useful" with high confidence are ever transmitted; silent hours stay local.
3. **Optional face blurring on decode.** Adds a post-processing step during playback using OpenCV face detection + Gaussian blur — demonstrable in the showcase.
4. **Event log is semantic, not photographic.** No raw pixel data is persisted in the log; only labels and timestamps.
5. **Federated-ready architecture.** Because the detector models are small and local, the system is compatible with federated fine-tuning (each home improves the model without sharing video). We note this as future work rather than implementing it.

The Sustainability criterion is tied in here too: less upload = less data-center energy = direct carbon reduction.

---

## 7. Technology stack

Everything is Python, all dependencies are pip-installable, all models are pretrained and openly licensed.

| Layer | Stack |
|---|---|
| Video I/O | `opencv-python`, `ffmpeg-python` (wrapper over system ffmpeg with libx264 / libx265) |
| Saliency | PyTorch + TASED-Net (pretrained weights), fallback: MobileNet-based saliency decoder |
| Object detection | Ultralytics `yolov8n.pt` (3.2 MB) |
| Audio events | TensorFlow Lite + YAMNet (`saved_model.pb`, ~3.7 MB) |
| Motion / flow | OpenCV MOG2, Lucas-Kanade |
| Metrics | `scikit-image` (PSNR, SSIM, MS-SSIM), `lpips` pip package, `libvmaf` CLI |
| Glue | Numpy, SciPy, small FastAPI/CLI wrapper so the demo is one command |
| Visualization | Matplotlib for metric plots; OpenCV overlays for live saliency heatmap during the demo |

All diagrams and architecture figures in the analysis document should be built in **Fractyl3D.com** — the competition guide explicitly rewards this ("thorough planning with diagrams & detailed explanations using STEAM IC's application").

---

## 8. Physical booth and showcase plan

The guide says a model must fit in 50×50×50 cm. Since this is CS, the "model" is a laptop running the live pipeline. Concretely:

- **On the table:** a laptop running the demo, a USB webcam (you likely already have one or can borrow one — avoids the hardware spend), a printed Fractyl3D system diagram poster (tabletop tri-fold if you want to hit the trifold visual-aid pattern from the guide).
- **The live demo loop (10 min including setup and Q&A, so ~6–7 min of content):**
  1. Point the webcam at the judge. The live saliency heatmap overlays their face — it lights up. (~30 s, hooks attention)
  2. Walk through one surveillance clip side-by-side: raw vs. saliency-compressed, with a real-time metric readout. Highlight that file size dropped Xx while MS-SSIM stayed ≥ 0.95. (~2 min)
  3. Show the sensor gate triggering: wave, clap, silence. The recording indicator turns on/off; event log scrolls. (~1.5 min)
  4. Pull up the slide/poster that compares to a pure neural codec. Show the latency/model-size delta. (~1.5 min)
  5. Close with sustainability: energy saved per-camera-per-year extrapolation, small bar chart. (~30 s)
- **Fallback:** pre-recorded video of the same demo on the laptop in case the webcam or audio misbehaves in the competition room. Always have the fallback ready.

Dress: business formal per the guide.

---

## 9. Evaluation plan (how we prove it works)

Evaluation drives both the analysis writing and the showcase visuals, so it's worth doing properly.

**Datasets.** Two sources:
- VIRAT or Avenue public surveillance datasets (freely available research datasets with realistic home / street footage).
- A handful of self-recorded clips with your webcam, labelled with ground-truth "useful / not useful" timestamps.

**Metrics:**
- **Compression metrics:** bitrate (Mbps), file size ratio vs H.265-at-QP-28 baseline.
- **Perceptual quality:** MS-SSIM, LPIPS, VMAF for a subset of clips.
- **Gate performance:** precision / recall / F1 of "useful footage" detection against your hand-labels.
- **System metrics:** end-to-end latency per frame, peak RAM, CPU utilization — proves the edge-feasibility claim.

**Comparisons:**
1. Baseline: uniform H.265 at QP=28.
2. Gate-only: record nothing outside useful windows, full-quality inside.
3. Saliency-only: saliency-aware QP on every frame, no gating.
4. **Ours: gate + saliency combined.**

Expected outcome: (4) beats (1) by 5–10× in storage and beats (2)(3) on the quality–storage Pareto frontier. Even if numbers come in modest, the comparative story is judge-friendly.

---

## 10. Timeline to May 1 submission

Today is Apr 24, 2026. 7 days. Aggressive but doable because every piece below is using pretrained components or a few lines of OpenCV.

| Day | Focus | Deliverable |
|---|---|---|
| **Day 1 (Fri Apr 24)** | Scope lock + environment setup. Review & approve this plan. Set up Python env, install dependencies, clone TASED-Net, confirm YOLOv8n and YAMNet load and run on one test frame. | Working dev environment, single-frame smoke tests of all models. |
| **Day 2 (Sat Apr 25)** | End-to-end v0: cheapest possible path (software blur fallback) running on one clip. Gate uses motion + person-detection only. | One command produces `out.mp4` smaller than `in.mp4` with visible quality preserved on faces. |
| **Day 3 (Sun Apr 26)** | Upgrade to per-frame average-QP encode (x264 via ffmpeg). Wire in audio event detection. Begin collecting metrics on 3–5 test clips. | v0.5 pipeline + first CSV of metrics. |
| **Day 4 (Mon Apr 27)** | Ablation study: vary threshold τ, vary saliency QP range Δ, vary baseline QP. Plot Pareto curves. Draft Fractyl3D architecture diagram. | Plots + diagram ready for analysis. |
| **Day 5 (Tue Apr 28)** | Draft full 5-page analysis. Structure per §11. First complete pass through every section. | 5-page draft of analysis.docx. |
| **Day 6 (Wed Apr 29)** | Revise analysis (Fractyl3D diagrams polished, references properly formatted, numbers consistent). Record screencast of live demo as a backup. Optional: attempt real x265 ROI encoding if time. | Analysis v2 + demo screencast. |
| **Day 7 (Thu Apr 30)** | Final proofread. Re-run all metrics to confirm consistency with the document. Submit to dropbox by end of day to buffer against May 1 cutoff. | **Submitted analysis.** |

**Between May 1 and May 28:** dedicated time for showcase-specific prep — polishing the live demo script, rehearsing to 10-minute timing, printing the Fractyl3D poster, stress-testing the fallback screencast, prepping answers for likely judge questions.

---

## 11. Structure of the 5-page analysis document

The guide caps the analysis at 5 pages, Times New Roman 12, 1.15 spacing, 1" margins. Rough layout:

| Page | Section | Notes |
|---|---|---|
| 1 | Title + problem statement + key insight (the 95%-unwatched stat framed as an engineering opportunity) | Hook fast. Set up the three sub-questions verbatim. |
| 2 | Scientific foundation: HVS principles, saliency, perceptual metrics | 5 principles from §4, each in ~3 sentences + citation. |
| 3 | System architecture: useful-footage gate + saliency compression core. One Fractyl3D diagram. | Answers sub-questions 2 and 3. |
| 4 | The neural-codec comparison + privacy + sustainability angle | Answers the "why not just AI?" question. Finish with the sustainability extrapolation. |
| 5 | Evaluation results + discussion + references | Table of metrics, one Pareto plot, honest discussion of limitations, APA-7 references. |

A small appendix can carry additional plots and a more detailed metric table — the guide puts appendices outside the page limit for the written format; less clear for the showcase analysis, so keep appendix minimal and stuff everything real into the 5 pages.

---

## 12. Criteria alignment (how we cover the 30-point rubric)

The showcase rubric totals 30 points across five criteria. Mapping each block of this plan to where it scores:

| Criterion (pts) | What the rubric asks | Where we score |
|---|---|---|
| Analysis & Presentation (5) | Clear structure, diagrams, engagement, pacing | Fractyl3D diagrams + 10-min showcase + eye contact + fallback video |
| Model (7) | Precision, functional tech components, within 50×50×50 | Laptop + webcam running live pipeline; tri-fold poster |
| Technology (7) | Innovative tech integration, functional, responsive, interactive, future-ready | Live saliency overlay + real-time event gate + neural-codec comparison as forward-looking angle |
| Scientific Application & Innovation (6) | Relevant science thoroughly explained, well-researched, original | HVS section (§4) + citations + ablation study |
| Sustainability (5) | Resource efficiency, minimized ecological footprint | The whole premise — less storage, less bandwidth, less data-center energy — quantified with a back-of-envelope extrapolation |

Notably, *functional technological components* show up in 4 of the 5 criteria. The live demo is our single highest-leverage asset, which is why it's worth spending day 6–May 28 polishing it.

---

## 13. Risks and fallbacks

| Risk | Likelihood | Mitigation |
|---|---|---|
| TASED-Net inference too slow on your laptop | Medium | Fall back to a smaller model (MobileNet-based saliency), or drop to 5 fps saliency and interpolate between frames. |
| libx265 ROI API too fiddly to wire in time | High | Day-2 path is per-frame average QP in x264; day-1 fallback is software-side blur before uniform encode. Both answer the principle. |
| YAMNet TF dependency breaks the environment | Medium | Remove audio gate entirely; the gate still works on motion + person detection. Audio becomes "future work." |
| Metrics come in flatter than hoped (e.g., only 20% file-size saving) | Low–medium | Frame the win honestly. 20% is still huge at data-center scale. The judge cares about the reasoning, not a 10× number. |
| Live demo fails on event day (webcam, audio, power) | Medium | Pre-recorded screencast on the laptop, ready to play in the same UI. Practice switching to it in under 10 seconds. |
| Analysis goes over 5 pages | High | Write a 6-page draft day 5, cut day 6. Appendix-ify anything non-essential. |
| Submission portal closes early on May 1 | Low | Submit end-of-day **April 30** at the latest. |

---

## 14. Key references (partial — will be finalized in the analysis)

*Human visual system & perceptual compression:*
- Campbell, F. W., & Robson, J. G. (1968). Application of Fourier analysis to the visibility of gratings. *J. Physiol.*, 197(3).
- Mannos, J. L., & Sakrison, D. J. (1974). The effects of a visual fidelity criterion on the encoding of images. *IEEE TIT*, 20(4).
- Itti, L., Koch, C., & Niebur, E. (1998). A model of saliency-based visual attention for rapid scene analysis. *IEEE TPAMI*, 20(11).
- Wang, Z., Simoncelli, E. P., & Bovik, A. C. (2003). Multiscale structural similarity for image quality assessment. *Asilomar*.
- Yang, X. et al. (2005). Just-noticeable-distortion model and its applications in video coding. *Signal Processing: Image Communication*, 20(7).
- Sullivan, G. J. et al. (2012). Overview of the High Efficiency Video Coding (HEVC) standard. *IEEE TCSVT*, 22(12).
- Wandell, B. A. (1995). *Foundations of Vision*. Sinauer.

*Neural video compression:*
- Lu, G. et al. (2019). DVC: An end-to-end deep video compression framework. *CVPR*.
- Ballé, J. et al. (2018). Variational image compression with a scale hyperprior. *ICLR*.
- He, D. et al. (2022). ELIC: Efficient learned image compression. *CVPR*.
- CompressAI (InterDigital): https://github.com/InterDigitalInc/CompressAI

*Evaluation metrics:*
- Li, Z. et al. (2016). Toward a better quality metric for the video community (VMAF). *SMPTE Motion Imaging Journal*.
- Zhang, R. et al. (2018). The unreasonable effectiveness of deep features as a perceptual metric (LPIPS). *CVPR*.
- Chandler, D. M., & Hemami, S. S. (2007). VSNR / PSNR-HVS variants. *IEEE TIP*, 16(10).

*Detection models we'll actually use:*
- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
- YAMNet: https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
- TASED-Net (check current canonical repo before citing exact URL).

*Inspiration:*
- Lague, S. (2025). *AI Compression is 300x Better (but we don't use it).* YouTube. https://www.youtube.com/watch?v=i6l3535vRjA

---

## 15. What I need from you before we build

Before moving from plan to build, a few decisions worth locking:

1. **Do you have a webcam** (laptop built-in is fine) and a mic? If not, we adapt the demo to pre-recorded clips only.
2. **Operating system / Python environment:** macOS, Linux, Windows? Affects a couple of ffmpeg / libx265 install paths.
3. **Any dataset preference?** I lean toward self-recording ~10 minutes of "empty room" + ~5 minutes of "someone walking in/out" as the honest test set, plus a public dataset for numbers.
4. **Option 1 graft depth:** are you happy with a written-only comparison to neural codecs (my recommendation given the timeline), or do you want to attempt a tiny autoencoder in code for side-by-side? The second is 2+ extra days of work and is the first thing I'd cut.
5. **Analysis tone:** heavier on the science (closer to a mini research paper) or heavier on the engineering narrative (closer to a product spec)? The rubric rewards the science, but a mix is fine.

Once those are answered, next step is to set up the dev environment and ship a Day-2 end-to-end skeleton so we're reasoning over real numbers as early as possible.

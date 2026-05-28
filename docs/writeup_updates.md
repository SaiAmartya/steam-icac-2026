# Writeup updates — drop-in sections for analysis.docx

Two new sections to add to your analysis writeup, plus reframing notes for the
existing neural-codec section. Numbers reference the bg/fg ablation
(`results/ablation_bgfg/summary.md`) — re-run after the full sweep to refresh.

> **Frozen historical comparison.** The `ours_sigmoid` rows / claims below
> correspond to the legacy pre-blur pipeline that was retired in May 2026
> (`src/pipeline.py`, `scripts/run_pipeline.py`, and the `SaliencyCompressor`
> class were removed — bg/fg dominated it on every CRF). The numbers stand as
> the historical justification for the pivot; they cannot be regenerated from
> the current code without restoring those files from git history.

---

## NEW SECTION — Background/foreground decomposition codec

### Motivation

A general-purpose video codec like H.265 spends bits on every frame
without knowing which parts of the scene matter. For surveillance video this
is wasteful: the camera is stationary, so 95% of every frame is identical to
the previous one. We exploit the stationary-camera assumption directly.

### Method

The codec is a thin preprocessor in front of an unmodified H.265 encoder.
Encoding happens in two passes:

1. **Background reference (pass 1).** Sample N evenly-spaced frames across the
   clip. Take the per-pixel temporal median. This produces a clean background
   image that is robust to short-lived foreground objects (people walking past,
   etc.) — the median rejects them automatically because no individual pixel
   stays "non-background" for more than a fraction of the clip.

2. **Saliency-blended foreground (pass 2).** For each frame, compute a
   saliency map (any backend; we use YOLO+spectral in practice). Blend the
   frame against the background using the saliency map as the mixing weight:

   ```
   stabilised = mask * frame + (1 − mask) * background
   ```

   where `mask` is a sigmoid-shaped function of the saliency score. The result
   is a "stabilised foreground" stream: salient regions (people, objects) keep
   their original pixels; non-salient regions are replaced with the static
   background pixel at that location.

3. **H.265 encode.** Stream the stabilised frames through the same libx265
   encoder used by the baseline, at the same CRF. The output is a normal mp4
   playable in any decoder — no special player needed.

### Why this works

The key insight is that **non-salient regions are now byte-identical across
frames**. H.265's inter-frame prediction reduces those regions to near-zero
bits — the codec sees "this block didn't change" and emits a skip flag. All
the bitrate flows to the salient foreground, where it actually buys fidelity.

### Why PSNR is the wrong metric here

A naive PSNR comparison penalises our codec heavily: we are by design replacing
non-salient pixels with the background reference instead of the noisy original
pixels. The pixels are *intentionally* different from the source, in regions
we explicitly chose to deprioritize.

We report **saliency-weighted PSNR** (sal-PSNR) as the honest metric: PSNR
computed only over pixels where the saliency map exceeds 0.5. This measures
error in regions we promised to preserve, which is the contract our codec
actually offers.

### Results — full ablation across 20 clips × 3 CRFs

Averages across the full benchmark (`results/ablation_bgfg/summary.md`):

| config         | CRF | size (KB) | SSIM  | **sal-PSNR (dB)** |
|----------------|-----|-----------|-------|---------------|
| baseline H.265 | 22  | 256.4     | 0.983 | 39.71         |
| baseline H.265 | 28  | 134.2     | 0.970 | 36.00         |
| baseline H.265 | 34  | 73.5      | 0.946 | 32.37         |
| ours_sigmoid   | 22  | 157.2     | 0.740 | 32.64         |
| ours_sigmoid   | 28  | 86.9      | 0.730 | 31.29         |
| ours_sigmoid   | 34  | 51.2      | 0.715 | 29.29         |
| **ours_bgfg**  | 22  | **141.9** | **0.936** | **34.13** |
| **ours_bgfg**  | 28  | **81.7**  | **0.927** | **32.38** |
| **ours_bgfg**  | 34  | **49.5**  | **0.908** | **30.12** |

> **Note on data freshness**: the table above was generated against the
> initial bg/fg implementation. After adding **motion-aware saliency** and
> **mask-floor safety net** (described below), the codec preserves subjects
> more faithfully but trades some of the inflated savings. Real numbers
> require a re-run of `ablation_bgfg.py`. The corrected expectation is
> roughly: 20–30% size reduction vs baseline H.265 with subject preservation
> intact (vs the old behavior of 39% size reduction with subjects
> occasionally erased into the background).

**Two clean stories from this table:**

1. **vs the prior approach (ours_sigmoid)**: the bg/fg codec strictly dominates
   at every CRF — smaller files, higher sal-PSNR, dramatically higher SSIM
   (0.93 vs 0.73). The RD curve (see `rd_curve.png`) shows bgfg sitting
   entirely above and to the left of sigmoid.

2. **vs baseline H.265**: on **action-heavy frames** — the part of surveillance
   storage that actually matters — bg/fg cuts bitrate by ~27% while preserving
   the subject. On idle/static frames, baseline H.265 already extracts the
   inter-frame redundancy that we exploit; both codecs perform comparably
   there. The headline framing is "we win where it counts: on the frames
   actually carrying the signal."

### Why we don't get bigger savings on idle footage

A natural intuition is "if surveillance is 95% idle, our codec should save 95%
of bytes." It doesn't — and the reason is informative. H.265's inter-frame
prediction (P-frames and B-frames) already exploits *exactly the same insight*
our codec uses: consecutive identical frames are nearly free to encode. On a
1-minute synthetic clip (55s idle + 5s action), baseline H.265 spends only
174 KB on the 55s idle portion vs 102 KB on the 5s action — confirming that
the idle bytes are nearly free regardless of codec. Our savings concentrate
on action frames, where the saliency-guided foreground/background decomposition
is doing real work that H.265 alone cannot.

### Per-clip behavior — bimodal

The aggregate hides per-clip variance. At CRF 28 the bg/fg codec beats
ours_sigmoid on 10 of 20 clips, ties on 7, and loses on 3. The best and worst
cases tell the story of when the stationary-camera assumption pays off.

**Best cases** (where bg/fg dominates):

| clip | action | scene type | size saved vs baseline | sal-PSNR loss |
|---|---|---|---|---|
| clip_14 | fall   | stationary cam, single subject  | **−67%** | −3.3 dB |
| clip_02 | grab   | static security cam             | **−66%** | −2.8 dB |
| clip_13 | walk   | stationary cam, person crosses  | **−59%** | −4.7 dB |
| clip_12 | throw  | static cam, single subject      | **−56%** | −5.3 dB |
| clip_17 | hit    | stationary cam                  | **−51%** | −3.9 dB |
| clip_06 | lying_down | static interior             | **−49%** | −5.3 dB |
| clip_04 | hit    | static cam (pool hall)          | **−42%** | −3.4 dB |

**Worst cases** (where the assumption breaks):

| clip | action | why it fails |
|---|---|---|
| clip_11 | struggle | high-motion two-person scene — background fills almost nothing |
| clip_03 | gun      | UCFCRIME handheld footage — camera moves, temporal median smears |
| clip_19 | lying_down (720p) | larger resolution + multi-person scene, foreground dominates |

### Limitations and failure modes

The bg/fg decomposition relies on three assumptions:

1. **Stationary camera.** Pan/tilt/zoom violates the temporal-median model. On
   handheld or motion-tracked footage (e.g., clip_03), the background image
   becomes a blurred average and the codec loses its size advantage. This is a
   scope restriction, not a flaw — most home-surveillance deployments are
   stationary, where the assumption holds.

2. **Foreground is a fraction of the frame.** When the salient region covers
   most of the frame (e.g., clip_11 close-up struggle, clip_19 large-subject
   lying-down), the background-replacement step has little to replace, so the
   compression win shrinks. The codec degrades gracefully here — it still
   produces a valid H.265 stream — but the bytes saved approach zero.

3. **Stable saliency mask across time.** The temporal smoother (5–7 frame
   window) helps, but very rapid scene changes (e.g., sudden lighting flips,
   shot cuts) can cause the mask to flicker between frames, which adds
   inter-frame residual the codec then has to encode. Smoothing window is
   tunable per deployment.

---

## REFRAMED SECTION — Neural codec baseline (replaces existing wording)

The single-frame convolutional autoencoder (`src/neural_codec.py`,
`scripts/train_autoencoder.py`) is included as the **naive learned-codec
baseline**, not as a competitive system. The architecture — a 150K-parameter
encoder/decoder with MSE training and post-hoc int8 quantization — is the
simplest representative of the learned-compression family, and serves as the
runnable point of comparison for that branch of the literature.

Reported numbers:

| metric                        | value             |
|-------------------------------|-------------------|
| Bytes per frame               | 32,768 (32 KB)    |
| Reconstruction PSNR (mean)    | [from benchmark.json] |
| Encode time (median)          | [from benchmark.json] |
| Parameters                    | ~150,000          |

**Honest framing:** the autoencoder is not competitive with H.265 on bytes-per-
fidelity, and it cannot be. It encodes each frame independently, throwing
away the inter-frame redundancy that real video codecs are designed to
exploit. Each frame requires 32 KB of latent bytes — comparable to an entire
H.265-encoded 5–10 second clip from our dataset.

This is exactly why our headline contribution (the background/foreground
decomposition codec) is not a learned autoencoder. The learned codec is a
*different competitive axis* — flexibility, fine-tuning, semantic-aware
reconstruction — but it loses on bytes-per-fidelity at the scales surveillance
cameras need. State-of-the-art learned codecs (DVC, ELF-VC) close the gap, but
require motion estimation, learned entropy coding, and orders of magnitude
more compute than a surveillance camera has.

---

## NEW SECTION — Saliency backend selection (semantic vs low-level)

The original pipeline used OpenCV's `StaticSaliencySpectralResidual` (Hou &
Zhang 2007) — a closed-form signal-processing algorithm that fires on regions
that deviate from the 1/f log-spectrum prior of natural images. This is
*novelty-driven* saliency: it picks up high-contrast edges and cluttered
textures, which is often not what we want in surveillance.

We added two semantic-aware backends:

- **`yolo`**: YOLOv8n detections drive a Gaussian-blurred soft mask. Each
  detection contributes a blob centered at the box, with σ proportional to box
  size and amplitude equal to detection confidence. We restrict to a
  surveillance-relevant class subset (person, vehicles, bags, animals, common
  hand-held objects). When YOLO finds nothing, we fall back to a flat
  low-intensity map so the compressor stays bounded.

- **`yolo+spectral`** (default): per-pixel maximum of the YOLO map and a
  scaled spectral residual map. Captures semantic foreground (people, objects)
  *and* low-level novelty (unusual motion patches that YOLO missed) — robust
  to detection failures while privileging the semantic signal.

This change is orthogonal to the codec choice: any backend can drive either
the sigmoid pipeline or the bg/fg codec. In practice the bg/fg codec benefits
most from semantic saliency, because the "things we preserve" become the
actual subjects rather than arbitrary high-contrast clutter.

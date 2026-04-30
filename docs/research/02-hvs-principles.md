# Research brief: human visual system (HVS) principles for perceptual compression

> **Note:** This brief is the scientific backbone for the Scientific Application & Innovation criterion. References below are canonical and should hold up to scrutiny — verify the exact volume/issue/page numbers when citing in APA-7.

## 1. Foveation and foveal vision
The human retina has a **fovea centralis** with vastly higher spatial resolution than peripheral retina. Visual acuity drops roughly 10× from foveal to peripheral regions — the basis for foveated compression.

- Wandell, B. A. (1995). *Foundations of Vision*. Sinauer Associates.
- Geisler, W. S., & Perry, J. S. (1998). A real-time foveated multiresolution system for low-bandwidth video communication. *Proc. SPIE Human Vision and Electronic Imaging*.
- Martínez-Conde, S., Macknik, S. L., & Hubel, D. H. (2004). The role of fixational eye movements in visual perception. *Nature Reviews Neuroscience*, 5(3), 229–240.

## 2. Contrast Sensitivity Function (CSF)
The CSF describes the eye's ability to detect sinusoidal gratings at different spatial frequencies. It peaks around 4 cycles/degree and falls off at very low and very high frequencies. JPEG's quantization matrix is CSF-derived.

- Campbell, F. W., & Robson, J. G. (1968). Application of Fourier analysis to the visibility of gratings. *J. Physiol.*, 197(3), 551–566.
- Watson, A. B., & Ahumada, A. J. (2005). A standard model for foveated luminance and chrominance sensitivity in the visual field. *J. Vision*, 5(4).
- ITU-T BT.500-14 (2019). Methodology for the subjective assessment of the quality of television pictures.

## 3. Just-Noticeable-Difference (JND)
The threshold below which distortion is invisible. Modern codecs tune quantization to stay under it. Both **spatial** JND (texture masking) and **temporal** JND (motion masking) apply to video.

- Mannos, J. L., & Sakrison, D. J. (1974). The effects of a visual fidelity criterion on the encoding of images. *IEEE TIT*, 20(4), 525–536.
- Yang, X., Ling, W., Lu, Z., Ong, E., & Yao, S. (2005). Just-noticeable-distortion model and its applications in video coding. *Signal Processing: Image Communication*, 20(7), 662–680.
- Wang, Z., & Bovik, A. C. (2009). Mean squared error: love it or leave it? *IEEE Signal Processing Magazine*, 26(1), 98–117.

## 4. Spatial and temporal masking
Distortion is harder to detect in high-activity regions (texture, edges, motion). HEVC and AV1 modulate QP by local activity for this reason.

- Winkler, S. (2005). *Digital Video Quality: Vision Models and Metrics*. John Wiley & Sons.
- Sullivan, G. J., Ohm, J.-R., Han, W.-J., & Wiegand, T. (2012). Overview of the High Efficiency Video Coding (HEVC) standard. *IEEE TCSVT*, 22(12), 1649–1668.

## 5. Saliency and visual attention
**Saliency** is bottom-up "visual loudness." **Attention** is top-down and task-driven. Saliency maps correlate well with eye-tracking ground truth.

- Itti, L., Koch, C., & Niebur, E. (1998). A model of saliency-based visual attention for rapid scene analysis. *IEEE TPAMI*, 20(11), 1254–1259.
- Borji, A., & Itti, L. (2013). State-of-the-art in visual attention modeling. *IEEE TPAMI*, 35(1), 185–207.
- Zhang, L. et al. (2008). SUN: top-down saliency using natural statistics. *Visual Cognition*, 16(7), 793–810.

## 6. Perceptual metrics
HVS-grounded vs ML-grounded:

**HVS-grounded:**
- **PSNR-HVS / PSNR-HVS-M** — Chandler & Hemami (2007). MSE with CSF-weighting and edge-preserving masking.
- **MS-SSIM** — Wang, Simoncelli & Bovik (2003). Multi-scale structural similarity. *Asilomar*.
- **Butteraugli** — Alakuijala et al. (2016). Spatial/temporal masking + CSF weighting; designed for near-transparency quality.

**ML-grounded:**
- **VMAF** — Li et al. (2016). Ensemble ML over subjective data. *SMPTE Motion Imaging J.*
- **LPIPS** — Zhang et al. (2018). Deep perceptual loss using AlexNet/VGG features. *CVPR*.

## How this maps onto our system
- The **saliency map** (principle 5) drives spatial bit allocation.
- **Temporal smoothing** of the map respects principle 4 (temporal masking).
- The **QP delta** range is bounded by principle 3 (JND).
- Our **evaluation stack** uses MS-SSIM + LPIPS (HVS-grounded + ML-grounded) so we cover both schools.

The analysis document's Scientific Application section should cite at least one reference per principle and explicitly trace the design decision to the principle it embodies.

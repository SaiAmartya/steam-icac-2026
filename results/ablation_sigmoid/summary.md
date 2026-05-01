# Sigmoid-mask RD curve (real CCTV, 20 clips)

**Operating point:** sigmoid mask, threshold 0.4, steepness 12, blur kernel 21, idle blur 51, gate 0.25.

| Config | CRF | Clips | KB | PSNR | Sal-PSNR | LPIPS |
|---|---|---|---|---|---|---|
| ours_sigmoid_b21_crf22 | 22 | 20 | 130 | 23.57 | 30.02 | 0.414 |
| ours_sigmoid_b21_crf34 | 34 | 20 | 46 | 23.03 | 26.29 | 0.436 |
| ours_sigmoid_b21_crf40 | 40 | 20 | 32 | 22.43 | 23.30 | 0.462 |

**Plus the CRF 28 point already measured in results/ablation_mask/:**
| ours_sigmoid_b21_crf28 | 28 | 20 | 74 | 23.39 | 28.61 | 0.422 |

**Reference baselines (uniform H.265 on the same 20 clips):**
| baseline_crf22 | 22 | 20 | 239 | 40.23 | 37.97 | 0.013 |
| baseline_crf28 | 28 | 20 | 128 | 36.65 | 33.69 | 0.029 |
| baseline_crf34 | 34 | 20 |  73 | 33.12 | 29.69 | 0.060 |
| baseline_crf40 | 40 | 20 |  44 | 29.60 | 25.85 | 0.117 |

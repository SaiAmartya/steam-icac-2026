# Mask-mode ablation (real CCTV, 20 clips, CRF 28)

| Config | Mask | Blur | Clips | KB | PSNR | Sal-PSNR | LPIPS | Trigger |
|---|---|---|---|---|---|---|---|---|
| ours_alpha_b21_crf28 | alpha | 21 | 20 | 71 | 23.69 | 22.01 | 0.371 | 0.652 |
| ours_binary_b21_crf28 | binary | 21 | 20 | 97 | 22.61 | 31.49 | 0.478 | 0.652 |
| ours_sigmoid_b21_crf28 | sigmoid | 21 | 20 | 74 | 23.39 | 28.61 | 0.422 | 0.652 |
| ours_sigmoid_b9_crf28 | sigmoid | 9 | 20 | 90 | 26.16 | 29.77 | 0.292 | 0.652 |

**Reference points (uniform H.265 baselines, same 20 clips):**
- baseline_crf28: 128 KB · PSNR 36.65 · sal-PSNR 33.69 · LPIPS 0.029
- baseline_crf34:  73 KB · PSNR 33.12 · sal-PSNR 29.69 · LPIPS 0.060
- baseline_crf40:  44 KB · PSNR 29.60 · sal-PSNR 25.85 · LPIPS 0.117

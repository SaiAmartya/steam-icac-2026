# ablation_bgfg — three-way comparison

Averages across all clips per (config, crf).  `sal_psnr` is the honest
metric for our codec: it only counts error in pixels we promised to preserve.

| config | CRF | size (KB) | PSNR | SSIM | **sal-PSNR** |
|---|---|---|---|---|---|
| baseline_h265 | 22 | 256.4 | 41.31 | 0.983 | **39.71** |
| baseline_h265 | 28 | 134.2 | 37.69 | 0.970 | **36.00** |
| baseline_h265 | 34 | 73.5 | 34.13 | 0.946 | **32.37** |
| ours_bgfg | 22 | 213.4 | 35.50 | 0.972 | **35.59** |
| ours_bgfg | 28 | 118.4 | 34.21 | 0.961 | **33.60** |
| ours_bgfg | 34 | 68.3 | 32.25 | 0.940 | **31.09** |
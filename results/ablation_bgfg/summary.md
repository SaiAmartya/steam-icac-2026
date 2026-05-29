# ablation_bgfg — three-way comparison

Averages across all clips per (config, crf).  `sal_psnr` is the honest
metric for our codec: it only counts error in pixels we promised to preserve.

| config | CRF | size (KB) | PSNR | SSIM | **sal-PSNR** |
|---|---|---|---|---|---|
| baseline_h265 | 22 | 11042.5 | 41.12 | 0.971 | **38.44** |
| ours_bgfg | 22 | 4919.8 | 35.75 | 0.956 | **35.15** |
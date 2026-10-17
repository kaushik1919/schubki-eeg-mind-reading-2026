# Controlled CLIP train-only results

All figures use identical shared video-ID holdouts. Each reported subject value is the mean across its saved folds; confidence intervals bootstrap the six paired subject means rather than treating overlapping fold rows as independent.

| Configuration | Top-1 | Delta top-1 (95% paired CI) | Top-5 | Delta top-5 (95% paired CI) | Mean runtime / subject-fold |
| --- | ---: | ---: | ---: | ---: | ---: |
| confirm_clip_final | 2.45% | +0.00% (+0.00%, +0.00%) | 12.25% | +0.00% (+0.00%, +0.00%) | 0.51s |
| confirm_clip_block8 | 3.10% | +0.65% (-0.82%, +2.61%) | 9.97% | -2.29% (-6.70%, +1.80%) | 0.57s |
| confirm_final_block8_50_50 | 3.27% | +0.82% (-0.49%, +2.29%) | 10.95% | -1.31% (-4.58%, +1.63%) | 1.08s |
| confirm_block8_audio_90_10 | 2.94% | +0.49% (-1.14%, +2.78%) | 11.27% | -0.98% (-6.05%, +3.76%) | 0.86s |
| confirm_final_block8_audio_45_45_10 | 3.27% | +0.82% (-0.16%, +2.12%) | 11.60% | -0.65% (-3.92%, +2.61%) | 1.37s |

## Per-subject means

| Configuration | 01 | 02 | 03 | 04 | 05 | 06 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| confirm_clip_final (top-1) | 4.90% | 2.94% | 3.92% | 0.00% | 2.94% | 0.00% |
| confirm_clip_final (top-5) | 14.71% | 19.61% | 14.71% | 5.88% | 10.78% | 7.84% |
| confirm_clip_block8 (top-1) | 2.94% | 3.92% | 2.94% | 4.90% | 2.94% | 0.98% |
| confirm_clip_block8 (top-5) | 14.71% | 7.84% | 13.73% | 11.76% | 8.82% | 2.94% |
| confirm_final_block8_50_50 (top-1) | 2.94% | 3.92% | 4.90% | 3.92% | 3.92% | 0.00% |
| confirm_final_block8_50_50 (top-5) | 18.63% | 11.76% | 12.75% | 7.84% | 10.78% | 3.92% |
| confirm_block8_audio_90_10 (top-1) | 2.94% | 1.96% | 2.94% | 5.88% | 2.94% | 0.98% |
| confirm_block8_audio_90_10 (top-5) | 19.61% | 7.84% | 13.73% | 12.75% | 10.78% | 2.94% |
| confirm_final_block8_audio_45_45_10 (top-1) | 3.92% | 2.94% | 4.90% | 3.92% | 3.92% | 0.00% |
| confirm_final_block8_audio_45_45_10 (top-5) | 18.63% | 11.76% | 12.75% | 10.78% | 8.82% | 6.86% |

## Paired per-subject differences from baseline

| Configuration | 01 | 02 | 03 | 04 | 05 | 06 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| confirm_clip_block8 (top-1 delta) | -1.96% | +0.98% | -0.98% | +4.90% | +0.00% | +0.98% |
| confirm_clip_block8 (top-5 delta) | +0.00% | -11.76% | -0.98% | +5.88% | -1.96% | -4.90% |
| confirm_final_block8_50_50 (top-1 delta) | -1.96% | +0.98% | +0.98% | +3.92% | +0.98% | +0.00% |
| confirm_final_block8_50_50 (top-5 delta) | +3.92% | -7.84% | -1.96% | +1.96% | +0.00% | -3.92% |
| confirm_block8_audio_90_10 (top-1 delta) | -1.96% | -0.98% | -0.98% | +5.88% | +0.00% | +0.98% |
| confirm_block8_audio_90_10 (top-5 delta) | +4.90% | -11.76% | -0.98% | +6.86% | +0.00% | -4.90% |
| confirm_final_block8_audio_45_45_10 (top-1 delta) | -0.98% | +0.00% | +0.98% | +3.92% | +0.98% | +0.00% |
| confirm_final_block8_audio_45_45_10 (top-5 delta) | +3.92% | -7.84% | -1.96% | +4.90% | -1.96% | -0.98% |

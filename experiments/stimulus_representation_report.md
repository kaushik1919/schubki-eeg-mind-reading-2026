# Controlled CLIP train-only results

All figures use identical shared video-ID holdouts. Each reported subject value is the mean of its three fixed-seed folds; therefore confidence intervals bootstrap six paired subject means, not the 18 overlapping folds.

| Configuration | Top-1 | Delta top-1 (95% paired CI) | Top-5 | Delta top-5 (95% paired CI) | Mean runtime / subject-fold |
| --- | ---: | ---: | ---: | ---: | ---: |
| clip_ridge_organizer_window | 2.45% | +0.00% (+0.00%, +0.00%) | 11.98% | +0.00% (+0.00%, +0.00%) | 0.46s |
| audio_simple_ridge | 1.96% | -0.49% (-1.36%, +0.38%) | 7.95% | -4.03% (-6.26%, -1.85%) | 0.28s |
| clip_audio_90_10 | 2.94% | +0.49% (+0.05%, +1.03%) | 11.87% | -0.11% (-0.49%, +0.38%) | 0.74s |
| clip_block4_ridge | 2.34% | -0.11% (-0.87%, +0.76%) | 10.62% | -1.36% (-3.32%, +0.93%) | 0.36s |
| clip_block8_ridge | 2.89% | +0.44% (-0.11%, +1.03%) | 12.09% | +0.11% (-1.74%, +2.18%) | 0.44s |
| clip_block12_ridge | 2.45% | -0.00% (-0.49%, +0.54%) | 10.95% | -1.03% (-2.12%, +0.05%) | 0.58s |

## Per-subject means

| Configuration | 01 | 02 | 03 | 04 | 05 | 06 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clip_ridge_organizer_window (top-1) | 3.92% | 3.27% | 2.61% | 1.31% | 1.96% | 1.63% |
| clip_ridge_organizer_window (top-5) | 15.03% | 15.03% | 13.73% | 8.82% | 10.78% | 8.50% |
| audio_simple_ridge (top-1) | 1.63% | 2.29% | 2.61% | 2.61% | 1.63% | 0.98% |
| audio_simple_ridge (top-5) | 7.52% | 8.50% | 6.54% | 8.17% | 9.48% | 7.52% |
| clip_audio_90_10 (top-1) | 4.58% | 3.27% | 2.29% | 2.94% | 2.29% | 2.29% |
| clip_audio_90_10 (top-5) | 15.03% | 14.38% | 13.40% | 8.17% | 10.78% | 9.48% |
| clip_block4_ridge (top-1) | 2.61% | 2.29% | 1.96% | 1.31% | 2.61% | 3.27% |
| clip_block4_ridge (top-5) | 12.09% | 10.13% | 11.44% | 6.86% | 14.05% | 9.15% |
| clip_block8_ridge (top-1) | 3.92% | 2.94% | 2.29% | 2.29% | 2.61% | 3.27% |
| clip_block8_ridge (top-5) | 13.40% | 13.40% | 10.78% | 10.46% | 11.76% | 12.75% |
| clip_block12_ridge (top-1) | 4.58% | 4.25% | 1.96% | 0.65% | 1.31% | 1.96% |
| clip_block12_ridge (top-5) | 14.38% | 12.09% | 13.40% | 5.88% | 10.78% | 9.15% |

# Controlled CLIP train-only results

All figures use identical shared video-ID holdouts. Each reported subject value is the mean of its three fixed-seed folds; therefore confidence intervals bootstrap six paired subject means, not the 18 overlapping folds.

| Configuration | Top-1 | Delta top-1 (95% paired CI) | Top-5 | Delta top-5 (95% paired CI) | Mean runtime / subject-fold |
| --- | ---: | ---: | ---: | ---: | ---: |
| clip_ridge_organizer_window | 2.45% | +0.00% (+0.00%, +0.00%) | 11.98% | +0.00% (+0.00%, +0.00%) | 0.46s |
| clip_temporal_two_segments_ridge | 2.72% | +0.27% (-0.11%, +0.87%) | 10.62% | -1.36% (-2.02%, -0.76%) | 0.56s |
| reverse_clip_ridge_50to600 | 1.14% | -1.31% (-1.91%, -0.71%) | 7.03% | -4.96% (-6.64%, -3.32%) | 0.22s |

## Per-subject means

| Configuration | 01 | 02 | 03 | 04 | 05 | 06 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clip_ridge_organizer_window (top-1) | 3.92% | 3.27% | 2.61% | 1.31% | 1.96% | 1.63% |
| clip_ridge_organizer_window (top-5) | 15.03% | 15.03% | 13.73% | 8.82% | 10.78% | 8.50% |
| clip_temporal_two_segments_ridge (top-1) | 3.92% | 3.59% | 4.25% | 1.31% | 1.63% | 1.63% |
| clip_temporal_two_segments_ridge (top-5) | 14.38% | 13.73% | 13.07% | 8.17% | 8.50% | 5.88% |
| reverse_clip_ridge_50to600 (top-1) | 1.63% | 0.98% | 1.31% | 0.98% | 0.98% | 0.98% |
| reverse_clip_ridge_50to600 (top-5) | 8.17% | 6.86% | 8.17% | 6.54% | 6.86% | 5.56% |

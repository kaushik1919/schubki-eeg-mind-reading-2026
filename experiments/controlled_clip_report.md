# Controlled CLIP train-only results

All figures use identical shared video-ID holdouts. Each reported subject value is the mean of its three fixed-seed folds; therefore confidence intervals bootstrap six paired subject means, not the 18 overlapping folds.

| Configuration | Top-1 | Delta top-1 (95% paired CI) | Top-5 | Delta top-5 (95% paired CI) | Mean runtime / subject-fold |
| --- | ---: | ---: | ---: | ---: | ---: |
| clip_ridge_organizer_window | 2.45% | +0.00% (+0.00%, +0.00%) | 11.98% | +0.00% (+0.00%, +0.00%) | 0.46s |
| clip_ridge_score_early_0to500 | 2.72% | +0.27% (-0.27%, +0.71%) | 9.80% | -2.18% (-2.45%, -1.91%) | 0.47s |
| clip_ridge_score_middle_100to800 | 2.12% | -0.33% (-0.87%, +0.38%) | 10.24% | -1.74% (-2.29%, -1.14%) | 0.46s |
| clip_ridge_score_late_200to1200 | 2.51% | +0.05% (-0.82%, +0.98%) | 9.26% | -2.72% (-3.92%, -1.25%) | 0.46s |
| clip_ridge_score_long_0to1500 | 2.18% | -0.27% (-0.71%, +0.22%) | 10.51% | -1.47% (-2.29%, -0.49%) | 0.47s |
| clip_ridge_targetpca64 | 1.85% | -0.60% (-1.20%, -0.00%) | 10.24% | -1.74% (-2.45%, -0.98%) | 0.25s |
| clip_pls_targetpca32 | 2.18% | -0.27% (-0.76%, +0.22%) | 9.53% | -2.45% (-3.27%, -1.58%) | 0.26s |
| clip_pooled_subjectz_ridge | 2.18% | -0.27% (-0.71%, +0.00%) | 9.48% | -2.51% (-3.54%, -1.53%) | 0.22s |

## Per-subject means

| Configuration | 01 | 02 | 03 | 04 | 05 | 06 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clip_ridge_organizer_window (top-1) | 3.92% | 3.27% | 2.61% | 1.31% | 1.96% | 1.63% |
| clip_ridge_organizer_window (top-5) | 15.03% | 15.03% | 13.73% | 8.82% | 10.78% | 8.50% |
| clip_ridge_score_early_0to500 (top-1) | 2.94% | 4.25% | 3.27% | 1.96% | 2.29% | 1.63% |
| clip_ridge_score_early_0to500 (top-5) | 13.40% | 13.07% | 11.11% | 6.86% | 8.50% | 5.88% |
| clip_ridge_score_middle_100to800 (top-1) | 2.61% | 4.58% | 1.96% | 0.98% | 1.63% | 0.98% |
| clip_ridge_score_middle_100to800 (top-5) | 13.40% | 14.38% | 11.44% | 6.54% | 8.17% | 7.52% |
| clip_ridge_score_late_200to1200 (top-1) | 2.61% | 1.96% | 2.29% | 3.27% | 2.61% | 2.29% |
| clip_ridge_score_late_200to1200 (top-5) | 10.78% | 11.44% | 9.15% | 9.15% | 8.82% | 6.21% |
| clip_ridge_score_long_0to1500 (top-1) | 2.94% | 2.61% | 2.29% | 0.65% | 2.61% | 1.96% |
| clip_ridge_score_long_0to1500 (top-5) | 12.75% | 12.09% | 12.09% | 7.84% | 11.44% | 6.86% |
| clip_ridge_targetpca64 (top-1) | 2.29% | 1.96% | 2.29% | 1.96% | 1.31% | 1.31% |
| clip_ridge_targetpca64 (top-5) | 12.75% | 12.09% | 11.44% | 8.50% | 8.82% | 7.84% |
| clip_pls_targetpca32 (top-1) | 3.92% | 2.29% | 1.63% | 1.63% | 1.31% | 2.29% |
| clip_pls_targetpca32 (top-5) | 12.09% | 11.11% | 10.46% | 7.19% | 8.50% | 7.84% |
| clip_pooled_subjectz_ridge (top-1) | 3.92% | 3.27% | 1.31% | 1.31% | 1.63% | 1.63% |
| clip_pooled_subjectz_ridge (top-5) | 13.73% | 14.05% | 9.48% | 5.23% | 7.19% | 7.19% |

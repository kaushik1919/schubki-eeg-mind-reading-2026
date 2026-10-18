# Temporal visual encoding audit and result

## Alignment audit

- The 1,000 training containers have durations of 3.000--3.067 s, but decoded frame counts range from 45 to 92 and frame rates from 15.0 to 30.5 fps. A fixed frame-count assumption is therefore incorrect.
- `extract_temporal_visual_features.py` decodes each container at its native frame count, derives luminance, contrast, frame-difference motion energy, Farneback optical-flow magnitude, and histogram scene-change distance, then resamples each trajectory to a common 90-bin grid.
- EEG is the organizer's 17 by 1,500 sample recording beginning at documented stimulus onset, FIR-decimated from 500 to 100 Hz. No organizer documentation supports an onset-offset correction.
- The new research holdout contains 102 shared training video IDs, disjoint from the three original screening folds, `final_holdout_ids.npy`, and `confirmation_holdout_ids.npy`. It was created before model fitting. Candidate columns and rows were verified as aligned by video ID.

## Frozen experiment

The causal temporal-response function uses all five low-level visual trajectories with lags 0--600 ms, independently per subject. Standardization and RidgeCV fitting use only non-holdout recordings. Retrieval scoring is Pearson correlation from 50--600 ms. This configuration was not tuned on this holdout.

| Model | Top-1 | Top-5 | MRR | Mean train + candidate inference / subject |
| --- | ---: | ---: | ---: | ---: |
| Temporal visual TRF | 0.98% | 5.72% | 0.058 | 3.69 s |
| Frozen Model C | 3.27% | 12.25% | 0.097 | 1.15 s |

| Subject | Visual TRF top-1 / top-5 / MRR | Model C top-1 / top-5 / MRR |
| --- | ---: | ---: |
| 01 | 0.00% / 4.90% / 0.052 | 5.88% / 16.67% / 0.128 |
| 02 | 0.00% / 5.88% / 0.055 | 2.94% / 15.69% / 0.107 |
| 03 | 0.98% / 1.96% / 0.044 | 2.94% / 8.82% / 0.088 |
| 04 | 0.00% / 5.88% / 0.045 | 0.98% / 9.80% / 0.076 |
| 05 | 2.94% / 7.84% / 0.078 | 4.90% / 13.73% / 0.112 |
| 06 | 1.96% / 7.84% / 0.072 | 1.96% / 8.82% / 0.072 |

The low-level visual trajectories do not provide a useful retrieval signal. They must not be fused with Model C, and the result does not justify nonlinear modeling.

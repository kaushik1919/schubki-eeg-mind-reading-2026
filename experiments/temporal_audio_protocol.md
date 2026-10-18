# Temporal audio TRF and scoring-window study: pre-specified protocol

Written before any temporal-audio model was fitted or scored.

## Validation status: exploratory
Only 17 of the 695 shared training videos have never served as validation (screening seeds 2026-28, confirmation, H2, H3, and the nested-rerank inner folds). No untouched 102-video holdout exists. All results here are **exploratory nested train-only cross-validation** on previously inspected video identities. They are not independent confirmation.

## Partitions
- Outer: 6 mutually disjoint folds of 102 shared video IDs (`disjoint_inner_folds(pool, exclude=[], n_folds=6, size=102, seed=2040)`). 612 IDs in total, each evaluated once. Candidates are the 102 fold IDs. Training uses all of the subject's other training recordings, including subject-specific IDs.
- Inner (TRF regularisation only): each training video ID is assigned to one of 5 blocks by `rng(2041).permutation(1..1000) mod 5`. Inner fold i = training videos in block i. All time samples of a video stay in one block.
- Model C constituents keep their frozen `RidgeCV` (efficient LOO over videos; one row per video, so already grouped by video).

## Models
- A `A_envelope_onset`: RMS envelope^0.3, positive envelope derivative, mean positive log-mel difference (onset strength), spectral flux. 4 features.
- B `B_logmel`: 32 log-mel bands, log(1 + 1e4 * power), 30-8000 Hz. 32 features.
- C `C_combined`: A + B. 36 features.
- V `V_visual_grouped` (audit control): the existing 5 low-level visual trajectories, refit with the corrected grouped regularisation.
- TRF: causal lags 0-600 ms (61 lags at 100 Hz), zero padding before onset (= silence), per-time-sample intercept (training-mean evoked response), training-fold centring and column standardisation, ridge alpha = lambda * n_samples with lambda in logspace(-4, 2, 13). Lambda is selected by mean inner-fold retrieval MRR, scored 50-600 ms among the inner fold's videos; ties go to the larger lambda.
- EEG: FIR decimation to 100 Hz, full 0-3000 ms target (as in Model C).
- Model C: frozen constituents and weights (0.45/0.45/0.10), refit per outer fold on identical training recordings.

## Metrics
Top-1, top-5 and MRR among 102 candidates; per-subject; 95% CI from 20,000 resamples of the 6 subjects; paired trial wins/losses against Model C. Chance: top-1 0.98%, top-5 4.90%, MRR 0.0509.

## Scoring windows (exploratory; Model C's frozen 50-600 ms is unchanged)
50-600, 100-1000, 200-1500, 500-2500 and 50-3000 ms, all applied to the same fitted predictions.

## Fusion gate (Phase 6)
Fusion is evaluated only if the audio TRF with the highest inner MRR among A/B/C has outer pooled top-5 >= 7.35% (1.5x chance) **and** a subject-bootstrap 95% CI lower bound for MRR above 0.0509. If so, a single pre-declared fusion is evaluated: 0.9 x rowz(Model C fused score) + 0.1 x rowz(TRF score), 50-600 ms. No weight search.

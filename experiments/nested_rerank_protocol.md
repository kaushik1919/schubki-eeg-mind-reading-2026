# Pre-registered nested comparison: conservative block-8 top-K reranking

Written before the outer holdout below was created or any model was fitted on it.

## Data partitions
- Shared pool: the 695 training-video IDs recorded for all six subjects.
- Outer holdout H2 (102 IDs): drawn with seed 2031 from shared IDs never used by the screening folds (seeds 2026/2027/2028) **or** the confirmation holdout (`manifests/confirmation_holdout_ids.npy`). Saved to `manifests/final_holdout_ids.npy` by `scripts/create_confirmation_holdout.py --exclude-ids ...`.
- Inner folds: 5 mutually disjoint folds of 102 IDs drawn (seed 2032) from the shared pool minus H2 (`src.controlled_cv.disjoint_inner_folds`). Inner folds may contain screening/confirmation IDs; they are training data relative to H2.
- Every fit uses only one subject's training recordings, with H2 always removed and, for inner folds, the inner fold removed too. No organizer test EEG or test labels are read.

## Constituent models (unchanged existing protocol)
Organizer Ridge forward model (`scripts/clip_controlled_cv.py::ForwardRegressor('ridge')`), 100 Hz 0-3000 ms target, correlation scoring 50-600 ms, on cached features:
`features/all_train_clip_features.npy` (final CLIP), `features/all_train_clip_intermediate_block8.npy` (block 8), `features/all_train_audio_simple_features.npy` (audio).

## Configurations (fixed; only K is selected)
- A: final CLIP.
- B: row-z fusion final/block8 0.50/0.50.
- C: row-z fusion final/block8/audio 0.45/0.45/0.10.
- D_K: rank by A; reorder only A's top-K candidates by B's score; ranks K+1..102 keep A's order. K in {3, 5}. Each EEG row is processed independently. Top-j accuracy for j >= K equals A's by construction.

## Selection rules (applied to inner-fold results only; frozen before the outer phase)
Inner metric = mean over 6 subjects of each subject's pooled inner top-1/top-5 (5 x 102 trials per subject).
1. K*: largest inner top-1 among D_3, D_5; tie -> K = 3.
2. Best top-1 model: largest inner top-1 among {A, B, C, D_K*}; tie -> higher top-5, then the earlier entry in that list.
3. Best top-5 model: largest inner top-5; tie -> higher top-1, then list order.
4. Balanced model: largest inner (top-1 + top-5) / 2. The challenge PDF says ranking is "determined by" top-1 and top-5 but gives no weighting, so an equal-weight mean of the two quoted accuracies is used. A chance-normalised variant (top1/(1/102) + top5/(5/102))/2 is reported as a sensitivity check only.
5. The balanced model is the one recommended for freezing. H2 results are reported for all configurations but are not used to change any selection.

## Inference reporting on H2
Per-subject top-1/top-5/MRR, paired trial wins/losses vs A, subject-level bootstrap CI (6 subjects as units), and an exact sign test on discordant trials (optimistic: treats trials as independent).

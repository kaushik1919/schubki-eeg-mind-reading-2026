# Frozen multilevel-fusion confirmation protocol

## Holdout and selection

- Shared video IDs were first intersected across all six training subjects: 695 IDs.
- The 265 IDs used by the earlier three screening folds (seeds 2026, 2027, and 2028) were excluded.
- Before fitting any model in this experiment, `scripts/create_confirmation_holdout.py` drew and saved 102 of the remaining 430 IDs to `manifests/confirmation_holdout_ids.npy` (seed 2030).
- The five configurations below, their weights, the 50--600 ms scoring window, and all feature sources were fixed before this holdout was evaluated. No development-test labels or root submission arrays were read or modified.

## Common evaluation contract

- All six subject-specific models train on only that subject's training recordings whose video IDs are not in the confirmation holdout.
- Candidate columns are the ascending IDs in `heldout_ids`; score rows and `target_ids` use the corresponding numeric recording-ID order. Every saved matrix is 102 by 102 and was checked for identical `heldout_ids` and `target_ids` across models and subjects.
- Each constituent model uses the existing organizer-compatible EEG target preprocessing and Ridge protocol: full 0--3000 ms target, target PCA with 64 components fitted within the subject training partition, and correlation scoring from 50--600 ms.
- Fusion is a fixed weighted sum of independently row-z-scored score matrices. Row z-scoring has no fitted parameters and uses neither true held-out IDs nor labels. No held-out-label calibration or weight search occurred.

## Frozen configurations and artifacts

| Configuration | Constituent score labels | Weights | Metrics CSV |
| --- | --- | --- | --- |
| Final CLIP | `confirm_clip_final` | 1.00 | `experiments/confirm_clip_final.csv` |
| CLIP block 8 | `confirm_clip_block8` | 1.00 | `experiments/confirm_clip_block8.csv` |
| Final + block 8 | `confirm_clip_final`, `confirm_clip_block8` | 0.50, 0.50 | `experiments/confirm_final_block8_50_50.csv` |
| Block 8 + audio | `confirm_clip_block8`, `confirm_audio_simple` | 0.90, 0.10 | `experiments/confirm_block8_audio_90_10.csv` |
| Final + block 8 + audio | `confirm_clip_final`, `confirm_clip_block8`, `confirm_audio_simple` | 0.45, 0.45, 0.10 | `experiments/confirm_final_block8_audio_45_45_10.csv` |

Each configuration has six compressed, aligned score artifacts in `experiments/paired_predictions/`, named `<label>_seedconfirmation_sub<subject>.npz`. Each artifact contains `heldout_ids`, `target_ids`, `scores`, and `rankings`.

The paired subject-level confidence intervals and per-subject results are in `experiments/multilevel_confirmation_report.md`.

## Erratum (2026-10-17, rank-error audit)
The "Common evaluation contract" above says constituents use "target PCA with 64 components". The saved CSVs record `method=ridge`; in `ForwardRegressor` the `ridge` method fits RidgeCV directly on the full 17 x 300 EEG target and ignores `target_components` (the CSV value 64 is an unused default). Re-running all three constituents reproduced every saved score matrix bit-exactly with `--method ridge`, so the reported metrics are correct; only this description was wrong. No result was changed.

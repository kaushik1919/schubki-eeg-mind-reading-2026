# EEG-to-Video Retrieval — SchubKI National AI Hackathon 2026

## Overview

Given a three-second EEG recording, rank 102 candidate videos from most to least likely to have been watched during that recording. The submitted system is a linear forward encoding model: frozen video and audio features predict the EEG response to each candidate, and candidates are ranked by correlation between predicted and measured EEG.

## Dataset

| Property | Value |
|---|---|
| Participants | 6 (sub-01 to sub-06) |
| EEG channels | 17 |
| Sampling rate | 500 Hz |
| Recording length | 3 s from video onset (17 x 1500 samples) |
| Training videos | 1,000 (`0001.mp4`-`1000.mp4`); 850-972 recordings per subject |
| Test candidates | 102 (`1001.mp4`-`1102.mp4`), one recording per video per subject |
| Hidden subjects | sub-05, sub-06 (test labels withheld) |

The organizer data is not redistributed. Place it at `eeg-mind-reading-challenge/` with subfolders `trn/`, `tst/`, `tst_stim/`, `stimuli_train/`, `stimuli_test/`.

## Method

| Stage | Implementation |
|---|---|
| EEG preprocessing | Zero-phase FIR decimation 500 -> 100 Hz (`scipy.signal.decimate`, factor 5); full 0-3000 ms target (17 x 300). No additional filtering. |
| Visual features | OpenCLIP ViT-B-32 (`openai` weights, QuickGELU), 8 uniformly sampled frames per video, mean-pooled. Final projected embedding (512-d) and CLS token after transformer block 8 (768-d). |
| Audio features | 16 kHz mono waveform; 12 RMS-envelope bins plus mean and standard deviation of 32 log-mel bands (76-d). Clips without audio map to a fixed silence vector. |
| Regression | Per subject and per feature set: StandardScaler, PCA (95% variance), StandardScaler, `RidgeCV` (20 alphas, 1e-3 to 1e6) predicting the full EEG target. All transforms are fitted on that subject's training recordings only. |
| Scoring | Pearson correlation between measured and predicted EEG over 50-600 ms, flattened across channels (organizer scoring). |
| Fusion | Each score matrix is z-scored per EEG row, then combined as a weighted mean: final CLIP 0.45, block 8 0.45, audio 0.10 (Model C). |
| Ranking | Stable descending sort per recording. Each recording is ranked independently; no joint assignment across recordings. |

## Models Evaluated

All entries below were implemented and run in this repository. Selection used train-only, video-identity-held-out validation (`experiments/audit_and_protocol.md`).

| Experiment | Outcome |
|---|---|
| Organizer baseline (AlexNet `features.3` + Ridge) | Reproduced; reference only |
| CLIP ViT-B/32 final embedding | Strongest single representation |
| DINOv2-S | Weaker than CLIP (2.18% / 9.04% vs 2.45% / 11.98% top-1 / top-5, screening folds) |
| VideoMAE, PANNs CNN14 | Features extracted and fitted in an early development-label screen; not carried forward |
| CLIP intermediate blocks 4 / 8 / 12 | Block 8 complementary to the final embedding; blocks 4 and 12 weaker alone and not fused |
| Simple audio features | Weak alone, small complementary gain in fusion |
| Multimodal score fusion | Final CLIP + block 8 + audio selected (Model C) |
| EEG scoring-window variants | Inconsistent top-1 changes; top-5 lower than 50-600 ms in all four variants |
| PLS regression, target PCA | Worse than direct Ridge |
| Pooled subject-normalized Ridge | Worse than subject-specific Ridge |
| Reverse EEG-to-CLIP Ridge | Substantially worse (1.14% / 7.03%) |
| Two-segment temporal CLIP | Top-1 +0.27 pp, top-5 -1.36 pp; not adopted |
| Top-K reranking of final CLIP by fused score | No benefit on independent holdout |

## Results

Independent holdout H2: 102 training-video identities shared by all six subjects and never used in any earlier fold. Models were trained on each subject's remaining training recordings. Configurations and selection rules were fixed before H2 was evaluated (`experiments/nested_rerank_protocol.md`).

| Model | Top-1 | Top-5 | MRR |
|---|---:|---:|---:|
| Final CLIP | 3.27% | 11.44% | 0.097 |
| Final CLIP + block 8 | 3.10% | 12.91% | 0.100 |
| Final CLIP + block 8 + audio (Model C) | 3.27% | 13.73% | 0.104 |
| Top-3 reranker | 3.43% | 11.44% | 0.098 |
| Top-5 reranker | 3.10% | 11.44% | 0.095 |

Chance is 0.98% top-1 and 4.90% top-5. These are validation results on held-out training videos, not hidden-test accuracy. Model C's top-5 gain over final CLIP was +2.29 percentage points with a 95% subject-resampled interval of -0.33 to +4.41 points, so it is not statistically conclusive. Top-1 did not improve. Accuracy on the hidden subjects is unknown.

Detailed analysis: `experiments/rank_error_analysis_report.md`.

## Reproduction

Environment (Python 3.11):

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Feature extraction (repeat with `--video-dir eeg-mind-reading-challenge/stimuli_test --ids manifests/test_candidate_ids.npy` and `test_*` output names for the candidates):

```bash
.venv/bin/python scripts/extract_clip_features.py --video-dir eeg-mind-reading-challenge/stimuli_train --ids manifests/all_training_ids.npy --output features/all_train_clip_features.npy
.venv/bin/python scripts/extract_clip_intermediate_features.py --video-dir eeg-mind-reading-challenge/stimuli_train --ids manifests/all_training_ids.npy --output-prefix features/all_train_clip_intermediate
.venv/bin/python scripts/extract_audio_features.py --video-dir eeg-mind-reading-challenge/stimuli_train --ids manifests/all_training_ids.npy --output features/all_train_audio_simple_features.npy
```

Validation (train-only; scripts refuse to overwrite existing results, so pass a new `--out-dir` when rerunning):

```bash
.venv/bin/python scripts/clip_controlled_cv.py --features features/all_train_clip_features.npy --label clip_final --output experiments/clip_final.csv --predictions-dir experiments/paired_predictions
.venv/bin/python scripts/nested_rerank_cv.py --phase inner --out-dir experiments/nested_rerank_repro
.venv/bin/python scripts/nested_rerank_cv.py --phase outer --out-dir experiments/nested_rerank_repro
```

Training and final prediction (fits on training recordings only, writes to a new versioned directory):

```bash
.venv/bin/python scripts/build_final_model_c.py --out-dir outputs/final_model_c_repro
```

Submission integrity and promotion:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/promote_submission.py            # validate only
.venv/bin/python scripts/promote_submission.py --confirm  # back up root files, then promote
```

## Repository Structure

| Path | Contents |
|---|---|
| `src/` | Data loading, preprocessing, retrieval scoring, fusion, reranking, rank analysis, frozen Model C |
| `scripts/` | Feature extraction, cross-validation, analysis, final build and promotion entry points |
| `configs/` | Frozen Model C configurations (`frozen_balanced.json`, `frozen_best_top1.json`, `frozen_best_top5.json`) |
| `manifests/` | Video-ID lists and precommitted holdouts |
| `experiments/` | Protocols, reports, CSV metrics and JSON run records |
| `outputs/final_model_c/` | Model C checkpoints, scores, submissions, `metadata.json`, `MANIFEST.sha256` |
| `outputs/previous_root_submissions/` | Backups of superseded root submissions |
| `tests/` | pytest suite (synthetic tests plus data-dependent integrity tests that skip without data) |

Excluded from Git and reproducible with the commands above: organizer data and documents, stimulus videos, feature caches (`features/*.npy`), legacy baseline checkpoints (`experiments/*.joblib`), and per-fold score archives (`experiments/**/*.npz`).

## Limitations

- About 850-970 training recordings per subject limit what a high-dimensional regression can learn.
- Performance varies across subjects; models are subject-specific and not shared.
- Sub-06 has the fewest recordings (850). The final CLIP Ridge alpha for sub-06 reaches the top of the grid, and fusion reduced sub-06 performance on both independent holdouts.
- 40 of 1,000 training and 3 of 102 test videos have no or near-silent audio, so the audio component carries no information for them.
- Each holdout has 102 videos; differences of 1-2 pp are within validation noise.

## Submission

| File | SHA-256 |
|---|---|
| `sub-05_ranked_candidate_list.npy` | `7921fbf58bc2729e188e669ff4ecb8e155884a98fbcbab243ebc10a467e2ad35` |
| `sub-06_ranked_candidate_list.npy` | `e252e4c8cc9c42bcd141bc1ced2084cc5225f87aed8c3148da32f289a4b0a1f1` |

Each file is an integer array of shape (102, 102). Row i corresponds to recording i of `tst/sub-XX_tst.npy` in original order and contains every candidate ID 1001-1102 exactly once, ordered from most to least likely.

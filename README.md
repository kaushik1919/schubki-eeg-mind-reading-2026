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
| Neural decoders | Shared compact EEGNet (about 66k parameters) with per-subject 17x17 adapters, trained on all six subjects. (1) Multi-positive EEG-to-CLIP-block-8 contrastive loss, 5 seeds. (2) Cross-subject EEG-to-EEG contrastive loss with a ridge readout from block-8 features, 5 seeds. S3 = equal-weight row-z fusion of both decoder ensembles and Model C. |
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
| Temporal audio TRF (envelope, onsets, log-mel) | Above chance but worse than Model C alone |
| Shared EEGNet contrastive decoder (subject adapters, multi-positive loss, CLIP block-8 target) | Clear gain over Model C; 5-seed ensemble frozen as candidate |
| Cross-subject EEG-to-EEG contrastive decoder with ridge readout | Strongest single decoder |
| Frozen pretrained CBraMod EEG encoder + linear probe | Near chance (1.33% / 6.15%) |
| DINOv2 (ViT-S, ViT-B) and CLAP contrastive targets | Worse than CLIP block 8, alone and combined |
| Hungarian / Sinkhorn / row-column calibration | Hungarian no gain; others within noise |

## Results

All numbers are top-1 / top-5 among exactly 102 candidates (chance 0.98% / 4.90%). None are hidden-test accuracy; performance on sub-05 and sub-06 test recordings is unknown.

### Current best: fresh-partition train-only validation

Six folds of 102 shared training videos (partition seed 2060, 3,672 trials). Every recording of a held-out video is excluded from training for all subjects. The systems and decision rule were fixed before this partition was evaluated (`experiments/fresh2060_protocol.md`). The video identities were seen in earlier studies, so this is a fresh split, not untouched data.

| System | Top-1 | Top-5 | MRR |
|---|---:|---:|---:|
| Model C | 3.08% | 13.04% | 0.100 |
| Contrastive decoder, 5 seeds | 4.66% | 17.37% | 0.130 |
| EEG-to-EEG decoder, 5 seeds | 5.56% | 22.00% | 0.152 |
| S1: contrastive decoder + Model C (frozen candidate) | 4.90% | 19.14% | 0.139 |
| **S3: contrastive decoder + EEG-to-EEG decoder + Model C** | **5.69%** | **22.98%** | **0.157** |
| S3 + Sinkhorn calibration (needs organizer approval) | 5.88% | 23.31% | 0.160 |

S3 vs S1 (paired, 6-subject bootstrap): top-1 +0.79 pp [+0.41, +1.14], top-5 +3.84 pp [+2.91, +4.77]. Per subject, S3 top-1 / top-5 was 6.4/27.6, 6.2/26.0, 6.2/22.1, 5.7/23.4, 4.4/18.3 and 5.2/20.6% for sub-01 to sub-06. Details: `experiments/max_accuracy_sprint_report.md`.

### Pre-registered development-subject check (sub-01 to sub-04 test recordings)

A single run of the frozen candidate S1. Labels were used only to score predictions saved and hashed beforehand (`experiments/frozen_candidate_confirmation_report.md`). These recordings were used in earlier baseline analysis, so this is a candidate check, not an untouched holdout.

| System | Top-1 | Top-5 | MRR |
|---|---:|---:|---:|
| Model C | 5.64% | 19.85% | 0.145 |
| Contrastive decoder, 5 seeds | 6.86% | 25.49% | 0.178 |
| S1 frozen candidate | 9.07% | 32.60% | 0.210 |

Both models score higher on these test sets than on held-out training videos, so absolute numbers from the two settings should not be compared.

### Earlier Model C holdout (H2)

H2: 102 training-video identities shared by all six subjects and never used in an earlier fold. Configurations and selection rules were fixed before H2 was evaluated (`experiments/nested_rerank_protocol.md`).

| Model | Top-1 | Top-5 | MRR |
|---|---:|---:|---:|
| Final CLIP | 3.27% | 11.44% | 0.097 |
| Final CLIP + block 8 | 3.10% | 12.91% | 0.100 |
| Final CLIP + block 8 + audio (Model C) | 3.27% | 13.73% | 0.104 |
| Top-3 reranker | 3.43% | 11.44% | 0.098 |
| Top-5 reranker | 3.10% | 11.44% | 0.095 |

Model C's top-5 gain over final CLIP was +2.29 pp with a 95% subject-resampled interval of -0.33 to +4.41 pp, so it is not statistically conclusive. Analysis: `experiments/rank_error_analysis_report.md`.

### Noise ceiling and unverified claims

Cross-subject repeatability suggests a perfect stimulus-to-mean-EEG model would reach roughly 15-19% top-1 with single-trial correlation retrieval (`experiments/contrastive_eeg_report.md`). A teammate's reported 13.9% top-1 has no artifacts in this repository and could not be reproduced: the best DINOv2 + CLAP model here reached 3.16% (`experiments/multimodal_contrastive_report.md`).


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
.venv/bin/python scripts/confirm_frozen_candidate.py --expected-config-sha256 afbbebb1121a39defd5c6b76b6f37a257104ebc831340e081c054418c7aa5daa   # one-shot; trains the 5 decoder seeds
.venv/bin/python scripts/build_final_ensemble.py --out-dir outputs/final_ensemble_repro   # S1 and S3 for sub-05/06 (+ sub-01 format check)
```

Fresh-partition validation of the decoders (seed 2060):

```bash
for s in 2050 2051 2052 2053 2054; do .venv/bin/python scripts/train_contrastive_eeg.py --variants D --seed $s --target features/all_train_clip_intermediate_block8.npy --outer-seed 2060 --out-dir experiments/fresh2060 --tag-suffix _block8_s$s; done
for s in 2050 2051 2052 2053 2054; do .venv/bin/python scripts/train_cross_subject_eeg.py --variants B --seed $s --outer-seed 2060 --out-dir experiments/fresh2060 --tag-suffix _s$s; done
.venv/bin/python scripts/modelc_folds.py --outer-seed 2060 --out-dir experiments/fresh2060/ModelC
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
| `outputs/contrastive_candidate/` | Frozen contrastive decoder checkpoints, normalisation, development-check scores |
| `outputs/final_ensemble_v1/` | Versioned S1 and S3 predictions for sub-05/06 (independent and Sinkhorn), scores, EEG-to-EEG decoder checkpoints, manifest |
| `outputs/previous_root_submissions/` | Backups of superseded root submissions |
| `tests/` | pytest suite (synthetic tests plus data-dependent integrity tests that skip without data) |

Excluded from Git and reproducible with the commands above: organizer data and documents, stimulus videos, feature caches (`features/*.npy`), legacy baseline checkpoints (`experiments/*.joblib`), and per-fold score archives (`experiments/**/*.npz`).

## Limitations

- About 850-970 training recordings per subject limit what a high-dimensional regression can learn.
- Performance varies across subjects. Model C is subject-specific; the neural decoders are shared across subjects with small subject adapters.
- Sub-06 has the fewest recordings (850). The final CLIP Ridge alpha for sub-06 reaches the top of the grid, and fusion reduced sub-06 performance on both independent holdouts.
- 40 of 1,000 training and 3 of 102 test videos have no or near-silent audio, so the audio component carries no information for them.
- Each holdout has 102 videos; differences of 1-2 pp are within validation noise.

## Submission

| File | SHA-256 |
|---|---|
| `sub-05_ranked_candidate_list.npy` | `7921fbf58bc2729e188e669ff4ecb8e155884a98fbcbab243ebc10a467e2ad35` |
| `sub-06_ranked_candidate_list.npy` | `e252e4c8cc9c42bcd141bc1ced2084cc5225f87aed8c3148da32f289a4b0a1f1` |

The root files above are frozen Model C. The stronger S3 predictions are stored separately and have not replaced them:

| File | SHA-256 |
|---|---|
| `outputs/final_ensemble_v1/S3/independent/sub-05_ranked_candidate_list.npy` | `8617a074c9921bbde1ff9abdb54c000067ffbb6be93e70bc47b114614557b102` |
| `outputs/final_ensemble_v1/S3/independent/sub-06_ranked_candidate_list.npy` | `c612371ab2622c4db02251be7d2e8a8a0454f5604c2ba44ac00ba8397c89212b` |

Each file is an integer array of shape (102, 102). Row i corresponds to recording i of `tst/sub-XX_tst.npy` in original order and contains every candidate ID 1001-1102 exactly once, ordered from most to least likely.

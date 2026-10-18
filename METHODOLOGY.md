# Methodology

This document describes how the submitted system (S3) was built and how every reported number was measured. Results are in `README.md`; full experiment records are in `experiments/`.

## 1. Task and data
- **Task:** for each 3-second EEG recording (17 channels, 500 Hz, 17 x 1500 samples from video onset), rank 102 candidate videos (IDs 1001-1102) from most to least likely.
- **Data:** six subjects have training recordings for 850-972 of the 1,000 training videos; 695 videos were recorded for all six.
- **Test sets:** each subject's test set has one recording of each of the 102 candidate videos, in shuffled order.
- **Labels:** test labels exist for sub-01..04 only. Hidden subjects sub-05 and sub-06 are scored by the organizers.
- **Organizer preprocessing:** 0.1-40 Hz filtering, blink-component removal and downsampling to 500 Hz. We add no further filtering.

## 2. Pipeline
### 2.1 EEG representation
- **Downsampling:** zero-phase FIR decimation by 5 (`scipy.signal.decimate`) to 100 Hz, giving 17 x 300 samples over 0-3000 ms.
- **Neural decoders:** per-subject, per-channel z-scoring, with mean and standard deviation computed over that subject's training recordings only and applied unchanged to its test recordings.
- **Row order:** test recordings are always processed in file order and never sorted.

### 2.2 Stimulus representation (frozen, public pretrained models)
- **CLIP:** OpenCLIP ViT-B/32 (OpenAI weights, QuickGELU) on 8 uniformly sampled frames, mean-pooled.
  - Final projected embedding: 512-d.
  - CLS token after transformer block 8: 768-d.
- **Simple audio:** 16 kHz mono; 12 RMS-envelope bins plus mean and standard deviation of 32 log-mel bands (76-d). Clips with no audio stream or a silent track (40 training, 3 candidate videos) map to a fixed silence vector.
- **Feature rows:** aligned to explicit video IDs (row = ID - 1 for training, ascending 1001-1102 for candidates) and checked on load.

### 2.3 Scorer 1: Model C (linear forward encoding)
- **Model:** per subject and per feature set, StandardScaler -> PCA (95% variance) -> StandardScaler -> RidgeCV (20 alphas, 1e-3 to 1e6) predicting the full 17 x 300 EEG response. Fitted on that subject's training recordings only.
- **Score:** Pearson correlation between measured and predicted EEG over 50-600 ms, flattened across channels (organizer scoring).
- **Fusion:** per-recording (row) z-scored scores combined as final CLIP 0.45, block 8 0.45, audio 0.10.

### 2.4 Scorer 2: cross-subject contrastive decoder
- **Encoder:** EEGNet-style, shared by all six subjects.
  - Per-subject 17 x 17 channel adapter, identity-initialised.
  - Temporal convolution (16 filters, 250 ms) and depthwise spatial convolution (x2).
  - Batch norm, ELU, average pooling, dropout 0.25.
  - Separable convolution (32), then a linear layer to a 128-d L2-normalised embedding.
  - About 66k encoder parameters.
- **Video side:** a linear head maps the frozen CLIP block-8 embedding to the same 128-d space.
- **Multi-positive contrastive loss:**
  - EEG-to-video cross-entropy over the distinct videos in a batch.
  - Video-to-EEG cross-entropy averaged over all recordings of that video.
  - A 0.5-weighted EEG-to-EEG term whose positives are the same video recorded from a different subject.
  - Recordings of the same video are never treated as negatives.
- **Batches:** all training recordings of 64 videos, so same-video recordings never straddle batches.
- **Training:**
  - AdamW (lr 1e-3, weight decay 0.1), learnable temperature (initial 0.07), gradient clipping 1.0.
  - Augmentation: +/-20 ms time shift and +/-10% amplitude jitter.
  - Up to 200 epochs, early stopping (patience 25) on retrieval MRR over a 10% inner-validation set of training video IDs.
- **Ensemble:** five seeds (2050-2054), averaged as row-z-scored cosine similarities. Frozen in `configs/frozen_contrastive_candidate.json`.

### 2.5 Scorer 3: EEG-to-EEG decoder
- **Encoder and training:** same encoder, adapters and training schedule as scorer 2.
- **Loss:** cross-subject supervised contrastive loss only. Positives are recordings of the same training video from different subjects; all other videos are negatives.
- **Readout:** after training, a RidgeCV readout (one row per training video) maps CLIP block-8 features to each video's mean EEG embedding. Candidates are scored by cosine similarity between the recording embedding and the readout prediction.
- **Ensemble:** five seeds, averaged after row z-scoring.

### 2.6 Fusion and ranking (S3)
- **Fusion:** S3 = (row-z decoder ensemble + row-z EEG-to-EEG ensemble + row-z Model C) / 3.
- **Ranking:** each recording is ranked independently by a stable descending sort over the explicit candidate IDs. No joint assignment across recordings is used in the submission.

## 3. Validation protocol
| Rule | How it is enforced |
|---|---|
| No test EEG in training | All fitting reads organizer `trn/` archives only. Test EEG is an inference input only. |
| Video-identity hold-out | Every outer fold removes all recordings of its 102 videos from training for **all** subjects (asserted in code). |
| Train-only transforms | Scalers, PCA, Ridge, channel normalisation, readouts and early stopping use training-fold data only. |
| Grouped inner validation | Hyperparameters, loss weights and early stopping use whole held-out training video IDs, never outer folds or test labels. |
| Fixed candidate set | Every evaluation ranks exactly 102 candidates. |
| Paired comparisons | All systems are scored on identical held-out videos and recordings. |
| Uncertainty | 95% bootstrap intervals resample subjects (the independent unit). Trial-level sign tests are reported as optimistic. |
| No templates | Development test EEG is never used as a reference response for hidden subjects. |

**Validation stages, in order:**
1. Screening folds and the H2 holdout (Model C selection).
2. Six exploratory outer folds (seed 2040) for neural decoder development.
3. A single pre-registered run of the frozen candidate on the sub-01..04 test recordings (the challenge PDF permits these labels for validation; predictions were hashed before labels were read).
4. A fresh outer partition (seed 2060) to confirm S3 against the frozen candidate under a rule written in advance.

All shared training videos have now been inspected at least once, so no untouched training-video holdout remains.

## 4. What was tried and rejected
| Approach | Finding |
|---|---|
| DINOv2 (ViT-S, ViT-B), VideoMAE, PANNs, CLAP | Weaker than CLIP as targets; adding them to CLIP block 8 hurt |
| Other CLIP blocks (4, 12), two-segment temporal CLIP | No consistent gain |
| PLS, target PCA, pooled Ridge, reverse EEG-to-CLIP Ridge | Worse than subject-specific forward Ridge |
| Alternative scoring windows | No consistent gain for Model C |
| Temporal audio and visual TRFs | Above chance, but weaker than Model C |
| Average-response regression target | No gain over contrastive training |
| Frozen pretrained CBraMod EEG encoder | Near chance |
| Compact EEG Transformer (teammate branch, 1 fold) | Weaker than EEGNet |
| Hungarian / Sinkhorn / row-column calibration | Hungarian no gain; others within noise (+0.2 pp top-1 for S3) |

## 5. Expected ceiling
Responses of different subjects to the same video correlate weakly (r = 0.06 per single-trial pair; match vs mismatch z = 33-45 against a permutation null). Extrapolating template identification with the Spearman-Brown relation suggests that even a perfect model of the mean EEG response would reach only about 15-19% top-1 with single-trial correlation retrieval. The present system reaches 5.7% on train-only validation.

## 6. Reproducibility
- **Software:** Python 3.11 with dependency versions recorded in each run's `metadata.json`.
- **Seeds:** fixed in the configuration files.
- **Checkpoints and hashes:** `outputs/contrastive_candidate/models/` (scorer 2), `outputs/final_ensemble_v1/models_B/` (scorer 3) and `outputs/final_model_c/models/` (Model C), with SHA-256 manifests.
- **Excluded from Git:** organizer data, stimulus videos and feature caches. Exact commands to regenerate them are in `README.md`.

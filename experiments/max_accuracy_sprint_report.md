# Maximum-accuracy sprint (2026-10-18)

The 20% top-1 target was **not reached**. The strongest verified system is **S3 = equal-thirds row-z fusion of the frozen block-8 contrastive decoder (5 seeds), the EEG<->EEG decoder B (5 seeds) and Model C**. On a fresh outer partition it scored **5.69% top-1 / 22.98% top-5 / MRR 0.157** (Model C: 3.08 / 13.04 / 0.100; frozen candidate S1: 4.90 / 19.14 / 0.139). Hidden-subject accuracy is unknown.

The noise-ceiling analysis (`experiments/contrastive_eeg_report.md`) puts a perfect stimulus -> mean-EEG model at roughly 15-19% top-1 for single-trial correlation retrieval. That was the reason for not spending the remaining budget on larger models.

## 1. Audit of existing models and the 13.9% claim
- **Repository-wide search:** no branch, artifact, report or score file contains a 13.9% result. Branches are `main` and `research/temporal-eeg-encoding` only. It is still unverifiable: candidate count, metric (top-1 vs top-5), held-out policy and subject exclusion are all unknown.
- **Teammate Transformer branch:** `experiments/eeg_transformer/E` has 1 of 6 folds, at 1.96% top-1 / 9.64% top-5. Not stronger.
- **Strongest existing models:** frozen candidate (S1) and EEG<->EEG decoder B (`experiments/cross_subject_eeg_report.md`).

## 2. Global assignment / calibration (label-free; `src/assignment.py`, `scripts/benchmark_assignment.py`)
Each method was applied per subject to the 102 x 102 held-out score matrix, with rows shuffled first. The true permutation is never used.

Seed-2040 folds (reused):
| System | Independent | Column z | Iterative row/col z | Sinkhorn tau 0.5 | Hungarian-first |
|---|---:|---:|---:|---:|---:|
| Model C | 2.97% | 3.40% | 3.35% | 3.43% | 2.97% |
| Decoder (5 seeds) | 4.90% | 4.79% | 4.68% | 5.12% | 4.60% |
| Decoder + Model C (S1) | 5.75% | 5.64% | 5.64% | 6.13% | 5.61% |
| B + Model C | 5.80% | 6.32% | 6.35% | 6.70% | 6.18% |
| Decoder + B + Model C | 6.13% | 6.70% | 6.75% | 6.89% | 6.24% |

Fresh seed-2060 partition (the confirmation; top-1):
| System | Independent | Column z | Iterative | Sinkhorn tau 0.5 | Hungarian + Sinkhorn | Hungarian + raw |
|---|---:|---:|---:|---:|---:|---:|
| Model C | 3.08% | 3.40% | 3.32% | 3.21% | 3.38% | 3.38% |
| S1 | 4.85% | 5.28% | 5.17% | 5.07% | 4.33% | 4.44% |
| S3 | 5.64% | 5.36% | 5.36% | 5.96% | 5.45% | 5.36% |

- **Hungarian:** no gain on either partition, and it hurts the decoder systems by up to 0.7 pp.
- **Sinkhorn:** gains are small (+0.2 to +0.4 pp on fresh S1/S3) and within noise (fresh S3: +0.19 pp, CI -0.30 to +0.82, top-1 W/L 73/66). With single-trial accuracy around 5%, the one-to-one constraint has little to exploit.
- **Perturb-and-MAP:** implemented but not run, given the null Hungarian/Sinkhorn results.
- **Permission:** these calibrations use the other recordings of the same subject. The PDF allows "unsupervised normalization of the test recordings" but does not address joint assignment, so organizer confirmation is needed before submitting any calibrated variant.

## 3. Fresh-partition confirmation (pre-specified in `experiments/fresh2060_protocol.md`)
- **What was retrained:** all decoders (5 + 5 seeds) and Model C on a new 6-fold partition (seed 2060; 3,672 trials). At most 19/102 videos per fold overlap any seed-2040 fold, and 72 of the 612 videos were never in a seed-2040 outer fold. The video identities were seen in earlier studies, so this is a fresh split, not untouched data.

| System | Top-1 | Top-5 | MRR | Train-only status |
|---|---:|---:|---:|---|
| Model C | 3.08% | 13.04% | 0.100 | frozen |
| Decoder 5 seeds | 4.66% | 17.37% | 0.130 | frozen recipe |
| B 5 seeds | 5.56% | 22.00% | 0.152 | recipe from cross-subject study |
| S1 frozen candidate | 4.90% | 19.14% | 0.139 | frozen |
| **S3 decoder + B + Model C** | **5.69%** | **22.98%** | **0.157** | **selected by pre-specified rule** |
| S3 + Sinkhorn tau 0.5 | 5.88% | 23.31% | 0.160 | permission-gated |

Paired on the fresh partition (6-subject bootstrap CIs; trial sign tests are optimistic):
- **S3 vs S1:** top-1 +0.79 pp [+0.41, +1.14] (W/L 70/41), top-5 +3.84 pp [+2.91, +4.77], MRR +0.018 [+0.013, +0.022].
- **S3 vs Model C:** top-1 +2.61 pp [+2.31, +2.89], top-5 +9.94 pp [+8.39, +11.11].

Per subject, top-1 / top-5:
| System | 01 | 02 | 03 | 04 | 05 | 06 |
|---|---:|---:|---:|---:|---:|---:|
| Model C | 3.6 / 16.5 | 3.8 / 15.4 | 3.4 / 12.6 | 3.1 / 11.8 | 2.5 / 12.1 | 2.1 / 10.0 |
| S1 | 5.1 / 24.2 | 4.9 / 20.3 | 5.9 / 18.0 | 4.6 / 20.1 | 4.2 / 16.3 | 4.7 / 16.0 |
| S3 | 6.4 / 27.6 | 6.2 / 26.0 | 6.2 / 22.1 | 5.7 / 23.4 | 4.4 / 18.3 | 5.2 / 20.6 |

S3 is at least as good as S1 for every subject on both metrics.

## 4. Versioned predictions (`outputs/final_ensemble_v1`, root files untouched)
- **Built by** `scripts/build_final_ensemble.py`:
  - Frozen decoder checkpoints from `outputs/contrastive_candidate/models`.
  - B retrained with 5 seeds on all training recordings, using the same inner-validation split (seed 3100) and normalisation statistics (asserted identical). Readouts were fitted on training videos only.
  - Model C scores from `outputs/final_model_c`, checked to be identical.
- **Variants:** S1 and S3, each with independent and Sinkhorn ranking. All are (102, 102) int64 permutations of 1001..1102 in test-file order. Covered by `tests/test_final_ensemble.py` and MANIFEST hashes.
- **Consistency check:** the S1 sub-01 ranking equals the frozen-candidate confirmation ranking exactly.
- **Format check, sub-01** (organizer-recommended; computed after outputs were fixed, used for nothing): S1 7.84% / 37.25%; S3 11.76% / 37.25%; S3 + Sinkhorn 8.82% / 37.25%.

| File | SHA-256 |
|---|---|
| `S3/independent/sub-05_ranked_candidate_list.npy` | `8617a074c9921bbde1ff9abdb54c000067ffbb6be93e70bc47b114614557b102` |
| `S3/independent/sub-06_ranked_candidate_list.npy` | `c612371ab2622c4db02251be7d2e8a8a0454f5604c2ba44ac00ba8397c89212b` |

## 5. Not pursued, and why
- **Bidirectional / CCA / PLS / kernel ridge:** PLS, target PCA and reverse EEG->CLIP ridge were all clearly worse in earlier controlled studies.
- **More CLIP layers or modalities:** blocks 4/12, DINOv2 and CLAP were worse or neutral.
- **Additional scoring windows:** tested twice, with no consistent gain.
- **Larger encoders:** the train/validation gap and the noise ceiling show capacity is not the bottleneck.

## Leaderboard (fresh partition seed 2060 unless noted; all train-only CV)
| Rank | Model | Top-1 | Top-5 | MRR | Valid | Retained |
|---:|---|---:|---:|---:|---|---|
| 1 | S3 + Sinkhorn 0.5 | 5.88% | 23.31% | 0.160 | yes; needs organizer OK | as alternative |
| 2 | **S3** | **5.69%** | **22.98%** | **0.157** | yes | **current best** |
| 3 | B 5 seeds | 5.56% | 22.00% | 0.152 | yes | component |
| 4 | S1 frozen candidate | 4.90% | 19.14% | 0.139 | yes | fallback |
| 5 | Decoder 5 seeds | 4.66% | 17.37% | 0.130 | yes | component |
| 6 | Model C | 3.08% | 13.04% | 0.100 | yes | current root submission |

# Cross-subject neural-consistency learning (exploratory)

Executed 2026-10-17/18 on `research/temporal-eeg-encoding` (Apple M5 Pro, MPS). Only organizer training archives were read. Development and hidden test EEG were not used. The frozen candidate (`configs/frozen_contrastive_candidate.json`) and Model C are unchanged.

**Validation status: exploratory.** These are the same 6 outer folds of 102 shared training videos (seed 2040) used by the frozen candidate's development, so the folds have already been inspected. Loss weights were selected per outer fold on grouped inner validation (10% of training video IDs, all subjects' recordings of them), never on outer-fold results. None of these variants has been evaluated on development test recordings.

## Variants
All variants use the same compact model and optimiser as the frozen candidate: shared EEGNet (64k parameters) + per-subject 17x17 adapters, AdamW lr 1e-3, weight decay 0.1, dropout 0.25, batches of all recordings of 64 videos, patience 25. Code: `src/cross_subject_learning.py`, `scripts/train_cross_subject_eeg.py`; config `configs/cross_subject_eeg.json`.

| Variant | Objective | Candidate scoring | Inner selection |
|---|---|---|---|
| A | Frozen-candidate decoder recipe: multi-positive EEG->CLIP-block-8 + 0.5 x EEG<->EEG | Learned linear video head | none (fixed) |
| B | **EEG<->EEG only**: supervised contrastive loss; positives = same training video from a different subject; all other videos are negatives | RidgeCV (LOO over videos) from block-8 features to each training video's mean EEG embedding (fit-train recordings only) | none (single setting) |
| C | EEG->CLIP multi-positive + w x EEG<->EEG | Video head | w in {1, 2}: chose 1 in 5 folds, 2 in 1 |
| D | C + lambda x MSE to a leave-own-subject-out average response of the same video (fit-train recordings only; train-fitted standardisation + PCA-64) | Video head | lambda in {0.25, 1}: chose 0.25 in 4 folds, 1 in 2 |

**Safeguards (unit-tested in `tests/test_cross_subject_learning.py`):**
- Positives are same-video, different-subject pairs only; same-video recordings are never negatives.
- Average-response targets use only fit-train recordings and never the anchor's own subject. Held-out and inner-validation EEG never contribute.
- The readout is fitted on fit-train recordings only.
- Per-subject normalisation is fitted on the fit-train set.
- Every recording of an outer-fold video is excluded for all subjects (asserted).

## Results (3,672 trials; one seed = 2050 unless noted)
| Model | Top-1 | Top-5 | MRR | dTop-1 vs Model C [95% CI] | dTop-5 vs Model C [95% CI] | Top-1 W/L vs C |
|---|---:|---:|---:|---:|---:|---:|
| Model C | 2.97% | 13.67% | 0.102 | - | - | - |
| A (candidate recipe, 1 seed) | 4.06% | 15.52% | 0.120 | +1.09 [+0.71, +1.39] | +1.85 [-0.11, +3.21] | 140/100 |
| **B EEG<->EEG + readout** | **5.36%** | **19.99%** | **0.143** | +2.40 [+1.61, +3.24] | +6.32 [+4.41, +8.01] | 182/94 |
| C EEG->CLIP + weighted EEG<->EEG | 4.11% | 16.04% | 0.120 | +1.14 [+0.76, +1.61] | +2.37 [+0.41, +4.00] | 144/102 |
| D C + average-response target | 4.17% | 15.93% | 0.121 | +1.20 [+0.82, +1.69] | +2.26 [+0.27, +3.89] | 145/101 |
| Frozen candidate decoder (5 seeds) | 4.90% | 18.08% | 0.134 | +1.93 [+1.20, +2.45] | +4.41 [+2.48, +6.24] | 168/97 |
| B, 5 seeds | 5.53% | 21.57% | 0.153 | +2.56 [+1.99, +3.16] | +7.90 [+6.37, +9.42] | 191/97 |
| Frozen candidate (decoder 5 seeds + Model C) | 5.64% | 20.32% | 0.147 | +2.67 [+2.26, +3.00] | +6.64 [+6.13, +7.22] | 152/54 |
| B 5 seeds + Model C (same 0.5/0.5 rule) | 5.75% | 22.41% | 0.156 | +2.78 [+2.02, +3.59] | +8.74 [+7.73, +9.89] | 141/39 |

Single-seed B for seeds 2050-2054: top-1 5.36 / 5.58 / 5.39 / 5.01 / 5.12%, top-5 19.7-20.8%. Every seed beats every single-seed A/C/D run.

Paired against the frozen candidate (`paired_vs_candidate.json`):
| Comparison | dTop-1 [CI] | dTop-5 [CI] | dMRR [CI] | Top-1 W/L (p) | Top-5 W/L (p) |
|---|---:|---:|---:|---:|---:|
| B 5 seeds vs candidate decoder 5 seeds | +0.63 [+0.14, +1.14] | +3.49 [+2.45, +4.74] | +0.019 [+0.013, +0.025] | 138/115 (0.17) | 392/264 (<1e-6) |
| B 5 seeds + Model C vs frozen candidate | +0.11 [-0.60, +0.82] | +2.10 [+1.44, +2.81] | +0.009 [+0.004, +0.014] | 92/88 (0.82) | 246/169 (<0.001) |
| B 5 seeds alone vs frozen candidate | -0.11 [-0.90, +0.76] | +1.25 [-0.11, +2.40] | +0.006 [-0.002, +0.014] | 152/156 (0.86) | 420/374 (0.11) |

Per subject, top-1 / top-5:
| Model | 01 | 02 | 03 | 04 | 05 | 06 |
|---|---:|---:|---:|---:|---:|---:|
| Model C | 4.2 / 19.0 | 3.4 / 15.2 | 2.5 / 13.4 | 3.1 / 12.6 | 2.5 / 11.6 | 2.1 / 10.3 |
| A | 4.6 / 16.2 | 4.4 / 16.8 | 3.9 / 17.2 | 4.6 / 15.0 | 3.9 / 14.2 | 2.9 / 13.7 |
| B | 8.2 / 21.2 | 5.2 / 23.0 | 6.0 / 19.3 | 4.7 / 19.1 | 4.7 / 17.5 | 3.3 / 19.8 |
| B 5 seeds | 5.7 / 23.7 | 7.0 / 24.8 | 5.2 / 21.2 | 5.2 / 19.6 | 4.6 / 19.0 | 5.4 / 21.1 |
| Frozen candidate | 7.0 / 25.5 | 5.2 / 21.9 | 5.4 / 20.4 | 6.4 / 18.6 | 4.9 / 17.3 | 4.9 / 18.1 |
| B 5 seeds + Model C | 7.4 / 27.0 | 4.9 / 25.3 | 7.0 / 21.7 | 5.1 / 20.6 | 4.9 / 18.6 | 5.2 / 21.2 |

## Representation diagnostics (means over 6 folds; held-out fold videos)
| Variant | Cross-subject same-video EEG retrieval top-1 | Effective rank (of 128) | Train-probe MRR / inner-val MRR | Best epoch | Train / fold | Inference / fold | Trainable params |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw EEG (Phase 1 reference) | 2.7% | - | - | - | - | - | - |
| A | 6.7% | 94.5 | 0.65 / 0.151 | 30 | 58 s | 0.02 s | 164.5k |
| B | **8.6%** | 93.7 | **0.48** / **0.171** | 73 | 92 s | 0.21 s (incl. readout) | 66.0k + ridge readout |
| C | 7.1% | 92.6 | 0.62 / 0.153 | 27 | 83 s (2 settings) | 0.02 s | 164.5k |
| D | 7.1% | 92.6 | 0.62 / 0.153 | 27 | 77 s (2 settings) | 0.01 s | 172.7k |

## Findings
- **EEG-to-EEG contrastive learning helps substantially as a standalone decoder.** B gives the best single-seed and 5-seed results, the best cross-subject same-video retrieval (8.6% vs 6.7% for A), and a smaller train/validation gap. It trains about 2.4x longer before early stopping. A plausible explanation (not tested separately): without the CLIP head in the loss, the encoder cannot overfit to the ~900 training-video embeddings. The stimulus mapping is learned afterwards by a ridge readout with one row per video.
- **Adding more EEG<->EEG weight to the CLIP objective (C) barely helps** (+0.05 pp top-1 over A).
- **Average-response supervision does not help.** D is within 0.1 pp of C on every metric, with identical diagnostics.
- **After fusion with Model C, B's advantage over the frozen candidate is in top-5 and MRR** (+2.1 pp top-5, CI +1.4 to +2.8), **not top-1** (+0.1 pp, CI -0.6 to +0.8).
- **No collapse or subject shortcut:** effective rank is 93-99 for every variant.

## Limitations
- The outer folds are reused, so these numbers are not independent validation. The 5-seed B ensemble and its fusion were run after single-seed B was seen to be best.
- The subject-level CIs use 6 units. Trial sign tests are optimistic.
- B has not been evaluated on development test recordings. The one-shot development check was spent on the frozen candidate, and a second look would no longer be a clean confirmation.

## Reproduction
```bash
.venv/bin/python scripts/train_cross_subject_eeg.py                     # B, C, D (seed 2050)
.venv/bin/python scripts/train_cross_subject_eeg.py --variants A        # A through the same code path
for s in 2051 2052 2053 2054; do .venv/bin/python scripts/train_cross_subject_eeg.py --variants B --seed $s --tag-suffix _s$s; done
X=experiments/cross_subject_eeg; .venv/bin/python scripts/evaluate_contrastive_eeg.py --variants D_block8 D_block8_s2051 D_block8_s2052 D_block8_s2053 D_block8_s2054 xA:$X/A xB:$X/B xC:$X/C xD:$X/D xB_s2051:$X/B_s2051 xB_s2052:$X/B_s2052 xB_s2053:$X/B_s2053 xB_s2054:$X/B_s2054 --ensembles Cand_dec=D_block8,D_block8_s2051,D_block8_s2052,D_block8_s2053,D_block8_s2054 xB_ens5=xB,xB_s2051,xB_s2052,xB_s2053,xB_s2054 --fuse-with-modelc Cand_dec xB xB_ens5 --out $X/summary.json
```

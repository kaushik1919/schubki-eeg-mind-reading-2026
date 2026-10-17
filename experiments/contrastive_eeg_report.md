# Cross-subject contrastive EEG decoding

Executed 2026-10-17 on `research/temporal-eeg-encoding` (Apple M5 Pro, PyTorch 2.14.1, MPS). Only organizer training archives were read. No test EEG or test labels were used, and Model C and the root submissions are unchanged.

**Validation status: exploratory.** All comparisons use the 6 disjoint 102-video outer folds (seed 2040) from `temporal_audio_protocol.md`. These video identities were inspected in earlier studies, and only 17 shared videos have never been used for validation. Nothing here is independent confirmation.

## Summary
| Question | Answer (measured) |
|---|---|
| Repeatable cross-subject stimulus information? | Yes. Match vs mismatch correlation gap z = 33-45 against a 200-permutation null (p = 0.005, the minimum). Single-trial cross-subject identification: 2.7% top-1 / 10.7% top-5 (chance 0.98 / 4.9). The 5-subject average template reaches 6.5% / 20.3%. |
| Ceiling for a perfect stimulus -> mean-EEG model | About 15-19% top-1 and 39-45% top-5 (Spearman-Brown + Gaussian extrapolation; method below). 30% top-1 appears to exceed what correlation-based single-trial retrieval can reach on this EEG. |
| Best single decoder | D: shared EEGNet + subject adapters + multi-positive loss, with a CLIP **block-8** target. 4.03% / 15.58% / 0.1195 (one seed); seeds range 3.65-5.07% top-1. |
| Best measured system | 5-seed D-block8 ensemble + Model C (0.5/0.5 row-z): **5.64% / 20.32% / 0.147**, vs Model C 2.97% / 13.67% / 0.102 on the same folds. |
| Milestone | 4-6% band: a modest, consistent improvement. Not a breakthrough, and not yet validated on untouched data. |
| Subject adapters | Help: B -> C top-5 +1.06 pp, MRR +0.005. |
| Multi-positive loss | Helps: C -> D top-1 +0.43 pp, top-5 +1.36 pp, MRR +0.0075. |
| Pretrained EEG model | CBraMod (frozen probe) is feasible and permitted, but weak: 1.33% / 6.15%. |

## Phase 1: Is there decodable, repeatable signal?
`scripts/cross_subject_reliability.py` -> `experiments/cross_subject_reliability.json`. The 695 videos recorded for all six subjects; FIR decimation to 100 Hz; per-subject channel z-scoring; optional subject-mean evoked-response removal; Pearson correlation over channels x time. Identification is within the six 102-video groups.

| Condition | Match r | Mismatch r | Gap z vs null | Pairwise top-1 / top-5 | 5-subject template top-1 / top-5 |
|---|---:|---:|---:|---:|---:|
| broadband, ERP removed, 50-600 ms | 0.080 | -0.000 | 37.7 | 2.66% / 10.67% | 4.63% / 17.81% |
| broadband, ERP kept, 50-600 ms | 0.080 | -0.000 | 35.8 | 2.69% / 10.62% | 4.71% / 17.73% |
| broadband, ERP removed, 0-3000 ms | 0.061 | -0.000 | 37.4 | 2.91% / 10.81% | 6.54% / 20.32% |
| 1-15 Hz, ERP removed, 50-600 ms | 0.082 | -0.000 | 32.6 | 2.20% / 9.37% | 3.62% / 14.62% |
| 1-15 Hz, ERP removed, 0-3000 ms | 0.053 | -0.000 | 44.6 | 2.87% / 11.26% | 6.05% / 21.02% |

Controls:
- **Subject identity:** only cross-subject pairs are compared, so subject identity cannot create matches.
- **Amplitude:** correlation is scale-invariant.
- **Shared artifacts:** removing the subject-mean response leaves the result unchanged; mismatched videos correlate at about 0.
- **Recording order:** order is not provided. Similarity to numeric-ID neighbours (v +- 1, 2, 5, 50) is about 0 (-0.006 to +0.002), so ID-adjacent drift is absent.
- **Timing:** shifting one subject by -100/-50/-20/0/+20/+50/+100 ms gives r = -0.012 / 0.028 / 0.078 / **0.089** / 0.075 / 0.033 / -0.006. The peak is at zero lag, so responses are stimulus-locked and aligned across subjects.
- **Class balance:** exactly one recording per video per subject.

**Noise ceiling** (`scripts/noise_ceiling.py`). Template-based identification improves with the number k of averaged subjects (0-3000 ms top-1: 2.9, 4.0, 4.8, 5.8, 6.5% for k = 1..5). Match correlations follow Spearman-Brown closely (predicted 0.083 / 0.099 / 0.111 / 0.121 vs measured 0.082 / 0.097 / 0.109 / 0.118). Extrapolating to a noise-free mean response (r_inf = sqrt(0.060) = 0.246) and using a Gaussian identification model gives 18.9% top-1 / 44.5% top-5 (0-3000 ms) and 15.2% / 38.6% (50-600 ms). The Gaussian model slightly *under*-predicts measured accuracy (5.0% vs 6.5% at k = 5), so the true ceiling may be somewhat higher. It is still far below 30% under this decoding family.

## Methods
- **Data:** all six subjects' training recordings (5,650). For each outer fold, every recording of the 102 held-out videos is removed for all subjects. Inner validation holds out 10% of the remaining video IDs (all subjects' recordings of those videos, `rng(3000 + fold)`) for early stopping. Per-subject channel z-scoring is fitted on fit-train recordings only.
- **Encoder** (`src/eeg_encoder.py`): 17 x 1500 input, FIR-decimated to 100 Hz (as in Model C). Optional per-subject 17x17 adapter (identity init) -> temporal conv (16 x 250 ms) -> depthwise spatial conv (x2) -> BN/ELU/avg-pool/dropout -> separable conv (32) -> pool -> linear to 128-d, L2-normalised. 64.4k encoder parameters, 1.8k for adapters, plus a linear video head (65.7k for 512-d CLIP, 98.4k for 768-d block 8).
- **Targets:** frozen cached stimulus embeddings (CLIP final, CLIP block 8, DINOv2-S), L2-normalised, through a learnable linear projection.
- **Losses** (`src/contrastive_learning.py`):
  - Naive symmetric InfoNCE (A, B, C).
  - Multi-positive (D): EEG -> video over the distinct videos in the batch; video -> EEG averaging over all of its recordings; and a 0.5-weighted EEG <-> EEG term whose positives are the same video recorded from *different* subjects.
  - Batches hold all recordings of 64 videos, so same-video recordings never straddle batches.
- **Training:** AdamW (lr 1e-3, weight decay 0.1), dropout 0.25, temperature init 0.07 (learnable, capped at 100), gradient clip 1.0, +-20 ms shift and +-10% amplitude jitter, up to 200 epochs with patience 25 on inner-validation MRR, fp32. Dropout and weight decay come from a 4-point grid on fold-0 inner validation only (`experiments/contrastive_eeg/search_selection.json`).
- **Evaluation:** cosine similarity to the 102 candidate embeddings for each subject's recordings. Model C scores come from `experiments/temporal_audio` (same folds, same training recordings, alignment asserted). Statistics: 95% CIs from 20,000 resamples of the 6 subjects; the trial-level sign test is optimistic.

## Results (6 folds x 6 subjects x 102 trials = 3,672)
| Model | Top-1 | Top-5 | MRR | dTop-1 vs C [95% CI] | dTop-5 vs C [95% CI] | Top-1 W/L |
|---|---:|---:|---:|---:|---:|---:|
| Frozen Model C | 2.97% | 13.67% | 0.1021 | - | - | - |
| A subject-specific | 2.56% | 11.47% | 0.0907 | -0.41 [-1.14, +0.11] | -2.21 [-4.66, -0.65] | 86/101 |
| B shared, naive loss | 3.00% | 11.96% | 0.0966 | +0.03 [-0.57, +0.60] | -1.72 [-4.00, +0.19] | 103/102 |
| C shared + adapters | 3.08% | 13.02% | 0.1018 | +0.11 [-0.35, +0.52] | -0.65 [-2.94, +0.65] | 106/102 |
| D + multi-positive (CLIP final) | 3.51% | 14.38% | 0.1093 | +0.54 [-0.16, +1.14] | +0.71 [-1.96, +2.56] | 124/104 |
| D, DINOv2-S target | 3.10% | 12.36% | 0.1004 | +0.14 [-0.46, +0.49] | -1.31 [-3.08, -0.16] | 103/98 |
| **D, CLIP block-8 target** | 4.03% | 15.58% | 0.1195 | +1.06 [+0.68, +1.36] | +1.91 [-0.08, +3.30] | 139/100 |
| P frozen CBraMod + linear probe | 1.33% | 6.15% | 0.0599 | -1.63 [-2.29, -0.98] | -7.52 [-9.64, -5.53] | 48/108 |
| D CLIP-final, 5 seeds | 3.79% | 15.93% | 0.1193 | +0.82 [+0.16, +1.58] | +2.26 [+0.11, +4.08] | 132/102 |
| D block-8, 5 seeds | 4.90% | 18.08% | 0.1338 | +1.93 [+1.20, +2.45] | +4.41 [+2.48, +6.24] | 168/97 |
| D final / block-8 / DINOv2 (1 seed each) | 4.58% | 18.19% | 0.1329 | +1.61 [+0.71, +2.42] | +4.52 [+3.05, +5.77] | 158/99 |
| D block-8 (1 seed) + Model C | 5.01% | 18.79% | 0.1382 | +2.04 [+1.61, +2.48] | +5.12 [+4.60, +5.66] | 133/58 |
| **D block-8, 5 seeds + Model C** | **5.64%** | **20.32%** | **0.1468** | **+2.67 [+2.26, +3.00]** | **+6.64 [+6.13, +7.22]** | 152/54 |

- **Seed robustness (block-8 target):** top-1 3.65 / 4.03 / 4.49 / 5.07 / 3.87%, top-5 15.6-16.3%. Every seed beats Model C on both metrics.
- **Target choice:** on inner validation (training data only), MRR was 0.150 for block 8, 0.130 for CLIP final and 0.126 for DINOv2. The outer-fold ranking agrees.
- **Complementarity:** decoder scores correlate only 0.19-0.22 with Model C per recording, which explains the fusion gain.

Per subject, top-1 / top-5 / MRR:
| Model | sub-01 | sub-02 | sub-03 | sub-04 | sub-05 | sub-06 |
|---|---:|---:|---:|---:|---:|---:|
| Model C | 4.2 / 19.0 / 0.128 | 3.4 / 15.2 / 0.113 | 2.5 / 13.4 / 0.098 | 3.1 / 12.6 / 0.099 | 2.5 / 11.6 / 0.092 | 2.1 / 10.3 / 0.084 |
| A subject-specific | 2.1 / 10.8 / 0.088 | 3.4 / 13.6 / 0.104 | 2.6 / 13.2 / 0.100 | 3.1 / 11.1 / 0.091 | 1.8 / 11.3 / 0.083 | 2.3 / 8.8 / 0.079 |
| D (CLIP final) | 3.3 / 13.6 / 0.107 | 5.1 / 17.5 / 0.131 | 3.3 / 13.2 / 0.105 | 3.8 / 14.5 / 0.107 | 3.4 / 14.9 / 0.110 | 2.3 / 12.6 / 0.095 |
| D block-8, 5 seeds | 4.4 / 19.3 / 0.135 | 5.4 / 18.6 / 0.141 | 5.1 / 18.1 / 0.138 | 5.2 / 17.2 / 0.132 | 4.6 / 17.0 / 0.124 | 4.7 / 18.3 / 0.132 |
| D block-8, 5 seeds + Model C | 7.0 / 25.5 / 0.172 | 5.2 / 21.9 / 0.148 | 5.4 / 20.4 / 0.145 | 6.4 / 18.6 / 0.147 | 4.9 / 17.3 / 0.134 | 4.9 / 18.1 / 0.135 |

The shared decoder improves every subject. The largest relative gain is for sub-06, the weakest subject under Model C (top-1 2.1% -> 4.9%, top-5 10.3% -> 18.1%), consistent with cross-subject sharing compensating for its small training set.

## Compute
| Model | Trainable params | Train / fold | Inference (612 trials) |
|---|---:|---:|---:|
| A (6 models) | 129.9k each | 50 s | 0.06 s |
| B / C / D (CLIP final) | 129.9k / 131.7k / 131.7k | 36 / 38 / 42 s | <0.1 s |
| D block 8 | 164.5k | 38 s | <0.1 s |
| P (frozen CBraMod 4.88M + probe) | 142.6k | 4 s extraction + 7 s | - |
| Model C | Ridge | 0.65 s / subject | <0.01 s |

Early stopping triggers after 5-30 epochs (about 1.5 s per epoch on MPS). All 124 trained models (94 fold-runs, including the search) took 56 minutes of recorded training time. Mixed precision was not used, because the model is too small to benefit. MPS reproduced the fold-0 D inner-validation MRR exactly on a rerun (0.13558), but results differ slightly from CPU runs.

## Phase 4: Pretrained EEG representations
| Check | CBraMod (ICLR 2025) |
|---|---|
| Weights | Public: `huggingface.co/weighting666/CBraMod`, uploaded 2024-12-11 (before the hackathon); MIT license; code commit b9e96100 |
| Organizer rule | Public pretrained models released before the hackathon are allowed |
| Channels | Montage-flexible (criss-cross attention); our 17 electrode names are not documented, so the spatial prior cannot be matched |
| Preprocessing | 200 Hz, 1 s patches; polyphase 500 -> 200 Hz, 3 patches |
| Hardware | 4.88M parameters; extraction for all 5,650 recordings takes about 3 s on MPS |
| Result | Frozen features + linear probe: 1.33% / 6.15%, near chance. Not pursued further |

EEGPT (58 channels, 256 Hz, 4 s windows) and LaBraM need named 10-20 channels. They were not attempted because the montage is unknown. Code and weights live in `third_party/` (gitignored); re-fetch commands are in the script docstring.

## Phase 8: Failure analysis (fold 0, `failure_analysis.json`)
- **No collapse:** held-out embeddings have mean pairwise cosine 0.061 and effective rank 97.7 of 128.
- **No subject shortcut:** subject identity is decodable from embeddings at 20.1% (raw normalised EEG 19.3%; chance 16.7%).
- **Cross-subject alignment improves:** EEG <-> EEG identification across subjects is 5.9% top-1 in embedding space vs 2.7% for raw EEG, on unseen videos.
- **Large train/validation gap:** train-probe MRR is 0.36-0.95 vs inner-validation 0.12-0.16, and early stopping triggers within 5-30 epochs. Capacity is not the bottleneck; data per video is. Seed variance is large (single-seed top-1 3.65-5.07%), which is why ensembling helps.
- **Alignment:** confirmed correct (Phase 1 timing control).
- **Targets:** DINOv2 is worse than CLIP; mid-level CLIP (block 8) is the best target.
- **Bug found and fixed:** the D loss *value* was logged as NaN (masked self-similarity -inf x 0). Gradients were unaffected, and the fold-0 rerun with the fix reproduced the same validation MRR. A regression test was added.

## Limitations
- Exploratory folds reuse previously inspected video identities. Target choice and ensemble/fusion variants were evaluated together on these folds (several were declared before results were seen, but several comparisons were made).
- The CIs use 6 subjects as units. Trial sign tests are optimistic.
- Hidden-subject performance is unknown. The decoder has not been run on test EEG.

## Recommended next steps
1. **Freeze** "D block-8, 5 seeds + Model C (0.5/0.5)" and run **one** pre-registered confirmation. No untouched shared training videos remain. The organizer permits the sub-01..04 test labels for validation; this needs your approval, because the project protocol currently excludes them.
2. **Highest-impact modelling experiment:** add the cross-subject EEG <-> EEG objective more strongly. Train the shared encoder on all 5,650 training recordings with the same-video multi-positive term as the main loss (it doubled cross-subject identification), and use the denoised cross-subject response as an additional target. This attacks the measured bottleneck: single-trial SNR (rho = 0.06) and roughly 6 recordings per video.

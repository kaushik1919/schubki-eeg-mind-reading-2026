# Temporal audio TRF, grouped regularisation and scoring windows

Executed 2026-10-17 on branch `research/temporal-eeg-encoding`. Protocol fixed beforehand in `experiments/temporal_audio_protocol.md`. Generated tables are in `experiments/temporal_audio/summary_tables.md` and `summary.json`; per-fold rows are in `results.csv`. No test EEG or test labels were read, and the frozen Model C submissions are unchanged.

## Summary
- Time-resolved audio TRFs are above chance (top-5 9.4-10.0%, MRR CI lower bound 0.072-0.079 vs chance 0.051), but every variant is clearly worse than Model C.
- Under the corrected video-grouped regularisation, the low-level visual TRF reaches 7.87% top-5 (it was 5.72% on H3, a different partition) and stays the weakest model.
- The pre-declared fusion (0.9 Model C + 0.1 combined audio TRF) passed its gate. It improved over Model C by top-1 +0.22 pp, top-5 +0.65 pp [+0.33, +0.93] and MRR +0.0038 [+0.0022, +0.0051]. Top-5 was not lower in any of the 6 folds, and MRR rose in 5 of 6 subjects. This is a small, consistent, **exploratory** gain on previously inspected video identities.
- For Model C, 50-600 ms remains the best scoring window. For the envelope/onset TRF, the full 50-3000 ms window was best (12.12% top-5), which is consistent with audio-driven activity sustained across the clip. This is post hoc and needs confirmation.

## Setup
- **Validation:** exploratory nested train-only CV. Only 17 shared training videos had never been used for validation, so no untouched holdout exists. 6 disjoint outer folds of 102 shared video IDs (seed 2040, 612 IDs) were each evaluated once, for 3,672 trials (6 subjects x 6 folds x 102).
- **Training per fold:** each subject's remaining training recordings (748-870 videos), identical for Model C and every TRF.
- **Inner CV (TRF regularisation only):** 5 video blocks (`rng(2041)` permutation of IDs mod 5). All time samples of a video stay in one block. Lambda is chosen by mean inner-fold retrieval MRR at 50-600 ms.
- **EEG:** organizer FIR decimation to 100 Hz, full 0-3000 ms target, sample k = k x 10 ms after video onset.
- **TRF:** causal lags 0-600 ms (61 lags), zero padding before onset (all features are 0 for silence), per-time-sample intercept (training-mean evoked response), training-fold centring and column standardisation from sufficient statistics, ridge alpha = lambda x n_samples, lambda in logspace(-4, 2, 13). Implementation: `src/grouped_trf.py`, which matches an explicit ridge solve in tests.
- **Model C:** frozen ForwardRegressor constituents and 0.45/0.45/0.10 row-z fusion, refit on each outer fold's training set.

## Audio timeline audit (all 1,102 files)
| Check | Training (1,000) | Candidates (102) |
|---|---|---|
| Container duration | 3.00-3.07 s | 2.97-3.07 s |
| Audio stream present | 968 | 100 |
| First audio pts | 0.000 s in all files with audio | 0.000 s |
| First video pts | 0.000 s (984), 0.033 s (16, all without audio) | 0.000 s (101), 0.033 s (1, no audio) |
| Audio-video offset applied | 0.000 s for all | 0.000 s for all |
| Decoded audio length | 2.97-3.07 s (none short) | 2.97-3.07 s |
| Status | ok 960, no stream 32, silent 8 | ok 99, no stream 2, silent 1 |
| Native sample rate | 8-96 kHz; 121 files at 8 or 11.025 kHz | 8-48 kHz |

Missing, silent and failed audio all map to all-zero features (= silence). Low-rate sources have no energy in upper mel bands. Per-file records: `features/*_temporal_audio_audit.json`.

## Features (`src/temporal_audio.py`, 300 x 36 per video on the EEG grid)
| Group | Features |
|---|---|
| A envelope + onset (4) | RMS envelope^0.3 in 25 ms Hann-centred windows; positive envelope derivative; onset strength (mean positive log-mel difference); spectral flux (L2 of positive magnitude difference) |
| B log-mel (32) | log(1 + 1e4 x mel power), 30-8000 Hz, 512-point FFT |
| C combined (36) | A + B |

## Results (50-600 ms scoring, 3,672 trials)
| Model | Top-1 | Top-5 | MRR [subj. 95% CI] | dTop-5 vs C [CI] | dMRR vs C [CI] | Top-5 W/L vs C |
|---|---:|---:|---:|---:|---:|---:|
| Model C | 2.97% | 13.67% | 0.1021 [0.0914, 0.1144] | - | - | - |
| Final CLIP only | 2.21% | 10.73% | 0.0858 | -2.94% [-3.98, -2.12] | -0.0163 | 164/272 |
| A envelope + onset | 2.37% | 9.99% | 0.0843 [0.0794, 0.0908] | -3.68% [-6.05, -1.61] | -0.0178 | 306/441 |
| B log-mel | 1.74% | 9.37% | 0.0760 [0.0729, 0.0796] | -4.30% [-7.14, -1.69] | -0.0261 | 292/450 |
| C combined | 1.69% | 9.83% | 0.0789 [0.0718, 0.0854] | -3.84% [-6.40, -1.31] | -0.0232 | 299/440 |
| V visual (grouped) | 1.77% | 7.87% | 0.0710 [0.0643, 0.0781] | -5.80% [-8.28, -3.68] | -0.0311 | 244/457 |
| **Fusion 0.9 C + 0.1 C-combined TRF** | **3.19%** | **14.32%** | **0.1059** | **+0.65% [+0.33, +0.93]** | **+0.0038 [+0.0022, +0.0051]** | 54/30 (sign p = 0.012, optimistic) |

Per subject, top-1 / top-5 / MRR:
| Model | sub-01 | sub-02 | sub-03 | sub-04 | sub-05 | sub-06 |
|---|---:|---:|---:|---:|---:|---:|
| Model C | 4.2 / 19.0 / 0.128 | 3.4 / 15.2 / 0.113 | 2.5 / 13.4 / 0.098 | 3.1 / 12.6 / 0.099 | 2.5 / 11.6 / 0.092 | 2.1 / 10.3 / 0.084 |
| A envelope + onset | 2.0 / 10.5 / 0.085 | 2.0 / 9.5 / 0.077 | 2.5 / 9.5 / 0.082 | 3.4 / 12.3 / 0.099 | 2.5 / 9.5 / 0.085 | 2.0 / 8.8 / 0.079 |
| C combined | 2.1 / 10.9 / 0.085 | 0.7 / 7.4 / 0.063 | 1.5 / 9.2 / 0.077 | 2.6 / 12.1 / 0.090 | 1.3 / 11.4 / 0.082 | 2.0 / 8.0 / 0.076 |
| Fusion | 4.7 / 19.8 / 0.133 | 3.1 / 15.5 / 0.113 | 2.8 / 14.5 / 0.102 | 3.4 / 13.4 / 0.104 | 2.6 / 11.6 / 0.096 | 2.5 / 11.1 / 0.087 |

Sub-06 is again the weakest subject for Model C, and it gains slightly from fusion here (+0.8 pp top-5). For sub-04, the audio-only envelope TRF matches Model C on MRR (0.099).

## Scoring windows (exploratory, same fitted predictions; top-1 / top-5 / MRR)
| Model | 50-600 | 100-1000 | 200-1500 | 500-2500 | 50-3000 |
|---|---:|---:|---:|---:|---:|
| Model C | 2.97 / **13.67** / **0.1021** | 3.02 / 12.09 / 0.0974 | 2.67 / 10.70 / 0.0875 | 1.55 / 7.19 / 0.0656 | 2.04 / 10.35 / 0.0826 |
| A envelope + onset | 2.37 / 9.99 / 0.0843 | 2.31 / 9.69 / 0.0826 | 2.26 / 9.45 / 0.0808 | 2.29 / 9.23 / 0.0785 | **3.16 / 12.12 / 0.0977** |
| C combined | 1.69 / 9.83 / 0.0789 | 2.53 / 9.42 / 0.0834 | 2.29 / 9.23 / 0.0790 | 2.31 / 9.18 / 0.0782 | 2.70 / 10.32 / 0.0867 |

Model C's information is concentrated early: performance falls steadily for later windows. The audio TRFs hold up over late windows (500-2500 ms is close to their 50-600 ms level), and their full-window score is the best. This is evidence that the soundtrack drives EEG throughout the clip, which the static-feature Model C cannot capture.

## Runtime (per subject-fold, M5 Pro CPU)
Sufficient statistics take about 2 s per subject for all atoms. TRF training including the 5-fold x 13-lambda inner search takes 0.6 s (A), 3.1 s (B) and 4.0 s (C), versus 0.65 s for Model C. Inference for 102 candidates takes under 0.03 s for every model. The full run took 334 s.

## Failure analysis
- Regularisation is poorly identified: the selected lambda ranged over 1e-4 to 30 across folds, and inner MRR (0.057-0.062) was barely above chance. The inner retrieval surface is flat, so the regulariser is chosen mostly by noise.
- Log-mel features (32 bands x 61 lags = 1,952 columns) are weaker than the 4 envelope/onset features. With ~250k correlated samples from ~800 videos, the extra spectral detail overfits rather than helps.
- The TRF is linear and subject-specific. It cannot share the cross-subject auditory response, and each subject only has ~850 three-second clips.
- The audit found no misalignment. Audio starts at video onset in every file with audio, so timing is not the cause of the weak standalone results.

## Comparison with Model C and decision
- Temporal audio alone: rejected as a replacement (top-5 -3.7 to -4.3 pp).
- Fusion: keep as the candidate next prototype, but do not promote it to the submission yet. The gain is small (+0.65 pp top-5, about 4 trials per subject over 612), measured on previously inspected IDs, and the fusion weight was fixed rather than validated.

## Limitations
- Exploratory only: the outer folds reuse video identities seen in earlier screening, confirmation, H2, H3 and nested-rerank evaluations. Model C's design was chosen on overlapping IDs, which may favour Model C, not the fusion.
- The trial-level sign test treats trials as independent and is optimistic. The subject CIs use only 6 units.
- The window comparison was not pre-registered for model selection, and the A 50-3000 ms result is post hoc.

## Recommended next experiment
A cross-subject shared response model. Fit a group-level stimulus-to-EEG mapping (Model C features plus the envelope/onset TRF) on the training recordings of all six subjects, with a per-subject linear readout (for example, reduced-rank or CCA-aligned channels). Evaluate it with the same 6-fold grouped protocol. This attacks the main limitation found here: each subject has only ~850 clips, while the same stimuli were seen by all six subjects (695 shared videos). Pre-declare the 50-3000 ms window for the envelope/onset component in that experiment, rather than adopting it from this post hoc result.

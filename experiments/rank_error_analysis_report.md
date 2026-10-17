# Top-1 vs top-5 rank-error analysis and nested top-K reranking test

All numbers below were executed on 2026-10-17 from train-only validation artifacts. No organizer test EEG, development test labels, or root submission files were read or modified. Full generated tables: `experiments/rank_error_analysis_tables.md` (+ `.json`); nested runs: `experiments/nested_rerank/{inner,outer}_results.json`, `selection.json`.

## 1. Conclusion first
- The "block 8 raises top-1 but lowers top-5" pattern is specific to the confirmation holdout. On the screening folds, the 5-fold inner validation, and the new untouched holdout H2, the final+block8(+audio) fusions *raised* top-5.
- The top-1 gain did not survive independent validation. On H2, final CLIP and the frozen fusion C both scored 3.27% top-1; B and D5 scored 3.10%.
- The conservative top-K reranker (D) keeps top-5 identical to final CLIP by construction. It gave +0.75 pp top-1 in inner validation (K*=5) but -0.16 pp on H2. It does not beat simple fusion.
- Recommended freeze: **C = final CLIP / block 8 / audio, 0.45 / 0.45 / 0.10 row-z score fusion**. It was chosen for all three roles by the prespecified inner rules. On H2 it matched final CLIP on top-1 and had +2.29 pp top-5 (subject CI -0.33 to +4.41 pp). That improvement is suggestive, not significant.

## 2. Why block 8 trades top-1 for top-5 (confirmation holdout, 612 trials)
| Config | Top-1 | Top-5 | MRR | Top-1 W/L vs final | Top-5 W/L vs final | 2-5 -> 1 | Top-5 -> out |
|---|---:|---:|---:|---:|---:|---:|---:|
| final | 2.45% | 12.25% | 0.0887 | - | - | - | - |
| block8 | 3.10% | 9.97% | 0.0923 | 18/14 | 47/61 | 2 | 61 |
| final+block8 | 3.27% | 10.95% | 0.0955 | 13/8 | 30/38 | 9 | 38 |
| final+block8+audio | 3.27% | 11.60% | 0.0981 | 13/8 | 33/37 | 7 | 37 |

Mechanism (measured):
- Block 8 is a largely different ranker, not a refinement of final CLIP. The row-wise correlation between their candidate scores is only 0.31 (screening 0.29), and their top-5 sets share just 0.70 of 5 candidates on average.
- Block 8 alone wins top-1 mostly by pulling targets up from outside the top 5 (16 of 18 wins came from rank > 5). It also pushes 61 of final's 75 top-5 hits out of its own top 5, so it behaves like an independent noisy ranker.
- Inside the fusion, promotions to rank 1 happen when block 8 also ranks the target highly (median block-8 rank 3 for 2-5 -> 1 promotions). Demotions out of the top 5 happen when block 8 places the target near chance (median block-8 rank 50). The top-5 losses are therefore block-8 noise, not systematic bias.
- Final CLIP's confidence carries no usable signal. Its top-1 vs top-2 z-margin averages 0.27 when correct and 0.30 when wrong (screening 0.33 vs 0.29), which rules out gating a rerank on base confidence.
- Promoted 2-5 targets trailed final's winner by a smaller z-gap (0.41 vs 0.51 for non-promoted).
- MRR improves for every fusion even here (+0.007 to +0.009). Rank mass moves from 26-102 into 6-25 (final+block8+audio: 382 -> 344 trials in ranks 26-102). This is a broad, modest gain in rank quality rather than a top-1-specific effect.

Concentration: the confirmation top-5 loss sits in sub-02 (final+block8 top-5 W/L 3/11) and sub-06 (2/6). The top-1 gain sits in sub-04 (4/0). Other subjects are mixed. One holdout of 102 videos cannot separate subject effects from video-set effects.

## 3. Consistency across evaluations (final+block8+audio minus final CLIP)
| Evaluation | Independence | Trials | dTop-1 | dTop-5 |
|---|---|---:|---:|---:|
| Screening seeds 2026-28 | overlapping folds; block 8 and weights chosen here | 1836 | +0.54 pp | +3.16 pp (W/L 130/72) |
| Confirmation | disjoint, 1 holdout | 612 | +0.82 pp | -0.65 pp |
| Nested inner (5 disjoint folds, H2 excluded) | IDs overlap screening/confirmation | 3060 | +1.04 pp | +2.71 pp |
| **H2 outer (untouched, seed 2031)** | disjoint from all of the above | 612 | **+0.00 pp** | **+2.29 pp** (W/L 37/23, sign p=0.09) |

## 4. Controlled comparison (pre-registered: `experiments/nested_rerank_protocol.md`)
D_K = order by final CLIP, then reorder only the top-K by the final+block8 fused score. Each EEG row is processed independently and the full 102-candidate list is preserved. K in {3,5} was selected on inner folds only, and the selection was written to `selection.json` before the outer phase, which refuses to run without it.

| Config | Inner top-1 | Inner top-5 | H2 top-1 | H2 top-5 | H2 MRR | H2 dTop-1 [subj. 95% CI] | H2 dTop-5 [subj. 95% CI] | H2 top-1 W/L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A final CLIP | 2.39% | 10.36% | 3.27% | 11.44% | 0.0969 | - | - | - |
| B final+block8 | 3.37% | 12.45% | 3.10% | 12.91% | 0.1000 | -0.16 [-1.80, +1.31] | +1.47 [-1.80, +4.41] | 12/13 |
| C final+block8+audio | **3.43%** | **13.07%** | 3.27% | **13.73%** | **0.1040** | +0.00 [-2.12, +2.29] | +2.29 [-0.33, +4.41] | 13/13 |
| D3 rerank top-3 | 2.84% | 10.36% | 3.43% | 11.44% | 0.0977 | +0.16 [-0.82, +1.14] | 0 (by construction) | 10/9 |
| D5 rerank top-5 (K*) | 3.14% | 10.36% | 3.10% | 11.44% | 0.0953 | -0.16 [-1.31, +1.14] | 0 (by construction) | 10/11 |

Selections (inner, frozen before H2): K* = 5; best top-1 = C; best top-5 = C; balanced ((top1+top5)/2) = C. The chance-normalised sensitivity check also ranks C first (inner 3.08, H2 3.07 vs A 2.27 / 2.83).

H2 per subject, top-1 / top-5:
| Config | sub-01 | sub-02 | sub-03 | sub-04 | sub-05 | sub-06 |
|---|---:|---:|---:|---:|---:|---:|
| A | 2.9 / 11.8 | 6.9 / 19.6 | 3.9 / 9.8 | 1.0 / 7.8 | 2.9 / 6.9 | 2.0 / 12.7 |
| C | 7.8 / 16.7 | 2.9 / 21.6 | 2.9 / 9.8 | 2.0 / 12.7 | 3.9 / 11.8 | 0.0 / 9.8 |
| D5 | 3.9 / 11.8 | 4.9 / 19.6 | 2.9 / 9.8 | 2.9 / 7.8 | 3.9 / 6.9 | 0.0 / 12.7 |

C vs A on H2: top-5 improved in sub-01, 02, 04, 05, was flat in sub-03, and regressed in sub-06 (W/L 5/8). Top-1 improved in sub-01 (6/1), 04, 05, and regressed in sub-02 (3/7), 03, 06. Sub-06 regressed for every fusion in both independent holdouts (confirmation and H2). It is the subject with the fewest training recordings (850).

Why D underperforms: D can only re-sort final CLIP's top-K, and its reranking signal comes from block 8. Block 8's information is mostly in candidates final CLIP ranks *outside* the top 5 (Section 2), so D gives up the top-5 recovery that drives C's MRR and top-5 gains. Inside the top 5, the reorder is close to a coin flip on H2 (10 wins / 11 losses).

## 5. Metric-trust audit
| Item | Finding |
|---|---|
| Video-ID leakage | Screening ∩ confirmation ∩ H2 are pairwise empty. Nested script asserts no H2 ID enters any fit and inner folds are disjoint from H2 and each other (also unit-tested). |
| Fold construction | Saved screening artifacts equal `shared_holdout_ids` for seeds 2026-28. Screening folds overlap (11-16 shared IDs), so they are correctly not treated as independent. |
| Preprocessing/normalisation | Scalers, feature PCA, and RidgeCV alpha are fit on in-fold training rows only. Row z-score fusion is parameter-free. |
| Reproducibility | Re-running final CLIP, block 8, and audio on the confirmation holdout reproduced all 18 saved score and ranking matrices bit-exactly. |
| Row/column alignment | Every artifact checked: `heldout_ids` ascending, `target_ids` a permutation of them, identical across fused constituents, and saved rankings equal `rank_candidates(scores)`. |
| Aggregation | Subject means are the bootstrap unit (6 units). Trial sign tests are labelled optimistic. |
| Same examples across methods | Enforced by assertions in `fuse_saved_scores_cv.py`, `rank_error_analysis.py`, and `nested_rerank_cv.py`. |
| Documentation error (fixed) | `multilevel_confirmation_protocol.md` claimed 64-component target PCA, but the runs used direct Ridge. Erratum appended. Metrics are unaffected. |
| Feature versions | Re-extracting training IDs 1/500/1000 reproduced the cached block-8 and audio rows exactly. |
| Audio features | 40/1000 training videos and 3/102 test videos have no or silent audio (verified: no audio stream in ffmpeg), so they map to an identical "silence" vector. Not leakage, but audio carries no information for these videos. |

No correctness bug affecting reported metrics was found, so no experiment needed rerunning.

## 6. Limits
- One untouched 102-video holdout leaves top-1 differences of ±1-2 pp inside the noise (a 1 pp change = 1 trial per subject). No improvement here is statistically significant.
- Inner-fold IDs overlap the screening folds that originally motivated block 8 and the 0.45/0.45/0.10 weights, so inner estimates for B and C may be optimistic. H2 is the clean estimate.
- H2 is now consumed. Any further selection needs a new partition. 226 shared IDs were never in screening, confirmation, or H2. Most of them served as inner-fold validation IDs here, so they are not pristine.

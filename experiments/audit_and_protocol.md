# Audit and leakage-free experiment protocol

## Submission provenance audit

The two root-level submission arrays currently hash to:

| File | SHA-256 |
| --- | --- |
| `sub-05_ranked_candidate_list.npy` | `76ceb6dd58aeda92d39a5d3675acc18bd8ecd5ba934a91997b6a41a017cc8dd7` |
| `sub-06_ranked_candidate_list.npy` | `8bce005860e33689f0923307cbdcabce607b298b96fd89a51e5c646c943ac2c0` |

They can be regenerated from the saved AlexNet, CLIP, and DINOv2-S hidden-score matrices with score-normalized weights 0.36, 0.04, and 0.60. The repository does not retain an earlier immutable copy or a creation log for a different root-level submission, so a claim about which submission was originally better is not supportable. No root-level submission is overwritten by the validation workflow.

## One-time development reference (not model selection)

The saved score matrices reproduce the following results against the development labels in their supplied EEG row order:

| Model | Sub-01 | Sub-02 | Sub-03 | Sub-04 | Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| AlexNet top-1 | 2.94% | 4.90% | 4.90% | 2.94% | 3.92% |
| AlexNet top-5 | 11.76% | 22.55% | 14.71% | 10.78% | 14.95% |
| Exploratory 3-way top-1 | 5.88% | 13.73% | 7.84% | 2.94% | 7.60% |
| Exploratory 3-way top-5 | 15.69% | 24.51% | 21.57% | 17.65% | 19.85% |

The three-way configuration was previously selected using those same development labels. It is therefore retained only as an exploratory reference and is excluded from future model selection.

## Primary validation protocol

`scripts/train_only_cv.py` is the selection mechanism from this point forward.

1. Intersect the training-video IDs available to all six subjects.
2. Select the same 102 IDs for validation using each fixed seed (2026, 2027, 2028).
3. Remove every selected ID from the fitting data for every subject.
4. Fit each subject-specific encoder only on its remaining training videos. Any channel normalization is fit only on those rows.
5. Retrieve among the complete 102 held-out IDs for each subject.
6. Aggregate 18 independent subject-split measurements; organizer test EEG and labels are never read.

## Initial train-only leaderboard

| Experiment | Features | EEG representation | Mean top-1 | Mean top-5 | Measurements |
| --- | --- | --- | ---: | ---: | ---: |
| DINOv2-S forward | DINOv2-S, frozen | 100 Hz, 0-3000 ms | 2.18% | 9.04% | 18 |
| DINOv2-S + channel z-score | DINOv2-S, frozen | 100 Hz, 0-3000 ms | 2.29% | 9.42% | 18 |
| DINOv2-S forward | DINOv2-S, frozen | 250 Hz, 0-3000 ms | 2.12% | 9.10% | 18 |
| DINOv2-S forward | DINOv2-S, frozen | 100 Hz, 0-500 ms | 2.45% | 8.77% | 18 |
| DINOv2-S forward | DINOv2-S, frozen | 100 Hz, 100-800 ms | 2.67% | 9.04% | 18 |
| CLIP forward | CLIP ViT-B/32, frozen | 100 Hz, 0-3000 ms | 2.45% | 11.98% | 18 |
| CLIP + DINOv2-S, equal score fusion | Frozen CLIP + DINOv2-S | 100 Hz, 0-3000 ms | 2.56% | 12.25% | 18 |
| CLIP forward | CLIP ViT-B/32, frozen | 100 Hz, theta/alpha/beta/low-gamma band power | 1.03% | 5.23% | 18 |

The observed differences are small: the equal CLIP+DINO fusion adds two top-1 trials over 1,836 held-out measurements compared with CLIP alone, and the 100-800 ms DINO window adds four trials over the baseline DINO setup. These findings are provisional, not a model-selection decision. Band-power features are clearly uncompetitive and are excluded. The next comparisons should use predefined alternatives and the same seeds; they must not use development labels for tuning.

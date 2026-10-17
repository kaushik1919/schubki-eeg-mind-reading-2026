# Development-subject confirmation of the frozen contrastive candidate

Pre-registration: `experiments/frozen_candidate_preregistration.md`. Raw record: `experiments/frozen_candidate_confirmation.json`.

| Item | Value |
|---|---|
| Candidate | 5-seed shared EEGNet (adapters, multi-positive loss, CLIP block-8 target) fused 0.5/0.5 with frozen Model C |
| Config SHA-256 | `afbbebb1121a39defd5c6b76b6f37a257104ebc831340e081c054418c7aa5daa` |
| Commit at run time | `6b4c1ff49fc06b78b1b43b1c585accd681684569` (config and code frozen in `bba102e`) |
| Runs | Exactly one (the script refuses reruns) |
| Training | Organizer training archives of all 6 subjects; 249 s on MPS; best inner-validation epochs 22/39/39/59/17 (inner MRR 0.142-0.151) |
| Inference | 4 x 102 development test recordings in file order; 5.5 s including Model C fits |
| Prediction hashes | `outputs/contrastive_candidate/dev_scores/PREDICTIONS.sha256`, written before labels were read |

**Status:** these development test recordings were used in earlier baseline analyses (and the sub-01 Model C format check). This is a pre-registered check of a frozen candidate, not an untouched external holdout.

## Results (408 trials; chance 0.98% / 4.90% / 0.051)
| Model | Top-1 | Top-5 | MRR |
|---|---:|---:|---:|
| Frozen Model C | 5.64% | 19.85% | 0.145 |
| Decoder ensemble alone | 6.86% | 25.49% | 0.178 |
| **Frozen candidate (decoder + Model C)** | **9.07%** | **32.60%** | **0.210** |

Candidate minus Model C, paired:
| Metric | Delta | 95% CI (4-subject bootstrap) | Trial wins/losses | Sign test p |
|---|---:|---:|---:|---:|
| Top-1 | +3.43 pp | +0.98 to +5.88 | 20 / 6 | 0.009 |
| Top-5 | +12.75 pp | +8.33 to +15.69 | 68 / 16 | < 1e-7 |
| MRR | +0.064 | +0.038 to +0.088 | - | - |

The decoder alone vs Model C: top-1 +1.23 pp (CI -1.72 to +3.92; W/L 23/18, p = 0.53), top-5 +5.64 pp (CI +0.98 to +11.03; W/L 74/51, p = 0.049).

Per subject (top-1 / top-5 / MRR):
| Subject | Model C | Decoder | Candidate |
|---|---:|---:|---:|
| sub-01 | 2.9 / 21.6 / 0.135 | 4.9 / 24.5 / 0.167 | 7.8 / 37.3 / 0.214 |
| sub-02 | 3.9 / 18.6 / 0.133 | 8.8 / 32.4 / 0.212 | 10.8 / 34.3 / 0.230 |
| sub-03 | 9.8 / 23.5 / 0.181 | 6.9 / 22.5 / 0.162 | 9.8 / 29.4 / 0.205 |
| sub-04 | 5.9 / 15.7 / 0.132 | 6.9 / 22.5 / 0.170 | 7.8 / 29.4 / 0.190 |

## Interpretation
- **The exploratory gain replicated in direction and in relative size.** On the 6 exploratory folds the candidate gave 1.9x Model C's top-1 and 1.5x its top-5. Here it gives 1.6x and 1.6x. Top-5 improves for every subject. Top-1 improves for 3 of 4 subjects and ties on sub-03.
- **Absolute accuracies are higher on the development test sets for both models** (Model C 5.64% here vs 2.97% on the training-video folds). Numbers from the two settings should not be compared directly. A likely reason is a difference between the test videos or test session and the training videos; this was not investigated.
- **Limits:**
  - 4 subjects is too few for a reliable bootstrap; the CI is indicative only.
  - The trial-level sign tests treat recordings as independent and are optimistic.
  - Sub-01's Model C value (2.94% / 21.57%) was seen in an earlier format check. It is reproduced exactly here.
- **Hidden subjects:** performance on sub-05 and sub-06 remains unknown, and no hidden-subject predictions were made.

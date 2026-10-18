# Pre-registration: development-subject check of the frozen contrastive candidate

Recorded before any development-test evaluation of this candidate.

| Item | Value |
|---|---|
| Candidate | `D_block8_ens5_plus_ModelC` |
| Frozen config | `configs/frozen_contrastive_candidate.json` |
| Config SHA-256 | `afbbebb1121a39defd5c6b76b6f37a257104ebc831340e081c054418c7aa5daa` |
| Code commit | `bba102e` (frozen config, confirmation script, training code) |
| Command | `.venv/bin/python scripts/confirm_frozen_candidate.py --expected-config-sha256 afbbebb1121a39defd5c6b76b6f37a257104ebc831340e081c054418c7aa5daa` |
| Outputs | `outputs/contrastive_candidate/{models/seed*.pt, normalisation.npz, dev_scores/sub-0[1-4]_scores.npz, dev_scores/PREDICTIONS.sha256}`, `experiments/frozen_candidate_confirmation.json` |

## Permission
The challenge PDF states that the test recordings and labels of sub-01 to sub-04 "may be used only for validation and model selection" and must not be used as reference templates for sub-05/06. This check uses them for evaluation only.

## Rules
- Run once. The script refuses to run if outputs exist, the checksum differs, or the config/code differs from the commit.
- All training (decoder and Model C) uses organizer training archives only. Test EEG is opened only after the 5 seeds are trained, and is used only as inference input in file order. Normalisation statistics come from training recordings.
- Early stopping uses grouped inner validation of training videos (seed 3100). No checkpoint, seed, weight or threshold is chosen from test data.
- Labels are loaded only after all predictions are saved and hashed.
- The outcome will be recorded regardless. Nothing about the candidate changes afterwards.

## Status of these recordings
The development test recordings were used earlier in the project's baseline analysis (`experiments/audit_and_protocol.md`, and the sub-01 Model C format check). This is a pre-registered check of a frozen candidate, not an untouched external holdout.

## Reported quantities
Top-1, top-5 and MRR for the candidate, the decoder ensemble alone and Model C. Per-subject results. Paired deltas against Model C with a 4-subject bootstrap CI (very few units; indicative only) and trial-level sign tests (optimistic: they treat trials as independent).

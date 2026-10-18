# DINOv2 + CLAP multimodal contrastive decoding (exploratory)

Executed 2026-10-18 on `research/temporal-eeg-encoding` (Apple M5 Pro, MPS). Only organizer training archives were read. No development or hidden test EEG was used, and Model C, the frozen contrastive candidate and the root submissions are unchanged.

**Validation status: exploratory.** These are the same 6 outer folds of 102 shared training videos (seed 2040) as all previous decoder work, so the folds have already been inspected. The EEG-to-EEG weight was chosen per outer fold on grouped inner validation only. Nothing was evaluated on development test recordings.

## Outcome
DINOv2 + CLAP does **not** outperform CLIP block 8 as a contrastive target, and adding DINOv2 + CLAP to block 8 makes it worse. The teammate's 13.9% top-1 is **not reproduced** under our protocol: the best DINOv2 + CLAP model reaches 3.16% top-1, about the level of Model C. The Phase 6 condition (meaningful multimodal improvement) was not met, so no multimodal seed ensemble, fusion or versioned predictions were produced.

## Stimulus features
| Modality | Source | Dim | Coverage |
|---|---|---:|---|
| CLIP block 8 | Cached OpenCLIP ViT-B/32 CLS, block 8, mean of 8 frames | 768 | all |
| DINOv2 ViT-S/14 | Cached `timm/vit_small_patch14_dinov2.lvd142m`, mean of 8 frames | 384 | all |
| DINOv2 ViT-B/14 | New: `timm/vit_base_patch14_dinov2.lvd142m` (uploaded 2023-05-09, Apache-2.0), same frame sampling | 768 | all |
| CLAP | New: `laion/larger_clap_general`, revision `ada0c23a` (created 2023-10-30, Apache-2.0), full soundtrack at 48 kHz mono, audio projection | 512 | train 960/1000, candidates 99/102 |

All checkpoints are public and predate the hackathon, as the challenge rules require. Clips without an audio stream or with a silent track (from the temporal-audio audit) are flagged `has_audio = False` and not passed to CLAP. Rows are ID-aligned (training IDs 1-1000, candidates 1001-1102), which is asserted on load. Re-extracting three candidates reproduced the cache exactly.

## Model (`src/multimodal_stimulus.py`, `scripts/train_multimodal_contrastive.py`)
- **EEG side:** unchanged from the frozen candidate (shared EEGNet, per-subject adapters, AdamW 1e-3 / weight decay 0.1, dropout 0.25, batches of all recordings of 64 videos, early stopping on inner-validation MRR, patience 25, seed 2050).
- **Stimulus side:** each modality is L2-normalised per row (no fitted parameters), then passed through LayerNorm and a linear projection to 128. Missing audio is replaced by a learned "no-audio" token. The projections are concatenated and passed through an MLP (256 hidden, GELU, dropout 0.1) to a 128-d L2-normalised embedding.
- **No ID shortcut:** the head sees only frozen features, never IDs or positions. Tests check permutation equivariance and that identical features give identical embeddings.
- **Loss:** symmetric multi-positive EEG<->stimulus contrastive loss; all recordings of a video are positives, never negatives. Optional cross-subject EEG<->EEG term with weight w: w = 0 in Phase 3; w in {0.5, 2} chosen per fold on inner validation in Phase 4.

## Results (3,672 trials; single seed 2050)
| Model | Top-1 | Top-5 | MRR | dTop-1 vs Model C [95% CI] | dTop-5 vs Model C [95% CI] |
|---|---:|---:|---:|---:|---:|
| Model C | 2.97% | 13.67% | 0.102 | - | - |
| A: CLIP block 8, MLP head | 4.47% | 16.94% | 0.128 | +1.50 [+0.46, +2.56] | +3.27 [+2.48, +4.00] |
| B: DINOv2 ViT-S | 3.05% | 11.47% | 0.094 | +0.08 [-0.63, +0.84] | -2.21 [-3.65, -0.52] |
| B2: DINOv2 ViT-B | 2.67% | 10.51% | 0.088 | -0.30 [-1.20, +0.54] | -3.16 [-5.20, -1.39] |
| C: CLAP | 2.21% | 10.05% | 0.082 | -0.76 [-1.58, -0.03] | -3.62 [-6.56, -1.14] |
| D: DINOv2-S + CLAP MLP | 2.64% | 12.31% | 0.095 | -0.33 [-0.60, -0.08] | -1.36 [-3.59, +0.22] |
| D2: DINOv2-B + CLAP MLP | 3.08% | 11.08% | 0.093 | +0.11 [-0.71, +0.93] | -2.59 [-5.36, -0.22] |
| E: block 8 + DINOv2-S + CLAP | 3.32% | 13.07% | 0.102 | +0.35 [-0.27, +0.95] | -0.60 [-2.37, +0.82] |
| A + EEG<->EEG (w selected) | **5.12%** | **18.41%** | **0.135** | +2.15 [+1.17, +3.08] | +4.74 [+3.02, +6.10] |
| D + EEG<->EEG (w selected) | 3.16% | 13.56% | 0.103 | +0.19 [-0.54, +0.93] | -0.11 [-2.07, +1.25] |
| Existing D_block8 (linear head, w = 0.5), seed 2050 | 4.03% | 15.58% | 0.120 | +1.06 [+0.68, +1.36] | +1.91 [-0.08, +3.30] |
| Existing 5-seed D_block8 decoder | 4.90% | 18.08% | 0.134 | +1.93 [+1.20, +2.45] | +4.41 [+2.48, +6.24] |
| Existing 5-seed decoder + Model C | 5.64% | 20.32% | 0.147 | +2.67 [+2.26, +3.00] | +6.64 [+6.13, +7.22] |

Paired, same trials (`paired_and_diagnostics.json`):
| Comparison | dTop-1 [CI] | dTop-5 [CI] | dMRR [CI] | Top-1 W/L |
|---|---:|---:|---:|---:|
| D (DINOv2 + CLAP) vs A (block 8) | -1.82 [-3.00, -0.63] | -4.63 [-6.40, -3.13] | -0.033 [-0.044, -0.022] | 86/153 |
| E (block 8 + DINOv2 + CLAP) vs A | -1.14 [-2.37, +0.05] | -3.87 [-5.23, -2.64] | -0.026 [-0.037, -0.017] | 109/151 |
| DINOv2 ViT-B vs ViT-S | -0.38 [-0.98, +0.44] | -0.95 [-2.72, +0.74] | -0.007 [-0.014, +0.002] | 92/106 |
| A + EEG<->EEG vs A | +0.65 [-0.98, +2.26] | +1.47 [+0.03, +3.10] | +0.008 [-0.006, +0.022] | 124/100 |
| D + EEG<->EEG vs D | +0.52 [-0.16, +1.14] | +1.25 [+0.68, +1.93] | +0.009 [+0.004, +0.013] | 88/69 |

Inner validation selected w = 0.5 in 3 of 6 folds for both A and D, and w = 2 in the other 3.

Per subject, top-1 / top-5 (sub-05 and sub-06 are listed last):
| Model | 01 | 02 | 03 | 04 | 05 | 06 |
|---|---:|---:|---:|---:|---:|---:|
| Model C | 4.2 / 19.0 | 3.4 / 15.2 | 2.5 / 13.4 | 3.1 / 12.6 | 2.5 / 11.6 | 2.1 / 10.3 |
| A block 8 | 6.9 / 20.9 | 4.9 / 17.8 | 6.0 / 17.5 | 4.4 / 16.8 | 2.9 / 13.9 | 1.6 / 14.7 |
| B DINOv2-S | 3.3 / 15.0 | 2.9 / 11.9 | 2.8 / 9.8 | 2.3 / 9.0 | 4.1 / 11.4 | 2.9 / 11.6 |
| C CLAP | 2.0 / 9.0 | 1.6 / 9.0 | 2.5 / 11.3 | 2.3 / 9.5 | 2.8 / 11.4 | 2.1 / 10.1 |
| D DINOv2-S + CLAP | 3.4 / 12.6 | 3.1 / 14.7 | 2.3 / 11.3 | 2.5 / 12.4 | 2.5 / 11.4 | 2.1 / 11.4 |
| D + EEG<->EEG | 3.1 / 14.4 | 3.9 / 15.2 | 3.4 / 14.1 | 3.3 / 13.1 | 1.6 / 12.3 | 3.6 / 12.4 |
| A + EEG<->EEG | 4.6 / 19.9 | 5.7 / 20.8 | 5.2 / 17.5 | 4.1 / 17.3 | 6.4 / 18.6 | 4.7 / 16.3 |
| 5-seed decoder + Model C | 7.0 / 25.5 | 5.2 / 21.9 | 5.4 / 20.4 | 6.4 / 18.6 | 4.9 / 17.3 | 4.9 / 18.1 |

## Diagnostics
| Model | Stimulus eff. rank | EEG eff. rank | Train-probe MRR / inner-val MRR | Best epoch | Train / fold | Trainable params |
|---|---:|---:|---:|---:|---:|---:|
| A block 8 | 48.7 | 84.9 | 0.61 / 0.157 | 46 | 46 s | 232k |
| A + EEG<->EEG | 47.8 | 88.4 | 0.65 / 0.159 | 40 | 72 s (2 settings) | 232k |
| B DINOv2-S | 64.8 | 80.5 | 0.56 / 0.119 | 22 | 31 s | 182k |
| C CLAP | 53.3 | 75.2 | 0.31 / 0.111 | 8 | 22 s | 199k |
| D DINOv2-S + CLAP | 69.7 | 86.5 | **0.92** / 0.129 | 29 | 35 s | 282k |
| D + EEG<->EEG | 69.1 | 97.1 | **0.99** / 0.137 | 54 | 81 s (2 settings) | 282k |
| E block 8 + DINOv2-S + CLAP | 68.9 | 85.6 | **0.90** / 0.133 | 34 | 39 s | 414k |

Inference for 612 trials takes under 0.05 s per fold for every model. Feature extraction took about 15 min (CLAP, 1,102 clips) and about 10 min (DINOv2 ViT-B) on MPS.

- **Overfitting:** the DINOv2 + CLAP heads nearly memorise the training videos (train-probe MRR 0.92-0.99) while generalising worse than block 8. Their richer stimulus space lets the model fit ~900 training videos without learning a transferable EEG mapping.
- **No collapse:** effective ranks are 48-70 for stimulus embeddings and 75-97 for EEG embeddings.
- **No ID leakage:** outer-fold videos are absent from every fit (asserted), the head sees only features, and its permutation equivariance is tested.
- **Missing-audio bias** (`missing_audio_audit.json`): 138 of 3,672 trials are no-audio videos.
  - They are much easier for CLAP-based models: MRR 0.17 vs 0.09 for D. That is plausibly a real "silent clip" EEG signature, but it affects only about 4% of trials (3 of 102 hidden candidates).
  - The audio-only model C gives every no-audio candidate an identical embedding, which creates exact score ties in 132 trials. Stable-sort ranking (used in all tables) breaks ties by candidate ID; pessimistic tie handling lowers no-audio MRR for C from 0.25 to 0.15.
  - The training loop's inner-validation metric counts ties optimistically (strict ">"), which only matters for this audio-only variant.

## Answers
1. **DINOv2 alone:** 3.05% / 11.47% / 0.094 (ViT-S); 2.67% / 10.51% / 0.088 (ViT-B). Roughly Model C on top-1, worse on top-5.
2. **CLAP alone:** 2.21% / 10.05% / 0.082.
3. **DINOv2 + CLAP:** 2.64% / 12.31% / 0.095 (ViT-S), 3.08% / 11.08% / 0.093 (ViT-B); with EEG<->EEG, 3.16% / 13.56% / 0.103.
4. **vs CLIP block 8:** DINOv2 + CLAP is 1.8 pp top-1 and 4.6 pp top-5 worse on identical trials. Adding it to block 8 also hurts.
5. **EEG<->EEG loss:** it helps consistently on top-5 and MRR (+1.25 to +1.47 pp top-5), while the top-1 change is within noise.
6. **Best single model here:** block 8 with the MLP head plus EEG<->EEG, 5.12% / 18.41% / 0.135 (one seed). This is not a multimodal model. The existing 5-seed decoder + Model C remains the best system (5.64% / 20.32% / 0.147).
7. **13.9% claim:** not reproducible under our protocol. Plausible reasons for a higher figure include a smaller candidate set, evaluation on training videos, or evaluation on development test recordings (where even Model C scores 5.6% and the frozen candidate 9.1%). The protocol and artifacts are needed to tell.

## Reproduction
```bash
.venv/bin/python scripts/extract_clap_features.py --video-dir eeg-mind-reading-challenge/stimuli_train --ids manifests/all_training_ids.npy --audit features/all_train_temporal_audio_audit.json --output features/all_train_clap_features.npz
.venv/bin/python scripts/extract_clap_features.py --video-dir eeg-mind-reading-challenge/stimuli_test --ids manifests/test_candidate_ids.npy --audit features/test_temporal_audio_audit.json --output features/test_clap_features.npz
.venv/bin/python scripts/extract_dinov2_features.py --model vit_base_patch14_dinov2.lvd142m --video-dir eeg-mind-reading-challenge/stimuli_train --ids manifests/all_training_ids.npy --output features/all_train_dinov2b_features.npy
.venv/bin/python scripts/extract_dinov2_features.py --model vit_base_patch14_dinov2.lvd142m --video-dir eeg-mind-reading-challenge/stimuli_test --ids manifests/test_candidate_ids.npy --output features/test_dinov2b_features.npy
for e in A_block8 B_dinov2 B2_dinov2b C_clap D_dinov2_clap D2_dinov2b_clap E_block8_dinov2_clap A_block8_e2e D_dinov2_clap_e2e; do .venv/bin/python scripts/train_multimodal_contrastive.py --experiment $e; done
M=experiments/multimodal_contrastive; .venv/bin/python scripts/evaluate_contrastive_eeg.py --variants D_block8 D_block8_s2051 D_block8_s2052 D_block8_s2053 D_block8_s2054 mA:$M/A_block8 mA_e2e:$M/A_block8_e2e mB:$M/B_dinov2 mB2:$M/B2_dinov2b mC:$M/C_clap mD:$M/D_dinov2_clap mD_e2e:$M/D_dinov2_clap_e2e mD2:$M/D2_dinov2b_clap mE:$M/E_block8_dinov2_clap --ensembles Cand_dec=D_block8,D_block8_s2051,D_block8_s2052,D_block8_s2053,D_block8_s2054 --fuse-with-modelc Cand_dec --out $M/summary.json
.venv/bin/python scripts/audit_multimodal_bias.py --dirs $M/A_block8 $M/B_dinov2 $M/C_clap $M/D_dinov2_clap $M/E_block8_dinov2_clap --out $M/missing_audio_audit.json
```

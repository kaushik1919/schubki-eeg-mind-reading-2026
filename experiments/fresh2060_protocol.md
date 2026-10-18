# Fresh-partition confirmation (outer seed 2060): pre-specified before any 2060 decoder result

**Why a new partition:** the seed-2040 folds were used to choose every recent component (block-8 target, EEG<->EEG decoder, fusions, assignment calibration). A new partition of the 695 shared videos (seed 2060) reshuffles the outer folds (at most 19 of 102 videos in common with any 2040 fold; 72 of its 612 videos were never in a 2040 outer fold). The video identities themselves were seen in earlier studies, so this is a **fresh split, not untouched data**.

**Fixed systems** (no retuning on 2060):
- S1 frozen candidate: rowz-mean(D_block8 seeds 2050-2054), fused 0.5/0.5 with Model C.
- S2: rowz-mean(EEG<->EEG decoder B seeds 2050-2054) + Model C, 0.5/0.5.
- S3: equal thirds of the D_block8 ensemble, B ensemble and Model C (row-z), implemented as the seed-level mean with Model C repeated 5x (as benchmarked).
- Calibration (label-free, uses the other recordings of the same subject): independent ranking; iterative row/column z-score; Sinkhorn with tau = 0.5 (best seed-2040 MRR for S3); Hungarian-first.

**Decision rules:**
- Prefer S3 over S1 if S3's 2060 MRR and top-1 are both >= S1's.
- Adopt a calibration only if it improves both top-1 and MRR on 2060 over independent ranking of the chosen system.
- Calibrations that use other test recordings are submission-eligible only with organizer approval (see report). The independent-ranking variant is always produced.

**Same training data rules:** all recordings of an outer-fold video are excluded for all subjects; inner early stopping uses 10% of training video IDs; no test EEG is used.

# Phase A diagnostic

## Exact current pipeline

- Input EEG is already organizer-preprocessed at 500 Hz, 17 channels, and 1,500 samples (3 s). The baseline applies zero-phase FIR `scipy.signal.decimate(..., factor=5)` to 100 Hz / 300 samples; it does not apply another band-pass or blink-removal stage.
- A subject-specific forward model fits, in order, a feature `StandardScaler`, full-SVD feature PCA retaining 95% variance, a second `StandardScaler`, and `RidgeCV` over 20 alphas from 1e-3 to 1e6. It predicts the full 17 x 300 time-domain EEG target.
- The organizer scoring representation is 50-600 ms after stimulus onset (100 Hz samples 5:60). Each recording and each candidate prediction is flattened across channels and time, independently mean-centered/scaled, then compared by their normalized dot product: per-recording Pearson correlation.

## Fold safety and alignment

- All learned transforms in `train_only_cv.py` and `clip_controlled_cv.py` fit only the subject's in-fold training EEG/videos. Target PCA and subject normalization are also fit in-fold.
- Training archives are numeric-sorted by video ID. Features use row `ID - 1`; held-out candidate columns are the explicit sorted held-out IDs. A fold asserts that every held-out ID is present in each subject before fitting.
- EEG recordings remain in archive order; no label-driven row sort is performed. The organizer prediction notebook sorts development EEG rows solely to evaluate against development labels. Reusing that operation for hidden rows would corrupt submission order; it is intentionally absent from the project inference and CV code.
- Similarity is computed one measured EEG recording at a time against every predicted candidate EEG. Its per-recording normalization is appropriate for amplitude-insensitive retrieval, but makes channel standardization a relative-channel-weighting choice rather than a global scale fix.

## Metrics, sample size, and timing

- The 18 raw measurements are six subjects x three shared held-out sets, each with 102 candidates and 748-870 subject-specific training videos. CSV means are micro/macro equivalent because every held-out fold has 102 trials.
- For uncertainty, the report averages the three overlapping folds within each subject, then uses the six subject means as paired bootstrap units. It does not incorrectly call the 18 overlapping splits independent.
- The stimulus-locked temporal origin is consistent with the organizer notebooks: 0 ms is video onset and the recording is exactly 0-3 s. The code converts all windows in milliseconds at 100 Hz and has regression tests for the organizer 50-600 ms mapping and clipped windows.

## Corrections made

- Added shared-ID and stimulus-window regression tests.
- Added saved fold-level score/ranking/target artifacts and metadata for paired analysis.
- Kept root submission files unmodified.

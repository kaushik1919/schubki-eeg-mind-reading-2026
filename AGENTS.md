# EEG-to-Video Retrieval Project Context

## Competition and deliverables

SchubKI National AI Hackathon 2026: Mind Reading Using Machine Learning and EEG.
Inputs are 3-second EEG recordings with shape `(17, 1500)` (17 channels, 500 Hz), from six subjects. The task is to rank 102 candidate videos for every hidden-subject EEG recording. Submission files are `sub-05_ranked_candidate_list.npy` and `sub-06_ranked_candidate_list.npy`. Each must be `(102, 102)`, every row must be a permutation of integer candidate IDs 1001 through 1102, and rows must remain in original test EEG order.

The challenge PDF is authoritative. Do not infer rules, filenames, splits, metadata, or video contents not confirmed by it.

## Non-negotiable scientific rules

- Never train or tune on hidden test EEG.
- Select models with video-identity-held-out validation; never allow a video identity in both train and validation.
- Do not use development-subject test EEG as a template for hidden subjects.
- Fit scalers, PCA, feature normalizers, and all learned transforms on training folds only.
- Preserve hidden EEG recording order and candidate-column/ID alignment.
- Do not overwrite organizer notebooks, challenge documents, or original data.
- Report only measurements actually executed, including top-1, top-5, per-subject results, train time, and inference time.

## Delivery sequence

1. Reproduce and validate the organizer baseline.
2. Independently evaluate frozen DINOv2 (visual), CLIP (semantic), VideoMAE (temporal), and PANNs CNN14 (audio; use a documented visual fallback if audio is unusable), all using shared split and score-matrix interfaces.
3. Select/ensemble only approaches with validated complementary errors; modularize feature extraction, EEG preprocessing, training, and retrieval/submission.
4. Explore research novelty only after the prototype works and support every added component with held-out experiments.

## Engineering standards

- Python 3.11; prefer NumPy, SciPy, scikit-learn, PyTorch, and appropriate EEG tooling.
- Set and record deterministic seeds where practical.
- Use GPU acceleration only when available and keep a CPU-compatible baseline.
- Put reusable code in `src/`, command entry points in `scripts/`, config in `configs/`, experiment records in `experiments/`, and synthetic tests in `tests/`.
- Candidate IDs must be represented explicitly, never inferred from score indices except by a checked mapping.
- Use `pytest`; tests must cover shapes, candidate alignment, valid rankings, deterministic grouped splits, and test-order preservation.

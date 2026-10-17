"""Score construction, ranking, and submission validation."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .data import CANDIDATE_IDS, validate_candidate_ids


def cosine_scores(predicted_features: np.ndarray, candidate_features: np.ndarray) -> np.ndarray:
    predicted_features, candidate_features = map(lambda x: np.asarray(x, dtype=np.float32), (predicted_features, candidate_features))
    if predicted_features.ndim != 2 or candidate_features.ndim != 2 or predicted_features.shape[1] != candidate_features.shape[1]:
        raise ValueError("predicted and candidate features must be 2D with equal feature width")
    left = predicted_features / np.maximum(np.linalg.norm(predicted_features, axis=1, keepdims=True), 1e-12)
    right = candidate_features / np.maximum(np.linalg.norm(candidate_features, axis=1, keepdims=True), 1e-12)
    return left @ right.T


def organizer_eeg_correlation_scores(measured_eeg: np.ndarray, predicted_eeg: np.ndarray, *, start: int = 5, stop: int = 60) -> np.ndarray:
    """Organizer baseline's per-trial Pearson-like similarity on 50-590 ms."""
    measured_eeg, predicted_eeg = np.asarray(measured_eeg), np.asarray(predicted_eeg)
    if measured_eeg.ndim != 3 or predicted_eeg.ndim != 3 or measured_eeg.shape[1:] != predicted_eeg.shape[1:]:
        raise ValueError("measured and predicted EEG must be (n, channels, time) with matching trailing shape")
    if not 0 <= start < stop <= measured_eeg.shape[-1]:
        raise ValueError("invalid scoring time window")
    measured = measured_eeg[:, :, start:stop].reshape(len(measured_eeg), -1).astype(np.float64)
    predicted = predicted_eeg[:, :, start:stop].reshape(len(predicted_eeg), -1).astype(np.float64)
    measured = (measured - measured.mean(1, keepdims=True)) / np.maximum(measured.std(1, keepdims=True), 1e-12)
    predicted = (predicted - predicted.mean(1, keepdims=True)) / np.maximum(predicted.std(1, keepdims=True), 1e-12)
    return (measured @ predicted.T / measured.shape[1]).astype(np.float32)


def row_zscore(scores: np.ndarray) -> np.ndarray:
    """Candidate-wise score normalization per EEG row for score-level ensembles."""
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError("scores must be 2D")
    return (scores - scores.mean(axis=1, keepdims=True)) / np.maximum(scores.std(axis=1, keepdims=True), 1e-12)


def weighted_score_ensemble(score_matrices: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Combine same-shaped score matrices after fold-independent row normalization."""
    weights = np.asarray(weights, dtype=np.float32)
    if len(score_matrices) < 2 or weights.shape != (len(score_matrices),) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("provide at least two non-negative weights with positive sum")
    matrices = [row_zscore(matrix) for matrix in score_matrices]
    if any(matrix.shape != matrices[0].shape for matrix in matrices):
        raise ValueError("score matrices must share shape")
    return np.average(np.stack(matrices), axis=0, weights=weights)


def weighted_rank_ensemble(score_matrices: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Combine candidate ranks; higher returned values denote more likely candidates."""
    weights = np.asarray(weights, dtype=np.float32)
    if len(score_matrices) < 2 or weights.shape != (len(score_matrices),) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("provide at least two non-negative weights with positive sum")
    scores = [np.asarray(matrix, dtype=np.float32) for matrix in score_matrices]
    if any(matrix.ndim != 2 or matrix.shape != scores[0].shape for matrix in scores): raise ValueError("score matrices must be same-shaped 2D arrays")
    ranks = [np.argsort(np.argsort(-matrix, axis=1, kind="stable"), axis=1, kind="stable") for matrix in scores]
    return -np.average(np.stack(ranks), axis=0, weights=weights)


def rank_candidates(scores: np.ndarray, candidate_ids: np.ndarray = CANDIDATE_IDS) -> np.ndarray:
    scores = np.asarray(scores)
    ids = np.asarray(candidate_ids)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError("candidate_ids must be a one-dimensional array of unique IDs")
    if scores.ndim != 2 or scores.shape[1] != len(ids):
        raise ValueError(f"scores must be (n_records, {len(ids)})")
    # Stable sort makes tie behavior reproducible and preserves supplied candidate-ID column order.
    return ids[np.argsort(-scores, axis=1, kind="stable")]


def validate_rankings(rankings: np.ndarray, *, n_records: int = 102) -> np.ndarray:
    rankings = np.asarray(rankings)
    if rankings.shape != (n_records, len(CANDIDATE_IDS)):
        raise ValueError(f"submission must have shape ({n_records}, {len(CANDIDATE_IDS)}), got {rankings.shape}")
    if not np.all(np.sort(rankings, axis=1) == CANDIDATE_IDS):
        raise ValueError("every ranking row must contain candidate IDs 1001..1102 exactly once")
    return rankings.astype(np.int64, copy=False)


def save_submission(path: str | Path, rankings: np.ndarray) -> None:
    np.save(Path(path), validate_rankings(rankings), allow_pickle=False)

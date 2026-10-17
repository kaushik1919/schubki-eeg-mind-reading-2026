"""Shared-ID, train-only cross-validation primitives for retrieval experiments."""
from __future__ import annotations

import numpy as np


def shared_holdout_ids(subject_ids: dict[str, np.ndarray], *, seed: int, size: int) -> np.ndarray:
    """Choose IDs present for every subject, independently of EEG values."""
    if not subject_ids:
        raise ValueError("at least one subject is required")
    common = set(next(iter(subject_ids.values())).tolist())
    for ids in subject_ids.values():
        common.intersection_update(np.asarray(ids).tolist())
    if len(common) < size:
        raise ValueError(f"only {len(common)} IDs are shared; requested {size}")
    return np.sort(np.random.default_rng(seed).choice(np.array(sorted(common)), size=size, replace=False)).astype(np.int64)


def common_ids(subject_ids: dict[str, np.ndarray]) -> np.ndarray:
    """Sorted IDs recorded for every subject."""
    common = set(next(iter(subject_ids.values())).tolist())
    for ids in subject_ids.values():
        common.intersection_update(np.asarray(ids).tolist())
    return np.array(sorted(common), dtype=np.int64)


def disjoint_inner_folds(pool: np.ndarray, *, exclude: np.ndarray, n_folds: int, size: int, seed: int) -> list[np.ndarray]:
    """Mutually disjoint sorted ID folds drawn from ``pool`` minus ``exclude``.

    Used for nested (inner) validation: ``exclude`` is the outer holdout, which
    therefore can never influence an inner-fold fit or selection.
    """
    available = np.setdiff1d(np.asarray(pool, dtype=np.int64), np.asarray(exclude, dtype=np.int64))
    if n_folds * size > len(available):
        raise ValueError(f"need {n_folds * size} IDs, only {len(available)} available")
    chosen = np.random.default_rng(seed).permutation(available)[: n_folds * size]
    return [np.sort(chosen[i * size:(i + 1) * size]) for i in range(n_folds)]


def millisecond_slice(*, representation_start_ms: int, representation_stop_ms: int, sampling_hz: int, window_ms: tuple[int, int]) -> slice:
    """Map an absolute stimulus-locked millisecond window into a representation."""
    start_ms, stop_ms = window_ms
    start = max(0, round((start_ms - representation_start_ms) * sampling_hz / 1000))
    stop = min(round((representation_stop_ms - representation_start_ms) * sampling_hz / 1000), round((stop_ms - representation_start_ms) * sampling_hz / 1000))
    if start >= stop:
        return slice(0, round((representation_stop_ms - representation_start_ms) * sampling_hz / 1000))
    return slice(start, stop)

"""Leakage-resistant split and retrieval metric utilities."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def video_heldout_split(video_ids: np.ndarray, *, validation_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split records such that every video identity is wholly train or validation."""
    ids = np.asarray(video_ids)
    if ids.ndim != 1 or len(ids) < 2:
        raise ValueError("video_ids must be a one-dimensional array with at least two records")
    splitter = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed)
    train, validation = next(splitter.split(np.zeros(len(ids)), groups=ids))
    if set(ids[train]) & set(ids[validation]):
        raise RuntimeError("grouped split leaked video identities")
    return train, validation


def retrieval_metrics(rankings: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    rankings, targets = np.asarray(rankings), np.asarray(targets)
    if rankings.ndim != 2 or targets.shape != (len(rankings),):
        raise ValueError("rankings must be (n, candidates) and targets must be (n,)")
    positions = (rankings == targets[:, None]).argmax(axis=1)
    if not np.all(rankings[np.arange(len(rankings)), positions] == targets):
        raise ValueError("each target must occur in its ranking")
    return {"top1": float(np.mean(positions == 0)), "top5": float(np.mean(positions < 5))}

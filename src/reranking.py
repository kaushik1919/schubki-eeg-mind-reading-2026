"""Per-recording conservative top-K reranking of a base score matrix.

Every operation here is row-independent: the ordering produced for one EEG
recording never depends on scores of any other recording, so no joint
(Hungarian-style) assignment across recordings is performed.
"""
from __future__ import annotations

import numpy as np

from .retrieval import row_zscore


def topk_rerank_order(base_scores: np.ndarray, rerank_scores: np.ndarray, k: int) -> np.ndarray:
    """Return column indices ordered best-first.

    The base ranking is computed with a stable descending sort.  Only its first
    ``k`` columns are reordered (stable descending sort by ``rerank_scores``);
    positions ``k..C-1`` keep the base order.  The top-K *set* is therefore
    identical to the base model's, so top-j accuracy for every j >= k is
    unchanged by construction.
    """
    base, rerank = np.asarray(base_scores, dtype=np.float64), np.asarray(rerank_scores, dtype=np.float64)
    if base.ndim != 2 or base.shape != rerank.shape:
        raise ValueError("base and rerank scores must be same-shaped 2D arrays")
    if not 1 <= int(k) <= base.shape[1]:
        raise ValueError(f"k must be in [1, {base.shape[1]}]")
    k = int(k)
    order = np.argsort(-base, axis=1, kind="stable")
    head = order[:, :k]
    head_scores = np.take_along_axis(rerank, head, axis=1)
    head = np.take_along_axis(head, np.argsort(-head_scores, axis=1, kind="stable"), axis=1)
    return np.concatenate([head, order[:, k:]], axis=1)


def topk_rerank_rankings(base_scores: np.ndarray, rerank_scores: np.ndarray, candidate_ids: np.ndarray, k: int) -> np.ndarray:
    """Map :func:`topk_rerank_order` through an explicit candidate-ID column map."""
    ids = np.asarray(candidate_ids)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids) or len(ids) != np.asarray(base_scores).shape[1]:
        raise ValueError("candidate_ids must be unique and match the score columns")
    return ids[topk_rerank_order(base_scores, rerank_scores, k)]


def fused_rerank_scores(score_matrices: list[np.ndarray], weights: list[float] | np.ndarray) -> np.ndarray:
    """Label-free row-z-scored weighted sum used as the within-top-K reranking score."""
    weights = np.asarray(weights, dtype=np.float64)
    if len(score_matrices) != len(weights) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("weights must be non-negative, positive-sum, one per matrix")
    stacked = np.stack([row_zscore(m).astype(np.float64) for m in score_matrices])
    return np.tensordot(weights / weights.sum(), stacked, axes=1)

"""Rank-error diagnostics for aligned retrieval artifacts (no fitting, no labels beyond targets)."""
from __future__ import annotations

import numpy as np


def target_positions(rankings: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    """0-based position of each row's correct ID inside its ranked candidate list."""
    rankings, targets = np.asarray(rankings), np.asarray(target_ids)
    if rankings.ndim != 2 or targets.shape != (len(rankings),):
        raise ValueError("rankings must be (n, C) and target_ids (n,)")
    hits = rankings == targets[:, None]
    if not np.all(hits.sum(axis=1) == 1):
        raise ValueError("every target must occur exactly once in its ranking row")
    return hits.argmax(axis=1)


def summarize_positions(positions: np.ndarray) -> dict[str, float]:
    p = np.asarray(positions)
    return {"n": int(len(p)), "top1": float(np.mean(p == 0)), "top5": float(np.mean(p < 5)), "mrr": float(np.mean(1.0 / (p + 1))),
            "median_rank": float(np.median(p + 1)), "mean_rank": float(np.mean(p + 1))}


def top_margin(scores: np.ndarray) -> np.ndarray:
    """Per-row gap between the best and second-best score (row-z-scored units if inputs are)."""
    s = np.sort(np.asarray(scores, dtype=np.float64), axis=1)
    return s[:, -1] - s[:, -2]


def transition_counts(base_pos: np.ndarray, new_pos: np.ndarray) -> dict[str, int]:
    """Paired wins/losses of a candidate configuration relative to a baseline."""
    b, n = np.asarray(base_pos), np.asarray(new_pos)
    if b.shape != n.shape:
        raise ValueError("paired position arrays must match")
    return {
        "top1_wins": int(np.sum((n == 0) & (b != 0))), "top1_losses": int(np.sum((n != 0) & (b == 0))),
        "top5_wins": int(np.sum((n < 5) & (b >= 5))), "top5_losses": int(np.sum((n >= 5) & (b < 5))),
        "promoted_2to5_to_1": int(np.sum((b >= 1) & (b < 5) & (n == 0))),
        "promoted_outside5_to_1": int(np.sum((b >= 5) & (n == 0))),
        "demoted_top5_to_outside": int(np.sum((b < 5) & (n >= 5))),
        "demoted_1_to_2to5": int(np.sum((b == 0) & (n >= 1) & (n < 5))),
        "rank_improved": int(np.sum(n < b)), "rank_worsened": int(np.sum(n > b)), "rank_equal": int(np.sum(n == b)),
    }


def exact_sign_test(wins: int, losses: int) -> float:
    """Two-sided exact binomial (McNemar) p-value on discordant pairs."""
    from scipy.stats import binomtest
    n = wins + losses
    return 1.0 if n == 0 else float(binomtest(wins, n, 0.5).pvalue)


def subject_bootstrap_ci(per_subject_deltas: np.ndarray, *, seed: int = 20261017, n_boot: int = 20000) -> tuple[float, float, float]:
    """Mean and 95% percentile CI resampling subjects (the independent unit)."""
    d = np.asarray(per_subject_deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))

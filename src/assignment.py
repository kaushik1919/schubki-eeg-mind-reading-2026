"""Transductive, label-free score calibration exploiting the one-recording-per-candidate structure.

All functions take an (n_recordings, n_candidates) score matrix for ONE subject's
test set and return a same-shaped score matrix (higher = more likely), so every
recording still gets a full ranking.  They use only scores, never labels.
NOTE: these methods use the other recordings of the same subject (column
statistics / one-to-one assumption).  Whether that is permitted for the hidden
subjects needs organizer confirmation; they are benchmarked on validation only.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp


def rowz(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, np.float64)
    return (s - s.mean(1, keepdims=True)) / np.maximum(s.std(1, keepdims=True), 1e-12)


def colz(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, np.float64)
    return (s - s.mean(0, keepdims=True)) / np.maximum(s.std(0, keepdims=True), 1e-12)


def iterative_rowcol_z(s: np.ndarray, n_iter: int = 10) -> np.ndarray:
    x = rowz(s)
    for _ in range(n_iter):
        x = rowz(colz(x))
    return x


def sinkhorn_log(s: np.ndarray, tau: float, n_iter: int = 200) -> np.ndarray:
    """log of the doubly stochastic matrix closest (entropic OT) to exp(rowz(s) / tau)."""
    logp = rowz(s) / tau
    for _ in range(n_iter):
        logp = logp - logsumexp(logp, axis=1, keepdims=True)
        logp = logp - logsumexp(logp, axis=0, keepdims=True)
    return logp - logsumexp(logp, axis=1, keepdims=True)


def hungarian_first(s: np.ndarray, base: np.ndarray | None = None) -> np.ndarray:
    """Put each row's Hungarian-assigned candidate first; order the rest by `base` (default s)."""
    s = np.asarray(s, np.float64); base = s if base is None else np.asarray(base, np.float64)
    r, c = linear_sum_assignment(-s)
    out = base - base.min(1, keepdims=True)
    out[r, c] = out.max() + 1.0
    return out


def perturb_and_map(s: np.ndarray, tau: float, n_samples: int = 200, seed: int = 0) -> np.ndarray:
    """Assignment marginals under Gumbel-perturbed scores; ties broken by the Sinkhorn score."""
    z = rowz(s) / tau; rng = np.random.default_rng(seed); counts = np.zeros_like(z)
    for _ in range(n_samples):
        r, c = linear_sum_assignment(-(z + rng.gumbel(size=z.shape))); counts[r, c] += 1
    return counts / n_samples + 1e-6 * rowz(sinkhorn_log(s, tau))

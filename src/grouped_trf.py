"""Video-grouped ridge temporal response functions via sufficient statistics.

Model per subject (EEG at 100 Hz, t = 0..T-1, channels c):
    y_v(t) = m(t) + sum_{f, l} w[f, l] x_v(t - l, f)
with a per-time-sample intercept m(t) (the training-mean evoked response),
causal lags l = 0..L-1 and zero padding before onset (zero == silence).

All statistics are accumulated per *atom* (a set of whole videos), so any
training set is a union of atoms and every time sample of a video stays in
the same fold.  Centering and column standardisation are derived only from
the training atoms of each fit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def lag_feature_major(x: np.ndarray, n_lags: int) -> np.ndarray:
    """(T, F) -> (T, F * n_lags); column f * n_lags + l holds x[t - l, f] (0 for t < l)."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or n_lags < 1:
        raise ValueError("x must be (T, F) and n_lags >= 1")
    T, F = x.shape
    out = np.zeros((T, F, n_lags), dtype=np.float32)
    for lag in range(n_lags):
        out[lag:, :, lag] = x[:T - lag]
    return out.reshape(T, F * n_lags)


def columns_for(features: list[int], n_lags: int) -> np.ndarray:
    return np.concatenate([np.arange(f * n_lags, (f + 1) * n_lags) for f in features])


@dataclass
class Stats:
    n: int            # videos
    G: np.ndarray     # (P, P) sum_v X_v^T X_v
    Sx: np.ndarray    # (T, P) sum_v X_v(t)
    Sxy: np.ndarray   # (P, C) sum_v X_v^T Y_v
    Sy: np.ndarray    # (T, C) sum_v Y_v(t)

    def __add__(self, o: "Stats") -> "Stats":
        return Stats(self.n + o.n, self.G + o.G, self.Sx + o.Sx, self.Sxy + o.Sxy, self.Sy + o.Sy)

    def __sub__(self, o: "Stats") -> "Stats":
        return Stats(self.n - o.n, self.G - o.G, self.Sx - o.Sx, self.Sxy - o.Sxy, self.Sy - o.Sy)


def accumulate(stimuli: np.ndarray, eeg: np.ndarray, n_lags: int, batch: int = 64) -> Stats:
    """stimuli (V, T, F) on the EEG grid; eeg (V, T, C)."""
    stimuli, eeg = np.asarray(stimuli, np.float32), np.asarray(eeg, np.float32)
    V, T, F = stimuli.shape; C = eeg.shape[2]; P = F * n_lags
    G = np.zeros((P, P)); Sx = np.zeros((T, P)); Sxy = np.zeros((P, C)); Sy = eeg.sum(0, dtype=np.float64)
    for s in range(0, V, batch):
        X = np.stack([lag_feature_major(v, n_lags) for v in stimuli[s:s + batch]])   # (b, T, P)
        flat, yflat = X.reshape(-1, P), eeg[s:s + batch].reshape(-1, C)
        G += flat.T @ flat; Sx += X.sum(0, dtype=np.float64); Sxy += flat.T @ yflat
    return Stats(V, G, Sx, Sxy, Sy)


def empty_like(s: Stats) -> Stats:
    return Stats(0, np.zeros_like(s.G), np.zeros_like(s.Sx), np.zeros_like(s.Sxy), np.zeros_like(s.Sy))


@dataclass
class TRFFit:
    cols: np.ndarray
    W: np.ndarray       # (n_lambda, p, C)
    xbar: np.ndarray    # (T, p)
    ybar: np.ndarray    # (T, C)
    lambdas: np.ndarray
    n_lags: int

    def predict(self, stimuli: np.ndarray, which: int | None = None) -> np.ndarray:
        """Return (n_lambda, V, T, C) or (V, T, C) predictions for full-feature stimuli (V, T, F)."""
        X = np.stack([lag_feature_major(v, self.n_lags)[:, self.cols] for v in np.asarray(stimuli, np.float32)]) - self.xbar[None].astype(np.float32)
        W = self.W if which is None else self.W[which:which + 1]
        pred = np.einsum("vtp,kpc->kvtc", X, W.astype(np.float32), optimize=True) + self.ybar[None, None].astype(np.float32)
        return pred if which is None else pred[0]


def fit(stats: Stats, cols: np.ndarray, n_lags: int, lambdas: np.ndarray) -> TRFFit:
    """Ridge on per-time-centred, column-standardised design; alpha = lambda * n_samples."""
    if stats.n < 2:
        raise ValueError("need at least two training videos")
    T = stats.Sx.shape[0]; N = stats.n * T
    xbar, ybar = stats.Sx[:, cols] / stats.n, stats.Sy / stats.n
    Gc = stats.G[np.ix_(cols, cols)] - stats.n * xbar.T @ xbar
    Cxy = stats.Sxy[cols] - stats.n * xbar.T @ ybar
    sd = np.sqrt(np.clip(np.diag(Gc) / N, 0, None)); sd[sd < 1e-12] = 1.0
    w, V = np.linalg.eigh(Gc / np.outer(sd, sd))
    b = V.T @ (Cxy / sd[:, None])
    W = np.stack([(V @ (b / (np.clip(w, 0, None) + lam * N)[:, None])) / sd[:, None] for lam in lambdas])
    return TRFFit(np.asarray(cols), W, xbar, ybar, np.asarray(lambdas), n_lags)

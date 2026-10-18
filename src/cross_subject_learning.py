"""Cross-subject neural-consistency objectives and leakage-safe target construction."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from .contrastive_learning import _multi_positive_ce


def cross_subject_positive_mask(video_index: torch.Tensor, subject: torch.Tensor) -> torch.Tensor:
    """pos[i, j] = same video, different subject (never i == j, never same subject)."""
    return (video_index[:, None] == video_index[None, :]) & (subject[:, None] != subject[None, :])


def eeg_eeg_supcon(eeg: torch.Tensor, video_index: torch.Tensor, subject: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """Supervised contrastive loss over EEG embeddings with cross-subject same-video positives.

    Same-subject same-video pairs cannot occur (one recording per video per subject);
    other recordings of the same video are never negatives because they are positives.
    Anchors without a positive in the batch are skipped.
    """
    B = len(eeg); eye = torch.eye(B, dtype=torch.bool, device=eeg.device)
    pos = cross_subject_positive_mask(video_index, subject) & ~eye
    logits = (logit_scale * eeg @ eeg.T).masked_fill(eye, float("-inf"))
    if not pos.any():
        return eeg.sum() * 0.0
    return _multi_positive_ce(logits, pos.float())


def loso_average_targets(X: np.ndarray, S: np.ndarray, V: np.ndarray, source_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Leave-own-subject-out average response for each recording.

    target[r] = mean of X over recordings with the same video, a *different*
    subject, and source_mask True.  Only `source_mask` recordings (the fit-train
    set) ever contribute, so held-out or inner-validation EEG cannot leak in.
    Returns (targets (n, ...), available (n,) bool).
    """
    X = np.asarray(X, np.float32); flat = X.reshape(len(X), -1)
    sums: dict[int, np.ndarray] = {}; counts: dict[int, int] = {}
    for r in np.where(source_mask)[0]:
        v = int(V[r]); sums[v] = sums.get(v, 0) + flat[r]; counts[v] = counts.get(v, 0) + 1
    out = np.zeros_like(flat); ok = np.zeros(len(X), bool)
    for r in range(len(X)):
        v = int(V[r])
        if v not in sums: continue
        own = source_mask[r]
        n = counts[v] - (1 if own else 0)
        if n < 1: continue
        out[r] = (sums[v] - (flat[r] if own else 0)) / n; ok[r] = True
    return out.reshape(X.shape), ok


class AverageTargetCoder:
    """Train-only standardisation + PCA of average-response targets."""

    def __init__(self, n_components: int = 64, seed: int = 0) -> None:
        self.scaler, self.pca = StandardScaler(), PCA(n_components=n_components, random_state=seed, svd_solver="randomized")

    def fit(self, targets: np.ndarray) -> "AverageTargetCoder":
        z = self.pca.fit_transform(self.scaler.fit_transform(targets.reshape(len(targets), -1)))
        self.sd = z.std(0) + 1e-6
        return self

    def transform(self, targets: np.ndarray) -> np.ndarray:
        return (self.pca.transform(self.scaler.transform(targets.reshape(len(targets), -1))) / self.sd).astype(np.float32)


class RidgeReadout:
    """Map stimulus features to the mean EEG embedding of each training video (one row per video)."""

    def fit(self, video_features: np.ndarray, mean_embeddings: np.ndarray) -> "RidgeReadout":
        self.scaler = StandardScaler().fit(video_features)
        self.model = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(self.scaler.transform(video_features), mean_embeddings)
        return self

    def predict(self, video_features: np.ndarray) -> np.ndarray:
        z = self.model.predict(self.scaler.transform(video_features))
        return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)


def mean_embedding_per_video(emb: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vids = np.unique(V)
    m = np.stack([emb[V == v].mean(0) for v in vids])
    return vids, m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)

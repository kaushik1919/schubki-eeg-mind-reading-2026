"""Organizer-style reproducible linear baseline: EEG -> stimulus feature space."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

from ..preprocessing import EEGPreprocessor


class RidgeEEGDecoder:
    def __init__(self, alpha: float = 100.0, start: int = 0, stop: int = 1500) -> None:
        self.eeg = EEGPreprocessor(start, stop)
        self.target = StandardScaler()
        self.regressor = Ridge(alpha=alpha)

    def fit(self, eeg: np.ndarray, features: np.ndarray) -> "RidgeEEGDecoder":
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or len(features) != len(eeg):
            raise ValueError("features must be 2D with one row per EEG record")
        self.regressor.fit(self.eeg.fit_transform(eeg), self.target.fit_transform(features))
        return self

    def predict(self, eeg: np.ndarray) -> np.ndarray:
        return self.target.inverse_transform(self.regressor.predict(self.eeg.transform(eeg))).astype(np.float32)


class OrganizerForwardEncoder:
    """Exact model direction of the supplied notebooks: video features -> EEG."""

    def __init__(self, alphas: np.ndarray | None = None, variance: float = 0.95) -> None:
        self.feature_scaler = StandardScaler()
        self.pca = PCA(n_components=variance, svd_solver="full")
        self.pca_scaler = StandardScaler()
        self.model = RidgeCV(alphas=np.logspace(-3, 6, 20) if alphas is None else alphas)

    def fit(self, features: np.ndarray, eeg: np.ndarray) -> "OrganizerForwardEncoder":
        features, eeg = np.asarray(features, dtype=np.float32), np.asarray(eeg, dtype=np.float32)
        if features.ndim != 2 or eeg.ndim != 3 or len(features) != len(eeg):
            raise ValueError("features must be (n,d) and EEG must be (n,channels,time)")
        x = self.feature_scaler.fit_transform(features)
        x = self.pca.fit_transform(x)
        x = self.pca_scaler.fit_transform(x)
        self.model.fit(x, eeg.reshape(len(eeg), -1))
        self.eeg_shape = eeg.shape[1:]
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = self.feature_scaler.transform(np.asarray(features, dtype=np.float32))
        x = self.pca.transform(x)
        x = self.pca_scaler.transform(x)
        return self.model.predict(x).reshape(-1, *self.eeg_shape).astype(np.float32)

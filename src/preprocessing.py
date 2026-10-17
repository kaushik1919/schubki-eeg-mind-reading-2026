"""Fold-safe EEG vectorization and normalization."""
from __future__ import annotations

import numpy as np
from scipy.signal import decimate
from sklearn.preprocessing import StandardScaler

from .data import validate_eeg


class EEGPreprocessor:
    """Per-feature standardization fitted only on the supplied training EEG."""

    def __init__(self, start: int = 0, stop: int = 1500) -> None:
        if not 0 <= start < stop <= 1500:
            raise ValueError("EEG window must satisfy 0 <= start < stop <= 1500")
        self.start, self.stop = start, stop
        self.scaler = StandardScaler()

    def _flatten(self, eeg: np.ndarray) -> np.ndarray:
        eeg = validate_eeg(eeg)
        return eeg[:, :, self.start : self.stop].reshape(len(eeg), -1)

    def fit(self, eeg: np.ndarray) -> "EEGPreprocessor":
        self.scaler.fit(self._flatten(eeg))
        return self

    def transform(self, eeg: np.ndarray) -> np.ndarray:
        return self.scaler.transform(self._flatten(eeg)).astype(np.float32)

    def fit_transform(self, eeg: np.ndarray) -> np.ndarray:
        return self.fit(eeg).transform(eeg)


def organizer_downsample(eeg: np.ndarray, factor: int = 5) -> np.ndarray:
    """The organizer baseline's zero-phase FIR 500 Hz -> 100 Hz conversion."""
    eeg = validate_eeg(eeg)
    if factor < 1 or 1500 % factor:
        raise ValueError("factor must be a positive divisor of 1500")
    return decimate(eeg, factor, axis=-1, ftype="fir", zero_phase=True).astype(np.float32)

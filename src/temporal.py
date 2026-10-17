"""Leakage-safe helpers for linear temporal response-function encoding."""
from __future__ import annotations

import numpy as np


def resample_trajectory(values: np.ndarray, target_samples: int) -> np.ndarray:
    """Linearly resample a (time, feature) stimulus trajectory."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or len(values) < 2 or target_samples < 2:
        raise ValueError("expected at least two time samples and a 2-D trajectory")
    source = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    target = np.linspace(0.0, 1.0, target_samples, dtype=np.float32)
    return np.stack([np.interp(target, source, values[:, i]) for i in range(values.shape[1])], axis=1).astype(np.float32)


def lagged_design(values: np.ndarray, max_lag_samples: int) -> np.ndarray:
    """Return causal zero-padded lags [0, ..., max_lag] for each timepoint."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or max_lag_samples < 0:
        raise ValueError("values must be (time, feature) and lag non-negative")
    padded = np.pad(values, ((max_lag_samples, 0), (0, 0)))
    return np.concatenate([padded[max_lag_samples - lag:max_lag_samples - lag + len(values)] for lag in range(max_lag_samples + 1)], axis=1)


def temporal_design(trajectories: np.ndarray, eeg_samples: int, max_lag_samples: int) -> np.ndarray:
    """Construct concatenated causal TRF samples for a batch of videos."""
    trajectories = np.asarray(trajectories, dtype=np.float32)
    if trajectories.ndim != 3:
        raise ValueError("trajectories must be (video, time, feature)")
    return np.concatenate([lagged_design(resample_trajectory(x, eeg_samples), max_lag_samples) for x in trajectories], axis=0)

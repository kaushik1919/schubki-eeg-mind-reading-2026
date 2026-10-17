"""Strict data loading and validation. No organizer layout is assumed here."""
from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np

N_CHANNELS: Final = 17
N_SAMPLES: Final = 1500
CANDIDATE_IDS: Final = np.arange(1001, 1103, dtype=np.int64)


def load_array(path: str | Path, *, name: str) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    return np.load(path, allow_pickle=False)


def validate_eeg(eeg: np.ndarray, *, name: str = "EEG") -> np.ndarray:
    eeg = np.asarray(eeg)
    if eeg.ndim != 3 or eeg.shape[1:] != (N_CHANNELS, N_SAMPLES):
        raise ValueError(
            f"{name} must have shape (n_records, {N_CHANNELS}, {N_SAMPLES}); got {eeg.shape}"
        )
    if not np.issubdtype(eeg.dtype, np.number) or not np.isfinite(eeg).all():
        raise ValueError(f"{name} must be finite numeric data")
    return eeg.astype(np.float32, copy=False)


def validate_candidate_ids(ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids)
    if ids.shape != (len(CANDIDATE_IDS),):
        raise ValueError(f"candidate IDs must have shape ({len(CANDIDATE_IDS)},), got {ids.shape}")
    if not np.array_equal(np.sort(ids), CANDIDATE_IDS):
        raise ValueError("candidate IDs must contain 1001..1102 exactly once")
    return ids.astype(np.int64, copy=False)


def load_eeg(path: str | Path, *, name: str = "EEG") -> np.ndarray:
    """Load EEG unchanged in its input row order after validation."""
    return validate_eeg(load_array(path, name=name), name=name)


def load_training_subject(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load organizer `sub-XX_trn.npz`, sorted by numeric stimulus ID."""
    archive = np.load(Path(path), allow_pickle=False)
    try:
        ids = np.array(sorted((int(key) for key in archive.files)), dtype=np.int64)
        eeg = validate_eeg(np.stack([archive[str(stimulus_id)] for stimulus_id in ids]), name="training EEG")
    finally:
        archive.close()
    return ids, eeg

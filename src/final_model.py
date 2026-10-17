"""Frozen Model C: final CLIP / CLIP block 8 / simple audio Ridge forward models with row-z score fusion.

This module only *composes* the exact validated components; it defines no new
hyperparameters.  The regressor and EEG representation are imported from
`scripts/clip_controlled_cv.py`, the implementation used in every validation run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.clip_controlled_cv import ForwardRegressor, representation
from src.controlled_cv import millisecond_slice
from src.data import CANDIDATE_IDS, validate_candidate_ids, validate_eeg
from src.retrieval import organizer_eeg_correlation_scores, rank_candidates, weighted_score_ensemble

TARGET_WINDOW_MS = (0, 3000)
SCORE_WINDOW_MS = (50, 600)
EXPECTED_CONFIG_ID = "C_final_block8_audio_45_45_10"
EXPECTED_WEIGHTS = {"final_clip": 0.45, "clip_block8": 0.45, "audio_simple": 0.10}


def load_frozen_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text())
    if config.get("config_id") != EXPECTED_CONFIG_ID:
        raise ValueError(f"unexpected config_id {config.get('config_id')}")
    weights = {c["name"]: c["weight"] for c in config["constituents"]}
    if weights != EXPECTED_WEIGHTS:
        raise ValueError(f"frozen weights changed: {weights}")
    return config


def score_slice() -> slice:
    return millisecond_slice(representation_start_ms=TARGET_WINDOW_MS[0], representation_stop_ms=TARGET_WINDOW_MS[1], sampling_hz=100, window_ms=SCORE_WINDOW_MS)


class FrozenModelC:
    """Subject-specific bundle of three ForwardRegressor('ridge') models."""

    def __init__(self, names: list[str], weights: list[float]) -> None:
        self.names, self.weights = list(names), np.asarray(weights, dtype=np.float32)
        self.models: dict[str, ForwardRegressor] = {}
        self.training_ids: np.ndarray | None = None

    def fit(self, training_ids: np.ndarray, training_eeg: np.ndarray, train_features: dict[str, np.ndarray]) -> "FrozenModelC":
        ids = np.asarray(training_ids, dtype=np.int64)
        if ids.ndim != 1 or len(np.unique(ids)) != len(ids) or ids.min() < 1 or ids.max() > 1000:
            raise ValueError("training IDs must be unique training-video IDs in 1..1000")
        target = representation(validate_eeg(training_eeg, name="training EEG"), TARGET_WINDOW_MS)
        for name in self.names:
            feats = np.asarray(train_features[name])
            if feats.shape[0] != 1000:
                raise ValueError(f"{name} training features must have rows for IDs 1..1000")
            self.models[name] = ForwardRegressor("ridge", 64).fit(feats[ids - 1], target)  # row = ID - 1
        self.training_ids = ids
        return self

    def constituent_scores(self, test_eeg: np.ndarray, candidate_features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Scores[i, j] = similarity of test EEG row i (file order) to candidate column j."""
        observed = representation(validate_eeg(test_eeg, name="test EEG"), TARGET_WINDOW_MS)
        sl = score_slice(); out = {}
        for name in self.names:
            feats = np.asarray(candidate_features[name])
            if feats.shape[0] != len(CANDIDATE_IDS):
                raise ValueError(f"{name} candidate features must have 102 rows")
            out[name] = organizer_eeg_correlation_scores(observed, self.models[name].predict(feats), start=sl.start, stop=sl.stop)
        return out

    def fused_scores(self, constituent: dict[str, np.ndarray]) -> np.ndarray:
        return weighted_score_ensemble([constituent[n] for n in self.names], self.weights)

    def rank(self, test_eeg: np.ndarray, candidate_features: dict[str, np.ndarray], candidate_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        ids = validate_candidate_ids(candidate_ids)
        constituent = self.constituent_scores(test_eeg, candidate_features)
        fused = self.fused_scores(constituent)
        if not np.isfinite(fused).all():
            raise ValueError("non-finite fused scores")
        return rank_candidates(fused, ids), fused, constituent

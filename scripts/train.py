#!/usr/bin/env python3
"""Train the organizer-compatible AlexNet-feature -> EEG forward baseline.

Features must be one row per numeric-sorted key in the organizer training NPZ.
Feature extraction is intentionally separate so it can be replaced by another frozen encoder.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import joblib
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_array, load_training_subject
from src.models import OrganizerForwardEncoder
from src.preprocessing import organizer_downsample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-npz", required=True)
    parser.add_argument("--train-features", required=True, help="Rows aligned to numeric-sorted NPZ keys")
    parser.add_argument("--feature-ids", default=None, help="Optional IDs for a shared feature cache; selects rows matching training NPZ keys")
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--metadata-out", default=None)
    args = parser.parse_args()
    ids, eeg = load_training_subject(args.train_npz)
    features = load_array(args.train_features, name="training video features")
    if args.feature_ids:
        feature_ids = load_array(args.feature_ids, name="feature IDs").astype(np.int64)
        if feature_ids.ndim != 1 or len(feature_ids) != len(features) or len(np.unique(feature_ids)) != len(feature_ids):
            raise ValueError("feature IDs must be unique and match feature-row count")
        lookup = {int(stimulus_id): row for row, stimulus_id in enumerate(feature_ids)}
        missing = sorted(set(ids) - set(lookup))
        if missing:
            raise ValueError(f"shared feature cache is missing {len(missing)} training IDs, e.g. {missing[:5]}")
        features = features[[lookup[int(stimulus_id)] for stimulus_id in ids]]
    if features.ndim != 2 or len(features) != len(ids):
        raise ValueError("train features must be 2D and aligned one-to-one with sorted training NPZ keys")
    started = time.perf_counter()
    model = OrganizerForwardEncoder().fit(features, organizer_downsample(eeg))
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "training_ids": ids, "decimation": 5, "score_window": [5, 60]}, args.model_out, compress=3)
    report = {"n_training_recordings": int(len(ids)), "pca_components": int(model.pca.n_components_), "ridge_alpha": float(model.model.alpha_), "train_seconds": time.perf_counter() - started}
    if args.metadata_out:
        Path(args.metadata_out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__": main()

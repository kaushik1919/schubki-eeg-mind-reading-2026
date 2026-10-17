#!/usr/bin/env python3
"""Train a subject-pooled forward encoder using only organizer training EEG."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_array, load_training_subject
from src.models import OrganizerForwardEncoder
from src.preprocessing import organizer_downsample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True); parser.add_argument("--features", required=True); parser.add_argument("--feature-ids", required=True)
    parser.add_argument("--subjects", nargs="+", default=["01", "02", "03", "04", "05", "06"])
    parser.add_argument("--model-out", required=True)
    args = parser.parse_args(); train_dir = Path(args.train_dir)
    feature_ids = load_array(args.feature_ids, name="feature IDs").astype(int); features = load_array(args.features, name="pooled training features")
    if len(feature_ids) != len(features) or len(np.unique(feature_ids)) != len(feature_ids): raise ValueError("feature IDs must uniquely align to feature rows")
    lookup = {int(identifier): row for row, identifier in enumerate(feature_ids)}; x_parts = []; eeg_parts = []
    for subject in args.subjects:
        ids, eeg = load_training_subject(train_dir / f"sub-{subject}_trn.npz")
        if set(ids) - set(lookup): raise ValueError(f"feature cache missing IDs for sub-{subject}")
        x_parts.append(features[[lookup[int(identifier)] for identifier in ids]])
        eeg_parts.append(organizer_downsample(eeg))
    model = OrganizerForwardEncoder().fit(np.concatenate(x_parts), np.concatenate(eeg_parts))
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "subjects": args.subjects, "decimation": 5, "score_window": [5, 60]}, args.model_out, compress=3)
    print({"subjects": args.subjects, "n_training_recordings": int(sum(len(x) for x in x_parts)), "pca_components": int(model.pca.n_components_), "ridge_alpha": float(model.model.alpha_)})


if __name__ == "__main__": main()

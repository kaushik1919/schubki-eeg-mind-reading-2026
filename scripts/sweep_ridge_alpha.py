#!/usr/bin/env python3
"""Development-only fixed-ridge-alpha sweep using already fitted training transforms."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import joblib
import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_array, load_training_subject, load_eeg
from src.evaluation import retrieval_metrics
from src.preprocessing import organizer_downsample
from src.retrieval import organizer_eeg_correlation_scores, rank_candidates, weighted_score_ensemble


def transformed_features(model, features: np.ndarray) -> np.ndarray:
    return model.pca_scaler.transform(model.pca.transform(model.feature_scaler.transform(features)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True); parser.add_argument("--experiments", default="experiments")
    parser.add_argument("--feature-ids", required=True); parser.add_argument("--alexnet-train", required=True); parser.add_argument("--alexnet-test", required=True)
    parser.add_argument("--clip-train", required=True); parser.add_argument("--clip-test", required=True)
    parser.add_argument("--alphas", nargs="+", type=float, default=[1e3, 3e3, 1e4, 3e4, 1e5, 3e5])
    args = parser.parse_args(); root, exp = Path(args.data_root), Path(args.experiments)
    feature_ids = load_array(args.feature_ids, name="feature IDs").astype(int); lookup = {identifier: row for row, identifier in enumerate(feature_ids)}
    all_train = [load_array(args.alexnet_train, name="AlexNet training features"), load_array(args.clip_train, name="CLIP training features")]
    candidates = [load_array(args.alexnet_test, name="AlexNet candidate features"), load_array(args.clip_test, name="CLIP candidate features")]
    prepared = {}
    for subject in ("01", "02", "03", "04"):
        ids, eeg = load_training_subject(root / "trn" / f"sub-{subject}_trn.npz")
        x_rows = [features[[lookup[int(identifier)] for identifier in ids]] for features in all_train]
        models = [joblib.load(exp / f"sub-{subject}_{name}_baseline.joblib")["model"] for name in ("alexnet", "clip")]
        prepared[subject] = ([transformed_features(model, x) for model, x in zip(models, x_rows)], [transformed_features(model, x) for model, x in zip(models, candidates)], organizer_downsample(eeg).reshape(len(eeg), -1), organizer_downsample(load_eeg(root / "tst" / f"sub-{subject}_tst.npy")), load_array(root / "tst_stim" / f"sub-{subject}_tst_stim.npy", name="development labels"), models[0].eeg_shape)
    for alpha in args.alphas:
        results = []
        for train_x, candidate_x, targets, observed, truth, eeg_shape in prepared.values():
            scores = []
            for x_train, x_candidate in zip(train_x, candidate_x):
                predicted = Ridge(alpha=alpha).fit(x_train, targets).predict(x_candidate).reshape(-1, *eeg_shape)
                scores.append(organizer_eeg_correlation_scores(observed, predicted, start=5, stop=60))
            results.append(retrieval_metrics(rank_candidates(weighted_score_ensemble(scores, np.array([0.9, 0.1])), np.arange(1001, 1103)), truth))
        print(f"alpha={alpha:g} top1={np.mean([r['top1'] for r in results]):.4f} top5={np.mean([r['top5'] for r in results]):.4f}")


if __name__ == "__main__": main()

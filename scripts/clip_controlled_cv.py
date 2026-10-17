#!/usr/bin/env python3
"""Run fixed train-only CLIP regression/retrieval experiments with saved pairs.

No organizer test file or development label is read.  Each run evaluates one
predeclared configuration on identical shared-ID subject folds.  Thus CSV/NPZ
artifacts support paired comparisons, but a configuration must not be selected
and claimed unbiased from these same outer folds.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

import numpy as np
from scipy.signal import decimate
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.controlled_cv import millisecond_slice, shared_holdout_ids
from src.data import load_training_subject
from src.evaluation import retrieval_metrics
from src.retrieval import organizer_eeg_correlation_scores, rank_candidates


class ForwardRegressor:
    """Feature PCA plus a fixed train-only regression/target reconstruction."""
    def __init__(self, method: str, target_components: int) -> None:
        self.method, self.target_components = method, target_components
        self.x_scale, self.x_pca, self.x_scale2 = StandardScaler(), PCA(n_components=.95, svd_solver="full"), StandardScaler()

    def _x_fit(self, x: np.ndarray) -> np.ndarray:
        return self.x_scale2.fit_transform(self.x_pca.fit_transform(self.x_scale.fit_transform(x)))

    def _x_transform(self, x: np.ndarray) -> np.ndarray:
        return self.x_scale2.transform(self.x_pca.transform(self.x_scale.transform(x)))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ForwardRegressor":
        self.y_shape = y.shape[1:]; x = self._x_fit(x); y2 = y.reshape(len(y), -1)
        if self.method == "ridge":
            self.model = RidgeCV(alphas=np.logspace(-3, 6, 20)).fit(x, y2)
            self.y_pca = self.y_scale = None
        else:
            self.y_scale = StandardScaler()
            y_scaled = self.y_scale.fit_transform(y2)
            self.y_pca = PCA(n_components=min(self.target_components, len(y_scaled) - 1), svd_solver="randomized", random_state=2026)
            y_latent = self.y_pca.fit_transform(y_scaled)
            if self.method == "ridge_target_pca":
                self.model = RidgeCV(alphas=np.logspace(-3, 6, 20)).fit(x, y_latent)
            elif self.method == "pls":
                self.model = PLSRegression(n_components=min(self.target_components, x.shape[1], y_latent.shape[1]), scale=False, max_iter=1000).fit(x, y_latent)
            else:
                raise ValueError(self.method)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        result = self.model.predict(self._x_transform(x))
        if self.y_pca is not None:
            result = self.y_scale.inverse_transform(self.y_pca.inverse_transform(result))
        return result.reshape(-1, *self.y_shape).astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", default="features/all_train_clip_features.npy")
    p.add_argument("--data-root", default="eeg-mind-reading-challenge")
    p.add_argument("--subjects", nargs="+", default=["01", "02", "03", "04", "05", "06"])
    p.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    p.add_argument("--holdout-size", type=int, default=102)
    p.add_argument("--holdout-ids", default=None, help="Optional precommitted shared-ID .npy holdout; bypasses random split creation")
    p.add_argument("--method", choices=["ridge", "ridge_target_pca", "pls"], default="ridge")
    p.add_argument("--target-components", type=int, default=64)
    p.add_argument("--target-window-ms", nargs=2, type=int, default=[0, 3000])
    p.add_argument("--score-window-ms", nargs=2, type=int, default=[50, 600])
    p.add_argument("--label", required=True); p.add_argument("--output", required=True); p.add_argument("--predictions-dir", required=True)
    return p.parse_args()


def representation(eeg: np.ndarray, window: tuple[int, int]) -> np.ndarray:
    eeg = decimate(eeg, 5, axis=-1, ftype="fir", zero_phase=True)
    selected = millisecond_slice(representation_start_ms=0, representation_stop_ms=3000, sampling_hz=100, window_ms=window)
    return eeg[..., selected].astype(np.float32)


def main() -> None:
    args = parse_args(); feature = np.load(args.features, mmap_mode="r")
    if feature.shape[0] != 1000: raise ValueError("features must align with IDs 1..1000")
    root = Path(args.data_root); subjects = {s: load_training_subject(root / "trn" / f"sub-{s}_trn.npz") for s in args.subjects}
    subject_ids = {s: x[0] for s, x in subjects.items()}; rows = []; destination = Path(args.predictions_dir); destination.mkdir(parents=True, exist_ok=True)
    folds = [(str(seed), shared_holdout_ids(subject_ids, seed=seed, size=args.holdout_size)) for seed in args.seeds]
    if args.holdout_ids:
        heldout = np.load(args.holdout_ids, allow_pickle=False).astype(np.int64)
        if heldout.shape != (args.holdout_size,) or len(np.unique(heldout)) != len(heldout): raise ValueError("holdout IDs must be unique and match --holdout-size")
        if not all(set(heldout).issubset(ids) for ids in subject_ids.values()): raise ValueError("holdout includes an ID absent for a subject")
        folds = [("confirmation", np.sort(heldout))]
    for seed, heldout in folds:
        heldout_set = set(heldout.tolist())
        for subject, (ids, raw_eeg) in subjects.items():
            mask = np.fromiter((i in heldout_set for i in ids), dtype=bool, count=len(ids))
            if mask.sum() != len(heldout): raise RuntimeError("shared ID missing in subject")
            started = time.perf_counter(); train_eeg = representation(raw_eeg[~mask], tuple(args.target_window_ms)); observed = representation(raw_eeg[mask], tuple(args.target_window_ms))
            model = ForwardRegressor(args.method, args.target_components).fit(feature[ids[~mask] - 1], train_eeg)
            predicted = model.predict(feature[heldout - 1])
            score_slice = millisecond_slice(representation_start_ms=args.target_window_ms[0], representation_stop_ms=args.target_window_ms[1], sampling_hz=100, window_ms=tuple(args.score_window_ms))
            scores = organizer_eeg_correlation_scores(observed, predicted, start=score_slice.start, stop=score_slice.stop)
            rankings = rank_candidates(scores, heldout); target_ids = ids[mask]; metrics = retrieval_metrics(rankings, target_ids)
            artifact = destination / f"{args.label}_seed{seed}_sub{subject}.npz"
            np.savez_compressed(artifact, heldout_ids=heldout, target_ids=target_ids, scores=scores, rankings=rankings)
            row = {"experiment": args.label, "method": args.method, "target_components": args.target_components, "target_window_ms": f"{args.target_window_ms[0]}:{args.target_window_ms[1]}", "score_window_ms": f"{args.score_window_ms[0]}:{args.score_window_ms[1]}", "seed": seed, "subject": subject, "n_train": int((~mask).sum()), "top1": metrics["top1"], "top5": metrics["top5"], "seconds": time.perf_counter()-started, "artifact": str(artifact)}
            rows.append(row); print(row, flush=True)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print({"mean_top1": float(np.mean([r["top1"] for r in rows])), "mean_top5": float(np.mean([r["top5"] for r in rows])), "mean_seconds": float(np.mean([r["seconds"] for r in rows]))})


if __name__ == "__main__": main()

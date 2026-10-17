#!/usr/bin/env python3
"""Evaluate a causal, subject-specific linear temporal-response function."""
from __future__ import annotations
import argparse, csv, sys, time
from pathlib import Path
import numpy as np
from scipy.signal import decimate
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.controlled_cv import millisecond_slice
from src.data import load_training_subject
from src.evaluation import retrieval_metrics
from src.retrieval import organizer_eeg_correlation_scores, rank_candidates
from src.temporal import temporal_design


def eeg_100hz(eeg: np.ndarray) -> np.ndarray:
    return decimate(eeg, 5, axis=-1, ftype="fir", zero_phase=True).transpose(0, 2, 1).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True); p.add_argument("--holdout-ids", required=True); p.add_argument("--label", required=True); p.add_argument("--output", required=True); p.add_argument("--predictions-dir", required=True)
    p.add_argument("--data-root", default="eeg-mind-reading-challenge"); p.add_argument("--subjects", nargs="+", default=["01", "02", "03", "04", "05", "06"]); p.add_argument("--max-lag-ms", type=int, default=600); p.add_argument("--score-window-ms", nargs=2, type=int, default=[50, 600]); args = p.parse_args()
    if args.max_lag_ms % 10: raise ValueError("max lag must be on the 100 Hz grid")
    features = np.load(args.features, mmap_mode="r"); heldout = np.load(args.holdout_ids, allow_pickle=False).astype(np.int64)
    if features.ndim != 3 or features.shape[0] != 1000 or len(np.unique(heldout)) != len(heldout): raise ValueError("invalid feature cache or holdout")
    destination = Path(args.predictions_dir); destination.mkdir(parents=True, exist_ok=True); rows = []; max_lag = args.max_lag_ms // 10
    for subject in args.subjects:
        ids, raw = load_training_subject(Path(args.data_root) / "trn" / f"sub-{subject}_trn.npz"); mask = np.isin(ids, heldout)
        if mask.sum() != len(heldout): raise ValueError(f"holdout not shared by subject {subject}")
        started = time.perf_counter(); train_y, observed = eeg_100hz(raw[~mask]), eeg_100hz(raw[mask])
        train_x = temporal_design(features[ids[~mask] - 1], train_y.shape[1], max_lag); candidate_x = temporal_design(features[heldout - 1], train_y.shape[1], max_lag)
        xscale, yscale = StandardScaler(), StandardScaler(); train_flat = train_y.reshape(-1, train_y.shape[-1]); model = RidgeCV(alphas=np.logspace(-2, 6, 17)).fit(xscale.fit_transform(train_x), yscale.fit_transform(train_flat))
        predicted = yscale.inverse_transform(model.predict(xscale.transform(candidate_x))).reshape(len(heldout), train_y.shape[1], train_y.shape[2]).transpose(0, 2, 1).astype(np.float32)
        observed_score = observed.transpose(0, 2, 1); score = millisecond_slice(representation_start_ms=0, representation_stop_ms=3000, sampling_hz=100, window_ms=tuple(args.score_window_ms)); scores = organizer_eeg_correlation_scores(observed_score, predicted, start=score.start, stop=score.stop); rankings = rank_candidates(scores, heldout); metrics = retrieval_metrics(rankings, ids[mask])
        artifact = destination / f"{args.label}_seedresearch_sub{subject}.npz"; np.savez_compressed(artifact, heldout_ids=heldout, target_ids=ids[mask], scores=scores, rankings=rankings)
        row = {"experiment": args.label, "subject": subject, "n_train": int((~mask).sum()), "max_lag_ms": args.max_lag_ms, "top1": metrics["top1"], "top5": metrics["top5"], "seconds": time.perf_counter() - started, "artifact": str(artifact)}; rows.append(row); print(row, flush=True)
    with Path(args.output).open("w", newline="") as h: writer = csv.DictWriter(h, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print({"mean_top1": float(np.mean([r["top1"] for r in rows])), "mean_top5": float(np.mean([r["top5"] for r in rows])), "mean_seconds": float(np.mean([r["seconds"] for r in rows]))})


if __name__ == "__main__": main()

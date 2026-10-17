#!/usr/bin/env python3
"""Leakage-free, video-level validation for a frozen feature representation.

Each split holds out the same video IDs for every supplied subject.  Those IDs
are excluded from every subject's regression fit, then each subject retrieves
only among its own held-out videos.  No organizer test EEG or test labels are
read by this script.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

import numpy as np
from scipy.signal import decimate, welch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_training_subject
from src.models.baseline import OrganizerForwardEncoder
from src.retrieval import organizer_eeg_correlation_scores, rank_candidates
from src.evaluation import retrieval_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="eeg-mind-reading-challenge")
    parser.add_argument("--features", required=True, help="Feature rows for IDs 1..1000")
    parser.add_argument("--subjects", nargs="+", default=["01", "02", "03", "04", "05", "06"])
    parser.add_argument("--holdout-size", type=int, default=102)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--output", required=True, help="CSV leaderboard output")
    parser.add_argument("--label", required=True)
    parser.add_argument("--downsample-factor", type=int, default=5, choices=[1, 2, 5])
    parser.add_argument("--window-ms", nargs=2, type=int, default=[0, 3000], metavar=("START", "STOP"))
    parser.add_argument("--channel-standardize", action="store_true", help="Fit per-channel mean/std on training videos only")
    parser.add_argument("--score-window-ms", nargs=2, type=int, default=[50, 600], metavar=("START", "STOP"))
    parser.add_argument("--eeg-representation", choices=["time", "bandpower"], default="time")
    return parser.parse_args()


def eeg_representation(eeg: np.ndarray, *, factor: int, window_ms: list[int]) -> np.ndarray:
    reduced = eeg if factor == 1 else decimate(eeg, factor, axis=-1, ftype="fir", zero_phase=True)
    hz = 500 // factor
    start, stop = (round(value * hz / 1000) for value in window_ms)
    if not 0 <= start < stop <= reduced.shape[-1]:
        raise ValueError(f"window {window_ms}ms is invalid after downsampling")
    return reduced[..., start:stop]


def bandpower(eeg: np.ndarray, *, sampling_hz: int) -> np.ndarray:
    """Training-safe per-channel Welch power in theta, alpha, beta, low-gamma."""
    frequencies, power = welch(eeg, fs=sampling_hz, axis=-1, nperseg=min(128, eeg.shape[-1]))
    bands = ((4, 8), (8, 13), (13, 30), (30, min(40, sampling_hz / 2)))
    values = []
    for low, high in bands:
        mask = (frequencies >= low) & (frequencies <= high)
        values.append(np.trapezoid(power[..., mask], frequencies[mask], axis=-1))
    return np.stack(values, axis=-1)


def main() -> None:
    args = parse_args()
    root = Path(args.data_root)
    features = np.load(args.features, mmap_mode="r")
    if features.ndim != 2 or len(features) != 1000:
        raise ValueError("features must be a (1000, D) array aligned to IDs 1..1000")

    loaded = {subject: load_training_subject(root / "trn" / f"sub-{subject}_trn.npz") for subject in args.subjects}
    shared_ids = set(loaded[args.subjects[0]][0].tolist())
    for subject in args.subjects[1:]:
        shared_ids.intersection_update(loaded[subject][0].tolist())
    shared_ids = np.array(sorted(shared_ids), dtype=np.int64)
    if len(shared_ids) < args.holdout_size:
        raise ValueError(f"only {len(shared_ids)} video IDs shared by all subjects")

    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        heldout_ids = np.sort(rng.choice(shared_ids, size=args.holdout_size, replace=False))
        heldout_set = set(heldout_ids.tolist())
        for subject, (ids, eeg) in loaded.items():
            test_mask = np.fromiter((identifier in heldout_set for identifier in ids), dtype=bool, count=len(ids))
            train_mask = ~test_mask
            if test_mask.sum() != args.holdout_size:
                raise AssertionError("shared holdout unexpectedly missing from a subject")
            started = time.perf_counter()
            train_eeg = eeg_representation(eeg[train_mask], factor=args.downsample_factor, window_ms=args.window_ms)
            observed = eeg_representation(eeg[test_mask], factor=args.downsample_factor, window_ms=args.window_ms)
            if args.channel_standardize:
                mean = train_eeg.mean(axis=(0, 2), keepdims=True)
                std = np.maximum(train_eeg.std(axis=(0, 2), keepdims=True), 1e-6)
                train_eeg = (train_eeg - mean) / std
                observed = (observed - mean) / std
            if args.eeg_representation == "bandpower":
                train_eeg = bandpower(train_eeg, sampling_hz=500 // args.downsample_factor)
                observed = bandpower(observed, sampling_hz=500 // args.downsample_factor)
            model = OrganizerForwardEncoder().fit(features[ids[train_mask] - 1], train_eeg)
            predicted = model.predict(features[heldout_ids - 1])
            representation_start = args.window_ms[0]
            score_start = 0 if args.eeg_representation == "bandpower" else max(0, round((args.score_window_ms[0] - representation_start) * (500 // args.downsample_factor) / 1000))
            score_stop = observed.shape[-1] if args.eeg_representation == "bandpower" else min(observed.shape[-1], round((args.score_window_ms[1] - representation_start) * (500 // args.downsample_factor) / 1000))
            if score_start >= score_stop:
                score_start, score_stop = 0, observed.shape[-1]
            scores = organizer_eeg_correlation_scores(observed, predicted, start=score_start, stop=score_stop)
            ranking = rank_candidates(scores, heldout_ids)
            metrics = retrieval_metrics(ranking, ids[test_mask])
            rows.append({
                "experiment": args.label, "seed": seed, "subject": subject,
                "n_train": int(train_mask.sum()), "n_heldout": int(test_mask.sum()), "downsample_factor": args.downsample_factor,
                "window_ms": f"{args.window_ms[0]}:{args.window_ms[1]}", "channel_standardize": args.channel_standardize,
                "eeg_representation": args.eeg_representation,
                "score_window_ms": f"{args.score_window_ms[0]}:{args.score_window_ms[1]}",
                "top1": metrics["top1"], "top5": metrics["top5"],
                "seconds": time.perf_counter() - started,
            })
            print(rows[-1], flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print({"mean_top1": float(np.mean([r["top1"] for r in rows])), "mean_top5": float(np.mean([r["top5"] for r in rows])), "output": str(output)})


if __name__ == "__main__":
    main()

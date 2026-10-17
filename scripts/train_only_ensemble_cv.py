#!/usr/bin/env python3
"""Train-only video-level CV for a predeclared equal-weight score ensemble."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_training_subject
from src.evaluation import retrieval_metrics
from src.models.baseline import OrganizerForwardEncoder
from src.preprocessing import organizer_downsample
from src.retrieval import organizer_eeg_correlation_scores, rank_candidates, weighted_score_ensemble


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="eeg-mind-reading-challenge")
    parser.add_argument("--features", nargs="+", required=True, help="One or more (1000, D) feature arrays")
    parser.add_argument("--subjects", nargs="+", default=["01", "02", "03", "04", "05", "06"])
    parser.add_argument("--holdout-size", type=int, default=102)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--label", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    features = [np.load(path, mmap_mode="r") for path in args.features]
    if any(array.ndim != 2 or len(array) != 1000 for array in features): raise ValueError("all feature arrays must align to IDs 1..1000")
    root = Path(args.data_root)
    loaded = {s: load_training_subject(root / "trn" / f"sub-{s}_trn.npz") for s in args.subjects}
    shared = set(loaded[args.subjects[0]][0].tolist())
    for s in args.subjects[1:]: shared.intersection_update(loaded[s][0].tolist())
    shared = np.array(sorted(shared), dtype=np.int64)
    rows = []
    for seed in args.seeds:
        heldout = np.sort(np.random.default_rng(seed).choice(shared, args.holdout_size, replace=False))
        heldout_set = set(heldout.tolist())
        for subject, (ids, eeg) in loaded.items():
            test = np.fromiter((i in heldout_set for i in ids), dtype=bool, count=len(ids))
            started = time.perf_counter(); observed = organizer_downsample(eeg[test])
            scores = []
            for feature in features:
                model = OrganizerForwardEncoder().fit(feature[ids[~test] - 1], organizer_downsample(eeg[~test]))
                scores.append(organizer_eeg_correlation_scores(observed, model.predict(feature[heldout - 1])))
            ranking = rank_candidates(weighted_score_ensemble(scores, np.ones(len(scores))), heldout)
            metric = retrieval_metrics(ranking, ids[test])
            row = {"experiment": args.label, "seed": seed, "subject": subject, "top1": metric["top1"], "top5": metric["top5"], "seconds": time.perf_counter() - started}
            rows.append(row); print(row, flush=True)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print({"mean_top1": float(np.mean([r["top1"] for r in rows])), "mean_top5": float(np.mean([r["top5"] for r in rows]))})


if __name__ == "__main__": main()

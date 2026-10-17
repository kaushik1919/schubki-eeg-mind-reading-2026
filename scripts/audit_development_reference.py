#!/usr/bin/env python3
"""Document existing score matrices against development labels exactly once.

This is an audit-only tool.  It must not be used to choose configurations.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import retrieval_metrics
from src.retrieval import rank_candidates, weighted_score_ensemble


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", default="experiments")
    parser.add_argument("--truth-dir", default="eeg-mind-reading-challenge/tst_stim")
    args = parser.parse_args()
    root, truth_root = Path(args.experiments), Path(args.truth_dir)
    configurations = {"alexnet": ("alexnet", [1.0]), "exploratory_3way": ("alexnet,clip,dinov2s", [0.36, 0.04, 0.60])}
    for name, (models, weights) in configurations.items():
        measurements = []
        for subject in ("01", "02", "03", "04"):
            matrices = [np.load(root / f"sub-{subject}_{model}_scores.npy") for model in models.split(",")]
            scores = matrices[0] if len(matrices) == 1 else weighted_score_ensemble(matrices, np.array(weights))
            result = retrieval_metrics(rank_candidates(scores), np.load(truth_root / f"sub-{subject}_tst_stim.npy"))
            measurements.append(result)
        print(name, [{key: round(value * 100, 2) for key, value in result.items()} for result in measurements], "mean", {key: round(float(np.mean([r[key] for r in measurements])) * 100, 2) for key in measurements[0]})


if __name__ == "__main__":
    main()

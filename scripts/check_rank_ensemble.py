#!/usr/bin/env python3
"""Compare a weighted rank-fusion ensemble on development labels only."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import retrieval_metrics
from src.retrieval import rank_candidates, weighted_rank_ensemble


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", nargs=3, required=True, type=float); parser.add_argument("--truth-dir", required=True); parser.add_argument("--experiments", default="experiments")
    args = parser.parse_args(); root, truth = Path(args.experiments), Path(args.truth_dir); metrics = []
    for subject in ("01", "02", "03", "04"):
        scores = [np.load(root / f"sub-{subject}_{name}_scores.npy") for name in ("alexnet", "clip", "dinov2s")]
        ranking = rank_candidates(weighted_rank_ensemble(scores, np.array(args.weights)), np.arange(1001, 1103))
        metrics.append(retrieval_metrics(ranking, np.load(truth / f"sub-{subject}_tst_stim.npy")))
    print({"weights": args.weights, "top1": float(np.mean([m["top1"] for m in metrics])), "top5": float(np.mean([m["top5"] for m in metrics]))})


if __name__ == "__main__": main()

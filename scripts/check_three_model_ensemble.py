#!/usr/bin/env python3
"""Evaluate specified AlexNet/CLIP/VideoMAE score weights on development labels only."""
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
    parser.add_argument("--weights", nargs=3, required=True, type=float, metavar=("ALEXNET", "CLIP", "VIDEOMAE"))
    parser.add_argument("--truth-dir", required=True); parser.add_argument("--experiments", default="experiments")
    parser.add_argument("--third-model", default="videomae")
    args = parser.parse_args(); root, truth = Path(args.experiments), Path(args.truth_dir); metrics = []
    for subject in ("01", "02", "03", "04"):
        scores = [np.load(root / f"sub-{subject}_{model}_scores.npy") for model in ("alexnet", "clip", args.third_model)]
        ranking = rank_candidates(weighted_score_ensemble(scores, np.array(args.weights)), np.arange(1001, 1103))
        metrics.append(retrieval_metrics(ranking, np.load(truth / f"sub-{subject}_tst_stim.npy")))
    print({"weights": args.weights, "top1": float(np.mean([m["top1"] for m in metrics])), "top5": float(np.mean([m["top5"] for m in metrics]))})


if __name__ == "__main__": main()

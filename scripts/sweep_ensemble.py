#!/usr/bin/env python3
"""Development-only AlexNet/CLIP ensemble-weight sweep."""
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
    parser.add_argument("--truth-dir", required=True)
    args = parser.parse_args(); root = Path(args.experiments); truth_dir = Path(args.truth_dir)
    ids = np.arange(1001, 1103)
    for clip_weight in np.linspace(0, 1, 11):
        subject_metrics = []
        for subject in ("01", "02", "03", "04"):
            scores = weighted_score_ensemble(
                [np.load(root / f"sub-{subject}_alexnet_scores.npy"), np.load(root / f"sub-{subject}_clip_scores.npy")],
                np.array([1 - clip_weight, clip_weight]),
            )
            subject_metrics.append(retrieval_metrics(rank_candidates(scores, ids), np.load(truth_dir / f"sub-{subject}_tst_stim.npy")))
        print(f"clip_weight={clip_weight:.1f} top1={np.mean([x['top1'] for x in subject_metrics]):.4f} top5={np.mean([x['top5'] for x in subject_metrics]):.4f}")


if __name__ == "__main__": main()

#!/usr/bin/env python3
"""Create a validated weighted score-level ensemble ranking."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_array, validate_candidate_ids
from src.retrieval import rank_candidates, save_submission, weighted_score_ensemble


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", required=True, type=float)
    parser.add_argument("--candidate-ids", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    scores = [load_array(path, name="score matrix") for path in args.scores]
    ids = validate_candidate_ids(load_array(args.candidate_ids, name="candidate IDs"))
    ensemble = weighted_score_ensemble(scores, np.array(args.weights))
    if ensemble.shape != (102, 102): raise ValueError(f"score matrices must be (102, 102), got {ensemble.shape}")
    save_submission(args.output, rank_candidates(ensemble, ids))


if __name__ == "__main__": main()

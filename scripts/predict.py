#!/usr/bin/env python3
"""Generate a valid submission without reordering test EEG recordings."""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_array, load_eeg, validate_candidate_ids
from src.preprocessing import organizer_downsample
from src.retrieval import organizer_eeg_correlation_scores, rank_candidates, save_submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True); parser.add_argument("--test-eeg", required=True)
    parser.add_argument("--candidate-features", required=True, help="102 feature rows aligned to candidate IDs")
    parser.add_argument("--candidate-ids", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--scores-out", default=None, help="Optional path for raw (EEG-row, candidate-column) scores")
    parser.add_argument("--score-start", type=int, default=None, help="Downsampled scoring-window start (default: model metadata)")
    parser.add_argument("--score-stop", type=int, default=None, help="Downsampled scoring-window stop (default: model metadata)")
    args = parser.parse_args(); started = time.perf_counter()
    # Deliberately no label sort: output row i stays input test EEG row i.
    test = organizer_downsample(load_eeg(args.test_eeg, name="test EEG"))
    if len(test) != 102: raise ValueError("submission test EEG must contain exactly 102 recordings")
    candidate_ids = validate_candidate_ids(load_array(args.candidate_ids, name="candidate IDs"))
    candidate_features = load_array(args.candidate_features, name="candidate features")
    if candidate_features.shape[0] != 102: raise ValueError("candidate features must have 102 rows")
    bundle = joblib.load(args.model)
    predicted = bundle["model"].predict(candidate_features)
    start, stop = bundle.get("score_window", [5, 60])
    scores = organizer_eeg_correlation_scores(test, predicted, start=start if args.score_start is None else args.score_start, stop=stop if args.score_stop is None else args.score_stop)
    if args.scores_out: np.save(args.scores_out, scores, allow_pickle=False)
    save_submission(args.output, rank_candidates(scores, candidate_ids))
    print(f"Saved {args.output} in {time.perf_counter() - started:.3f}s; input order preserved.")

if __name__ == "__main__": main()

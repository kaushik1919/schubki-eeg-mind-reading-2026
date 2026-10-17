#!/usr/bin/env python3
"""Development-only time-window sweep for an existing score-model ensemble."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_array, load_eeg
from src.evaluation import retrieval_metrics
from src.preprocessing import organizer_downsample
from src.retrieval import organizer_eeg_correlation_scores, rank_candidates, weighted_score_ensemble


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True); parser.add_argument("--experiments", default="experiments")
    parser.add_argument("--alexnet-features", required=True); parser.add_argument("--clip-features", required=True)
    parser.add_argument("--windows", nargs="+", default=["0:30", "0:45", "5:45", "5:60", "10:80", "0:100", "20:100"])
    args = parser.parse_args(); root, experiment = Path(args.data_root), Path(args.experiments)
    features = [load_array(args.alexnet_features, name="AlexNet candidate features"), load_array(args.clip_features, name="CLIP candidate features")]
    models = {}
    observations, truths = {}, {}
    for subject in ("01", "02", "03", "04"):
        models[subject] = [joblib.load(experiment / f"sub-{subject}_{name}_baseline.joblib")["model"] for name in ("alexnet", "clip")]
        observations[subject] = organizer_downsample(load_eeg(root / "tst" / f"sub-{subject}_tst.npy"))
        truths[subject] = load_array(root / "tst_stim" / f"sub-{subject}_tst_stim.npy", name="development labels")
    for spec in args.windows:
        start, stop = map(int, spec.split(":")); metrics = []
        for subject in models:
            scores = [organizer_eeg_correlation_scores(observations[subject], model.predict(feature), start=start, stop=stop) for model, feature in zip(models[subject], features)]
            ranking = rank_candidates(weighted_score_ensemble(scores, np.array([0.9, 0.1])), np.arange(1001, 1103))
            metrics.append(retrieval_metrics(ranking, truths[subject]))
        print(f"window={spec} top1={np.mean([m['top1'] for m in metrics]):.4f} top5={np.mean([m['top5'] for m in metrics]):.4f}")


if __name__ == "__main__": main()

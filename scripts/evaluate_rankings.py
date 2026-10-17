#!/usr/bin/env python3
"""Score a development-subject ranking against its supplied organizer labels."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import retrieval_metrics
from src.retrieval import validate_rankings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings", required=True); parser.add_argument("--truth", required=True)
    args = parser.parse_args()
    rankings = validate_rankings(np.load(args.rankings, allow_pickle=False))
    truth = np.load(args.truth, allow_pickle=False)
    print(retrieval_metrics(rankings, truth))


if __name__ == "__main__": main()

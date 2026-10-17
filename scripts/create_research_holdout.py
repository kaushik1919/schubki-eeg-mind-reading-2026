#!/usr/bin/env python3
"""Reserve an untouched common training-video holdout for new research."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.controlled_cv import shared_holdout_ids
from src.data import load_training_subject


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--output", required=True); p.add_argument("--seed", type=int, default=2031); p.add_argument("--size", type=int, default=102); p.add_argument("--data-root", default="eeg-mind-reading-challenge"); args = p.parse_args()
    ids = {s: load_training_subject(Path(args.data_root) / "trn" / f"sub-{s}_trn.npz")[0] for s in ("01", "02", "03", "04", "05", "06")}
    common = set.intersection(*(set(x) for x in ids.values()))
    used = set().union(*(set(shared_holdout_ids(ids, seed=s, size=args.size)) for s in (2026, 2027, 2028)))
    for path in (Path("manifests/final_holdout_ids.npy"), Path("manifests/confirmation_holdout_ids.npy")):
        used.update(np.load(path, allow_pickle=False).astype(int).tolist())
    available = np.array(sorted(common - used), dtype=np.int64)
    if len(available) < args.size: raise ValueError(f"only {len(available)} unused shared IDs")
    choice = np.sort(np.random.default_rng(args.seed).choice(available, size=args.size, replace=False))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); np.save(args.output, choice, allow_pickle=False)
    print({"common": len(common), "previously_reserved": len(used), "available": len(available), "holdout": len(choice)})


if __name__ == "__main__": main()

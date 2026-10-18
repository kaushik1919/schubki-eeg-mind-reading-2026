#!/usr/bin/env python3
"""Frozen Model C score matrices on the 6 shared-video outer folds for a given partition seed.

Same estimator as experiments/temporal_audio (ModelC__50_600): frozen ForwardRegressor
constituents fitted on each subject's training recordings outside the fold, 0.45/0.45/0.10 row-z fusion.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_contrastive_eeg import SUBJECTS
from src.controlled_cv import common_ids, disjoint_inner_folds
from src.data import load_training_subject
from src.final_model import FrozenModelC, load_frozen_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--outer-seed", type=int, required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-root", default="eeg-mind-reading-challenge"); a = ap.parse_args(); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = load_frozen_config("configs/frozen_balanced.json"); names = [c["name"] for c in cfg["constituents"]]; w = [c["weight"] for c in cfg["constituents"]]
    feats = {c["name"]: np.load(c["train_features"]) for c in cfg["constituents"]}
    data = {s: load_training_subject(Path(a.data_root) / "trn" / f"sub-{s}_trn.npz") for s in SUBJECTS}
    folds = disjoint_inner_folds(common_ids({s: v[0] for s, v in data.items()}), exclude=np.array([], np.int64), n_folds=6, size=102, seed=a.outer_seed)
    for o, f in enumerate(folds):
        for s, (ids, eeg) in data.items():
            test = np.isin(ids, f); assert np.array_equal(ids[test], f)
            m = FrozenModelC(names, w).fit(ids[~test], eeg[~test], feats)
            cand = {n: x[f - 1] for n, x in feats.items()}
            scores = m.fused_scores(m.constituent_scores(eeg[test], {n: v for n, v in cand.items()}))
            np.savez_compressed(out / f"fold{o}_sub{s}.npz", heldout_ids=f, target_ids=ids[test], scores=scores)
        print("fold", o, flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase 1 diagnostic: is there repeatable, video-specific EEG across subjects?

Training EEG only (organizer `trn/`), restricted to the 695 videos recorded for
all six subjects.  No model is fitted.  Normalisation is per subject and
label-free (channel z-scoring and subject-mean evoked-response removal).
Identification is evaluated within the six 102-video groups used by the
temporal-audio protocol (seed 2040), so chance is 1/102.
"""
from __future__ import annotations

import argparse
import json
from itertools import permutations
from pathlib import Path
import sys

import numpy as np
from scipy.signal import butter, decimate, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.controlled_cv import common_ids, disjoint_inner_folds
from src.data import load_training_subject

SUBJECTS = ["01", "02", "03", "04", "05", "06"]


def rowz(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(1, keepdims=True); return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def prepare(eeg: np.ndarray, *, band: str, remove_erp: bool, window: tuple[int, int], shift: int = 0) -> np.ndarray:
    x = decimate(eeg, 5, axis=-1, ftype="fir", zero_phase=True)                       # 100 Hz
    if band == "1-15Hz":
        x = sosfiltfilt(butter(4, [1, 15], btype="band", fs=100, output="sos"), x, axis=-1)
    x = (x - x.mean(axis=(0, 2), keepdims=True)) / x.std(axis=(0, 2), keepdims=True)  # per-subject channel z-score
    if remove_erp:
        x = x - x.mean(0, keepdims=True)                                                # subject-mean evoked response
    a, b = window[0] // 10 + shift, window[1] // 10 + shift
    a, b = max(a, 0), min(b, x.shape[-1])
    return rowz(x[:, :, a:b].reshape(len(x), -1))


def identification(C: np.ndarray, groups: list[np.ndarray]) -> dict[str, float]:
    """C[q, c]: similarity of query recording q to candidate c, both indexed by shared-video position."""
    pos = []
    for g in groups:
        sub = C[np.ix_(g, g)]
        pos.append((sub > np.diag(sub)[:, None]).sum(1))
    p = np.concatenate(pos)
    return {"top1": float(np.mean(p == 0)), "top5": float(np.mean(p < 5)), "mrr": float(np.mean(1 / (p + 1)))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--data-root", default="eeg-mind-reading-challenge"); ap.add_argument("--out", default="experiments/cross_subject_reliability.json")
    ap.add_argument("--n-perm", type=int, default=200); a = ap.parse_args()
    data = {s: load_training_subject(Path(a.data_root) / "trn" / f"sub-{s}_trn.npz") for s in SUBJECTS}
    shared = common_ids({s: v[0] for s, v in data.items()})
    raw = {s: data[s][1][np.searchsorted(data[s][0], shared)] for s in SUBJECTS}
    assert all(np.array_equal(data[s][0][np.searchsorted(data[s][0], shared)], shared) for s in SUBJECTS)
    folds = disjoint_inner_folds(shared, exclude=np.array([], np.int64), n_folds=6, size=102, seed=2040)
    groups = [np.searchsorted(shared, f) for f in folds]
    rng = np.random.default_rng(7); results = {"n_shared": int(len(shared)), "conditions": {}}
    conditions = [("broadband", True, (50, 600)), ("broadband", False, (50, 600)), ("broadband", True, (0, 3000)), ("1-15Hz", True, (50, 600)), ("1-15Hz", True, (0, 3000)), ("1-15Hz", True, (50, 1000))]
    for band, rm, win in conditions:
        name = f"{band}|erp_removed={rm}|{win[0]}-{win[1]}ms"
        X = {s: prepare(raw[s], band=band, remove_erp=rm, window=win) for s in SUBJECTS}
        pair_id, match, mismatch, null_gap = [], [], [], []
        for i, j in permutations(SUBJECTS, 2):
            C = X[i] @ X[j].T
            pair_id.append(identification(C, groups)); d = np.diag(C)
            match.append(float(d.mean())); mismatch.append(float((C.sum() - d.sum()) / (C.size - len(d))))
        # Permutation null for the pooled match-minus-mismatch gap (shuffle video labels of the candidate subject).
        Cs = {(i, j): X[i] @ X[j].T for i, j in permutations(SUBJECTS, 2)}
        obs_gap = float(np.mean(match) - np.mean(mismatch))
        for _ in range(a.n_perm):
            perm = rng.permutation(len(shared))
            null_gap.append(np.mean([np.mean(C[np.arange(len(shared)), perm]) - (C.sum() - C[np.arange(len(shared)), perm].sum()) / (C.size - len(shared)) for C in Cs.values()]))
        null_gap = np.array(null_gap)
        # Leave-one-subject-out template: mean of the other five subjects' responses to each video.
        loso = {}
        for s in SUBJECTS:
            T = rowz(np.mean([X[o] for o in SUBJECTS if o != s], axis=0))
            loso[s] = identification(X[s] @ T.T, groups)
        # Ordering control: similarity of video v to its numeric-ID neighbours v+k (k = 1..5) vs distant videos.
        nb = {}
        for k in (1, 2, 5, 50):
            vals = [];
            for (i, j), C in Cs.items():
                idx = np.where(np.isin(shared + k, shared))[0]; vals.append(np.mean(C[idx, np.searchsorted(shared, shared[idx] + k)]))
            nb[k] = float(np.mean(vals))
        results["conditions"][name] = {
            "pairwise_identification_mean": {m: float(np.mean([p[m] for p in pair_id])) for m in ("top1", "top5", "mrr")},
            "pairwise_identification_min_max_top1": [float(min(p["top1"] for p in pair_id)), float(max(p["top1"] for p in pair_id))],
            "match_corr": float(np.mean(match)), "mismatch_corr": float(np.mean(mismatch)), "gap": obs_gap,
            "perm_null_gap_mean": float(null_gap.mean()), "perm_null_gap_sd": float(null_gap.std()), "perm_p": float((1 + np.sum(null_gap >= obs_gap)) / (1 + len(null_gap))),
            "gap_z": float((obs_gap - null_gap.mean()) / null_gap.std()),
            "loso_template_identification": loso, "loso_mean": {m: float(np.mean([loso[s][m] for s in SUBJECTS])) for m in ("top1", "top5", "mrr")},
            "id_neighbour_similarity": nb}
        print(name, json.dumps({k: v for k, v in results["conditions"][name].items() if k not in ("loso_template_identification",)}), flush=True)
    # Timing control: shift one subject by +-k samples (10 ms) relative to the other, 1-15 Hz, 150-600 ms.
    shifts = {}
    base = {s: prepare(raw[s], band="1-15Hz", remove_erp=True, window=(150, 600)) for s in SUBJECTS}
    for k in (-10, -5, -2, 0, 2, 5, 10):
        Xk = {s: prepare(raw[s], band="1-15Hz", remove_erp=True, window=(150, 600), shift=k) for s in SUBJECTS}
        shifts[k] = float(np.mean([np.mean(np.sum(base[i] * Xk[j], axis=1)) for i, j in permutations(SUBJECTS, 2)]))
    results["timing_shift_match_corr_1-15Hz_150-600"] = shifts; print("shifts", shifts)
    results["chance"] = {"top1": 1 / 102, "top5": 5 / 102, "mrr": float(np.mean(1 / np.arange(1, 103)))}
    Path(a.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

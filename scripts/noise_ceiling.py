#!/usr/bin/env python3
"""Estimate the retrieval ceiling of a perfect stimulus -> mean-EEG model (training EEG only).

For each subject s and k = 1..5, the template for video v is the mean response
of k other subjects (all subsets).  We measure (i) top-1/top-5 identification of
s's single trial among the 102 videos of each shared group and (ii) the
standardised match separation z_k = (mean match corr - mean mismatch corr) /
sd(mismatch corr).  A Gaussian model acc(z) = int phi(x - z) Phi(x)^(N-1) dx is
checked against the measured accuracies and extrapolated to k -> infinity with
the Spearman-Brown relation r_k = r_1 * sqrt(k / (1 + (k - 1) rho)), where rho is
the between-subject response reliability (= r_1 between two single trials).
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.cross_subject_reliability import SUBJECTS, prepare, rowz
from src.controlled_cv import common_ids, disjoint_inner_folds
from src.data import load_training_subject


def gaussian_acc(z: float, n: int = 102, k: int = 1) -> float:
    x = np.linspace(-8, 8 + z, 4001)
    cdf = norm.cdf(x); dens = norm.pdf(x - z)
    # P(rank < k) = P(at most k-1 of n-1 mismatches exceed x)
    from scipy.stats import binom
    return float(np.trapezoid(dens * binom.cdf(k - 1, n - 1, 1 - cdf), x))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--data-root", default="eeg-mind-reading-challenge"); ap.add_argument("--out", default="experiments/noise_ceiling.json"); a = ap.parse_args()
    data = {s: load_training_subject(Path(a.data_root) / "trn" / f"sub-{s}_trn.npz") for s in SUBJECTS}
    shared = common_ids({s: v[0] for s, v in data.items()})
    groups = [np.searchsorted(shared, f) for f in disjoint_inner_folds(shared, exclude=np.array([], np.int64), n_folds=6, size=102, seed=2040)]
    out = {}
    for band, win in (("broadband", (0, 3000)), ("broadband", (50, 600))):
        X = {s: prepare(data[s][1][np.searchsorted(data[s][0], shared)], band=band, remove_erp=True, window=win) for s in SUBJECTS}
        rows = {}
        for k in range(1, 6):
            acc1, acc5, zs, rm = [], [], [], []
            for s in SUBJECTS:
                for others in combinations([o for o in SUBJECTS if o != s], k):
                    T = rowz(np.mean([X[o] for o in others], axis=0)); C = X[s] @ T.T
                    for g in groups:
                        sub = C[np.ix_(g, g)]; d = np.diag(sub); off = sub[~np.eye(len(g), dtype=bool)]
                        p = (sub > d[:, None]).sum(1); acc1.append(np.mean(p == 0)); acc5.append(np.mean(p < 5))
                        zs.append((d.mean() - off.mean()) / off.std()); rm.append(d.mean())
            z = float(np.mean(zs)); rows[k] = {"top1": float(np.mean(acc1)), "top5": float(np.mean(acc5)), "z": z, "match_r": float(np.mean(rm)),
                                              "gaussian_top1": gaussian_acc(z), "gaussian_top5": gaussian_acc(z, k=5)}
        rho = rows[1]["match_r"]                                   # single-trial between-subject reliability
        r_inf = rows[1]["match_r"] / np.sqrt(rho)                   # Spearman-Brown, k -> infinity
        z_inf = rows[1]["z"] * r_inf / rows[1]["match_r"]           # sd of mismatches ~ constant (checked across k)
        out[f"{band}|{win[0]}-{win[1]}ms"] = {"by_k": rows, "rho_single": rho, "r_inf": float(r_inf), "z_inf": float(z_inf),
                                              "ceiling_top1": gaussian_acc(z_inf), "ceiling_top5": gaussian_acc(z_inf, k=5),
                                              "sb_predicted_r_k": {k: float(rho * np.sqrt(k / (1 + (k - 1) * rho)) / np.sqrt(rho) * np.sqrt(rho)) for k in range(1, 6)}}
        print(json.dumps(out[f"{band}|{win[0]}-{win[1]}ms"], indent=1), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

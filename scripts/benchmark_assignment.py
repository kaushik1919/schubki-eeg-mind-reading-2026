#!/usr/bin/env python3
"""Benchmark label-free global-assignment calibrations on saved held-out score matrices.

Sources are saved fold score matrices (6 outer folds x 6 subjects, 102 x 102,
heldout_ids ascending == target_ids, rows sorted by ID).  Because rows are
sorted by ID in the artifacts, the true permutation is the identity; the
methods never see it.  Rows are shuffled with a fixed permutation before
calibration as a guard against any row-order dependence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.assignment import hungarian_first, iterative_rowcol_z, perturb_and_map, rowz, sinkhorn_log, colz
from src.rank_analysis import subject_bootstrap_ci, summarize_positions, target_positions
from src.retrieval import rank_candidates

SUBJECTS = ["01", "02", "03", "04", "05", "06"]


def load_source(spec: str, o: int, s: str) -> tuple[np.ndarray, np.ndarray]:
    """spec = 'dir' | 'dir@key' | 'a+b+c' (mean of row-z of parts) | 'fuse(a;b)' handled by '+'."""
    parts = spec.split("+"); mats = []; h = None
    for p in parts:
        d, key = (p.split("@") + [None])[:2]
        z = np.load(Path(d) / f"fold{o}_sub{s}.npz"); h0 = z["heldout_ids"]
        assert np.array_equal(h0, z["target_ids"]) or key is not None
        if h is None: h = h0
        assert np.array_equal(h, h0)
        mats.append(rowz(z[key] if key else z["scores"]))
    return h, np.mean(mats, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", nargs="+", required=True, help="name=spec")
    ap.add_argument("--taus", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.5])
    ap.add_argument("--pam", action="store_true"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); perm = np.random.default_rng(99).permutation(102); res = {}
    for item in a.sources:
        name, spec = item.split("=", 1); pos = {}
        methods = {"A_independent": lambda x: x, "colz": lambda x: rowz(colz(x)), "D_iter_rowcol": iterative_rowcol_z}
        for t in a.taus:
            methods[f"C_sinkhorn_t{t}"] = (lambda t: lambda x: sinkhorn_log(x, t))(t)
            methods[f"B_hungarian+sinkhorn_t{t}"] = (lambda t: lambda x: hungarian_first(sinkhorn_log(x, t)))(t)
        methods["B_hungarian+raw"] = lambda x: hungarian_first(x)
        if a.pam:
            for t in a.taus: methods[f"G_pam_t{t}"] = (lambda t: lambda x: perturb_and_map(x, t, 100))(t)
        for o in range(6):
            for s in SUBJECTS:
                h, sc = load_source(spec, o, s)
                sc_p = sc[perm]; targets = h[perm]                   # shuffled recording order
                for m, f in methods.items():
                    p = target_positions(rank_candidates(f(sc_p), h), targets)
                    pos.setdefault(m, {}).setdefault(s, []).append(p)
        res[name] = {}
        for m, by in pos.items():
            by = {s: np.concatenate(v) for s, v in by.items()}; per = {s: summarize_positions(by[s]) for s in SUBJECTS}
            res[name][m] = {k: float(np.mean([per[s][k] for s in SUBJECTS])) for k in ("top1", "top5", "mrr")} | {"per_subject_top1": {s: per[s]["top1"] for s in SUBJECTS}, "_pos": by}
        base = res[name]["A_independent"]
        print(f"\n== {name}")
        for m, r in res[name].items():
            d1 = subject_bootstrap_ci(np.array([summarize_positions(r["_pos"][s])["top1"] - summarize_positions(base["_pos"][s])["top1"] for s in SUBJECTS]))
            r["d_top1_vs_independent"] = d1
            print(f"{m:28s} top1 {r['top1']:.2%} top5 {r['top5']:.2%} mrr {r['mrr']:.4f}  dTop1 {d1[0]:+.2%} [{d1[1]:+.2%},{d1[2]:+.2%}]")
    for n in res:
        for m in res[n]: res[n][m].pop("_pos")
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

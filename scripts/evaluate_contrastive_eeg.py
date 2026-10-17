#!/usr/bin/env python3
"""Paired evaluation of contrastive EEG decoders against frozen Model C on identical folds.

Model C scores come from experiments/temporal_audio (same 6 outer folds, same
training recordings).  Candidate/target alignment is asserted per file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rank_analysis import exact_sign_test, subject_bootstrap_ci, summarize_positions, target_positions, transition_counts
from src.retrieval import rank_candidates, row_zscore

SUBJECTS = ["01", "02", "03", "04", "05", "06"]


def load_scores(d: Path, key: str | None, o: int, s: str):
    z = np.load(d / f"fold{o}_sub{s}.npz")
    return z["heldout_ids"], z["target_ids"], z[key] if key else z["scores"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="experiments/contrastive_eeg"); ap.add_argument("--modelc-dir", default="experiments/temporal_audio")
    ap.add_argument("--variants", nargs="+", default=["A", "B", "C", "D"]); ap.add_argument("--ensembles", nargs="*", default=[], help="name=variant1,variant2,... (mean of row-z scores)")
    ap.add_argument("--fuse-with-modelc", nargs="*", default=[], help="variants fused 0.5/0.5 with Model C (row-z), pre-declared")
    ap.add_argument("--out", default="experiments/contrastive_eeg/summary.json")
    a = ap.parse_args(); d = Path(a.dir)
    sources = {"ModelC": lambda o, s: load_scores(Path(a.modelc_dir), "ModelC__50_600", o, s)}
    for v in a.variants: sources[v] = (lambda v: lambda o, s: load_scores(d / v, None, o, s))(v)
    for spec in a.ensembles:
        name, parts = spec.split("="); parts = parts.split(",")
        def ens(o, s, parts=parts):
            loaded = [sources[p](o, s) for p in parts]; return loaded[0][0], loaded[0][1], np.mean([row_zscore(x[2]) for x in loaded], axis=0)
        sources[name] = ens
    for v in a.fuse_with_modelc:
        def fused(o, s, v=v):
            h, t, x = sources[v](o, s); _, _, c = sources["ModelC"](o, s); return h, t, 0.5 * row_zscore(x) + 0.5 * row_zscore(c)
        sources[f"{v}+ModelC"] = fused
    pos, corr = {}, {}
    for name, fn in sources.items():
        pos[name] = {s: [] for s in SUBJECTS}
        for o in range(6):
            for s in SUBJECTS:
                h, t, x = fn(o, s); hc, tc, c = sources["ModelC"](o, s)
                assert np.array_equal(h, hc) and np.array_equal(t, tc), "unaligned folds"
                pos[name][s].append(target_positions(rank_candidates(x, h), t))
                if name != "ModelC": corr.setdefault(name, []).append(float(np.mean([np.corrcoef(row_zscore(x)[i], row_zscore(c)[i])[0, 1] for i in range(len(x))])))
        pos[name] = {s: np.concatenate(v) for s, v in pos[name].items()}
    res = {}; base = pos["ModelC"]
    for name, p in pos.items():
        per = {s: summarize_positions(p[s]) for s in SUBJECTS}
        r = {m: float(np.mean([per[s][m] for s in SUBJECTS])) for m in ("top1", "top5", "mrr")} | {"per_subject": per}
        if name != "ModelC":
            t = transition_counts(np.concatenate([base[s] for s in SUBJECTS]), np.concatenate([p[s] for s in SUBJECTS]))
            r |= {f"d_{m}": subject_bootstrap_ci(np.array([per[s][m] - res["ModelC"]["per_subject"][s][m] for s in SUBJECTS])) for m in ("top1", "top5", "mrr")}
            r |= {"top1_WL": (t["top1_wins"], t["top1_losses"]), "top1_sign_p": exact_sign_test(t["top1_wins"], t["top1_losses"]),
                  "top5_WL": (t["top5_wins"], t["top5_losses"]), "top5_sign_p": exact_sign_test(t["top5_wins"], t["top5_losses"]), "score_corr_with_ModelC": float(np.mean(corr[name]))}
        res[name] = r
    for v in a.variants:
        logs = [json.loads((d / v / f"fold{o}.json").read_text()) for o in range(6)]
        lg = [l for rec in logs for l in rec["logs"].values()]
        res[v]["compute"] = {"train_s_per_fold": float(np.mean([r["train_s"] for r in logs])), "eval_s_per_fold": float(np.mean([r["eval_s"] for r in logs])),
                             "n_params_per_model": int(lg[0]["n_params"]), "n_models_per_fold": len(logs[0]["logs"]),
                             "best_epoch_mean": float(np.mean([l["best_epoch"] for l in lg])), "inner_val_mrr_mean": float(np.mean([l["inner_val_mrr"] for l in lg])),
                             "train_probe_mrr_mean": float(np.mean([m["train_subset_mrr"] for rec in logs for m in rec["metrics"].values()]))}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("| Model | Top-1 | Top-5 | MRR | dTop-1 vs C [CI] | dTop-5 vs C [CI] | dMRR [CI] | Top-1 W/L (p) | corr w/ C |")
    for n, r in res.items():
        if n == "ModelC": print(f"| {n} | {r['top1']:.2%} | {r['top5']:.2%} | {r['mrr']:.4f} | | | | | |"); continue
        f = lambda k, pct=True: f"{r[k][0]:+.2%} [{r[k][1]:+.2%}, {r[k][2]:+.2%}]" if pct else f"{r[k][0]:+.4f} [{r[k][1]:+.4f}, {r[k][2]:+.4f}]"
        print(f"| {n} | {r['top1']:.2%} | {r['top5']:.2%} | {r['mrr']:.4f} | {f('d_top1')} | {f('d_top5')} | {f('d_mrr', False)} | {r['top1_WL'][0]}/{r['top1_WL'][1]} ({r['top1_sign_p']:.3f}) | {r['score_corr_with_ModelC']:.3f} |")
    print("\nper subject top1/top5/mrr")
    for n, r in res.items():
        print(n, " | ".join(f"{s}: {r['per_subject'][s]['top1']:.1%}/{r['per_subject'][s]['top5']:.1%}/{r['per_subject'][s]['mrr']:.3f}" for s in SUBJECTS))
    for v in a.variants: print(v, res[v]["compute"])


if __name__ == "__main__":
    main()

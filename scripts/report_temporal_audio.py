#!/usr/bin/env python3
"""Summarise experiments/temporal_audio artifacts; applies the pre-declared fusion gate."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rank_analysis import exact_sign_test, subject_bootstrap_ci, summarize_positions, target_positions, transition_counts
from src.retrieval import rank_candidates, row_zscore

SUBJECTS = ["01", "02", "03", "04", "05", "06"]
AUDIO = ["A_envelope_onset", "B_logmel", "C_combined"]
CHANCE_MRR = float(np.mean(1 / np.arange(1, 103)))


def positions(d: Path, key: str) -> dict[str, np.ndarray]:
    out = {}
    for s in SUBJECTS:
        ps = []
        for f in sorted(d.glob(f"fold*_sub{s}.npz")):
            z = np.load(f); ps.append(target_positions(rank_candidates(z[key] if not callable(key) else key(z), z["heldout_ids"]), z["target_ids"]))
        out[s] = np.concatenate(ps)
    return out


def summary(pos: dict[str, np.ndarray]) -> dict:
    per = {s: summarize_positions(pos[s]) for s in SUBJECTS}
    return {"per_subject": per, **{m: float(np.mean([per[s][m] for s in SUBJECTS])) for m in ("top1", "top5", "mrr")}}


def compare(pos: dict, base: dict) -> dict:
    sm, sb = summary(pos), summary(base)
    t = transition_counts(np.concatenate([base[s] for s in SUBJECTS]), np.concatenate([pos[s] for s in SUBJECTS]))
    return {"d_" + m: subject_bootstrap_ci(np.array([sm["per_subject"][s][m] - sb["per_subject"][s][m] for s in SUBJECTS])) for m in ("top1", "top5", "mrr")} | {
        "top1_WL": (t["top1_wins"], t["top1_losses"]), "top5_WL": (t["top5_wins"], t["top5_losses"]), "top5_sign_p": exact_sign_test(t["top5_wins"], t["top5_losses"])}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--dir", default="experiments/temporal_audio"); a = p.parse_args(); d = Path(a.dir)
    rows = list(csv.DictReader((d / "results.csv").open()))
    models = ["ModelC", "final_clip", *AUDIO, "V_visual_grouped"]; windows = ["50_600", "100_1000", "200_1500", "500_2500", "50_3000"]
    pos = {(m, w): positions(d, f"{m}__{w}") for m in models for w in windows}
    res = {"main": {}, "windows": {}, "lambda": {}, "runtime": {}, "inner_mrr": {}}
    base = pos[("ModelC", "50_600")]
    for m in models:
        res["main"][m] = summary(pos[(m, "50_600")]) | ({} if m == "ModelC" else {"vs_ModelC": compare(pos[(m, "50_600")], base)})
        res["main"][m]["mrr_subject_ci"] = subject_bootstrap_ci(np.array([res["main"][m]["per_subject"][s]["mrr"] for s in SUBJECTS]))
        res["windows"][m] = {w: {k: summary(pos[(m, w)])[k] for k in ("top1", "top5", "mrr")} for w in windows}
        r = [x for x in rows if x["experiment"] == m and x["window"] == "50_600"]
        res["runtime"][m] = {"train_s_per_subject_fold": float(np.mean([float(x["train_s"]) for x in r])), "infer_s_per_subject_fold": float(np.mean([float(x["infer_s"]) for x in r]))}
        if r[0]["lam"] not in ("", "None"):
            res["lambda"][m] = dict(sorted(defaultdict(int, {k: sum(1 for x in r if x["lam"] == k) for k in {x["lam"] for x in r}}).items(), key=lambda kv: float(kv[0])))
            res["inner_mrr"][m] = float(np.mean([float(x["inner_mrr"]) for x in r]))
    # ---- pre-declared fusion gate ----
    best = max(AUDIO, key=lambda m: res["inner_mrr"][m]); b = res["main"][best]
    gate = {"best_audio_by_inner_mrr": best, "outer_top5": b["top5"], "mrr_ci_low": b["mrr_subject_ci"][1], "passes": bool(b["top5"] >= 0.0735 and b["mrr_subject_ci"][1] > CHANCE_MRR)}
    if gate["passes"]:
        fused = positions(d, lambda z: 0.9 * row_zscore(z["ModelC__50_600"]) + 0.1 * row_zscore(z[f"{best}__50_600"]))
        res["fusion"] = summary(fused) | {"vs_ModelC": compare(fused, base)}
        corr = [float(np.corrcoef(row_zscore(np.load(f)["ModelC__50_600"])[i], row_zscore(np.load(f)[f"{best}__50_600"])[i])[0, 1]) for f in sorted(d.glob("fold*_sub*.npz")) for i in range(102)]
        res["fusion"]["score_corr_with_ModelC"] = float(np.mean(corr))
    res["fusion_gate"] = gate; res["chance_mrr"] = CHANCE_MRR
    (d / "summary.json").write_text(json.dumps(res, indent=1))
    L = ["| Model | Top-1 | Top-5 | MRR [subj. 95% CI] | dTop-1 vs C | dTop-5 vs C | dMRR vs C | Top-5 W/L vs C |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in models:
        r = res["main"][m]; ci = r["mrr_subject_ci"]
        if m == "ModelC": L.append(f"| {m} | {r['top1']:.2%} | {r['top5']:.2%} | {r['mrr']:.4f} [{ci[1]:.4f}, {ci[2]:.4f}] | - | - | - | - |"); continue
        c = r["vs_ModelC"]; f = lambda k, pct=True: (f"{c[k][0]:+.2%} [{c[k][1]:+.2%}, {c[k][2]:+.2%}]" if pct else f"{c[k][0]:+.4f} [{c[k][1]:+.4f}, {c[k][2]:+.4f}]")
        L.append(f"| {m} | {r['top1']:.2%} | {r['top5']:.2%} | {r['mrr']:.4f} [{ci[1]:.4f}, {ci[2]:.4f}] | {f('d_top1')} | {f('d_top5')} | {f('d_mrr', False)} | {c['top5_WL'][0]}/{c['top5_WL'][1]} |")
    L += ["", "| Model | " + " | ".join(f"sub-{s}" for s in SUBJECTS) + " |", "|---|" + "---:|" * 6]
    for m in models:
        L.append(f"| {m} | " + " | ".join(f"{res['main'][m]['per_subject'][s]['top1']:.1%} / {res['main'][m]['per_subject'][s]['top5']:.1%} / {res['main'][m]['per_subject'][s]['mrr']:.3f}" for s in SUBJECTS) + " |")
    L += ["", "| Model | " + " | ".join(windows) + " |", "|---|" + "---:|" * len(windows)]
    for m in models:
        L.append(f"| {m} | " + " | ".join(f"{res['windows'][m][w]['top1']:.2%} / {res['windows'][m][w]['top5']:.2%} / {res['windows'][m][w]['mrr']:.4f}" for w in windows) + " |")
    L += ["", f"lambda counts: {res['lambda']}", f"inner MRR: {res['inner_mrr']}", f"runtime: {res['runtime']}", f"fusion gate: {gate}", f"fusion: {res.get('fusion', 'not evaluated (gate failed)') if not gate['passes'] else {k: v for k, v in res['fusion'].items() if k != 'per_subject'}}"]
    (d / "summary_tables.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Rank-error analysis of saved, aligned train-only CV artifacts.

Reads only `experiments/paired_predictions/*.npz` (train-only validation
artifacts).  No model is fitted, no organizer test EEG/labels are read, and
derived fusions use fixed, predeclared weights with label-free row z-scoring.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rank_analysis import exact_sign_test, subject_bootstrap_ci, summarize_positions, target_positions, top_margin, transition_counts
from src.retrieval import rank_candidates, row_zscore, weighted_score_ensemble

SUBJECTS = ["01", "02", "03", "04", "05", "06"]
# name -> (constituent labels, weights); single-label entries are used as saved.
STUDIES = {
    "confirmation": {"folds": ["confirmation"], "baseline": "final", "configs": {
        "final": (["confirm_clip_final"], [1.0]), "block8": (["confirm_clip_block8"], [1.0]),
        "final+block8": (["confirm_clip_final", "confirm_clip_block8"], [.5, .5]),
        "block8+audio": (["confirm_clip_block8", "confirm_audio_simple"], [.9, .1]),
        "final+block8+audio": (["confirm_clip_final", "confirm_clip_block8", "confirm_audio_simple"], [.45, .45, .1])}},
    "screening": {"folds": ["2026", "2027", "2028"], "baseline": "final", "configs": {
        "final": (["clip_ridge_organizer_window"], [1.0]), "block8": (["clip_block8_ridge"], [1.0]),
        "final+block8": (["clip_ridge_organizer_window", "clip_block8_ridge"], [.5, .5]),
        "block8+audio": (["clip_block8_ridge", "audio_simple_ridge"], [.9, .1]),
        "final+block8+audio": (["clip_ridge_organizer_window", "clip_block8_ridge", "audio_simple_ridge"], [.45, .45, .1])}},
}


def load_fold(directory: Path, labels: list[str], weights: list[float], fold: str, subject: str) -> dict[str, np.ndarray]:
    arrays = [np.load(directory / f"{label}_seed{fold}_sub{subject}.npz") for label in labels]
    ref = arrays[0]
    for other in arrays[1:]:
        if not (np.array_equal(ref["heldout_ids"], other["heldout_ids"]) and np.array_equal(ref["target_ids"], other["target_ids"])):
            raise ValueError(f"unaligned artifacts for {labels} fold {fold} sub {subject}")
    heldout, targets = ref["heldout_ids"], ref["target_ids"]
    if not (np.all(np.diff(heldout) > 0) and np.array_equal(np.sort(targets), heldout)):
        raise ValueError("heldout_ids must be ascending and target_ids a permutation of them")
    scores = ref["scores"] if len(arrays) == 1 else weighted_score_ensemble([a["scores"] for a in arrays], np.array(weights))
    rankings = rank_candidates(scores, heldout)
    if len(arrays) == 1 and not np.array_equal(rankings, ref["rankings"]):
        raise ValueError("saved rankings disagree with saved scores")
    return {"heldout": heldout, "targets": targets, "scores": scores, "pos": target_positions(rankings, targets)}


def analyze(directory: Path, study: dict) -> dict:
    data = {name: {(f, s): load_fold(directory, *spec, f, s) for f in study["folds"] for s in SUBJECTS} for name, spec in study["configs"].items()}
    base = data[study["baseline"]]
    out: dict = {"configs": {}}
    for name, cells in data.items():
        pos_all = np.concatenate([cells[k]["pos"] for k in sorted(cells)])
        base_all = np.concatenate([base[k]["pos"] for k in sorted(cells)])
        per_subject = {s: summarize_positions(np.concatenate([cells[(f, s)]["pos"] for f in study["folds"]])) for s in SUBJECTS}
        base_subject = {s: summarize_positions(np.concatenate([base[(f, s)]["pos"] for f in study["folds"]])) for s in SUBJECTS}
        per_fold = {f: summarize_positions(np.concatenate([cells[(f, s)]["pos"] for s in SUBJECTS])) for f in study["folds"]}
        trans = transition_counts(base_all, pos_all)
        deltas = {m: subject_bootstrap_ci(np.array([per_subject[s][m] - base_subject[s][m] for s in SUBJECTS])) for m in ("top1", "top5", "mrr")}
        hist = {"1": int(np.sum(pos_all == 0)), "2-5": int(np.sum((pos_all >= 1) & (pos_all < 5))), "6-10": int(np.sum((pos_all >= 5) & (pos_all < 10))),
                "11-25": int(np.sum((pos_all >= 10) & (pos_all < 25))), "26-102": int(np.sum(pos_all >= 25))}
        out["configs"][name] = {"overall": summarize_positions(pos_all), "transitions_vs_baseline": trans,
                                "top1_sign_test_p": exact_sign_test(trans["top1_wins"], trans["top1_losses"]),
                                "top5_sign_test_p": exact_sign_test(trans["top5_wins"], trans["top5_losses"]),
                                "subject_bootstrap_delta": deltas, "per_subject": per_subject, "per_fold": per_fold, "rank_histogram": hist,
                                "per_subject_transitions": {s: transition_counts(np.concatenate([base[(f, s)]["pos"] for f in study["folds"]]),
                                                                                 np.concatenate([cells[(f, s)]["pos"] for f in study["folds"]])) for s in SUBJECTS},
                                "per_fold_transitions": {f: transition_counts(np.concatenate([base[(f, s)]["pos"] for s in SUBJECTS]),
                                                                              np.concatenate([cells[(f, s)]["pos"] for s in SUBJECTS])) for f in study["folds"]}}
    # Margin and agreement diagnostics: final vs block8 (both row-z-scored, label-free).
    margin = {"base_correct_margin": [], "base_wrong_margin": []}
    fused_name = "final+block8"; fb = data[fused_name]; b8 = data["block8"]
    gaps_promoted, gaps_not_promoted, b8_pos_on_demote, b8_pos_on_promote, corr = [], [], [], [], []
    for key in sorted(base):
        z = row_zscore(base[key]["scores"]); m = top_margin(z); bp = base[key]["pos"]; fp = fb[key]["pos"]
        margin["base_correct_margin"] += m[bp == 0].tolist(); margin["base_wrong_margin"] += m[bp != 0].tolist()
        rows = np.arange(len(bp)); target_col = np.searchsorted(base[key]["heldout"], base[key]["targets"])
        gap = z.max(axis=1) - z[rows, target_col]   # how far the target trails the base winner
        sel = (bp >= 1) & (bp < 5)
        gaps_promoted += gap[sel & (fp == 0)].tolist(); gaps_not_promoted += gap[sel & (fp != 0)].tolist()
        b8_pos_on_demote += b8[key]["pos"][(bp < 5) & (fp >= 5)].tolist(); b8_pos_on_promote += b8[key]["pos"][(bp >= 1) & (bp < 5) & (fp == 0)].tolist()
        zb = row_zscore(b8[key]["scores"]); corr += [float(np.corrcoef(z[i], zb[i])[0, 1]) for i in range(len(z))]
    agree = []
    for key in sorted(base):
        a = np.argsort(-base[key]["scores"], axis=1, kind="stable")[:, :5]; c = np.argsort(-b8[key]["scores"], axis=1, kind="stable")[:, :5]
        agree += [len(set(a[i]) & set(c[i])) for i in range(len(a))]
    out["diagnostics"] = {
        "final_top_margin_when_correct_mean": float(np.mean(margin["base_correct_margin"])) if margin["base_correct_margin"] else None,
        "final_top_margin_when_wrong_mean": float(np.mean(margin["base_wrong_margin"])),
        "n_target_rank2to5_in_final": len(gaps_promoted) + len(gaps_not_promoted),
        "z_gap_to_winner_promoted_by_fusion_mean": float(np.mean(gaps_promoted)) if gaps_promoted else None,
        "z_gap_to_winner_not_promoted_mean": float(np.mean(gaps_not_promoted)) if gaps_not_promoted else None,
        "block8_target_rank_when_fusion_demotes_out_of_top5": [int(x) + 1 for x in b8_pos_on_demote],
        "block8_target_rank_when_fusion_promotes_2to5_to_1": [int(x) + 1 for x in b8_pos_on_promote],
        "final_vs_block8_rowwise_score_correlation_mean": float(np.mean(corr)), "final_vs_block8_rowwise_score_correlation_sd": float(np.std(corr)),
        "final_vs_block8_top5_overlap_mean": float(np.mean(agree)),
    }
    return out


def table(study_name: str, res: dict, folds: list[str]) -> list[str]:
    L = [f"## {study_name.capitalize()} ({len(folds)} fold(s) x 6 subjects x 102 trials)", "",
         "| Config | Top-1 | Top-5 | MRR | Median rank | dTop-1 [95% subj. CI] | dTop-5 [95% subj. CI] | dMRR [95% subj. CI] |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in res["configs"].items():
        o, d = r["overall"], r["subject_bootstrap_delta"]
        L.append(f"| {name} | {o['top1']:.2%} | {o['top5']:.2%} | {o['mrr']:.4f} | {o['median_rank']:.0f} | {d['top1'][0]:+.2%} [{d['top1'][1]:+.2%}, {d['top1'][2]:+.2%}] | "
                 f"{d['top5'][0]:+.2%} [{d['top5'][1]:+.2%}, {d['top5'][2]:+.2%}] | {d['mrr'][0]:+.4f} [{d['mrr'][1]:+.4f}, {d['mrr'][2]:+.4f}] |")
    L += ["", "Paired transitions vs final CLIP (pooled trials; sign-test p treats trials as independent and is optimistic):", "",
          "| Config | Top-1 W/L (p) | Top-5 W/L (p) | 2-5 -> 1 | >5 -> 1 | Top-5 -> out | 1 -> 2-5 | Rank better/worse/equal |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in res["configs"].items():
        if name == "final": continue
        t = r["transitions_vs_baseline"]
        L.append(f"| {name} | {t['top1_wins']}/{t['top1_losses']} ({r['top1_sign_test_p']:.2f}) | {t['top5_wins']}/{t['top5_losses']} ({r['top5_sign_test_p']:.2f}) | {t['promoted_2to5_to_1']} | "
                 f"{t['promoted_outside5_to_1']} | {t['demoted_top5_to_outside']} | {t['demoted_1_to_2to5']} | {t['rank_improved']}/{t['rank_worsened']}/{t['rank_equal']} |")
    L += ["", "Per-subject top-1 / top-5 / MRR:", "", "| Config | " + " | ".join(f"sub-{s}" for s in SUBJECTS) + " |", "|---|" + "---:|" * 6]
    for name, r in res["configs"].items():
        L.append(f"| {name} | " + " | ".join(f"{r['per_subject'][s]['top1']:.1%} / {r['per_subject'][s]['top5']:.1%} / {r['per_subject'][s]['mrr']:.3f}" for s in SUBJECTS) + " |")
    L += ["", "Per-subject top-1 W/L and top-5 W/L vs final:", "", "| Config | " + " | ".join(f"sub-{s}" for s in SUBJECTS) + " |", "|---|" + "---:|" * 6]
    for name, r in res["configs"].items():
        if name == "final": continue
        L.append(f"| {name} | " + " | ".join(f"{t['top1_wins']}/{t['top1_losses']} ; {t['top5_wins']}/{t['top5_losses']}" for t in (r["per_subject_transitions"][s] for s in SUBJECTS)) + " |")
    if len(folds) > 1:
        L += ["", "Per-fold top-1 / top-5 (folds overlap in video IDs; not independent):", "", "| Config | " + " | ".join(f"seed {f}" for f in folds) + " |", "|---|" + "---:|" * len(folds)]
        for name, r in res["configs"].items():
            L.append(f"| {name} | " + " | ".join(f"{r['per_fold'][f]['top1']:.2%} / {r['per_fold'][f]['top5']:.2%}" for f in folds) + " |")
    L += ["", "Rank histogram of the correct video (counts):", "", "| Config | 1 | 2-5 | 6-10 | 11-25 | 26-102 |", "|---|---:|---:|---:|---:|---:|"]
    for name, r in res["configs"].items():
        h = r["rank_histogram"]; L.append(f"| {name} | {h['1']} | {h['2-5']} | {h['6-10']} | {h['11-25']} | {h['26-102']} |")
    d = res["diagnostics"]
    L += ["", "Diagnostics (row-z-scored units):", "",
          f"- Final-CLIP top-1 vs top-2 margin: {d['final_top_margin_when_correct_mean']:.3f} when correct, {d['final_top_margin_when_wrong_mean']:.3f} when wrong.",
          f"- Trials with target at final rank 2-5: {d['n_target_rank2to5_in_final']}; mean z-gap to the final winner: {d['z_gap_to_winner_promoted_by_fusion_mean']} when final+block8 promotes to 1, {d['z_gap_to_winner_not_promoted_mean']} otherwise.",
          f"- Block-8 rank of the target when final+block8 pushes a final top-5 hit out: {d['block8_target_rank_when_fusion_demotes_out_of_top5']}.",
          f"- Block-8 rank of the target when final+block8 promotes 2-5 -> 1: {d['block8_target_rank_when_fusion_promotes_2to5_to_1']}.",
          f"- Row-wise correlation of final vs block-8 candidate scores: {d['final_vs_block8_rowwise_score_correlation_mean']:.3f} (sd {d['final_vs_block8_rowwise_score_correlation_sd']:.3f}); mean top-5 overlap {d['final_vs_block8_top5_overlap_mean']:.2f}/5.", ""]
    return L


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts-dir", default="experiments/paired_predictions")
    p.add_argument("--json-out", default="experiments/rank_error_analysis.json")
    p.add_argument("--tables-out", default="experiments/rank_error_analysis_tables.md")
    a = p.parse_args(); directory = Path(a.artifacts_dir); results = {}; lines = ["# Rank-error tables (generated by scripts/rank_error_analysis.py)", ""]
    for name, study in STUDIES.items():
        results[name] = analyze(directory, study); lines += table(name, results[name], study["folds"])
    for path in (a.json_out, a.tables_out):
        if Path(path).exists(): raise FileExistsError(f"refusing to overwrite {path}")
    Path(a.json_out).write_text(json.dumps(results, indent=1)); Path(a.tables_out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

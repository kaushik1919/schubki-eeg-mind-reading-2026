#!/usr/bin/env python3
"""Nested, train-only comparison of A/B/C and conservative top-K reranking (D_K).

See experiments/nested_rerank_protocol.md.  Phase `inner` fits on the
subject training data minus the outer holdout H2, evaluates on disjoint inner
folds, applies the frozen selection rules and writes a selection record.  Phase
`outer` refuses to run without that record, then fits on training minus H2 and
reports H2 metrics.  Organizer test EEG/labels are never read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.clip_controlled_cv import ForwardRegressor, representation
from src.controlled_cv import common_ids, disjoint_inner_folds, millisecond_slice
from src.data import load_training_subject
from src.rank_analysis import exact_sign_test, subject_bootstrap_ci, summarize_positions, target_positions, transition_counts
from src.reranking import fused_rerank_scores, topk_rerank_rankings
from src.retrieval import organizer_eeg_correlation_scores, rank_candidates

SUBJECTS = ["01", "02", "03", "04", "05", "06"]
FEATURES = {"final": "features/all_train_clip_features.npy", "block8": "features/all_train_clip_intermediate_block8.npy", "audio": "features/all_train_audio_simple_features.npy"}
K_VALUES = (3, 5)
INNER_SEED, N_INNER, FOLD_SIZE = 2032, 5, 102
TARGET_WINDOW, SCORE_WINDOW = (0, 3000), (50, 600)


def sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def constituent_scores(features: dict[str, np.ndarray], ids: np.ndarray, eeg: np.ndarray, heldout: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    """Fit each constituent on `ids` not in `heldout`; return aligned 102x102 scores."""
    mask = np.isin(ids, heldout)
    if mask.sum() != len(heldout):
        raise RuntimeError("a held-out ID is missing for this subject")
    started = time.perf_counter()
    train_eeg, observed = representation(eeg[~mask], TARGET_WINDOW), representation(eeg[mask], TARGET_WINDOW)
    sl = millisecond_slice(representation_start_ms=TARGET_WINDOW[0], representation_stop_ms=TARGET_WINDOW[1], sampling_hz=100, window_ms=SCORE_WINDOW)
    scores = {}
    for name, feat in features.items():
        model = ForwardRegressor("ridge", 64).fit(feat[ids[~mask] - 1], train_eeg)
        scores[name] = organizer_eeg_correlation_scores(observed, model.predict(feat[heldout - 1]), start=sl.start, stop=sl.stop)
    return scores, ids[mask], time.perf_counter() - started


def config_rankings(s: dict[str, np.ndarray], heldout: np.ndarray) -> dict[str, np.ndarray]:
    b = fused_rerank_scores([s["final"], s["block8"]], [.5, .5])
    c = fused_rerank_scores([s["final"], s["block8"], s["audio"]], [.45, .45, .10])
    out = {"A_final": rank_candidates(s["final"], heldout), "B_final_block8": rank_candidates(b, heldout), "C_final_block8_audio": rank_candidates(c, heldout)}
    for k in K_VALUES:
        out[f"D{k}_rerank_top{k}"] = topk_rerank_rankings(s["final"], b, heldout, k)
    return out


def subject_means(pos: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, float]]:
    """pos[config][subject] -> pooled positions; returns per-config macro means over subjects."""
    res = {}
    for cfg, by_s in pos.items():
        per = {s: summarize_positions(by_s[s]) for s in SUBJECTS}
        res[cfg] = {"top1": float(np.mean([per[s]["top1"] for s in SUBJECTS])), "top5": float(np.mean([per[s]["top5"] for s in SUBJECTS])),
                    "mrr": float(np.mean([per[s]["mrr"] for s in SUBJECTS])), "per_subject": per}
        res[cfg]["balanced"] = (res[cfg]["top1"] + res[cfg]["top5"]) / 2
        res[cfg]["chance_normalised"] = (res[cfg]["top1"] * 102 + res[cfg]["top5"] * 102 / 5) / 2
    return res


def select(m: dict[str, dict]) -> dict[str, str]:
    d = {k: m[f"D{k}_rerank_top{k}"]["top1"] for k in K_VALUES}
    k_star = max(K_VALUES, key=lambda k: (d[k], -k))  # tie -> smaller K
    pool = ["A_final", "B_final_block8", "C_final_block8_audio", f"D{k_star}_rerank_top{k_star}"]
    order = {c: i for i, c in enumerate(pool)}
    best = lambda key: max(pool, key=lambda c: (round(key(m[c]), 12), -order[c]))
    return {"K_star": k_star, "best_top1": best(lambda r: r["top1"] + 1e-6 * r["top5"]), "best_top5": best(lambda r: r["top5"] + 1e-6 * r["top1"]),
            "balanced": best(lambda r: r["balanced"]), "candidates": pool}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["inner", "outer"], required=True)
    p.add_argument("--data-root", default="eeg-mind-reading-challenge")
    p.add_argument("--outer-holdout", default="manifests/final_holdout_ids.npy")
    p.add_argument("--out-dir", default="experiments/nested_rerank")
    a = p.parse_args(); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    result_path, selection_path = out / f"{a.phase}_results.json", out / "selection.json"
    if result_path.exists(): raise FileExistsError(f"refusing to overwrite {result_path}")
    if a.phase == "outer" and not selection_path.exists(): raise RuntimeError("run --phase inner first; selection must precede the outer evaluation")
    t0 = time.perf_counter(); root = Path(a.data_root)
    subjects = {s: load_training_subject(root / "trn" / f"sub-{s}_trn.npz") for s in SUBJECTS}
    features = {k: np.load(v) for k, v in FEATURES.items()}
    if any(f.shape[0] != 1000 for f in features.values()): raise ValueError("features must align with IDs 1..1000")
    outer = np.sort(np.load(a.outer_holdout, allow_pickle=False).astype(np.int64)); pool = common_ids({s: v[0] for s, v in subjects.items()})
    if outer.shape != (102,) or not set(outer).issubset(pool): raise ValueError("outer holdout must be 102 shared IDs")
    if a.phase == "inner":
        folds = disjoint_inner_folds(pool, exclude=outer, n_folds=N_INNER, size=FOLD_SIZE, seed=INNER_SEED)
        assert all(not set(f) & set(outer) for f in folds)
        evaluations = [(f"inner{i}", fold, outer) for i, fold in enumerate(folds)]  # (name, eval IDs, extra IDs removed from training)
    else:
        evaluations = [("h2", outer, np.array([], dtype=np.int64))]
    positions: dict[str, dict[str, list]] = {}; runtimes = []; artifacts = []
    for name, heldout, also_excluded in evaluations:
        for s, (ids, eeg) in subjects.items():
            keep = ~np.isin(ids, also_excluded)  # inner: drop H2 from training as well
            scores, targets, secs = constituent_scores(features, ids[keep], eeg[keep], heldout); runtimes.append(secs)
            assert not set(ids[keep][~np.isin(ids[keep], heldout)]) & set(outer), "outer holdout leaked into a fit"
            ranks = config_rankings(scores, heldout)
            for cfg, r in ranks.items():
                positions.setdefault(cfg, {}).setdefault(s, []).append(target_positions(r, targets))
            path = out / f"{name}_sub{s}.npz"
            if path.exists(): raise FileExistsError(path)
            np.savez_compressed(path, heldout_ids=heldout, target_ids=targets, **{f"scores_{k}": v for k, v in scores.items()}, **{f"rankings_{k}": v for k, v in ranks.items()}); artifacts.append(str(path))
            print(name, s, {c: int((r[:, 0] == targets).sum()) for c, r in ranks.items()}, f"{secs:.2f}s", flush=True)
    pos = {c: {s: np.concatenate(v) for s, v in by_s.items()} for c, by_s in positions.items()}
    metrics = subject_means(pos)
    record = {"phase": a.phase, "seeds": {"outer_holdout": 2031, "inner_folds": INNER_SEED, "pca_random_state": 2026}, "k_values": list(K_VALUES),
              "n_inner_folds": N_INNER if a.phase == "inner" else 0, "outer_holdout_sha256": sha256(a.outer_holdout),
              "features": {k: {"path": v, "sha256": sha256(v)} for k, v in FEATURES.items()}, "target_window_ms": TARGET_WINDOW, "score_window_ms": SCORE_WINDOW,
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
              "mean_fit_seconds_per_subject_fold_all_constituents": float(np.mean(runtimes)), "total_seconds": time.perf_counter() - t0,
              "metrics": metrics, "artifacts": artifacts}
    if a.phase == "inner":
        record["selection"] = select(metrics)
        if selection_path.exists(): raise FileExistsError(selection_path)
        selection_path.write_text(json.dumps(record["selection"], indent=1))
    else:
        record["selection"] = json.loads(selection_path.read_text()); base = pos["A_final"]
        record["paired_vs_A"] = {}
        for cfg in pos:
            if cfg == "A_final": continue
            t = transition_counts(np.concatenate([base[s] for s in SUBJECTS]), np.concatenate([pos[cfg][s] for s in SUBJECTS]))
            record["paired_vs_A"][cfg] = {"transitions": t, "top1_sign_p": exact_sign_test(t["top1_wins"], t["top1_losses"]), "top5_sign_p": exact_sign_test(t["top5_wins"], t["top5_losses"]),
                                          "per_subject_transitions": {s: transition_counts(base[s], pos[cfg][s]) for s in SUBJECTS},
                                          **{f"delta_{m}_subject_ci": subject_bootstrap_ci(np.array([metrics[cfg]["per_subject"][s][m] - metrics["A_final"]["per_subject"][s][m] for s in SUBJECTS])) for m in ("top1", "top5", "mrr")}}
    result_path.write_text(json.dumps(record, indent=1))
    print(json.dumps({c: {k: round(v, 4) for k, v in r.items() if k != "per_subject"} for c, r in metrics.items()}, indent=1))
    print("selection:", record["selection"])


if __name__ == "__main__":
    main()

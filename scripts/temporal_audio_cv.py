#!/usr/bin/env python3
"""Exploratory nested train-only CV: video-grouped temporal audio TRFs vs frozen Model C.

Protocol: experiments/temporal_audio_protocol.md.  Reads only organizer
training archives (`trn/`).  No test EEG or test labels are read.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.clip_controlled_cv import ForwardRegressor, representation
from src.controlled_cv import common_ids, disjoint_inner_folds, millisecond_slice
from src.data import load_training_subject
from src.grouped_trf import Stats, accumulate, columns_for, empty_like, fit
from src.rank_analysis import summarize_positions, target_positions
from src.retrieval import organizer_eeg_correlation_scores, rank_candidates, weighted_score_ensemble
from src.temporal import resample_trajectory
from src.temporal_audio import GROUPS

SUBJECTS = ["01", "02", "03", "04", "05", "06"]
OUTER_SEED, INNER_SEED, N_OUTER, FOLD, N_INNER = 2040, 2041, 6, 102, 5
N_LAGS = 61                                   # 0..600 ms at 100 Hz
LAMBDAS = np.logspace(-4, 2, 13)
WINDOWS = [(50, 600), (100, 1000), (200, 1500), (500, 2500), (50, 3000)]
SELECT_WINDOW = (50, 600)
MODEL_C = {"final_clip": ("features/all_train_clip_features.npy", .45), "clip_block8": ("features/all_train_clip_intermediate_block8.npy", .45), "audio_simple": ("features/all_train_audio_simple_features.npy", .10)}


def wslice(window):
    return millisecond_slice(representation_start_ms=0, representation_stop_ms=3000, sampling_hz=100, window_ms=window)


def scores_for(observed: np.ndarray, predicted: np.ndarray, window) -> np.ndarray:
    s = wslice(window); return organizer_eeg_correlation_scores(observed, predicted, start=s.start, stop=s.stop)


def mrr_of(scores: np.ndarray, ids: np.ndarray) -> float:
    return summarize_positions(target_positions(rank_candidates(scores, ids), ids))["mrr"]


def predict_chunked(model, stim, which=None, chunk=24):
    return np.concatenate([model.predict(stim[i:i + chunk], which) for i in range(0, len(stim), chunk)], axis=-3)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="eeg-mind-reading-challenge")
    p.add_argument("--audio", default="features/all_train_temporal_audio.npz")
    p.add_argument("--visual", default="features/all_train_temporal_visual.npy")
    p.add_argument("--subjects", nargs="+", default=SUBJECTS)
    p.add_argument("--out-dir", default="experiments/temporal_audio")
    a = p.parse_args(); out = Path(a.out_dir)
    if (out / "results.csv").exists(): raise FileExistsError(out / "results.csv")
    out.mkdir(parents=True, exist_ok=True)
    audio_npz = np.load(a.audio); audio = audio_npz["features"]
    if not np.array_equal(audio_npz["ids"], np.arange(1, 1001)): raise ValueError("audio cache must be ID-aligned 1..1000")
    visual = np.stack([resample_trajectory(v, 300) for v in np.load(a.visual)])
    cfeat = {k: np.load(v[0]) for k, v in MODEL_C.items()}
    subjects = {s: load_training_subject(Path(a.data_root) / "trn" / f"sub-{s}_trn.npz") for s in a.subjects}
    pool = common_ids({s: v[0] for s, v in subjects.items()}) if len(a.subjects) == 6 else None
    if pool is None: raise ValueError("all six subjects are required to define shared folds")
    folds = disjoint_inner_folds(pool, exclude=np.array([], dtype=np.int64), n_folds=N_OUTER, size=FOLD, seed=OUTER_SEED)
    block = np.random.default_rng(INNER_SEED).permutation(1000) % N_INNER   # block[id - 1]
    experiments = {**{k: ("audio", v) for k, v in GROUPS.items()}, "V_visual_grouped": ("visual", list(range(5)))}
    rows = []; t_all = time.perf_counter()
    for s, (ids, raw) in subjects.items():
        eeg = representation(raw, (0, 3000))                       # (n, 17, 300), organizer decimation
        Y = eeg.transpose(0, 2, 1)                                  # (n, 300, 17)
        outer_of = np.full(len(ids), N_OUTER)
        for o, f in enumerate(folds): outer_of[np.isin(ids, f)] = o
        inner_of = block[ids - 1]
        t0 = time.perf_counter()
        atoms = {fam: {(o, i): accumulate(stim[ids[(outer_of == o) & (inner_of == i)] - 1], Y[(outer_of == o) & (inner_of == i)], N_LAGS)
                       for o in range(N_OUTER + 1) for i in range(N_INNER)} for fam, stim in (("audio", audio), ("visual", visual))}
        stats_s = time.perf_counter() - t0; print(f"sub-{s}: sufficient statistics in {stats_s:.1f}s", flush=True)
        stim_of = {"audio": audio, "visual": visual}
        for o, heldout in enumerate(folds):
            test_mask = outer_of == o; train_mask = ~test_mask
            assert np.array_equal(ids[test_mask], heldout) and not set(ids[train_mask]) & set(heldout)
            observed = eeg[test_mask]; record = {"heldout_ids": heldout, "target_ids": ids[test_mask]}
            # ---- frozen Model C on identical training recordings ----
            t0 = time.perf_counter(); preds = {}
            for name, (path, _) in MODEL_C.items():
                preds[name] = ForwardRegressor("ridge", 64).fit(cfeat[name][ids[train_mask] - 1], eeg[train_mask]).predict(cfeat[name][heldout - 1])
            c_train = time.perf_counter() - t0; t0 = time.perf_counter()
            for w in WINDOWS:
                parts = [scores_for(observed, preds[n], w) for n in MODEL_C]
                record[f"ModelC__{w[0]}_{w[1]}"] = weighted_score_ensemble(parts, np.array([v[1] for v in MODEL_C.values()]))
                record[f"final_clip__{w[0]}_{w[1]}"] = parts[0]
            c_inf = time.perf_counter() - t0
            rows += [dict(experiment=k.split("__")[0], window=k.split("__")[1], fold=o, subject=s, lam=None, inner_mrr=None, train_s=c_train, infer_s=c_inf / len(WINDOWS),
                          **{m: v for m, v in summarize_positions(target_positions(rank_candidates(record[k], heldout), ids[test_mask])).items() if m in ("top1", "top5", "mrr")}) for k in list(record) if "__" in k]
            # ---- grouped TRFs ----
            for exp, (fam, feats) in experiments.items():
                t0 = time.perf_counter(); cols = columns_for(feats, N_LAGS); A = atoms[fam]
                train_total = sum((A[(oo, i)] for oo in range(N_OUTER + 1) if oo != o for i in range(N_INNER)), empty_like(A[(0, 0)]))
                inner_mrr = np.zeros(len(LAMBDAS))
                for i in range(N_INNER):
                    val_stats = sum((A[(oo, i)] for oo in range(N_OUTER + 1) if oo != o), empty_like(A[(0, 0)]))
                    model = fit(train_total - val_stats, cols, N_LAGS, LAMBDAS)
                    vmask = train_mask & (inner_of == i); vids = ids[vmask]
                    pred = predict_chunked(model, stim_of[fam][vids - 1])          # (k, v, T, C)
                    inner_mrr += [mrr_of(scores_for(eeg[vmask], pred[k].transpose(0, 2, 1), SELECT_WINDOW), vids) for k in range(len(LAMBDAS))]
                inner_mrr /= N_INNER
                best = max(range(len(LAMBDAS)), key=lambda k: (round(inner_mrr[k], 10), k))   # tie -> larger lambda
                final = fit(train_total, cols, N_LAGS, LAMBDAS[best:best + 1]); tr = time.perf_counter() - t0
                t0 = time.perf_counter(); pred = predict_chunked(final, stim_of[fam][heldout - 1], which=0).transpose(0, 2, 1)
                for w in WINDOWS: record[f"{exp}__{w[0]}_{w[1]}"] = scores_for(observed, pred, w)
                inf = time.perf_counter() - t0
                for w in WINDOWS:
                    k = f"{exp}__{w[0]}_{w[1]}"; m = summarize_positions(target_positions(rank_candidates(record[k], heldout), ids[test_mask]))
                    rows.append(dict(experiment=exp, window=f"{w[0]}_{w[1]}", fold=o, subject=s, lam=float(LAMBDAS[best]), inner_mrr=float(inner_mrr[best]), train_s=tr, infer_s=inf / len(WINDOWS), top1=m["top1"], top5=m["top5"], mrr=m["mrr"]))
            np.savez_compressed(out / f"fold{o}_sub{s}.npz", **record)
            sel = {r["experiment"]: (round(r["top1"] * 102), round(r["top5"] * 102), r["lam"]) for r in rows if r["fold"] == o and r["subject"] == s and r["window"] == "50_600"}
            print(f"sub-{s} fold{o}: {sel}", flush=True)
    with (out / "results.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "run.json").write_text(json.dumps({"folds": [f.tolist() for f in folds], "outer_seed": OUTER_SEED, "inner_seed": INNER_SEED, "n_lags": N_LAGS, "lambdas": LAMBDAS.tolist(),
                                              "windows": WINDOWS, "python": platform.python_version(), "numpy": np.__version__, "total_seconds": time.perf_counter() - t_all}, indent=1))
    print(f"done in {time.perf_counter() - t_all:.0f}s")


if __name__ == "__main__":
    main()

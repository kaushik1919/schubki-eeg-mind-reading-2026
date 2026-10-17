#!/usr/bin/env python3
"""Failure analysis for the contrastive decoder on one outer fold (training EEG only).

Retrains a variant on fold `--fold` with the frozen config, then on the held-out
fold videos measures: embedding collapse (mean pairwise cosine, effective rank),
subject decodability from embeddings (5-fold logistic regression, chance 1/6),
cross-subject EEG<->EEG identification in embedding space vs. raw EEG, and the
train/validation gap.
"""
from __future__ import annotations

import argparse
import json
from itertools import permutations
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_contrastive_eeg import OUTER_SEED, SUBJECTS, load_all, normalise, train_one
from src.controlled_cv import common_ids, disjoint_inner_folds


def eff_rank(z: np.ndarray) -> float:
    s = np.linalg.svd(z - z.mean(0), compute_uv=False); p = s / s.sum(); return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def ident(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    a /= np.linalg.norm(a, axis=1, keepdims=True); b /= np.linalg.norm(b, axis=1, keepdims=True)
    c = a @ b.T; return float(np.mean((c > np.diag(c)[:, None]).sum(1) == 0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--fold", type=int, default=0); ap.add_argument("--variant", default="D")
    ap.add_argument("--config", default="configs/contrastive_eeg.json"); ap.add_argument("--out", default="experiments/contrastive_eeg/failure_analysis.json")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu"); a = ap.parse_args()
    cfg_all = json.loads(Path(a.config).read_text()); cfg = cfg_all["base"]
    ids, raw, S = load_all(Path("eeg-mind-reading-challenge")); feats = np.load(cfg_all["target_features"]).astype(np.float32); feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    shared = common_ids({s: ids[S == k] for k, s in enumerate(SUBJECTS)})
    fold = disjoint_inner_folds(shared, exclude=np.array([], np.int64), n_folds=6, size=102, seed=OUTER_SEED)[a.fold]
    test = np.isin(ids, fold); tv = np.unique(ids[~test])
    val_videos = np.random.default_rng(3000 + a.fold).choice(tv, int(round(cfg_all["inner_val_fraction"] * len(tv))), replace=False)
    val = np.isin(ids, val_videos); fit = ~test & ~val; X = normalise(raw, S, fit); log = {}
    enc, head, adapters = train_one(cfg, X, S, ids, feats, fit, val, a.device, cfg["seed"], a.variant, log)
    enc.eval()
    def embed(mask):
        idx = np.where(mask)[0]; idx = idx[np.lexsort((ids[idx], S[idx]))]
        with torch.no_grad():
            e = enc(torch.from_numpy(X[idx]).to(a.device), torch.from_numpy(S[idx]).to(a.device) if adapters else None).cpu().numpy()
        return idx, e
    ti, te = embed(test)
    probe = np.isin(ids, np.random.default_rng(1).choice(np.unique(ids[fit]), 102, replace=False)) & fit
    pi, pe = embed(probe)
    res = {"fold": a.fold, "variant": a.variant, "best_epoch": log["best_epoch"], "inner_val_mrr": log["inner_val_mrr"],
           "test_mean_pairwise_cos": float(np.mean((te @ te.T)[~np.eye(len(te), dtype=bool)])), "test_effective_rank": eff_rank(te), "embed_dim": int(te.shape[1]),
           "subject_decoding_acc_test_embeddings": float(cross_val_score(LogisticRegression(max_iter=2000), te, S[ti], cv=5).mean()),
           "subject_decoding_acc_raw_eeg": float(cross_val_score(LogisticRegression(max_iter=2000), X[ti].reshape(len(ti), -1), S[ti], cv=5).mean()),
           "chance_subject": 1 / 6}
    by = {k: te[S[ti] == k] for k in range(6)}; raw_by = {k: X[ti][S[ti] == k].reshape(-1, 17 * 300) for k in range(6)}
    res["cross_subject_eeg_ident_top1_embedding"] = float(np.mean([ident(by[i], by[j]) for i, j in permutations(range(6), 2)]))
    res["cross_subject_eeg_ident_top1_raw"] = float(np.mean([ident(raw_by[i][:, 5 * 17:], raw_by[j][:, 5 * 17:]) for i, j in permutations(range(6), 2)]))
    res["train_probe_mean_pairwise_cos"] = float(np.mean((pe @ pe.T)[~np.eye(len(pe), dtype=bool)]))
    res["train_probe_effective_rank"] = eff_rank(pe)
    res["loss_history_tail"] = log["history"][-5:]
    Path(a.out).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

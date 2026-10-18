#!/usr/bin/env python3
"""Multimodal-target contrastive EEG decoding on the 6 exploratory outer folds.

Encoder, adapters, optimiser, batches, early stopping and inner validation are
identical to the frozen candidate (`train_contrastive_eeg.train_one`, variant D).
Only the stimulus side changes: a `MultimodalStimulusHead` over cached frozen
features.  The loss is the symmetric multi-positive EEG<->stimulus objective
(same-video recordings are positives, never negatives) plus an optional
cross-subject EEG<->EEG term with weight w; w is either fixed or chosen per outer
fold from a grid on grouped inner validation (10% of training video IDs).

Only organizer training archives are read; outer-fold videos are removed from
training for every subject.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_contrastive_eeg import OUTER_SEED, SUBJECTS, load_all, normalise, positions_by_subject, train_one
from src.controlled_cv import common_ids, disjoint_inner_folds
from src.multimodal_stimulus import MultimodalStimulusHead, assemble


def eff_rank(z):
    s = np.linalg.svd(z - z.mean(0), compute_uv=False); p = s / s.sum(); return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/multimodal_contrastive.json"); ap.add_argument("--experiment", required=True, help="key in config['experiments']")
    ap.add_argument("--folds", nargs="+", type=int, default=list(range(6))); ap.add_argument("--seed", type=int, default=None); ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--data-root", default="eeg-mind-reading-challenge"); ap.add_argument("--out-dir", default="experiments/multimodal_contrastive")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args(); cfg_all = json.loads(Path(a.config).read_text()); exp = cfg_all["experiments"][a.experiment]
    base = {**cfg_all["base"], **({"seed": a.seed} if a.seed is not None else {})}; tag = a.experiment + a.tag_suffix
    out = Path(a.out_dir) / tag; out.mkdir(parents=True, exist_ok=True)
    feats, spec = assemble(exp["modalities"]); head_cfg = cfg_all["head"]
    ids, raw, S = load_all(Path(a.data_root))
    folds = disjoint_inner_folds(common_ids({s: ids[S == k] for k, s in enumerate(SUBJECTS)}), exclude=np.array([], np.int64), n_folds=6, size=102, seed=OUTER_SEED)
    grid = exp.get("eeg_eeg_grid", [exp.get("eeg_eeg_weight", 0.0)])
    for o in a.folds:
        if (out / f"fold{o}.json").exists(): print("skip", tag, o); continue
        test = np.isin(ids, folds[o]); tv = np.unique(ids[~test])
        val_videos = np.random.default_rng(3000 + o).choice(tv, int(round(cfg_all["inner_val_fraction"] * len(tv))), replace=False)
        val = np.isin(ids, val_videos); fit = ~test & ~val
        assert not set(ids[fit]) & set(folds[o]) and not set(ids[fit]) & set(val_videos) and not set(ids[val]) & set(folds[o])
        X = normalise(raw, S, fit); t0 = time.perf_counter(); cands = []
        for w in grid:
            cfg = {**base, "eeg_eeg_weight": float(w)}; log = {}
            enc, head, adapters = train_one(cfg, X, S, ids, feats, fit, val, a.device, cfg["seed"], "D", log,
                                            make_head=lambda: MultimodalStimulusHead(spec, cfg["embed_dim"], head_cfg["proj_dim"], head_cfg["hidden"], head_cfg["dropout"]))
            cands.append((log["inner_val_mrr"], -len(cands), w, enc, head, adapters, log))
        best = max(cands, key=lambda c: (round(c[0], 6), c[1])); _, _, w, enc, head, adapters, log = best
        train_s = time.perf_counter() - t0; t0 = time.perf_counter(); metrics = {}
        for k, s in enumerate(SUBJECTS):
            pos, scores, vids = positions_by_subject(enc, head, X, S, ids, feats, test & (S == k), a.device, adapters)[s]
            assert np.array_equal(vids, folds[o])
            np.savez_compressed(out / f"fold{o}_sub{s}.npz", heldout_ids=vids, target_ids=vids, scores=scores)
            probe = np.random.default_rng(o).choice(np.unique(ids[fit & (S == k)]), 102, replace=False)
            tp = positions_by_subject(enc, head, X, S, ids, feats, fit & (S == k) & np.isin(ids, probe), a.device, adapters)[s][0]
            metrics[s] = {"top1": float(np.mean(pos == 0)), "top5": float(np.mean(pos < 5)), "mrr": float(np.mean(1 / (pos + 1))), "train_subset_mrr": float(np.mean(1 / (tp + 1)))}
        infer_s = time.perf_counter() - t0
        with torch.no_grad():
            head.eval(); stim = head(torch.from_numpy(feats[folds[o] - 1]).to(a.device)).cpu().numpy()
            enc.eval(); ti = np.where(test)[0]; eeg = enc(torch.from_numpy(X[ti]).to(a.device), torch.from_numpy(S[ti]).to(a.device)).cpu().numpy()
        has_audio = feats[folds[o] - 1, spec.mask_col] > 0.5
        mrr_by_audio = {}
        if (~has_audio).any():
            sel = np.isin(ids[ti], folds[o][~has_audio])
            allpos = np.concatenate([positions_by_subject(enc, head, X, S, ids, feats, test & (S == k), a.device, adapters)[s][0] for k, s in enumerate(SUBJECTS)])
            noa = np.tile(~has_audio, 6); mrr_by_audio = {"with_audio": float(np.mean(1 / (allpos[~noa] + 1))), "without_audio": float(np.mean(1 / (allpos[noa] + 1))), "n_without": int(noa.sum())}
        rec = {"fold": o, "experiment": a.experiment, "modalities": exp["modalities"], "seed": base["seed"], "selected_eeg_eeg_weight": w,
               "grid_inner_val_mrr": {str(c[2]): c[0] for c in cands}, "train_s": train_s, "eval_s": infer_s, "metrics": metrics,
               "logs": {"shared": {k: log[k] for k in ("best_epoch", "epochs_run", "inner_val_mrr", "n_params")} | {"history": log["history"]}},
               "stimulus_effective_rank": eff_rank(stim), "eeg_effective_rank": eff_rank(eeg), "mrr_by_audio": mrr_by_audio}
        (out / f"fold{o}.json").write_text(json.dumps(rec, indent=1))
        print(f"fold{o} {tag}: w={w} grid={rec['grid_inner_val_mrr']} {train_s:.0f}s best_ep={log['best_epoch']} test(top1,top5)={ {s: (round(m['top1']*102), round(m['top5']*102)) for s, m in metrics.items()} } rank(stim/eeg)={rec['stimulus_effective_rank']:.0f}/{rec['eeg_effective_rank']:.0f}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Variant P: frozen pretrained CBraMod (ICLR 2025, MIT) features + contrastive linear probe.

Weights: huggingface.co/weighting666/CBraMod `pretrained_weights.pth` (repo sha
500543c7, uploaded 2024-12-11, i.e. before the hackathon).  Code: github.com/
wjq-learning/CBraMod (commit b9e96100), cloned into third_party/ (not committed).
Per fold: per-subject channel z-score fitted on fit-train recordings, polyphase
resampling 500 -> 200 Hz, three 1 s patches, frozen encoder, mean over
channels -> 3 x 200 = 600-d, per-subject feature standardisation (fit-train),
then the same multi-positive objective, inner validation and outer folds as A-D.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.signal import resample_poly
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "third_party" / "CBraMod"))
from models.cbramod import CBraMod  # noqa: E402
from scripts.train_contrastive_eeg import SUBJECTS, OUTER_SEED, positions_by_subject, train_one  # noqa: E402
from src.controlled_cv import common_ids, disjoint_inner_folds  # noqa: E402
from src.data import load_training_subject  # noqa: E402
from src.eeg_encoder import ProbeEncoder  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/contrastive_eeg.json"); ap.add_argument("--folds", nargs="+", type=int, default=list(range(6)))
    ap.add_argument("--weights", default="third_party/CBraMod/pretrained_weights/pretrained_weights.pth")
    ap.add_argument("--data-root", default="eeg-mind-reading-challenge"); ap.add_argument("--out-dir", default="experiments/contrastive_eeg")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args(); cfg_all = json.loads(Path(a.config).read_text()); cfg = {**cfg_all["base"], "max_shift": 0}
    out = Path(a.out_dir) / "P"; out.mkdir(parents=True, exist_ok=True)
    ids, raw, S = [], [], []
    for k, s in enumerate(SUBJECTS):
        i, x = load_training_subject(Path(a.data_root) / "trn" / f"sub-{s}_trn.npz"); ids.append(i); raw.append(x); S.append(np.full(len(i), k))
    ids, raw, S = np.concatenate(ids), np.concatenate(raw), np.concatenate(S)
    feats = np.load(cfg_all["target_features"]).astype(np.float32); feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    folds = disjoint_inner_folds(common_ids({s: ids[S == k] for k, s in enumerate(SUBJECTS)}), exclude=np.array([], np.int64), n_folds=6, size=102, seed=OUTER_SEED)
    model = CBraMod(); model.load_state_dict(torch.load(a.weights, map_location="cpu")); model.proj_out = torch.nn.Identity(); model = model.to(a.device).eval()
    for o in a.folds:
        if (out / f"fold{o}.json").exists(): print("skip", o); continue
        test_mask = np.isin(ids, folds[o]); train_videos = np.unique(ids[~test_mask])
        val_videos = np.random.default_rng(3000 + o).choice(train_videos, int(round(cfg_all["inner_val_fraction"] * len(train_videos))), replace=False)
        val_mask = np.isin(ids, val_videos); fit_mask = ~test_mask & ~val_mask
        t0 = time.perf_counter(); Z = np.empty((len(ids), 600), np.float32)
        for k in range(len(SUBJECTS)):
            m = S == k; ref = raw[m & fit_mask]; mu, sd = ref.mean(axis=(0, 2), keepdims=True), ref.std(axis=(0, 2), keepdims=True)
            x = resample_poly((raw[m] - mu) / sd, 2, 5, axis=-1).astype(np.float32).reshape(m.sum(), 17, 3, 200)
            with torch.no_grad():
                Z[m] = np.concatenate([model(torch.from_numpy(x[j:j + 128]).to(a.device)).mean(1).flatten(1).cpu().numpy() for j in range(0, len(x), 128)])
            zr = Z[m & fit_mask]; Z[m] = (Z[m] - zr.mean(0)) / (zr.std(0) + 1e-6)
        extract_s = time.perf_counter() - t0; t0 = time.perf_counter(); log = {}
        enc, head, _ = train_one(cfg, Z, S, ids, feats, fit_mask, val_mask, a.device, cfg["seed"], "P", log, make_encoder=lambda: ProbeEncoder(600, cfg["embed_dim"], cfg["dropout"]))
        train_s = time.perf_counter() - t0; metrics = {}
        for k, s in enumerate(SUBJECTS):
            pos, scores, vids = positions_by_subject(enc, head, Z, S, ids, feats, test_mask & (S == k), a.device, False)[s]
            assert np.array_equal(vids, folds[o])
            np.savez_compressed(out / f"fold{o}_sub{s}.npz", heldout_ids=vids, target_ids=vids, scores=scores)
            probe = np.random.default_rng(o).choice(np.unique(ids[fit_mask & (S == k)]), 102, replace=False)
            tp = positions_by_subject(enc, head, Z, S, ids, feats, fit_mask & (S == k) & np.isin(ids, probe), a.device, False)[s][0]
            metrics[s] = {"top1": float(np.mean(pos == 0)), "top5": float(np.mean(pos < 5)), "mrr": float(np.mean(1 / (pos + 1))), "train_subset_mrr": float(np.mean(1 / (tp + 1)))}
        log["n_params"] = int(log["n_params"]); log["frozen_cbramod_params"] = int(sum(p.numel() for p in model.parameters()))
        (out / f"fold{o}.json").write_text(json.dumps({"fold": o, "variant": "P", "config": cfg, "extract_s": extract_s, "train_s": train_s + extract_s, "eval_s": 0.0, "metrics": metrics, "logs": {"shared": log}}, indent=1))
        print(f"fold{o} P: extract {extract_s:.0f}s train {train_s:.0f}s val={log['inner_val_mrr']:.4f}@{log['best_epoch']} test={ {s: (round(v['top1']*102), round(v['top5']*102)) for s, v in metrics.items()} }", flush=True)


if __name__ == "__main__":
    main()

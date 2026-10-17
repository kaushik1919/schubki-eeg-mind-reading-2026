#!/usr/bin/env python3
"""Train/evaluate contrastive EEG->video encoders on the 6 shared-video outer folds.

Variants (identical folds, hyperparameters and inner validation):
  A  subject-specific encoder, CLIP loss
  B  shared encoder over all subjects, naive CLIP loss (same-video pairs are negatives)
  C  shared encoder + per-subject spatial adapters, naive CLIP loss
  D  shared encoder + adapters + multi-positive loss (dedup videos, cross-subject EEG-EEG positives)

Leakage rules: every recording of an outer-fold video is removed from training
for all subjects; inner validation (early stopping, hyperparameter search)
holds out whole video identities from the training fold only; per-subject
channel normalisation is fitted on that subject's fit-train recordings only.
Only organizer training archives are read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.signal import decimate
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.contrastive_learning import VideoGroupedBatches, multi_positive_loss, naive_infonce, retrieval_positions
from src.controlled_cv import common_ids, disjoint_inner_folds
from src.data import load_training_subject
from src.eeg_encoder import EEGNetEncoder, VideoHead, n_parameters

SUBJECTS = ["01", "02", "03", "04", "05", "06"]
OUTER_SEED = 2040


def load_all(root: Path):
    ids, eeg, subj = [], [], []
    for k, s in enumerate(SUBJECTS):
        i, x = load_training_subject(root / "trn" / f"sub-{s}_trn.npz")
        ids.append(i); eeg.append(decimate(x, 5, axis=-1, ftype="fir", zero_phase=True).astype(np.float32)); subj.append(np.full(len(i), k))
    return np.concatenate(ids), np.concatenate(eeg), np.concatenate(subj)


def normalise(eeg, subj, fit_mask):
    out = np.empty_like(eeg)
    for k in range(len(SUBJECTS)):
        m = subj == k; ref = eeg[m & fit_mask]
        mu, sd = ref.mean(axis=(0, 2), keepdims=True), ref.std(axis=(0, 2), keepdims=True)
        out[m] = (eeg[m] - mu) / sd
    return out


def positions_by_subject(enc, head, X, S, V, feats, mask, device, adapters):
    """Retrieval among the videos in `mask`, separately per subject."""
    enc.eval(); head.eval(); res = {}
    with torch.no_grad():
        for k in range(len(SUBJECTS)):
            m = mask & (S == k)
            if m.sum() < 2: continue
            vids = np.sort(V[m]); order = np.argsort(V[m]); idx = np.where(m)[0][order]
            e = enc(torch.from_numpy(X[idx]).to(device), torch.from_numpy(S[idx]).to(device) if adapters else None).cpu().numpy()
            c = head(torch.from_numpy(feats[vids - 1]).to(device)).cpu().numpy()
            res[SUBJECTS[k]] = (retrieval_positions(e, c, vids, vids), e @ c.T, vids)
    return res


def train_one(cfg, X, S, V, feats, train_mask, val_mask, device, seed, variant, log, make_encoder=None):
    torch.manual_seed(seed); np.random.seed(seed)
    adapters = variant in ("C", "D")
    enc = (make_encoder() if make_encoder else EEGNetEncoder(dropout=cfg["dropout"], embed_dim=cfg["embed_dim"], n_subjects=len(SUBJECTS) if adapters else None)).to(device)
    head = VideoHead(feats.shape[1], cfg["embed_dim"]).to(device)
    logit_scale = torch.nn.Parameter(torch.tensor(np.log(1 / cfg["temperature"]), dtype=torch.float32, device=device))
    params = list(enc.parameters()) + list(head.parameters()) + [logit_scale]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    tr = np.where(train_mask)[0]; sampler = VideoGroupedBatches(V[tr], cfg["videos_per_batch"], seed)
    Xt, St, Vt = torch.from_numpy(X).to(device), torch.from_numpy(S).to(device), torch.from_numpy(V).to(device)
    F = torch.from_numpy(feats).to(device); rng = np.random.default_rng(seed)
    best, best_state, bad, history = -1.0, None, 0, []
    for epoch in range(cfg["max_epochs"]):
        enc.train(); head.train(); total = 0.0
        for batch in sampler:
            b = torch.from_numpy(tr[batch]).to(device)
            x = Xt[b]
            shift = int(rng.integers(-cfg["max_shift"], cfg["max_shift"] + 1)) if cfg["max_shift"] else 0
            if shift: x = torch.roll(x, shift, dims=-1)
            jitter = cfg.get("amp_jitter", 0.1)
            if jitter: x = x * torch.empty(len(x), *([1] * (x.dim() - 1)), device=device).uniform_(1 - jitter, 1 + jitter)
            e = enc(x, St[b] if adapters else None); scale = logit_scale.clamp(max=np.log(100.0)).exp()
            if variant in ("D", "P"):
                uniq, inv = torch.unique(Vt[b], return_inverse=True)
                loss = multi_positive_loss(e, head(F[uniq - 1]), inv, scale, subject=St[b], eeg_eeg_weight=cfg["eeg_eeg_weight"])
            else:
                loss = naive_infonce(e, head(F[Vt[b] - 1]), scale)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"]); opt.step(); total += float(loss.detach())
        pv = positions_by_subject(enc, head, X, S, V, feats, val_mask, device, adapters)
        val_mrr = float(np.mean([np.mean(1 / (p + 1)) for p, _, _ in pv.values()]))
        history.append((epoch, total / len(sampler), val_mrr))
        if val_mrr > best + 1e-5:
            best, bad = val_mrr, 0
            best_state = ({k: v.detach().clone() for k, v in enc.state_dict().items()}, {k: v.detach().clone() for k, v in head.state_dict().items()}, epoch)
        else:
            bad += 1
            if bad >= cfg["patience"]: break
    enc.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])
    log.update(best_epoch=best_state[2], epochs_run=len(history), inner_val_mrr=best, history=history, n_params=n_parameters(enc, head) + 1)
    return enc, head, adapters


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/contrastive_eeg.json")
    ap.add_argument("--variants", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--folds", nargs="+", type=int, default=list(range(6)))
    ap.add_argument("--search", action="store_true", help="hyperparameter search on fold-0 inner validation only")
    ap.add_argument("--data-root", default="eeg-mind-reading-challenge")
    ap.add_argument("--out-dir", default="experiments/contrastive_eeg")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=None, help="override base seed")
    ap.add_argument("--target", default=None, help="override target feature cache (rows = IDs 1..1000)")
    ap.add_argument("--tag-suffix", default="", help="appended to the variant directory name")
    a = ap.parse_args(); cfg_all = json.loads(Path(a.config).read_text()); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    if a.seed is not None: cfg_all["base"]["seed"] = a.seed
    if a.target: cfg_all["target_features"] = a.target
    ids, raw, S = load_all(Path(a.data_root)); feats = np.load(cfg_all["target_features"]).astype(np.float32)
    if feats.shape[0] != 1000: raise ValueError("target features must have one row per training ID 1..1000")
    feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    shared = common_ids({s: ids[S == k] for k, s in enumerate(SUBJECTS)})
    folds = disjoint_inner_folds(shared, exclude=np.array([], np.int64), n_folds=6, size=102, seed=OUTER_SEED)
    grid = cfg_all["search_grid"] if a.search else [cfg_all["base"]]
    for o in ([0] if a.search else a.folds):
        test_mask = np.isin(ids, folds[o]); train_videos = np.unique(ids[~test_mask])
        val_videos = np.random.default_rng(3000 + o).choice(train_videos, int(round(cfg_all["inner_val_fraction"] * len(train_videos))), replace=False)
        val_mask = np.isin(ids, val_videos); fit_mask = ~test_mask & ~val_mask
        assert not set(ids[fit_mask]) & set(folds[o]) and not set(ids[fit_mask]) & set(val_videos)
        X = normalise(raw, S, fit_mask)
        for gi, cfg in enumerate(grid):
            cfg = {**cfg_all["base"], **cfg}
            for variant in (["D"] if a.search else a.variants):
                tag = f"search{gi}_D" if a.search else variant + a.tag_suffix
                if (out / tag / f"fold{o}.json").exists(): print("skip", tag, o); continue
                (out / tag).mkdir(exist_ok=True); t0 = time.perf_counter(); logs = {}
                if variant == "A":
                    models = {}
                    for k, s in enumerate(SUBJECTS):
                        logs[s] = {}; models[s] = train_one(cfg, X, S, ids, feats, fit_mask & (S == k), val_mask & (S == k), a.device, cfg["seed"] + k, variant, logs[s])
                else:
                    logs["shared"] = {}; m = train_one(cfg, X, S, ids, feats, fit_mask, val_mask, a.device, cfg["seed"], variant, logs["shared"]); models = {s: m for s in SUBJECTS}
                train_s = time.perf_counter() - t0; t0 = time.perf_counter(); metrics = {}
                for k, s in enumerate(SUBJECTS):
                    enc, head, adapters = models[s]
                    if not a.search:
                        pos, scores, vids = positions_by_subject(enc, head, X, S, ids, feats, test_mask & (S == k), a.device, adapters)[s]
                        assert np.array_equal(vids, folds[o])
                        np.savez_compressed(out / tag / f"fold{o}_sub{s}.npz", heldout_ids=vids, target_ids=vids, scores=scores)
                        metrics[s] = {"top1": float(np.mean(pos == 0)), "top5": float(np.mean(pos < 5)), "mrr": float(np.mean(1 / (pos + 1)))}
                    probe = np.random.default_rng(o).choice(np.unique(ids[fit_mask & (S == k)]), 102, replace=False)  # train-gap probe
                    tp = positions_by_subject(enc, head, X, S, ids, feats, fit_mask & (S == k) & np.isin(ids, probe), a.device, adapters)
                    metrics.setdefault(s, {})["train_subset_mrr"] = float(np.mean(1 / (tp[s][0] + 1))) if s in tp else None
                infer_s = time.perf_counter() - t0
                rec = {"fold": o, "variant": variant, "target_features": cfg_all["target_features"], "config": cfg, "train_s": train_s, "eval_s": infer_s, "metrics": metrics, "logs": logs}
                (out / tag / f"fold{o}.json").write_text(json.dumps(rec, indent=1))
                summ = {s: (round(v.get("top1", 0) * 102), round(v.get("top5", 0) * 102), round(v["train_subset_mrr"], 3)) for s, v in metrics.items()}
                vm = {k: (round(v["inner_val_mrr"], 4), v["best_epoch"]) for k, v in logs.items()}
                print(f"fold{o} {tag}: {train_s:.0f}s val={vm} test(top1,top5,trainMRR)={summ}", flush=True)


if __name__ == "__main__":
    main()

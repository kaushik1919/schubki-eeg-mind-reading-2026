#!/usr/bin/env python3
"""Cross-subject neural-consistency decoders on the 6 exploratory outer folds.

Variants (shared EEGNet + per-subject adapters, CLIP block-8 stimulus space):
  A   frozen candidate decoder recipe (single seed 2050): multi-positive EEG->CLIP + 0.5 x EEG<->EEG,
      retrained through this script so that all diagnostics use the same code path
  B   EEG<->EEG cross-subject supervised contrastive only; candidates scored through a RidgeCV readout
      from block-8 features to each training video's mean EEG embedding
  C   EEG->CLIP multi-positive + w x EEG<->EEG, w in {1, 2} selected per outer fold on grouped inner validation
  D   C (selected w) + lambda x average-response regression; target = leave-own-subject-out mean
      response of the same training video (fit-train recordings only), PCA-64; lambda in {0.25, 1}
      selected per outer fold on grouped inner validation
Inner validation (10% of training video IDs, all subjects) uses one metric for every variant:
mean per-subject retrieval MRR among inner-validation videos with the variant's own scoring path.
Only organizer training archives are read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_contrastive_eeg import OUTER_SEED, SUBJECTS, load_all, normalise
from src.contrastive_learning import VideoGroupedBatches, multi_positive_loss
from src.controlled_cv import common_ids, disjoint_inner_folds
from src.cross_subject_learning import AverageTargetCoder, RidgeReadout, eeg_eeg_supcon, loso_average_targets, mean_embedding_per_video
from src.eeg_encoder import EEGNetEncoder, VideoHead, n_parameters

TARGET = "features/all_train_clip_intermediate_block8.npy"


def eff_rank(z):
    s = np.linalg.svd(z - z.mean(0), compute_uv=False); p = s / s.sum(); return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


class Trainer:
    def __init__(self, cfg, X, S, V, feats, device):
        self.cfg, self.X, self.S, self.V, self.feats, self.device = cfg, X, S, V, feats, device
        self.Xt, self.St, self.Vt = torch.from_numpy(X).to(device), torch.from_numpy(S).to(device), torch.from_numpy(V).to(device)
        self.Ft = torch.from_numpy(feats).to(device)

    @torch.no_grad()
    def embed(self, enc, mask):
        idx = np.where(mask)[0]; enc.eval()
        return idx, np.concatenate([enc(self.Xt[torch.from_numpy(idx[j:j + 1024]).to(self.device)], self.St[torch.from_numpy(idx[j:j + 1024]).to(self.device)]).cpu().numpy() for j in range(0, len(idx), 1024)])

    def candidate_embeddings(self, enc, head, readout_mask, cand_ids):
        if head is not None:
            with torch.no_grad(): head.eval(); return head(self.Ft[torch.from_numpy(cand_ids - 1).to(self.device)]).cpu().numpy()
        idx, e = self.embed(enc, readout_mask); vids, m = mean_embedding_per_video(e, self.V[idx])
        return RidgeReadout().fit(self.feats[vids - 1], m).predict(self.feats[cand_ids - 1])

    def retrieval(self, enc, head, eval_mask, readout_mask):
        """{subject: (positions, scores, sorted video ids)} among the videos of eval_mask, per subject."""
        idx, e = self.embed(enc, eval_mask); out = {}
        cand_all = np.unique(self.V[idx]); c_all = self.candidate_embeddings(enc, head, readout_mask, cand_all)
        for k, s in enumerate(SUBJECTS):
            sel = self.S[idx] == k
            if sel.sum() < 2: continue
            order = np.argsort(self.V[idx][sel]); vids = self.V[idx][sel][order]; es = e[sel][order]
            c = c_all[np.searchsorted(cand_all, vids)]; sc = es @ c.T
            out[s] = ((sc > np.diag(sc)[:, None]).sum(1), sc, vids)
        return out

    def train(self, mode, fit_mask, val_mask, seed, w_e2e=0.5, lam_avg=0.0, log=None):
        cfg, dev = self.cfg, self.device; torch.manual_seed(seed); np.random.seed(seed)
        enc = EEGNetEncoder(dropout=cfg["dropout"], embed_dim=cfg["embed_dim"], n_subjects=6).to(dev)
        head = VideoHead(self.feats.shape[1], cfg["embed_dim"]).to(dev) if mode != "e2e" else None
        scale_p = torch.nn.Parameter(torch.tensor(np.log(1 / cfg["temperature"]), dtype=torch.float32, device=dev))
        params = list(enc.parameters()) + ([*head.parameters()] if head else []) + [scale_p]
        avg_head = None
        if lam_avg > 0:
            tgt, ok = loso_average_targets(self.X, self.S, self.V, fit_mask)
            coder = AverageTargetCoder(64, seed).fit(tgt[fit_mask & ok]); Y = np.zeros((len(self.X), 64), np.float32)
            Y[fit_mask & ok] = coder.transform(tgt[fit_mask & ok]); Yt, OKt = torch.from_numpy(Y).to(dev), torch.from_numpy(fit_mask & ok).to(dev)
            avg_head = torch.nn.Linear(cfg["embed_dim"], 64).to(dev); params += list(avg_head.parameters())
        opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        tr = np.where(fit_mask)[0]; sampler = VideoGroupedBatches(self.V[tr], cfg["videos_per_batch"], seed); rng = np.random.default_rng(seed)
        best, state, bad, hist = -1.0, None, 0, []
        for epoch in range(cfg["max_epochs"]):
            enc.train(); head and head.train(); total = 0.0
            for batch in sampler:
                b = torch.from_numpy(tr[batch]).to(dev); x = self.Xt[b]
                sh = int(rng.integers(-cfg["max_shift"], cfg["max_shift"] + 1));  x = torch.roll(x, sh, dims=-1) if sh else x
                x = x * torch.empty(len(x), 1, 1, device=dev).uniform_(0.9, 1.1)
                e = enc(x, self.St[b]); scale = scale_p.clamp(max=np.log(100.0)).exp(); uniq, inv = torch.unique(self.Vt[b], return_inverse=True)
                if mode == "e2e":
                    loss = eeg_eeg_supcon(e, inv, self.St[b], scale)
                else:
                    loss = multi_positive_loss(e, head(self.Ft[uniq - 1]), inv, scale, subject=self.St[b], eeg_eeg_weight=w_e2e)
                if avg_head is not None:
                    m = OKt[b]
                    if m.any(): loss = loss + lam_avg * F.mse_loss(avg_head(e[m]), Yt[b][m])
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"]); opt.step(); total += float(loss.detach())
            r = self.retrieval(enc, head, val_mask, fit_mask); val = float(np.mean([np.mean(1 / (p + 1)) for p, _, _ in r.values()]))
            hist.append((epoch, total / len(sampler), val))
            if val > best + 1e-5:
                best, bad = val, 0; state = ({k: v.detach().clone() for k, v in enc.state_dict().items()}, {k: v.detach().clone() for k, v in head.state_dict().items()} if head else None, epoch)
            else:
                bad += 1
                if bad >= cfg["patience"]: break
        enc.load_state_dict(state[0]); head and head.load_state_dict(state[1])
        if log is not None: log.update(best_epoch=state[2], inner_val_mrr=best, epochs=len(hist), n_params=n_parameters(enc, *( [head] if head else [])) + 1 + (n_parameters(avg_head) if avg_head else 0))
        return enc, head


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/cross_subject_eeg.json"); ap.add_argument("--folds", nargs="+", type=int, default=list(range(6)))
    ap.add_argument("--variants", nargs="+", default=["B", "C", "D"]); ap.add_argument("--out-dir", default="experiments/cross_subject_eeg")
    ap.add_argument("--data-root", default="eeg-mind-reading-challenge"); ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=None); ap.add_argument("--tag-suffix", default="")
    a = ap.parse_args(); cfg_all = json.loads(Path(a.config).read_text()); cfg = cfg_all["base"]; out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    if a.seed is not None: cfg["seed"] = a.seed
    ids, raw, S = load_all(Path(a.data_root)); feats = np.load(TARGET).astype(np.float32); feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    folds = disjoint_inner_folds(common_ids({s: ids[S == k] for k, s in enumerate(SUBJECTS)}), exclude=np.array([], np.int64), n_folds=6, size=102, seed=OUTER_SEED)
    for o in a.folds:
        test = np.isin(ids, folds[o]); tv = np.unique(ids[~test])
        val_videos = np.random.default_rng(3000 + o).choice(tv, int(round(cfg_all["inner_val_fraction"] * len(tv))), replace=False)
        val = np.isin(ids, val_videos); fit = ~test & ~val
        assert not set(ids[fit]) & set(folds[o]) and not set(ids[fit]) & set(val_videos)
        X = normalise(raw, S, fit); T = Trainer(cfg, X, S, ids, feats, a.device)
        for variant in a.variants:
            tag = variant + a.tag_suffix
            if (out / tag / f"fold{o}.json").exists(): print("skip", tag, o); continue
            (out / tag).mkdir(exist_ok=True); t0 = time.perf_counter(); cands = {}
            grid = {"A": [dict(mode="e2v", w_e2e=0.5)], "B": [dict(mode="e2e")], "C": [dict(mode="e2v", w_e2e=w) for w in cfg_all["grid_w_e2e"]]}
            if variant == "D":
                cw = json.loads((out / "C" / f"fold{o}.json").read_text())["selected"]["w_e2e"]
                grid["D"] = [dict(mode="e2v", w_e2e=cw, lam_avg=l) for l in cfg_all["grid_lam_avg"]]
            for gi, g in enumerate(grid[variant]):
                log = {}; enc, head = T.train(fit_mask=fit, val_mask=val, seed=cfg["seed"], log=log, **g); cands[gi] = (log["inner_val_mrr"], g, enc, head, log)
            gi = max(cands, key=lambda k: (round(cands[k][0], 6), -k)); _, g, enc, head, log = cands[gi]
            train_s = time.perf_counter() - t0; t0 = time.perf_counter()
            res = T.retrieval(enc, head, test, fit); infer_s = time.perf_counter() - t0
            metrics = {}
            for s, (p, sc, vids) in res.items():
                assert np.array_equal(vids, folds[o])
                np.savez_compressed(out / tag / f"fold{o}_sub{s}.npz", heldout_ids=vids, target_ids=vids, scores=sc)
                metrics[s] = {"top1": float(np.mean(p == 0)), "top5": float(np.mean(p < 5)), "mrr": float(np.mean(1 / (p + 1)))}
            # diagnostics on held-out videos: cross-subject same-video retrieval and effective rank; train gap probe
            idx, e = T.embed(enc, test); xs = []
            for i in range(6):
                for j in range(6):
                    if i == j: continue
                    ai, bj = idx[S[idx] == i], idx[S[idx] == j]
                    ea, eb = e[S[idx] == i][np.argsort(ids[ai])], e[S[idx] == j][np.argsort(ids[bj])]; c = ea @ eb.T
                    xs.append(float(np.mean((c > np.diag(c)[:, None]).sum(1) == 0)))
            probe = fit & np.isin(ids, np.random.default_rng(o).choice(np.unique(ids[fit]), 102, replace=False))
            tp = T.retrieval(enc, head, probe, fit)
            rec = {"fold": o, "variant": variant, "seed": cfg["seed"], "selected": g, "grid_inner_val_mrr": {str(k): v[0] for k, v in cands.items()}, "train_s": train_s, "infer_s": infer_s,
                   "metrics": metrics, "best_epoch": log["best_epoch"], "inner_val_mrr": log["inner_val_mrr"], "n_params": log["n_params"],
                   "cross_subject_eeg_ident_top1": float(np.mean(xs)), "effective_rank": eff_rank(e), "train_probe_mrr": float(np.mean([np.mean(1 / (p + 1)) for p, _, _ in tp.values()]))}
            (out / tag / f"fold{o}.json").write_text(json.dumps(rec, indent=1))
            print(f"fold{o} {tag}: sel={g} grid={rec['grid_inner_val_mrr']} {train_s:.0f}s test(top1,top5)={ {s: (round(m['top1']*102), round(m['top5']*102)) for s, m in metrics.items()} } xsub={rec['cross_subject_eeg_ident_top1']:.3f} rank={rec['effective_rank']:.0f} gap={rec['train_probe_mrr']:.2f}/{rec['inner_val_mrr']:.3f}", flush=True)


if __name__ == "__main__":
    main()

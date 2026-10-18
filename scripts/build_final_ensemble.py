#!/usr/bin/env python3
"""Build versioned hidden-subject (and dev-format-check) predictions for the final ensembles.

Components, all fitted on organizer training recordings only:
  * frozen contrastive decoder: 5 checkpoints in outputs/contrastive_candidate/models (trained on all
    training recordings, inner-val seed 3100) with its saved per-subject normalisation;
  * EEG<->EEG decoder B: 5 seeds trained here on all training recordings (same inner-val split),
    ridge readout from block-8 features to the mean fit-train EEG embedding per video;
  * frozen Model C: fused scores from outputs/final_model_c/scores (hidden) or refit (dev).
Test EEG is used only as inference input in file order.  Root submission files are not written.
Outputs (per subject): S1 = decoder+ModelC, S3 = decoder+B+ModelC, each with independent
ranking and with Sinkhorn(tau=0.5) calibration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.signal import decimate
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_contrastive_eeg import SUBJECTS, load_all, normalise
from scripts.train_cross_subject_eeg import Trainer
from src.assignment import rowz, sinkhorn_log
from src.cross_subject_learning import RidgeReadout, mean_embedding_per_video
from src.data import CANDIDATE_IDS, load_eeg, load_training_subject
from src.eeg_encoder import EEGNetEncoder, VideoHead
from src.final_model import FrozenModelC, load_frozen_config
from src.retrieval import rank_candidates, validate_rankings


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def l2(x): return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="outputs/final_ensemble_v1"); ap.add_argument("--data-root", default="eeg-mind-reading-challenge")
    ap.add_argument("--subjects", nargs="+", default=["05", "06", "01"]); ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args(); out, root = Path(a.out_dir), Path(a.data_root)
    if out.exists(): raise FileExistsError(out)
    (out / "models_B").mkdir(parents=True); (out / "scores").mkdir(); t0 = time.perf_counter()
    cand = json.loads(Path("configs/frozen_contrastive_candidate.json").read_text()); dec = cand["decoder"]; hp = dec["hyperparameters"]
    xcfg = json.loads(Path("configs/cross_subject_eeg.json").read_text())["base"]
    ids, raw, S = load_all(root)
    tv = np.unique(ids); val_videos = np.random.default_rng(dec["early_stopping"]["inner_val_video_seed"]).choice(tv, int(round(0.1 * len(tv))), replace=False)
    val = np.isin(ids, val_videos); fit = ~val
    X = normalise(raw, S, fit)
    norm = np.load("outputs/contrastive_candidate/normalisation.npz")
    stats = {k: (norm[f"mean_{k}"], norm[f"std_{k}"]) for k in range(6)}
    for k in range(6):   # identical statistics to the frozen candidate
        ref = raw[(S == k) & fit]; assert np.allclose(ref.mean(axis=(0, 2), keepdims=True), stats[k][0], atol=1e-5)
    b8_tr = l2(np.load(dec["target_features"]["train"]).astype(np.float32)); b8_te = l2(np.load(dec["target_features"]["candidates"]).astype(np.float32))
    # ---- frozen decoder ----
    D = []
    for seed in dec["seeds"]:
        ck = torch.load(f"outputs/contrastive_candidate/models/seed{seed}.pt", map_location=a.device)
        enc = EEGNetEncoder(dropout=hp["dropout"], embed_dim=hp["embed_dim"], n_subjects=6).to(a.device); enc.load_state_dict(ck["encoder"]); enc.eval()
        head = VideoHead(768, hp["embed_dim"]).to(a.device); head.load_state_dict(ck["video_head"]); head.eval()
        with torch.no_grad(): D.append((enc, head(torch.from_numpy(b8_te).to(a.device)).cpu().numpy()))
    # ---- EEG<->EEG decoder B on all training data ----
    T = Trainer(xcfg, X, S, ids, b8_tr, a.device); B = []; blog = {}
    for seed in dec["seeds"]:
        log = {}; enc, _ = T.train("e2e", fit, val, seed, log=log); blog[seed] = {k: log[k] for k in ("best_epoch", "inner_val_mrr", "n_params")}
        idx, e = T.embed(enc, fit); vids, m = mean_embedding_per_video(e, ids[idx]); readout = RidgeReadout().fit(b8_tr[vids - 1], m)
        torch.save({"encoder": enc.state_dict(), "seed": seed, "readout_coef": readout.model.coef_, "readout_intercept": readout.model.intercept_}, out / "models_B" / f"seed{seed}.pt")
        B.append((enc.eval(), readout.predict(b8_te))); print("B seed", seed, blog[seed], flush=True)
    # ---- Model C ----
    mc_cfg = load_frozen_config("configs/frozen_balanced.json"); names = [c["name"] for c in mc_cfg["constituents"]]; w = [c["weight"] for c in mc_cfg["constituents"]]
    mc_tr = {c["name"]: np.load(c["train_features"]) for c in mc_cfg["constituents"]}; mc_te = {c["name"]: np.load(c["test_features"]) for c in mc_cfg["constituents"]}
    record = {"B_train_log": blog, "subjects": {}}
    for s in a.subjects:
        k = SUBJECTS.index(s); tst = root / "tst" / f"sub-{s}_tst.npy"; test = load_eeg(tst)
        x = torch.from_numpy(((decimate(test, 5, axis=-1, ftype="fir", zero_phase=True) - stats[k][0]) / stats[k][1]).astype(np.float32)).to(a.device); subj = torch.full((len(test),), k, device=a.device)
        with torch.no_grad():
            dsc = [enc(x, subj).cpu().numpy() @ c.T for enc, c in D]; bsc = [enc(x, subj).cpu().numpy() @ c.T for enc, c in B]
        saved = Path(f"outputs/final_model_c/scores/sub-{s}_scores.npz")
        if saved.exists():
            mc = np.load(saved)["fused"]
        else:
            tid, teeg = load_training_subject(root / "trn" / f"sub-{s}_trn.npz"); mc = FrozenModelC(names, w).fit(tid, teeg, mc_tr).rank(test, mc_te, CANDIDATE_IDS)[1]
        dz, bz, cz = np.mean([rowz(v) for v in dsc], 0), np.mean([rowz(v) for v in bsc], 0), rowz(mc)
        systems = {"S1": 0.5 * rowz(dz) + 0.5 * cz, "S3": (dz + bz + cz) / 3}
        files = {}
        for name, sc in systems.items():
            for cal, f in (("independent", lambda z: z), (f"sinkhorn_t{a.tau}", lambda z: sinkhorn_log(z, a.tau))):
                r = validate_rankings(rank_candidates(f(sc), CANDIDATE_IDS)).astype(np.int64)
                p = out / name / cal / f"sub-{s}_ranked_candidate_list.npy"; p.parent.mkdir(parents=True, exist_ok=True); np.save(p, r, allow_pickle=False); files[f"{name}/{cal}"] = sha(p)
        np.savez_compressed(out / "scores" / f"sub-{s}_scores.npz", candidate_ids=CANDIDATE_IDS, decoder=dz, eeg_eeg=bz, model_c=mc, S1=systems["S1"], S3=systems["S3"])
        record["subjects"][s] = {"test_eeg_sha256": sha(tst), "files": files, "role": "hidden" if s in ("05", "06") else "format_check"}
        print(s, files, flush=True)
    record["total_s"] = time.perf_counter() - t0
    (out / "metadata.json").write_text(json.dumps(record, indent=1))
    fl = sorted(x for x in out.rglob("*") if x.is_file() and x.name != "MANIFEST.sha256")
    (out / "MANIFEST.sha256").write_text("".join(f"{sha(f)}  {f.relative_to(out)}\n" for f in fl))


if __name__ == "__main__":
    main()

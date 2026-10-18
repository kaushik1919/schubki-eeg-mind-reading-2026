#!/usr/bin/env python3
"""One-shot, pre-registered development-subject check of the frozen contrastive candidate.

Order of operations (enforced):
  1. Verify the frozen config checksum and that it is committed unmodified.
  2. Train the 5-seed decoder on organizer *training* recordings only
     (early stopping on grouped inner validation of training videos).  No test
     file is opened before training finishes.
  3. Fit frozen Model C per development subject on training recordings.
  4. Score development test EEG in file order; save scores/rankings and hash them.
  5. Only then load tst_stim labels and compute metrics.
Refuses to run twice (existing outputs or result file).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.signal import decimate
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_contrastive_eeg import SUBJECTS, load_all, normalise, train_one
from src.data import CANDIDATE_IDS, load_eeg, load_training_subject
from src.final_model import FrozenModelC, load_frozen_config
from src.rank_analysis import exact_sign_test, subject_bootstrap_ci, summarize_positions, target_positions, transition_counts
from src.retrieval import rank_candidates, row_zscore, validate_rankings

CONFIG = "configs/frozen_contrastive_candidate.json"


def sha(p) -> str: return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fuse(decoder_scores: list[np.ndarray], model_c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ens = np.mean([row_zscore(s) for s in decoder_scores], axis=0)
    return ens, 0.5 * row_zscore(ens) + 0.5 * row_zscore(model_c)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expected-config-sha256", required=True); ap.add_argument("--data-root", default="eeg-mind-reading-challenge")
    ap.add_argument("--out-dir", default="outputs/contrastive_candidate"); ap.add_argument("--result", default="experiments/frozen_candidate_confirmation.json")
    a = ap.parse_args(); out, root = Path(a.out_dir), Path(a.data_root)
    if Path(a.result).exists() or out.exists(): raise FileExistsError("confirmation already run; this check is one-shot")
    if sha(CONFIG) != a.expected_config_sha256: raise ValueError("frozen config checksum mismatch")
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", CONFIG, "scripts/train_contrastive_eeg.py", "src/eeg_encoder.py", "src/contrastive_learning.py", "src/final_model.py", __file__]).returncode != 0:
        raise RuntimeError("frozen config or code differs from the committed version")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    cfg = json.loads(Path(CONFIG).read_text()); dec = cfg["decoder"]; hp = dec["hyperparameters"]
    assert SUBJECTS == list(dec["subject_index"]) and all(dec["subject_index"][s] == k for k, s in enumerate(SUBJECTS))
    device = dec["device"] if (dec["device"] != "mps" or torch.backends.mps.is_available()) else "cpu"
    (out / "models").mkdir(parents=True); (out / "dev_scores").mkdir(); t0 = time.perf_counter()

    # ---- 2. decoder training: training archives only ----
    ids, raw, S = load_all(root)
    tv = np.unique(ids); val_videos = np.random.default_rng(dec["early_stopping"]["inner_val_video_seed"]).choice(tv, int(round(dec["early_stopping"]["inner_val_fraction"] * len(tv))), replace=False)
    val_mask = np.isin(ids, val_videos); fit_mask = ~val_mask
    X = normalise(raw, S, fit_mask)
    stats = {k: (raw[(S == k) & fit_mask].mean(axis=(0, 2), keepdims=True), raw[(S == k) & fit_mask].std(axis=(0, 2), keepdims=True)) for k in range(6)}
    np.savez(out / "normalisation.npz", **{f"mean_{k}": v[0] for k, v in stats.items()}, **{f"std_{k}": v[1] for k, v in stats.items()})
    tfeat = np.load(dec["target_features"]["train"]).astype(np.float32); tfeat /= np.linalg.norm(tfeat, axis=1, keepdims=True)
    cfeat = np.load(dec["target_features"]["candidates"]).astype(np.float32); cfeat /= np.linalg.norm(cfeat, axis=1, keepdims=True)
    assert tfeat.shape == (1000, 768) and cfeat.shape == (102, 768)
    models, train_log = [], {}
    for seed in dec["seeds"]:
        log = {}; enc, head, adapters = train_one({**hp, "seed": seed}, X, S, ids, tfeat, fit_mask, val_mask, device, seed, dec["variant"], log)
        torch.save({"encoder": enc.state_dict(), "video_head": head.state_dict(), "seed": seed, "best_epoch": log["best_epoch"]}, out / "models" / f"seed{seed}.pt")
        models.append((enc.eval(), head.eval())); train_log[seed] = {k: log[k] for k in ("best_epoch", "epochs_run", "inner_val_mrr", "n_params")}
        print("seed", seed, train_log[seed], flush=True)
    train_s = time.perf_counter() - t0

    # ---- 3. Model C per development subject (training recordings only) ----
    mc_cfg = load_frozen_config(cfg["model_c"]["config"])
    names = [c["name"] for c in mc_cfg["constituents"]]; weights = [c["weight"] for c in mc_cfg["constituents"]]
    mc_train = {c["name"]: np.load(c["train_features"]) for c in mc_cfg["constituents"]}; mc_test = {c["name"]: np.load(c["test_features"]) for c in mc_cfg["constituents"]}

    # ---- 4. inference on development test EEG (file order) ----
    t1 = time.perf_counter(); pred_hashes = {}
    for s in cfg["confirmation"]["subjects"]:
        k = dec["subject_index"][s]
        test = load_eeg(root / "tst" / f"sub-{s}_tst.npy")                                 # inference input only
        x = ((decimate(test, 5, axis=-1, ftype="fir", zero_phase=True) - stats[k][0]) / stats[k][1]).astype(np.float32)
        dscores = []
        with torch.no_grad():
            for enc, head in models:
                e = enc(torch.from_numpy(x).to(device), torch.full((len(x),), k, device=device)).cpu().numpy()
                dscores.append(e @ head(torch.from_numpy(cfeat).to(device)).cpu().numpy().T)
        tid, teeg = load_training_subject(root / "trn" / f"sub-{s}_trn.npz")
        mc = FrozenModelC(names, weights).fit(tid, teeg, mc_train)
        _, mc_fused, _ = mc.rank(test, mc_test, CANDIDATE_IDS)
        ens, fused = fuse(dscores, mc_fused)
        rk = {n: validate_rankings(rank_candidates(v, CANDIDATE_IDS)) for n, v in (("candidate", fused), ("decoder", ens), ("model_c", mc_fused))}
        path = out / "dev_scores" / f"sub-{s}_scores.npz"
        np.savez_compressed(path, candidate_ids=CANDIDATE_IDS, test_eeg_sha256=sha(root / "tst" / f"sub-{s}_tst.npy"), **{f"seed{sd}": v for sd, v in zip(dec["seeds"], dscores)},
                            decoder_ensemble=ens, model_c=mc_fused, candidate=fused, **{f"rankings_{n}": v for n, v in rk.items()})
        pred_hashes[path.name] = sha(path)
    (out / "dev_scores" / "PREDICTIONS.sha256").write_text("".join(f"{h}  {n}\n" for n, h in pred_hashes.items()))
    infer_s = time.perf_counter() - t1

    # ---- 5. labels: metrics only ----
    pos = {n: {} for n in ("candidate", "decoder", "model_c")}
    for s in cfg["confirmation"]["subjects"]:
        z = np.load(out / "dev_scores" / f"sub-{s}_scores.npz"); truth = np.load(root / "tst_stim" / f"sub-{s}_tst_stim.npy").astype(np.int64)
        for n in pos: pos[n][s] = target_positions(z[f"rankings_{n}"], truth)
    subs = cfg["confirmation"]["subjects"]; res = {"commit": commit, "config_sha256": sha(CONFIG), "device": device, "train_s": train_s, "inference_s": infer_s, "train_log": train_log, "prediction_hashes": pred_hashes}
    for n, p in pos.items():
        per = {s: summarize_positions(p[s]) for s in subs}
        res[n] = {m: float(np.mean([per[s][m] for s in subs])) for m in ("top1", "top5", "mrr")} | {"per_subject": per}
    for n in ("candidate", "decoder"):
        t = transition_counts(np.concatenate([pos["model_c"][s] for s in subs]), np.concatenate([pos[n][s] for s in subs]))
        res[n]["vs_model_c"] = {f"d_{m}": subject_bootstrap_ci(np.array([res[n]["per_subject"][s][m] - res["model_c"]["per_subject"][s][m] for s in subs])) for m in ("top1", "top5", "mrr")} | {
            "top1_WL": [t["top1_wins"], t["top1_losses"]], "top1_sign_p": exact_sign_test(t["top1_wins"], t["top1_losses"]), "top5_WL": [t["top5_wins"], t["top5_losses"]], "top5_sign_p": exact_sign_test(t["top5_wins"], t["top5_losses"])}
    Path(a.result).write_text(json.dumps(res, indent=1)); print(json.dumps({k: v for k, v in res.items() if k not in ("train_log",)}, indent=1))


if __name__ == "__main__":
    main()

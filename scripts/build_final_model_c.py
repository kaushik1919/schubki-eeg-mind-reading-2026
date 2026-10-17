#!/usr/bin/env python3
"""Refit frozen Model C on all training recordings and write versioned submissions.

Fitting uses only `trn/sub-XX_trn.npz` (training EEG of that subject).  Test EEG
is used only at inference, in file order, never for fitting.  Development test
labels are read only for the organizer-recommended format check (--format-check),
after all outputs are fixed, and influence nothing.  Root submission files are
never written by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shlex
import sys
import time

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import CANDIDATE_IDS, load_eeg, load_training_subject
from src.evaluation import retrieval_metrics
from src.final_model import FrozenModelC, load_frozen_config
from src.retrieval import validate_rankings


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def versions() -> dict[str, str]:
    import scipy, sklearn, torch, open_clip, cv2, imageio_ffmpeg
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "joblib": joblib.__version__,
            "torch": torch.__version__, "open_clip_torch": open_clip.__version__, "opencv": cv2.__version__, "imageio-ffmpeg": imageio_ffmpeg.__version__, "ffmpeg": imageio_ffmpeg.get_ffmpeg_version(),
            "platform": platform.platform()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/frozen_balanced.json")
    p.add_argument("--data-root", default="eeg-mind-reading-challenge")
    p.add_argument("--candidate-ids", default="manifests/test_candidate_ids.npy")
    p.add_argument("--validated-record", default="experiments/nested_rerank/outer_results.json", help="Run record whose training-feature hashes were validated")
    p.add_argument("--subjects", nargs="+", default=["05", "06"])
    p.add_argument("--format-check", nargs="*", default=["01"], help="Development subjects scored only as an organizer format check")
    p.add_argument("--out-dir", default="outputs/final_model_c")
    a = p.parse_args(); t0 = time.perf_counter(); out = Path(a.out_dir)
    if out.exists(): raise FileExistsError(f"refusing to overwrite {out}; choose a new versioned directory")
    config = load_frozen_config(a.config); names = [c["name"] for c in config["constituents"]]; weights = [c["weight"] for c in config["constituents"]]

    # --- Feature provenance: training features must be the exact validated caches. ---
    validated = json.loads(Path(a.validated_record).read_text())["features"]
    key = {"final_clip": "final", "clip_block8": "block8", "audio_simple": "audio"}
    train_feats, test_feats, feature_record = {}, {}, {}
    for c in config["constituents"]:
        tr, te = c["train_features"], c["test_features"]
        if validated[key[c["name"]]]["path"] != tr or validated[key[c["name"]]]["sha256"] != sha256(tr):
            raise ValueError(f"{tr} differs from the validated feature cache")
        train_feats[c["name"]], test_feats[c["name"]] = np.load(tr), np.load(te)
        if train_feats[c["name"]].shape[0] != 1000 or test_feats[c["name"]].shape[0] != 102 or train_feats[c["name"]].shape[1] != test_feats[c["name"]].shape[1]:
            raise ValueError(f"{c['name']} feature shapes are inconsistent")
        if not (np.isfinite(train_feats[c["name"]]).all() and np.isfinite(test_feats[c["name"]]).all()): raise ValueError("non-finite features")
        feature_record[c["name"]] = {"train": tr, "train_sha256": sha256(tr), "test": te, "test_sha256": sha256(te), "dim": int(train_feats[c["name"]].shape[1])}
    candidate_ids = np.load(a.candidate_ids, allow_pickle=False).astype(np.int64)
    if not np.array_equal(candidate_ids, CANDIDATE_IDS): raise ValueError("candidate feature rows must be IDs 1001..1102 ascending")

    (out / "models").mkdir(parents=True); (out / "scores").mkdir(); (out / "format_check").mkdir()
    subjects_meta = {}
    for subject in list(a.subjects) + list(a.format_check):
        is_hidden = subject in a.subjects
        trn, tst = Path(a.data_root) / "trn" / f"sub-{subject}_trn.npz", Path(a.data_root) / "tst" / f"sub-{subject}_tst.npy"
        ids, eeg = load_training_subject(trn)
        started = time.perf_counter(); model = FrozenModelC(names, weights).fit(ids, eeg, train_feats); fit_s = time.perf_counter() - started
        test_eeg = load_eeg(tst, name=f"sub-{subject} test EEG")  # file order, never sorted
        if test_eeg.shape[0] != 102: raise ValueError("expected 102 test recordings")
        started = time.perf_counter(); rankings, fused, constituent = model.rank(test_eeg, test_feats, candidate_ids); infer_s = time.perf_counter() - started
        rankings = validate_rankings(rankings).astype(np.int64)
        # Determinism: an independent refit must reproduce the scores bit-exactly.
        again, fused2, _ = FrozenModelC(names, weights).fit(ids, eeg, train_feats).rank(test_eeg, test_feats, candidate_ids)
        if not (np.array_equal(fused, fused2) and np.array_equal(rankings, again)): raise RuntimeError("non-deterministic refit/inference")
        ties = int(sum(len(r) - len(np.unique(r)) for r in fused))
        dest = out if is_hidden else out / "format_check"
        sub_path = dest / f"sub-{subject}_ranked_candidate_list.npy"; np.save(sub_path, rankings, allow_pickle=False)
        ckpt = out / "models" / f"sub-{subject}_model_c.joblib"
        joblib.dump({"model": model, "config_id": config["config_id"], "training_ids": ids, "train_npz_sha256": sha256(trn), "feature_record": feature_record}, ckpt, compress=3)
        np.savez_compressed(out / "scores" / f"sub-{subject}_scores.npz", candidate_ids=candidate_ids, fused=fused, **{f"scores_{k}": v for k, v in constituent.items()})
        meta = {"role": "hidden" if is_hidden else "format_check_only", "train_npz": str(trn), "train_npz_sha256": sha256(trn), "test_eeg": str(tst), "test_eeg_sha256": sha256(tst),
                "n_training_recordings": int(len(ids)), "training_video_ids": ids.tolist(), "fit_seconds": fit_s, "inference_seconds": infer_s, "exact_fused_score_ties": ties,
                "ridge_alpha": {n: float(model.models[n].model.alpha_) for n in names}, "feature_pca_components": {n: int(model.models[n].x_pca.n_components_) for n in names},
                "submission": str(sub_path), "submission_sha256": sha256(sub_path), "checkpoint": str(ckpt), "checkpoint_sha256": sha256(ckpt)}
        if not is_hidden:  # organizer-recommended format check; computed after outputs are fixed
            truth = np.load(Path(a.data_root) / "tst_stim" / f"sub-{subject}_tst_stim.npy").astype(np.int64)
            meta["format_check_metrics"] = retrieval_metrics(rankings, truth)
        subjects_meta[subject] = meta
        print(subject, {k: v for k, v in meta.items() if k != "training_video_ids"}, flush=True)

    root = {}
    for subject in a.subjects:
        r = Path(f"sub-{subject}_ranked_candidate_list.npy")
        if r.exists():
            old, new = np.load(r), np.load(out / r.name)
            root[subject] = {"root_path": str(r), "root_sha256": sha256(r), "new_sha256": sha256(out / r.name), "identical": bool(np.array_equal(old, new)),
                             "top1_agreement": float(np.mean(old[:, 0] == new[:, 0])),
                             "mean_top5_overlap": float(np.mean([len(set(o[:5]) & set(n[:5])) for o, n in zip(old, new)]))}
    metadata = {"config_id": config["config_id"], "config_path": a.config, "config_sha256": sha256(a.config), "weights": dict(zip(names, weights)),
                "command": " ".join(shlex.quote(x) for x in [".venv/bin/python", *sys.argv]), "seeds": {"note": "RidgeCV and full-SVD PCA are deterministic; ForwardRegressor random_state=2026 applies only to unused target-PCA paths"},
                "features": feature_record, "candidate_ids_sha256": sha256(a.candidate_ids), "versions": versions(), "subjects": subjects_meta,
                "comparison_with_root_submissions": root, "total_seconds": time.perf_counter() - t0}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=1))
    files = sorted(x for x in out.rglob("*") if x.is_file() and x.name != "MANIFEST.sha256")
    (out / "MANIFEST.sha256").write_text("".join(f"{sha256(f)}  {f.relative_to(out)}\n" for f in files))
    print(json.dumps(root, indent=1)); print(f"done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()

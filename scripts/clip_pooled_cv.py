#!/usr/bin/env python3
"""Pooled CLIP forward regression with training-fitted subject normalization."""
from __future__ import annotations
import argparse, csv, sys, time
from pathlib import Path
import numpy as np
from scipy.signal import decimate
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.controlled_cv import shared_holdout_ids
from src.data import load_training_subject
from src.evaluation import retrieval_metrics
from src.retrieval import organizer_eeg_correlation_scores, rank_candidates


def downsample(eeg: np.ndarray) -> np.ndarray:
    return decimate(eeg, 5, axis=-1, ftype="fir", zero_phase=True).astype(np.float32)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", default="features/all_train_clip_features.npy"); p.add_argument("--data-root", default="eeg-mind-reading-challenge")
    p.add_argument("--subjects", nargs="+", default=["01","02","03","04","05","06"]); p.add_argument("--seeds", nargs="+", type=int, default=[2026,2027,2028])
    p.add_argument("--holdout-size", type=int, default=102); p.add_argument("--label", required=True); p.add_argument("--output", required=True); p.add_argument("--predictions-dir", required=True)
    a=p.parse_args(); feature=np.load(a.features, mmap_mode="r"); root=Path(a.data_root)
    loaded={s:load_training_subject(root/"trn"/f"sub-{s}_trn.npz") for s in a.subjects}; ids_by_subject={s:x[0] for s,x in loaded.items()}
    rows=[]; out=Path(a.predictions_dir); out.mkdir(parents=True,exist_ok=True)
    for seed in a.seeds:
        heldout=shared_holdout_ids(ids_by_subject,seed=seed,size=a.holdout_size); heldset=set(heldout.tolist()); x_train=[]; y_train=[]; observed={}
        for subject,(ids,eeg) in loaded.items():
            held=np.fromiter((i in heldset for i in ids),dtype=bool,count=len(ids)); train=~held; train_eeg=downsample(eeg[train]); mean=train_eeg.mean(axis=(0,2),keepdims=True); std=np.maximum(train_eeg.std(axis=(0,2),keepdims=True),1e-6)
            x_train.append(feature[ids[train]-1]); y_train.append(((train_eeg-mean)/std).reshape(train.sum(),-1)); observed[subject]=((downsample(eeg[held])-mean)/std)
        started=time.perf_counter(); sx=StandardScaler(); pca=PCA(n_components=.95,svd_solver="full"); sx2=StandardScaler(); X=sx2.fit_transform(pca.fit_transform(sx.fit_transform(np.concatenate(x_train))))
        model=RidgeCV(alphas=np.logspace(-3,6,20)).fit(X,np.concatenate(y_train)); candidate=model.predict(sx2.transform(pca.transform(sx.transform(feature[heldout-1])))).reshape(len(heldout),17,300)
        elapsed_per_subject=(time.perf_counter()-started)/len(loaded)
        for subject,(ids,_) in loaded.items():
            scores=organizer_eeg_correlation_scores(observed[subject],candidate,start=5,stop=60); rankings=rank_candidates(scores,heldout); held=np.fromiter((i in heldset for i in ids),dtype=bool,count=len(ids)); metric=retrieval_metrics(rankings,ids[held])
            artifact=out/f"{a.label}_seed{seed}_sub{subject}.npz"; np.savez_compressed(artifact,heldout_ids=heldout,target_ids=ids[held],scores=scores,rankings=rankings)
            row={"experiment":a.label,"seed":seed,"subject":subject,"n_train_total":len(X),"top1":metric["top1"],"top5":metric["top5"],"seconds":elapsed_per_subject,"artifact":str(artifact)}; rows.append(row); print(row,flush=True)
    with Path(a.output).open("w",newline="") as h: w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print({"mean_top1":float(np.mean([r["top1"] for r in rows])),"mean_top5":float(np.mean([r["top5"] for r in rows]))})
if __name__=="__main__":main()

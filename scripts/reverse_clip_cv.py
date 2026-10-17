#!/usr/bin/env python3
"""Leakage-safe subject-specific EEG-to-CLIP Ridge decoding on shared ID folds."""
from __future__ import annotations
import argparse,csv,sys,time
from pathlib import Path
import numpy as np
from scipy.signal import decimate
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.controlled_cv import shared_holdout_ids,millisecond_slice
from src.data import load_training_subject
from src.evaluation import retrieval_metrics
from src.retrieval import rank_candidates

def main()->None:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--features",default="features/all_train_clip_features.npy");p.add_argument("--data-root",default="eeg-mind-reading-challenge");p.add_argument("--subjects",nargs="+",default=["01","02","03","04","05","06"]);p.add_argument("--seeds",nargs="+",type=int,default=[2026,2027,2028]);p.add_argument("--holdout-size",type=int,default=102);p.add_argument("--eeg-window-ms",nargs=2,type=int,default=[50,600]);p.add_argument("--label",required=True);p.add_argument("--output",required=True);p.add_argument("--predictions-dir",required=True);a=p.parse_args()
 feature=np.load(a.features,mmap_mode="r");root=Path(a.data_root);loaded={s:load_training_subject(root/"trn"/f"sub-{s}_trn.npz") for s in a.subjects};id_map={s:v[0] for s,v in loaded.items()};dest=Path(a.predictions_dir);dest.mkdir(parents=True,exist_ok=True);rows=[]
 for seed in a.seeds:
  heldout=shared_holdout_ids(id_map,seed=seed,size=a.holdout_size);heldset=set(heldout.tolist());candidates=feature[heldout-1];candidates=candidates/np.maximum(np.linalg.norm(candidates,axis=1,keepdims=True),1e-12)
  for subject,(ids,eeg) in loaded.items():
   held=np.fromiter((i in heldset for i in ids),dtype=bool,count=len(ids));start=time.perf_counter();down=decimate(eeg,5,axis=-1,ftype="fir",zero_phase=True);window=millisecond_slice(representation_start_ms=0,representation_stop_ms=3000,sampling_hz=100,window_ms=tuple(a.eeg_window_ms));x_train=down[~held,:,window].reshape((~held).sum(),-1);x_test=down[held,:,window].reshape(held.sum(),-1)
   xs=StandardScaler();xp=PCA(n_components=.95,svd_solver="full");xs2=StandardScaler();x_train=xs2.fit_transform(xp.fit_transform(xs.fit_transform(x_train)));x_test=xs2.transform(xp.transform(xs.transform(x_test)));ys=StandardScaler();model=RidgeCV(alphas=np.logspace(-3,6,20)).fit(x_train,ys.fit_transform(feature[ids[~held]-1]));pred=ys.inverse_transform(model.predict(x_test));pred=pred/np.maximum(np.linalg.norm(pred,axis=1,keepdims=True),1e-12);scores=(pred@candidates.T).astype(np.float32);rankings=rank_candidates(scores,heldout);metric=retrieval_metrics(rankings,ids[held]);artifact=dest/f"{a.label}_seed{seed}_sub{subject}.npz";np.savez_compressed(artifact,heldout_ids=heldout,target_ids=ids[held],scores=scores,rankings=rankings)
   row={"experiment":a.label,"seed":seed,"subject":subject,"eeg_window_ms":f"{a.eeg_window_ms[0]}:{a.eeg_window_ms[1]}","top1":metric["top1"],"top5":metric["top5"],"seconds":time.perf_counter()-start,"artifact":str(artifact)};rows.append(row);print(row,flush=True)
 with Path(a.output).open("w",newline="") as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print({"mean_top1":float(np.mean([r["top1"] for r in rows])),"mean_top5":float(np.mean([r["top5"] for r in rows]))})
if __name__=="__main__":main()

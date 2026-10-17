#!/usr/bin/env python3
"""Create a precommitted shared-ID confirmation holdout unused by prior seeds."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.controlled_cv import shared_holdout_ids
from src.data import load_training_subject

def main()->None:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data-root",default="eeg-mind-reading-challenge");p.add_argument("--prior-seeds",nargs="+",type=int,default=[2026,2027,2028]);p.add_argument("--size",type=int,default=102);p.add_argument("--seed",type=int,default=2030);p.add_argument("--output",required=True);p.add_argument("--exclude-ids",nargs="*",default=[],help="Additional saved holdout .npy files whose IDs must not be reused");a=p.parse_args();root=Path(a.data_root);subject_ids={f"{i:02d}":load_training_subject(root/'trn'/f'sub-{i:02d}_trn.npz')[0] for i in range(1,7)}
 if Path(a.output).exists():raise FileExistsError(f"refusing to overwrite precommitted holdout {a.output}")
 common=set(next(iter(subject_ids.values())).tolist());[common.intersection_update(ids.tolist()) for ids in subject_ids.values()];used=set();[used.update(shared_holdout_ids(subject_ids,seed=s,size=a.size).tolist()) for s in a.prior_seeds];[used.update(np.load(f,allow_pickle=False).astype(int).tolist()) for f in a.exclude_ids];available=np.array(sorted(common-used),dtype=np.int64)
 if len(available)<a.size:raise ValueError('not enough untouched shared IDs')
 chosen=np.sort(np.random.default_rng(a.seed).choice(available,a.size,replace=False));Path(a.output).parent.mkdir(parents=True,exist_ok=True);np.save(a.output,chosen,allow_pickle=False);print({'output':a.output,'shared':len(common),'prior_used':len(used),'remaining':len(available),'holdout':len(chosen),'first_ids':chosen[:5].tolist()})
if __name__=='__main__':main()

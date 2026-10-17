#!/usr/bin/env python3
"""Create a fixed-weight, leakage-free fusion from aligned saved CV score matrices."""
from __future__ import annotations
import argparse,csv,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation import retrieval_metrics
from src.retrieval import rank_candidates,weighted_score_ensemble

def main()->None:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--artifacts-dir",default="experiments/paired_predictions");p.add_argument("--labels",nargs="+",required=True);p.add_argument("--csvs",nargs="+",required=True);p.add_argument("--weights",nargs="+",type=float,required=True);p.add_argument("--label",required=True);p.add_argument("--output",required=True);p.add_argument("--predictions-dir",required=True);a=p.parse_args();dest=Path(a.predictions_dir);dest.mkdir(parents=True,exist_ok=True);rows=[];source=Path(a.artifacts_dir)
 if len(a.labels)<2 or len(a.labels)!=len(a.csvs) or len(a.labels)!=len(a.weights):raise ValueError('labels, csvs, and weights must have the same length >= 2')
 runtime={}
 for path in a.csvs:
  with open(path,newline='') as h:
   for row in csv.DictReader(h):runtime[(row['seed'],row['subject'])]=runtime.get((row['seed'],row['subject']),0.0)+float(row['seconds'])
 for left in sorted(source.glob(f"{a.labels[0]}_seed*_sub*.npz")):
  suffix=left.name.split("_seed",1)[1]
  arrays=[np.load(left)]
  for label in a.labels[1:]:
   candidates=list(source.glob(f"{label}_seed{suffix}"))
   if len(candidates)!=1:raise FileNotFoundError(f"expected one artifact for {label}/{suffix}")
   arrays.append(np.load(candidates[0]))
  if any(not np.array_equal(arrays[0]['heldout_ids'],x['heldout_ids']) or not np.array_equal(arrays[0]['target_ids'],x['target_ids']) for x in arrays[1:]):raise ValueError('unaligned fold artifacts')
  scores=weighted_score_ensemble([x['scores'] for x in arrays],np.array(a.weights));rankings=rank_candidates(scores,arrays[0]['heldout_ids']);metric=retrieval_metrics(rankings,arrays[0]['target_ids']);seed=suffix.split('_sub')[0];subject=suffix.rsplit('_sub',1)[1].removesuffix('.npz');artifact=dest/f"{a.label}_seed{seed}_sub{subject}.npz";np.savez_compressed(artifact,heldout_ids=arrays[0]['heldout_ids'],target_ids=arrays[0]['target_ids'],scores=scores,rankings=rankings);row={"experiment":a.label,"seed":seed,"subject":subject,"top1":metric['top1'],"top5":metric['top5'],"seconds":runtime[(seed,subject)],"artifact":str(artifact)};rows.append(row);print(row)
 with Path(a.output).open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print({'mean_top1':float(np.mean([r['top1'] for r in rows])),'mean_top5':float(np.mean([r['top5'] for r in rows]))})
if __name__=='__main__':main()

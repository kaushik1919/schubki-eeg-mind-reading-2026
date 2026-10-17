#!/usr/bin/env python3
"""Cache frozen ViT-B/32 CLS embeddings from predefined CLIP transformer blocks."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
import open_clip
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.extract_clip_features import sample_frames,video_path

BLOCKS=(3,7,11)  # zero-indexed blocks: early, middle, final
def main()->None:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--video-dir",required=True);p.add_argument("--ids",required=True);p.add_argument("--output-prefix",required=True);p.add_argument("--batch-size",type=int,default=16);p.add_argument("--device",choices=("auto","mps","cpu"),default="auto");a=p.parse_args();ids=np.load(a.ids,allow_pickle=False).astype(int);device="mps" if a.device=="auto" and torch.backends.mps.is_available() else ("cpu" if a.device=="auto" else a.device)
 model,_,preprocess=open_clip.create_model_and_transforms("ViT-B-32",pretrained="openai",force_quick_gelu=True,device=device);model.eval();results=[[] for _ in BLOCKS]
 with torch.inference_mode():
  for n,sid in enumerate(ids,1):
   images=[preprocess(image) for image in sample_frames(video_path(Path(a.video_dir),int(sid)),8)]; chunks=[[] for _ in BLOCKS]
   for start in range(0,len(images),a.batch_size):
    output=model.visual.forward_intermediates(torch.stack(images[start:start+a.batch_size]).to(device),indices=list(BLOCKS),normalize_intermediates=True,intermediates_only=True,output_fmt="NLC",output_extra_tokens=True)
    for i,prefix in enumerate(output["image_intermediates_prefix"]): chunks[i].append(prefix[:,0].float().cpu())
   for i in range(len(BLOCKS)):results[i].append(torch.cat(chunks[i]).mean(0).numpy().astype(np.float32))
   if n%25==0 or n==len(ids):print(f"{n}/{len(ids)}",flush=True)
 for block,rows in zip(BLOCKS,results):np.save(f"{a.output_prefix}_block{block+1}.npy",np.stack(rows),allow_pickle=False);print(block+1,np.stack(rows).shape)
if __name__=="__main__":main()

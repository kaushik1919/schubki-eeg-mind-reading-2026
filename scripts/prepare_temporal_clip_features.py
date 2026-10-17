#!/usr/bin/env python3
"""Turn chronological CLIP frames into mean or two-segment ordered features."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--frames",required=True);p.add_argument("--mode",choices=["mean","two_segments"],required=True);p.add_argument("--output",required=True);a=p.parse_args()
    x=np.load(a.frames,mmap_mode="r")
    if x.ndim!=3 or x.shape[1]!=8: raise ValueError("expected (n, 8, embedding) chronological frame cache")
    result=x.mean(1) if a.mode=="mean" else np.concatenate([x[:,:4].mean(1),x[:,4:].mean(1)],axis=1)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);np.save(a.output,result.astype(np.float32),allow_pickle=False)
    print(result.shape)
if __name__=="__main__":main()

#!/usr/bin/env python3
"""Cache simple 3-second mono acoustic-envelope and log-mel features from MP4s."""
from __future__ import annotations
import argparse, subprocess
from pathlib import Path
import imageio_ffmpeg
import numpy as np
from scipy.signal import stft

SAMPLE_RATE=16000

def video_path(directory: Path, sid: int) -> Path:
    for candidate in (directory/f"{sid:04d}.mp4",directory/f"{sid}.mp4"):
        if candidate.exists(): return candidate
    raise FileNotFoundError(sid)

def audio(path: Path) -> np.ndarray:
    command=[imageio_ffmpeg.get_ffmpeg_exe(),"-v","error","-i",str(path),"-vn","-ac","1","-ar",str(SAMPLE_RATE),"-f","f32le","pipe:1"]
    run=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False); x=np.frombuffer(run.stdout,dtype=np.float32); target=3*SAMPLE_RATE
    return np.pad(x[:target],(0,max(0,target-len(x))),mode="constant") if run.returncode==0 and len(x) else np.zeros(target,dtype=np.float32)

def mel_filter(n_fft: int, n_mels: int) -> np.ndarray:
    hz=np.linspace(0,SAMPLE_RATE/2,n_fft//2+1); mel=2595*np.log10(1+hz/700); anchors=np.linspace(mel[0],mel[-1],n_mels+2); bins=np.interp(anchors,mel,np.arange(len(hz)))
    bank=np.zeros((n_mels,len(hz)),dtype=np.float32)
    for i in range(n_mels):
        left,mid,right=bins[i:i+3]; idx=np.arange(len(hz)); bank[i]=np.maximum(0,np.minimum((idx-left)/max(mid-left,1e-6),(right-idx)/max(right-mid,1e-6)))
    return bank

FILTER=mel_filter(512,32)
def features(x: np.ndarray) -> np.ndarray:
    _,_,z=stft(x,fs=SAMPLE_RATE,nperseg=512,noverlap=352,boundary=None); power=(np.abs(z)**2).astype(np.float32); logmel=np.log1p(FILTER@power)
    # 12 ordered RMS envelope bins + mean/std of 32 log-mel bands = 76 dimensions.
    envelope=np.array([np.sqrt(np.mean(chunk**2)+1e-12) for chunk in np.array_split(x,12)],dtype=np.float32)
    return np.concatenate([envelope,logmel.mean(1),logmel.std(1)]).astype(np.float32)

def main()->None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--video-dir",required=True);p.add_argument("--ids",required=True);p.add_argument("--output",required=True);a=p.parse_args();ids=np.load(a.ids,allow_pickle=False).astype(int);directory=Path(a.video_dir);rows=[]
    for n,sid in enumerate(ids,1):
        rows.append(features(audio(video_path(directory,int(sid)))))
        if n%50==0 or n==len(ids):print(f"{n}/{len(ids)}",flush=True)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);np.save(a.output,np.stack(rows),allow_pickle=False);print(np.stack(rows).shape)
if __name__=="__main__":main()

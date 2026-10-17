#!/usr/bin/env python3
"""Extract frozen VideoMAE temporal embeddings from uniformly sampled video frames."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, VideoMAEModel

MODEL_ID = "MCG-NJU/videomae-base-finetuned-kinetics"


def video_path(video_dir: Path, stimulus_id: int) -> Path:
    for candidate in (video_dir / f"{stimulus_id:04d}.mp4", video_dir / f"{stimulus_id}.mp4"):
        if candidate.exists(): return candidate
    raise FileNotFoundError(f"No MP4 for stimulus {stimulus_id} in {video_dir}")


def sample_frames(path: Path, count: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0: raise ValueError(f"Could not read frame count: {path}")
        frames = []
        for frame_number in np.linspace(0, total - 1, count).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
            ok, frame = cap.read()
            if not ok: raise ValueError(f"Could not decode frame {frame_number}: {path}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return frames
    finally:
        cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", required=True); parser.add_argument("--ids", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=16, help="The Kinetics checkpoint expects 16 frames")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    if args.frames != 16: raise ValueError("this VideoMAE checkpoint is configured for 16 frames")
    ids = np.load(args.ids, allow_pickle=False).astype(np.int64)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids): raise ValueError("ids must be a one-dimensional array of unique stimulus IDs")
    device = "mps" if args.device == "auto" and torch.backends.mps.is_available() else ("cpu" if args.device == "auto" else args.device)
    if device == "mps" and not torch.backends.mps.is_available(): raise RuntimeError("MPS was requested but is unavailable")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = VideoMAEModel.from_pretrained(MODEL_ID).to(device).eval()
    vectors = []; video_dir = Path(args.video_dir)
    with torch.inference_mode():
        for position, stimulus_id in enumerate(ids, 1):
            inputs = processor(sample_frames(video_path(video_dir, int(stimulus_id)), args.frames), return_tensors="pt")
            hidden = model(pixel_values=inputs["pixel_values"].to(device)).last_hidden_state
            vectors.append(hidden.mean(dim=1).squeeze(0).float().cpu().numpy())
            if position % 10 == 0 or position == len(ids): print(f"{position}/{len(ids)} videos", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, np.stack(vectors).astype(np.float32), allow_pickle=False)
    print(f"Saved {args.output}: {(len(ids), len(vectors[0]))}; model={MODEL_ID}; device={device}")


if __name__ == "__main__": main()

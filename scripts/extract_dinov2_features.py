#!/usr/bin/env python3
"""Extract mean eight-frame frozen DINOv2 ViT-B/14 embeddings from videos."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import timm
from timm.data import create_transform, resolve_model_data_config
import torch

MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"


def video_path(video_dir: Path, stimulus_id: int) -> Path:
    for candidate in (video_dir / f"{stimulus_id:04d}.mp4", video_dir / f"{stimulus_id}.mp4"):
        if candidate.exists(): return candidate
    raise FileNotFoundError(f"No MP4 for stimulus {stimulus_id} in {video_dir}")


def sample_frames(path: Path, count: int) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0: raise ValueError(f"Could not read frame count: {path}")
        frames = []
        for frame_number in np.linspace(0, total - 1, count).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number)); ok, frame = cap.read()
            if not ok: raise ValueError(f"Could not decode frame {frame_number}: {path}")
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        return frames
    finally:
        cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", required=True); parser.add_argument("--ids", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=8); parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--checkpoint-every", type=int, default=25, help="Persist partial feature matrix every N videos; reruns resume it")
    args = parser.parse_args(); ids = np.load(args.ids, allow_pickle=False).astype(np.int64)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids): raise ValueError("ids must be a one-dimensional array of unique stimulus IDs")
    device = "mps" if args.device == "auto" and torch.backends.mps.is_available() else ("cpu" if args.device == "auto" else args.device)
    if device == "mps" and not torch.backends.mps.is_available(): raise RuntimeError("MPS was requested but is unavailable")
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0).to(device).eval()
    preprocess = create_transform(**resolve_model_data_config(model), is_training=False)
    output = Path(args.output)
    vectors = list(np.load(output, allow_pickle=False)) if output.exists() else []
    if len(vectors) > len(ids): raise ValueError("existing output has more rows than supplied IDs")
    video_dir = Path(args.video_dir)
    with torch.inference_mode():
        for position, stimulus_id in enumerate(ids[len(vectors):], len(vectors) + 1):
            images = [preprocess(image) for image in sample_frames(video_path(video_dir, int(stimulus_id)), args.frames)]
            outputs = []
            for start in range(0, len(images), args.batch_size): outputs.append(model(torch.stack(images[start:start + args.batch_size]).to(device)).float().cpu())
            vectors.append(torch.cat(outputs).mean(dim=0).numpy())
            if position % args.checkpoint_every == 0 or position == len(ids):
                output.parent.mkdir(parents=True, exist_ok=True)
                np.save(output, np.stack(vectors).astype(np.float32), allow_pickle=False)
                print(f"{position}/{len(ids)} videos", flush=True)
    print(f"Saved {output}: {(len(ids), len(vectors[0]))}; model={MODEL_NAME}; device={device}")


if __name__ == "__main__": main()

#!/usr/bin/env python3
"""Extract frozen OpenCLIP image embeddings, optionally retaining frame order."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import open_clip


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
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
            ok, frame = cap.read()
            if not ok: raise ValueError(f"Could not decode frame {frame_number}: {path}")
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        return frames
    finally:
        cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", required=True); parser.add_argument("--ids", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=8); parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--keep-frames", action="store_true", help="save (video, frame, embedding) instead of mean pooling")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    ids = np.load(args.ids, allow_pickle=False).astype(np.int64)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids): raise ValueError("ids must be a one-dimensional array of unique stimulus IDs")
    device = "mps" if args.device == "auto" and torch.backends.mps.is_available() else ("cpu" if args.device == "auto" else args.device)
    if device == "mps" and not torch.backends.mps.is_available(): raise RuntimeError("MPS was requested but is unavailable")
    # The OpenAI ViT-B/32 checkpoint was trained with QuickGELU; make that architecture explicit.
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", force_quick_gelu=True, device=device)
    model.eval(); video_dir = Path(args.video_dir); embeddings = []
    with torch.inference_mode():
        for position, stimulus_id in enumerate(ids, 1):
            images = [preprocess(image) for image in sample_frames(video_path(video_dir, int(stimulus_id)), args.frames)]
            chunks = []
            for start in range(0, len(images), args.batch_size):
                chunks.append(model.encode_image(torch.stack(images[start:start + args.batch_size]).to(device)).float().cpu())
            frame_embeddings = torch.cat(chunks).numpy().astype(np.float32)
            embeddings.append(frame_embeddings if args.keep_frames else frame_embeddings.mean(0))
            if position % 25 == 0 or position == len(ids): print(f"{position}/{len(ids)} videos", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, np.stack(embeddings), allow_pickle=False)
    print(f"Saved {args.output}: {np.stack(embeddings).shape}; model=ViT-B-32/openai; device={device}")


if __name__ == "__main__": main()

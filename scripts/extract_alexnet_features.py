#!/usr/bin/env python3
"""Extract mean eight-frame torchvision AlexNet `features.3` embeddings from videos.

Writes one float32 feature row per supplied ID, in exactly the ID-file order. This is a
self-contained public-model replacement for the organizer notebook's Net2Brain call.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torchvision.models import AlexNet_Weights, alexnet


def video_path(video_dir: Path, stimulus_id: int) -> Path:
    candidates = (video_dir / f"{stimulus_id:04d}.mp4", video_dir / f"{stimulus_id}.mp4")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No MP4 for stimulus {stimulus_id} in {video_dir}")


def sample_frames(path: Path, count: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise ValueError(f"Could not read frame count: {path}")
        frames = []
        for frame_number in np.linspace(0, total - 1, count).astype(int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Could not decode frame {frame_number}: {path}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return frames
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--ids", required=True, help=".npy one-dimensional stimulus-ID array")
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    ids = np.load(args.ids, allow_pickle=False).astype(np.int64)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError("ids must be a one-dimensional array of unique stimulus IDs")
    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    else:
        device = args.device

    weights = AlexNet_Weights.IMAGENET1K_V1
    transform = weights.transforms()
    model = alexnet(weights=weights).features.to(device).eval()
    activation: list[torch.Tensor] = []
    handle = model[3].register_forward_hook(lambda _module, _inputs, output: activation.append(output.detach()))
    video_dir = Path(args.video_dir)
    vectors: list[np.ndarray] = []
    try:
        with torch.inference_mode():
            for position, stimulus_id in enumerate(ids, 1):
                images = [transform(torch.from_numpy(frame).permute(2, 0, 1)) for frame in sample_frames(video_path(video_dir, int(stimulus_id)), args.frames)]
                chunks = []
                for start in range(0, len(images), args.batch_size):
                    activation.clear()
                    model(torch.stack(images[start : start + args.batch_size]).to(device))
                    chunks.append(activation.pop().flatten(1).cpu())
                vectors.append(torch.cat(chunks).mean(0).numpy().astype(np.float32))
                if position % 25 == 0 or position == len(ids):
                    print(f"{position}/{len(ids)} videos", flush=True)
    finally:
        handle.remove()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, np.stack(vectors), allow_pickle=False)
    print(f"Saved {args.output}: {(len(ids), len(vectors[0]))}; device={device}; weights={weights}")


if __name__ == "__main__":
    main()

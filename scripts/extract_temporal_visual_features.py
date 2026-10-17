#!/usr/bin/env python3
"""Extract chronological low-level visual trajectories from MP4 stimuli."""
from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np


def path_for(directory: Path, video_id: int) -> Path:
    path = directory / f"{video_id:04d}.mp4"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def video_features(path: Path, frames: int = 90) -> np.ndarray:
    cap = cv2.VideoCapture(str(path)); result = []; previous = None; previous_hist = None
    while True:
        ok, image = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(cv2.resize(image, (96, 54)), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        luminance, contrast = float(gray.mean()), float(gray.std())
        hist = cv2.calcHist([(gray * 255).astype(np.uint8)], [0], None, [16], [0, 256]).ravel(); hist /= max(hist.sum(), 1.0)
        if previous is None:
            motion = flow = scene = 0.0
        else:
            motion = float(np.mean(np.abs(gray - previous)))
            field = cv2.calcOpticalFlowFarneback(previous, gray, None, 0.5, 2, 15, 2, 5, 1.2, 0)
            flow = float(np.mean(np.linalg.norm(field, axis=2)))
            scene = float(np.abs(hist - previous_hist).sum() / 2.0)
        result.append([luminance, contrast, motion, flow, scene]); previous, previous_hist = gray, hist
    cap.release()
    if len(result) < 2:
        raise ValueError(f"{path} yielded too few decoded frames: {len(result)}")
    # Containers are not uniformly 90 decoded frames.  Keep native timing
    # information through the feature calculation, then interpolate to the
    # common 3-second grid required by the temporal encoder.
    native = np.asarray(result, dtype=np.float32)
    source = np.linspace(0.0, 1.0, len(native)); target = np.linspace(0.0, 1.0, frames)
    return np.stack([np.interp(target, source, native[:, column]) for column in range(native.shape[1])], axis=1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", required=True); parser.add_argument("--ids", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); ids = np.load(args.ids, allow_pickle=False).astype(int); rows = []
    for number, video_id in enumerate(ids, 1):
        rows.append(video_features(path_for(Path(args.video_dir), int(video_id))))
        if number % 50 == 0 or number == len(ids): print(f"{number}/{len(ids)}", flush=True)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); np.save(output, np.stack(rows), allow_pickle=False)


if __name__ == "__main__":
    main()

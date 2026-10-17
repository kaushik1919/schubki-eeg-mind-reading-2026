#!/usr/bin/env python3
"""Extract frozen PANNs CNN14 audio embeddings from video soundtracks."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import imageio_ffmpeg
import numpy as np
import torch
from panns_inference.config import classes_num
from panns_inference.models import Cnn14

SAMPLE_RATE = 32000


def video_path(video_dir: Path, stimulus_id: int) -> Path:
    for candidate in (video_dir / f"{stimulus_id:04d}.mp4", video_dir / f"{stimulus_id}.mp4"):
        if candidate.exists(): return candidate
    raise FileNotFoundError(f"No MP4 for stimulus {stimulus_id} in {video_dir}")


def decode_audio(path: Path) -> np.ndarray:
    """Decode/resample audio. Silent or unreadable audio becomes zeros, not an exception."""
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    audio = np.frombuffer(completed.stdout, dtype=np.float32)
    target_length = SAMPLE_RATE * 3
    if completed.returncode != 0 or len(audio) == 0:
        return np.zeros(target_length, dtype=np.float32)
    return np.pad(audio[:target_length], (0, max(0, target_length - len(audio))), mode="constant")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", required=True); parser.add_argument("--ids", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=str(Path.home() / "panns_data" / "Cnn14_mAP=0.431.pth"))
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    args = parser.parse_args(); ids = np.load(args.ids, allow_pickle=False).astype(np.int64)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids): raise ValueError("ids must be a one-dimensional array of unique stimulus IDs")
    if not Path(args.checkpoint).exists() or Path(args.checkpoint).stat().st_size < 300_000_000:
        raise FileNotFoundError(f"PANNs checkpoint is missing or incomplete: {args.checkpoint}")
    device = "mps" if args.device == "auto" and torch.backends.mps.is_available() else ("cpu" if args.device == "auto" else args.device)
    if device == "mps" and not torch.backends.mps.is_available(): raise RuntimeError("MPS was requested but is unavailable")
    model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=classes_num)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model"])
    model.to(device).eval()
    vectors = []; video_dir = Path(args.video_dir)
    with torch.inference_mode():
        for position, stimulus_id in enumerate(ids, 1):
            waveform = torch.from_numpy(decode_audio(video_path(video_dir, int(stimulus_id))))[None, :].to(device)
            vectors.append(model(waveform, None)["embedding"].squeeze(0).float().cpu().numpy())
            if position % 25 == 0 or position == len(ids): print(f"{position}/{len(ids)} videos", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, np.stack(vectors), allow_pickle=False)
    print(f"Saved {args.output}: {(len(ids), len(vectors[0]))}; model=CNN14; device={device}")


if __name__ == "__main__": main()

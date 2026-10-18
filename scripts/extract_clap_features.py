#!/usr/bin/env python3
"""Extract frozen CLAP audio embeddings (laion/larger_clap_general) for stimulus videos.

Checkpoint: huggingface.co/laion/larger_clap_general, revision ada0c23a36c4e8582805bb38fec3905903f18b41
(created 2023-10-30, Apache-2.0) -- public and released before the hackathon.
Audio: the full soundtrack decoded with ffmpeg to 48 kHz mono (CLAP's native rate).
Missing audio is explicit: files without an audio stream, or whose track is
silent per the temporal-audio audit (RMS < 1e-5), get a zero vector and
has_audio = False; CLAP is not run on them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import imageio_ffmpeg
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
MODEL, REVISION, SR = "laion/larger_clap_general", "ada0c23a36c4e8582805bb38fec3905903f18b41", 48000


def decode(path: Path) -> np.ndarray | None:
    r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(path), "-map", "0:a:0", "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"], capture_output=True)
    x = np.frombuffer(r.stdout, dtype=np.float32)
    return x if r.returncode == 0 and len(x) and np.isfinite(x).all() else None


def main() -> None:
    from transformers import ClapModel, ClapProcessor
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video-dir", required=True); ap.add_argument("--ids", required=True); ap.add_argument("--audit", required=True, help="*_temporal_audio_audit.json")
    ap.add_argument("--output", required=True); ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args(); out = Path(a.output)
    if out.exists(): raise FileExistsError(out)
    ids = np.load(a.ids).astype(np.int64); audit = {r["id"]: r for r in json.loads(Path(a.audit).read_text())}
    model = ClapModel.from_pretrained(MODEL, revision=REVISION).to(a.device).eval(); proc = ClapProcessor.from_pretrained(MODEL, revision=REVISION)
    feats = np.zeros((len(ids), model.config.projection_dim), np.float32); has = np.zeros(len(ids), bool); status = []
    for n, sid in enumerate(ids):
        st = audit[int(sid)]["status"]; status.append(st)
        if st in ("no_audio_stream", "silent", "decode_failed"): continue
        x = decode(Path(a.video_dir) / f"{int(sid):04d}.mp4")
        if x is None: status[-1] = "decode_failed"; continue
        inp = proc(audio=[x], sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            f = model.get_audio_features(**{k: v.to(a.device) for k, v in inp.items()})
        f = getattr(f, "pooler_output", f)
        feats[n] = f.float().cpu().numpy()[0]; has[n] = True
        if (n + 1) % 100 == 0: print(f"{n + 1}/{len(ids)}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, ids=ids, features=feats, has_audio=has, status=np.array(status), model=MODEL, revision=REVISION)
    print({"n": len(ids), "with_audio": int(has.sum()), "dim": feats.shape[1]})


if __name__ == "__main__":
    main()

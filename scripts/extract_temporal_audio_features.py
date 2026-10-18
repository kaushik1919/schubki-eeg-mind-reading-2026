#!/usr/bin/env python3
"""Audit each soundtrack's timeline and extract onset-locked 100 Hz audio features.

Per file: container duration, first video-frame pts and first audio-frame pts
(ffmpeg showinfo/ashowinfo with -copyts), native sample rate, decoded length.
Audio is decoded to 16 kHz mono and placed at (audio_pts - video_pts) on a
buffer whose sample 0 is video onset (= EEG onset per the challenge PDF).
Status codes: ok, no_audio_stream, silent, short_audio, decode_failed.
Failures never raise; they yield deterministic all-silence features.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.temporal_audio import FEATURE_NAMES, N_BINS, SAMPLE_RATE, onset_locked_buffer, temporal_audio_features

EXE = imageio_ffmpeg.get_ffmpeg_exe()
SILENCE_RMS = 1e-5       # below this full-file RMS (~ -100 dBFS) the track is treated as silent
SHORT_S = 2.9            # audio ending before this (relative to onset) is flagged short


def video_path(directory: Path, sid: int) -> Path:
    for p in (directory / f"{sid:04d}.mp4", directory / f"{sid}.mp4"):
        if p.exists(): return p
    raise FileNotFoundError(sid)


def first_pts(path: Path, kind: str) -> float | None:
    flt = ["-af", "ashowinfo"] if kind == "a" else ["-vf", "showinfo"]
    r = subprocess.run([EXE, "-hide_banner", "-copyts", "-i", str(path), "-map", f"0:{kind}:0", *flt, f"-frames:{kind}", "1", "-f", "null", "-"], capture_output=True, text=True, errors="replace")
    m = re.search(r"pts_time:\s*([-\d.e]+)", r.stderr)
    return float(m.group(1)) if (r.returncode == 0 and m) else None


def audit_and_extract(path: Path) -> tuple[np.ndarray, dict]:
    info = subprocess.run([EXE, "-hide_banner", "-i", str(path)], capture_output=True, text=True, errors="replace").stderr
    dur = re.search(r"Duration: (\d+):(\d+):([\d.]+)", info)
    rec = {"container_s": (int(dur.group(1)) * 3600 + int(dur.group(2)) * 60 + float(dur.group(3))) if dur else None}
    a = re.search(r"Stream #\S+.*Audio: (\w+).*?(\d+) Hz", info)
    rec["video_pts0"] = first_pts(path, "v")
    rec["video_decode_ok"] = rec["video_pts0"] is not None
    if not a:
        rec.update(status="no_audio_stream", codec=None, native_hz=None, audio_pts0=None, offset_s=0.0, decoded_s=0.0, rms=0.0)
        return np.zeros((N_BINS, len(FEATURE_NAMES)), np.float32), rec
    rec.update(codec=a.group(1), native_hz=int(a.group(2)), audio_pts0=first_pts(path, "a"))
    d = subprocess.run([EXE, "-v", "error", "-i", str(path), "-map", "0:a:0", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"], capture_output=True)
    x = np.frombuffer(d.stdout, dtype=np.float32)
    if d.returncode != 0 or len(x) == 0 or not np.isfinite(x).all():
        rec.update(status="decode_failed", offset_s=0.0, decoded_s=float(len(x)) / SAMPLE_RATE, rms=0.0)
        return np.zeros((N_BINS, len(FEATURE_NAMES)), np.float32), rec
    v0 = rec["video_pts0"] if rec["video_pts0"] is not None else 0.0
    offset = (rec["audio_pts0"] if rec["audio_pts0"] is not None else 0.0) - v0
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    rec.update(offset_s=offset, decoded_s=len(x) / SAMPLE_RATE, rms=rms)
    rec["status"] = "silent" if rms < SILENCE_RMS else ("short_audio" if offset + len(x) / SAMPLE_RATE < SHORT_S else "ok")
    feats = temporal_audio_features(onset_locked_buffer(x, offset))
    return feats, rec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video-dir", required=True); p.add_argument("--ids", required=True); p.add_argument("--output", required=True)
    a = p.parse_args(); out = Path(a.output)
    if out.exists(): raise FileExistsError(out)
    ids = np.load(a.ids, allow_pickle=False).astype(np.int64)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids): raise ValueError("ids must be unique")
    feats, recs = [], []
    for n, sid in enumerate(ids, 1):
        f, r = audit_and_extract(video_path(Path(a.video_dir), int(sid))); r["id"] = int(sid); feats.append(f); recs.append(r)
        if n % 100 == 0 or n == len(ids): print(f"{n}/{len(ids)}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, ids=ids, features=np.stack(feats), feature_names=np.array(FEATURE_NAMES), status=np.array([r["status"] for r in recs]), offset_s=np.array([r["offset_s"] for r in recs]))
    Path(str(out).replace(".npz", "_audit.json")).write_text(json.dumps(recs, indent=0))
    from collections import Counter
    print(Counter(r["status"] for r in recs), "offsets:", sorted({round(r["offset_s"], 4) for r in recs})[:10], "video_pts0:", sorted({r["video_pts0"] for r in recs}, key=str)[:5])


if __name__ == "__main__":
    main()

"""Time-resolved audio features on the 100 Hz EEG grid (pure NumPy, deterministic).

Time convention: EEG sample k (after FIR decimation 500 -> 100 Hz) is at
t_k = k * 10 ms after video onset.  Audio feature bin k is computed from a
25 ms Hann window centred on t_k.  Audio is placed on an onset-locked
buffer using its measured stream offset; anything before the first audio
sample or after the last one is silence (zeros).  Every feature equals 0 for
silence, so zero lag-padding before onset also means silence.
"""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
GRID_HZ = 100
N_BINS = 300                      # 0..2990 ms, matches the 17 x 300 EEG target
WIN = 400                         # 25 ms
HOP = SAMPLE_RATE // GRID_HZ      # 160 samples = 10 ms
N_MELS = 32
LOGMEL_SCALE = 1e4                # log-mel = log(1 + scale * mel power); silence -> 0
ENV_COMPRESSION = 0.3             # power-law compression of RMS envelope
FEATURE_NAMES = ("envelope", "envelope_derivative_pos", "onset_strength", "spectral_flux") + tuple(f"logmel_{i:02d}" for i in range(N_MELS))
GROUPS = {"A_envelope_onset": list(range(4)), "B_logmel": list(range(4, 4 + N_MELS)), "C_combined": list(range(4 + N_MELS))}


def onset_locked_buffer(waveform: np.ndarray, offset_s: float, n_bins: int = N_BINS) -> np.ndarray:
    """Place mono audio on a buffer whose sample 0 is video onset; pad/truncate with silence."""
    x = np.asarray(waveform, dtype=np.float32).ravel()
    total = n_bins * HOP + WIN  # enough for the last centred window
    out = np.zeros(total, dtype=np.float32)
    shift = int(round(offset_s * SAMPLE_RATE))
    src_start, dst_start = max(0, -shift), max(0, shift)
    n = max(0, min(len(x) - src_start, total - dst_start))
    if n:
        out[dst_start:dst_start + n] = x[src_start:src_start + n]
    return out


def mel_filterbank(n_fft: int = 512, n_mels: int = N_MELS, fmin: float = 30.0, fmax: float = 8000.0) -> np.ndarray:
    hz = np.linspace(0, SAMPLE_RATE / 2, n_fft // 2 + 1)
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    inv = lambda m: 700 * (10 ** (m / 2595) - 1)
    edges = inv(np.linspace(mel(fmin), mel(fmax), n_mels + 2))
    bank = np.zeros((n_mels, len(hz)), dtype=np.float64)
    for i in range(n_mels):
        lo, mid, hi = edges[i:i + 3]
        bank[i] = np.clip(np.minimum((hz - lo) / (mid - lo), (hi - hz) / (hi - mid)), 0, None)
    return bank


_BANK = mel_filterbank()
_HANN = np.hanning(WIN)


def frames(buffer: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """(n_bins, WIN) windows centred on t_k = k * 10 ms; samples before onset are silence."""
    padded = np.concatenate([np.zeros(WIN // 2, dtype=np.float32), buffer])
    idx = np.arange(n_bins)[:, None] * HOP + np.arange(WIN)[None, :]
    return padded[idx]


def temporal_audio_features(buffer: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Return (n_bins, 36) features in FEATURE_NAMES order for one onset-locked buffer."""
    f = frames(buffer, n_bins).astype(np.float64)
    rms = np.sqrt(np.mean(f ** 2, axis=1))
    env = rms ** ENV_COMPRESSION
    env_d = np.maximum(0.0, np.diff(env, prepend=0.0))                       # silence before onset
    mag = np.abs(np.fft.rfft(f * _HANN, n=512, axis=1))
    logmel = np.log1p(LOGMEL_SCALE * (mag ** 2) @ _BANK.T)
    onset = np.maximum(0.0, np.diff(logmel, axis=0, prepend=0.0)).mean(axis=1)
    flux = np.sqrt((np.maximum(0.0, np.diff(mag, axis=0, prepend=0.0)) ** 2).sum(axis=1))
    return np.column_stack([env, env_d, onset, flux, logmel]).astype(np.float32)

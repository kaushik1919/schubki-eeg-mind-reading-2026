"""Multimodal stimulus encoder: per-modality projections + small fusion MLP -> shared embedding.

Inputs are frozen, cached per-video features only (no video IDs, positions or
indices), so the head cannot learn an ID or dataset-order shortcut: two videos
with identical features always get identical embeddings.

Feature matrix layout: blocks concatenated column-wise as described by
`ModalitySpec`; each block is L2-normalised per row (label-free, no fitted
parameters).  Missing audio is encoded by an explicit 0/1 mask column; masked
audio blocks are replaced by a single learned "no audio" embedding instead of
being projected from the zero vector.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

FEATURE_FILES = {
    "clip_block8": "features/all_train_clip_intermediate_block8.npy",
    "dinov2s": "features/all_train_dinov2s_features.npy",
    "dinov2b": "features/all_train_dinov2b_features.npy",
    "clap": "features/all_train_clap_features.npz",
}
TEST_FEATURE_FILES = {
    "clip_block8": "features/test_clip_intermediate_block8.npy",
    "dinov2s": "features/test_dinov2s_features.npy",
    "dinov2b": "features/test_dinov2b_features.npy",
    "clap": "features/test_clap_features.npz",
}
AUDIO_MODALITIES = {"clap"}


@dataclass(frozen=True)
class ModalitySpec:
    names: tuple[str, ...]
    dims: tuple[int, ...]

    @property
    def mask_col(self) -> int:          # final column = has_audio (always present for a fixed layout)
        return sum(self.dims)

    def slices(self) -> list[slice]:
        out, start = [], 0
        for d in self.dims:
            out.append(slice(start, start + d)); start += d
        return out


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return np.where(n > 0, x / np.maximum(n, 1e-12), 0.0).astype(np.float32)


def load_modality(name: str, *, test: bool = False, root: Path = Path(".")) -> tuple[np.ndarray, np.ndarray]:
    """Return (features, has_signal) with rows aligned to IDs 1..1000 (train) or 1001..1102 (test)."""
    path = root / (TEST_FEATURE_FILES if test else FEATURE_FILES)[name]
    expected = np.arange(1001, 1103) if test else np.arange(1, 1001)
    if path.suffix == ".npz":
        z = np.load(path)
        if not np.array_equal(z["ids"], expected): raise ValueError(f"{path} rows are not ID-aligned")
        return z["features"].astype(np.float32), z["has_audio"].astype(bool)
    x = np.load(path).astype(np.float32)
    if len(x) != len(expected): raise ValueError(f"{path} has {len(x)} rows, expected {len(expected)}")
    return x, np.ones(len(x), bool)


def assemble(names: list[str] | tuple[str, ...], *, test: bool = False, root: Path = Path(".")) -> tuple[np.ndarray, ModalitySpec]:
    """Concatenate per-row L2-normalised modality blocks plus a has_audio column."""
    blocks, dims, has_audio = [], [], None
    for n in names:
        x, ok = load_modality(n, test=test, root=root)
        blocks.append(_l2(x) * ok[:, None]); dims.append(x.shape[1])
        if n in AUDIO_MODALITIES: has_audio = ok
    if has_audio is None:
        has_audio = np.ones(len(blocks[0]), bool)
    mat = np.concatenate(blocks + [has_audio[:, None].astype(np.float32)], axis=1)
    return mat, ModalitySpec(tuple(names), tuple(dims))


class MultimodalStimulusHead(nn.Module):
    """LayerNorm + Linear projection per modality -> concat -> MLP -> L2-normalised embedding."""

    def __init__(self, spec: ModalitySpec, embed_dim: int = 128, proj_dim: int = 128, hidden: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.spec = spec
        self.proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, proj_dim)) for d in spec.dims])
        self.missing = nn.ParameterDict({n: nn.Parameter(torch.zeros(proj_dim)) for n in spec.names if n in AUDIO_MODALITIES})
        width = proj_dim * len(spec.dims)
        self.fuse = nn.Sequential(nn.GELU(), nn.Dropout(dropout), nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        has_audio = x[:, self.spec.mask_col:self.spec.mask_col + 1]
        parts = []
        for name, sl, proj in zip(self.spec.names, self.spec.slices(), self.proj):
            z = proj(x[:, sl])
            if name in self.missing:
                z = has_audio * z + (1 - has_audio) * self.missing[name]
            parts.append(z)
        return F.normalize(self.fuse(torch.cat(parts, dim=1)), dim=-1)

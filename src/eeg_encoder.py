"""Compact EEGNet-style encoder with optional per-subject spatial adapters."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SubjectAdapter(nn.Module):
    """Per-subject 17x17 channel mixing (identity init) plus channel bias."""

    def __init__(self, n_subjects: int, n_channels: int = 17) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.eye(n_channels).repeat(n_subjects, 1, 1))
        self.bias = nn.Parameter(torch.zeros(n_subjects, n_channels, 1))

    def forward(self, x: torch.Tensor, subject: torch.Tensor) -> torch.Tensor:  # x: (B, C, T)
        return torch.bmm(self.weight[subject], x) + self.bias[subject]


class EEGNetEncoder(nn.Module):
    """Input (B, 17, T) at 100 Hz -> L2-normalised (B, embed_dim)."""

    def __init__(self, n_channels: int = 17, n_samples: int = 300, f1: int = 16, depth: int = 2, f2: int = 32,
                 kernel: int = 25, dropout: float = 0.5, embed_dim: int = 128, n_subjects: int | None = None) -> None:
        super().__init__()
        self.adapter = SubjectAdapter(n_subjects, n_channels) if n_subjects else None
        self.temporal = nn.Sequential(nn.Conv2d(1, f1, (1, kernel), padding=(0, kernel // 2), bias=False), nn.BatchNorm2d(f1))
        self.spatial = nn.Sequential(nn.Conv2d(f1, f1 * depth, (n_channels, 1), groups=f1, bias=False), nn.BatchNorm2d(f1 * depth), nn.ELU(),
                                     nn.AvgPool2d((1, 4)), nn.Dropout(dropout))
        self.separable = nn.Sequential(nn.Conv2d(f1 * depth, f1 * depth, (1, 16), padding=(0, 8), groups=f1 * depth, bias=False),
                                       nn.Conv2d(f1 * depth, f2, 1, bias=False), nn.BatchNorm2d(f2), nn.ELU(), nn.AvgPool2d((1, 5)), nn.Dropout(dropout))
        with torch.no_grad():
            n_flat = self.separable(self.spatial(self.temporal(torch.zeros(1, 1, n_channels, n_samples)))).numel()
        self.head = nn.Linear(n_flat, embed_dim)

    def forward(self, x: torch.Tensor, subject: torch.Tensor | None = None) -> torch.Tensor:
        if self.adapter is not None:
            if subject is None:
                raise ValueError("subject indices required when adapters are enabled")
            x = self.adapter(x, subject)
        h = self.separable(self.spatial(self.temporal(x.unsqueeze(1))))
        return F.normalize(self.head(h.flatten(1)), dim=-1)


class VideoHead(nn.Module):
    """Learnable linear projection of frozen stimulus embeddings into the shared space."""

    def __init__(self, in_dim: int, embed_dim: int = 128) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, embed_dim)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(v), dim=-1)


def n_parameters(*modules: nn.Module) -> int:
    return sum(p.numel() for m in modules for p in m.parameters())


class ProbeEncoder(nn.Module):
    """Dropout + linear projection of frozen pretrained-EEG features (linear probe)."""

    def __init__(self, in_dim: int, embed_dim: int = 128, dropout: float = 0.25) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, embed_dim))

    def forward(self, x: torch.Tensor, subject: torch.Tensor | None = None) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)

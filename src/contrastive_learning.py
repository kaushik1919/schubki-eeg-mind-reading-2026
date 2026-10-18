"""Contrastive objectives and video-grouped batch sampling for EEG-to-video retrieval."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def naive_infonce(eeg: torch.Tensor, video: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """Symmetric CLIP loss over recording-video pairs; duplicated videos are (wrongly) negatives."""
    logits = logit_scale * eeg @ video.T
    target = torch.arange(len(eeg), device=eeg.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


def _multi_positive_ce(logits: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    """Mean over rows of -mean_{p in P(i)} log softmax(logits_i)[p]; rows without positives are skipped."""
    logp = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    counts = positive.sum(1)
    keep = counts > 0
    # where() instead of a product: masked (-inf) self-similarities would give -inf * 0 = NaN in the value.
    picked = torch.where(positive > 0, logp, torch.zeros_like(logp))
    return -(picked.sum(1)[keep] / counts[keep]).mean()


def multi_positive_loss(eeg: torch.Tensor, video_unique: torch.Tensor, video_index: torch.Tensor, logit_scale: torch.Tensor,
                        subject: torch.Tensor | None = None, eeg_eeg_weight: float = 0.5) -> torch.Tensor:
    """False-negative-free objective.

    eeg (B, d); video_unique (U, d) one row per distinct video in the batch;
    video_index (B,) maps each recording to its row of video_unique.
    Terms: EEG->video CE over distinct videos; video->EEG multi-positive CE;
    EEG<->EEG supervised-contrastive term whose positives are other recordings
    of the same video (from different subjects when `subject` is given).
    """
    B = len(eeg)
    e2v = logit_scale * eeg @ video_unique.T                           # (B, U)
    loss = F.cross_entropy(e2v, video_index)
    pos_v = (video_index[None, :] == torch.arange(len(video_unique), device=eeg.device)[:, None]).float()  # (U, B)
    loss = loss + _multi_positive_ce(e2v.T, pos_v)
    if eeg_eeg_weight > 0:
        same = video_index[:, None] == video_index[None, :]
        eye = torch.eye(B, dtype=torch.bool, device=eeg.device)
        pos = same & ~eye
        if subject is not None:
            pos = pos & (subject[:, None] != subject[None, :])
        e2e = (logit_scale * eeg @ eeg.T).masked_fill(eye, float("-inf"))
        if pos.any():
            loss = loss + eeg_eeg_weight * _multi_positive_ce(e2e, pos.float())
    return loss


class VideoGroupedBatches:
    """Yield batches containing *all* training recordings of a random set of videos.

    Grouping guarantees that same-video recordings from different subjects
    co-occur (needed for multi-positive terms) and never straddle batches.
    """

    def __init__(self, video_ids: np.ndarray, videos_per_batch: int, seed: int) -> None:
        self.video_ids = np.asarray(video_ids)
        self.unique = np.unique(self.video_ids)
        self.by_video = {v: np.where(self.video_ids == v)[0] for v in self.unique}
        self.k, self.rng = videos_per_batch, np.random.default_rng(seed)

    def __iter__(self):
        order = self.rng.permutation(self.unique)
        for s in range(0, len(order), self.k):
            yield np.concatenate([self.by_video[v] for v in order[s:s + self.k]])

    def __len__(self) -> int:
        return int(np.ceil(len(self.unique) / self.k))


def retrieval_positions(eeg_emb: np.ndarray, cand_emb: np.ndarray, cand_ids: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    scores = eeg_emb @ cand_emb.T
    target_col = np.searchsorted(cand_ids, target_ids)
    return (scores > scores[np.arange(len(scores)), target_col][:, None]).sum(1)

"""Synthetic tests for the EEG encoder, contrastive losses and grouped batching."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.contrastive_learning import VideoGroupedBatches, multi_positive_loss, naive_infonce, retrieval_positions
from src.eeg_encoder import EEGNetEncoder, SubjectAdapter, VideoHead, n_parameters


def test_encoder_shapes_normalisation_and_size():
    enc = EEGNetEncoder(n_subjects=6).eval()
    z = enc(torch.randn(4, 17, 300), torch.tensor([0, 1, 2, 5]))
    assert z.shape == (4, 128) and torch.allclose(z.norm(dim=1), torch.ones(4), atol=1e-5)
    assert n_parameters(enc) < 100_000
    with pytest.raises(ValueError): enc(torch.randn(2, 17, 300))


def test_adapter_starts_as_identity_and_is_subject_specific():
    ad = SubjectAdapter(3)
    x = torch.randn(2, 17, 10)
    assert torch.allclose(ad(x, torch.tensor([0, 2])), x)
    with torch.no_grad(): ad.weight[1].mul_(2)
    assert torch.allclose(ad(x, torch.tensor([1, 0]))[0], 2 * x[0]) and torch.allclose(ad(x, torch.tensor([1, 0]))[1], x[1])


def test_multi_positive_loss_has_no_false_negatives():
    """Two recordings of the same video: dedup loss is minimal when both match it, naive loss penalises them."""
    v = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=1)
    eeg = F.normalize(torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), dim=1)
    idx = torch.tensor([0, 0, 1]); scale = torch.tensor(20.0)
    moved = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]), dim=1)  # recording 1 pushed off its video
    mp = multi_positive_loss(eeg, v, idx, scale, eeg_eeg_weight=0.0)
    assert mp < multi_positive_loss(moved, v, idx, scale, eeg_eeg_weight=0.0)
    e2v_only = F.cross_entropy(scale * eeg @ v.T, idx)
    naive_e2v = F.cross_entropy(scale * eeg @ v[idx].T, torch.arange(3))  # duplicate column counted as a negative
    assert e2v_only < 1e-3 and naive_e2v > 2 / 3 * np.log(2) - 1e-3


def test_eeg_eeg_term_uses_only_cross_subject_same_video_pairs():
    e = F.normalize(torch.randn(4, 8), dim=1); v = F.normalize(torch.randn(2, 8), dim=1)
    idx = torch.tensor([0, 0, 1, 1]); scale = torch.tensor(10.0)
    same_subject = multi_positive_loss(e, v, idx, scale, subject=torch.tensor([0, 0, 1, 1]), eeg_eeg_weight=1.0)
    only_video = multi_positive_loss(e, v, idx, scale, eeg_eeg_weight=0.0)
    assert torch.isclose(same_subject, only_video)  # no cross-subject positives -> term skipped
    cross = multi_positive_loss(e, v, idx, scale, subject=torch.tensor([0, 1, 0, 1]), eeg_eeg_weight=1.0)
    assert not torch.isclose(cross, only_video)


def test_multi_positive_loss_value_and_gradients_are_finite():
    torch.manual_seed(1)
    e = F.normalize(torch.randn(6, 8), dim=1).requires_grad_(); v = F.normalize(torch.randn(3, 8), dim=1)
    loss = multi_positive_loss(e, v, torch.tensor([0, 0, 1, 1, 2, 2]), torch.tensor(14.0), subject=torch.tensor([0, 1, 0, 1, 0, 1]))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(e.grad).all()


def test_video_grouped_batches_keep_videos_whole_and_cover_everything():
    vids = np.array([5, 5, 7, 9, 9, 9, 11, 12])
    batches = list(VideoGroupedBatches(vids, videos_per_batch=2, seed=0))
    seen = np.concatenate(batches); assert sorted(seen) == list(range(len(vids)))
    owner = {}
    for b, idx in enumerate(batches):
        for v in vids[idx]: assert owner.setdefault(v, b) == b


def test_retrieval_positions_uses_explicit_candidate_ids():
    cand_ids = np.array([3, 8, 20]); cand = np.eye(3)
    eeg = np.array([[0, 1.0, 0], [0, 0, 1.0]])
    assert list(retrieval_positions(eeg, cand, cand_ids, np.array([8, 3]))) == [0, 1]


def test_one_training_step_reduces_loss_on_learnable_toy_problem():
    torch.manual_seed(0)
    enc, head = EEGNetEncoder(dropout=0.0), VideoHead(16)
    feats = F.normalize(torch.randn(32, 16), dim=1)
    x = torch.randn(32, 17, 300) + feats[:, :1, None]
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=1e-2); losses = []
    for _ in range(30):
        loss = naive_infonce(enc(x), head(feats), torch.tensor(10.0)); opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]

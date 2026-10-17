"""Positive pairs, leakage of average-response targets, subject mapping and shapes."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.cross_subject_learning import (AverageTargetCoder, RidgeReadout, cross_subject_positive_mask, eeg_eeg_supcon,
                                        loso_average_targets, mean_embedding_per_video)
from src.eeg_encoder import EEGTransformerEncoder


def test_positive_mask_is_same_video_different_subject_only():
    v = torch.tensor([7, 7, 7, 9, 9]); s = torch.tensor([0, 1, 1, 0, 2])
    pos = cross_subject_positive_mask(v, s)
    expect = torch.tensor([[0, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0]], dtype=torch.bool)
    assert torch.equal(pos, expect)


def test_supcon_is_finite_low_for_aligned_pairs_and_skips_anchors_without_positives():
    e = F.normalize(torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.], [1., 1.]]), dim=1)
    v = torch.tensor([0, 0, 1, 1, 2]); s = torch.tensor([0, 1, 0, 1, 0])
    aligned = eeg_eeg_supcon(e, v, s, torch.tensor(20.0))
    swapped = eeg_eeg_supcon(e[[0, 2, 1, 3, 4]], v, s, torch.tensor(20.0))
    assert torch.isfinite(aligned) and aligned < swapped
    assert eeg_eeg_supcon(e, torch.arange(5), s, torch.tensor(20.0)) == 0  # no positives -> zero loss


def test_loso_targets_exclude_own_subject_and_non_source_recordings():
    V = np.array([1, 1, 1, 2, 2]); S = np.array([0, 1, 2, 0, 1])
    X = np.array([[1.], [2.], [100.], [5.], [7.]], np.float32)
    source = np.array([True, True, False, True, True])           # recording 2 is held out / inner validation
    t, ok = loso_average_targets(X, S, V, source)
    assert ok.all()
    np.testing.assert_allclose(t[:, 0], [2.0, 1.0, 1.5, 7.0, 5.0])  # held-out value 100 never contributes
    t2, ok2 = loso_average_targets(X, S, V, np.array([True, False, False, False, False]))
    assert ok2.tolist() == [False, True, True, False, False]       # only-own-subject source -> unavailable


def test_target_coder_and_readout_shapes_are_train_fitted():
    rng = np.random.default_rng(0)
    coder = AverageTargetCoder(8, 0).fit(rng.normal(size=(50, 17, 30)))
    assert coder.transform(rng.normal(size=(4, 17, 30))).shape == (4, 8)
    feats = rng.normal(size=(40, 12)); emb = rng.normal(size=(40, 6))
    out = RidgeReadout().fit(feats, emb).predict(feats[:5])
    assert out.shape == (5, 6) and np.allclose(np.linalg.norm(out, axis=1), 1)


def test_mean_embedding_per_video_groups_by_explicit_ids():
    e = np.array([[1., 0.], [0., 1.], [1., 0.]]); V = np.array([5, 3, 5])
    vids, m = mean_embedding_per_video(e, V)
    assert vids.tolist() == [3, 5] and np.allclose(m, [[0, 1], [1, 0]])


def test_small_transformer_encoder_needs_subjects_and_returns_normalized_embedding():
    encoder = EEGTransformerEncoder(model_dim=32, depth=1, heads=4, embed_dim=16, n_subjects=2, dropout=0.0)
    x, subject = torch.randn(3, 17, 300), torch.tensor([0, 1, 0])
    with pytest.raises(ValueError):
        encoder(x)
    out = encoder(x, subject)
    assert out.shape == (3, 16)
    assert torch.allclose(out.norm(dim=1), torch.ones(3), atol=1e-5)

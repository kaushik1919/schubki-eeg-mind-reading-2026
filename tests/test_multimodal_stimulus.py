"""Feature alignment, missing-audio handling and shortcut/leakage guards for the multimodal head."""
import numpy as np
import pytest
import torch

from src.multimodal_stimulus import AUDIO_MODALITIES, ModalitySpec, MultimodalStimulusHead, _l2, assemble, load_modality


def _matrix(rng, n=6, dims=(5, 4), has_audio=None):
    blocks = [_l2(rng.normal(size=(n, d))) for d in dims]
    ha = np.ones(n, bool) if has_audio is None else has_audio
    blocks[1] = blocks[1] * ha[:, None]
    return np.concatenate(blocks + [ha[:, None].astype(np.float32)], 1).astype(np.float32)


def test_spec_slices_and_mask_column():
    spec = ModalitySpec(("dinov2s", "clap"), (5, 4))
    assert spec.slices() == [slice(0, 5), slice(5, 9)] and spec.mask_col == 9


def test_head_output_is_normalised_and_depends_only_on_features():
    torch.manual_seed(0); rng = np.random.default_rng(0)
    spec = ModalitySpec(("dinov2s", "clap"), (5, 4)); head = MultimodalStimulusHead(spec, 8, 6, 10).eval()
    x = torch.from_numpy(_matrix(rng))
    z = head(x)
    assert z.shape == (6, 8) and torch.allclose(z.norm(dim=1), torch.ones(6), atol=1e-5)
    perm = torch.randperm(6)
    assert torch.allclose(head(x[perm]), z[perm], atol=1e-6)            # no row-position / ordering shortcut
    dup = torch.cat([x[:1], x[:1]])
    assert torch.allclose(head(dup)[0], head(dup)[1])                    # identical features -> identical embedding


def test_missing_audio_uses_learned_token_not_zero_projection():
    torch.manual_seed(0); rng = np.random.default_rng(1)
    spec = ModalitySpec(("dinov2s", "clap"), (5, 4)); head = MultimodalStimulusHead(spec, 8, 6, 10).eval()
    x = torch.from_numpy(_matrix(rng, has_audio=np.array([True, False, True, False, True, True])))
    with torch.no_grad(): head.missing["clap"].fill_(3.0)
    y = x.clone(); y[1, 5:9] = torch.randn(4)                             # garbage in a masked audio block is ignored
    assert torch.allclose(head(x)[1], head(y)[1], atol=1e-6)
    assert "clap" in AUDIO_MODALITIES and "dinov2s" not in head.missing


def test_cached_features_are_id_aligned_and_missing_audio_is_explicit():
    try:
        x, ok = load_modality("clap", test=True); d, _ = load_modality("dinov2s", test=True)
    except FileNotFoundError:
        pytest.skip("feature caches absent")
    assert x.shape[0] == d.shape[0] == 102 and ok.dtype == bool
    assert np.all(x[~ok] == 0) and np.allclose(np.linalg.norm(x[ok], axis=1), 1, atol=1e-4)
    m, spec = assemble(["dinov2s", "clap"], test=True)
    assert m.shape == (102, sum(spec.dims) + 1) and np.array_equal(m[:, spec.mask_col] > 0.5, ok)


def test_training_features_cover_ids_1_to_1000_only():
    try:
        m, spec = assemble(["clip_block8", "dinov2s"])
    except FileNotFoundError:
        pytest.skip("feature caches absent")
    assert m.shape[0] == 1000 and np.all(m[:, spec.mask_col] == 1)

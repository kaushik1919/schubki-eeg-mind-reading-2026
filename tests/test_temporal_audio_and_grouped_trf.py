"""Synthetic tests for temporal audio features and video-grouped TRF regression."""
import numpy as np
import pytest

from src.grouped_trf import accumulate, columns_for, empty_like, fit, lag_feature_major
from src.temporal_audio import FEATURE_NAMES, HOP, N_BINS, SAMPLE_RATE, onset_locked_buffer, temporal_audio_features

# ---------------- temporal audio ----------------

def test_feature_shape_and_names():
    f = temporal_audio_features(onset_locked_buffer(np.random.default_rng(0).normal(size=3 * SAMPLE_RATE) * .1, 0.0))
    assert f.shape == (N_BINS, len(FEATURE_NAMES)) == (300, 36) and f.dtype == np.float32 and np.isfinite(f).all()


def test_silence_and_missing_audio_give_all_zero_features():
    assert not temporal_audio_features(onset_locked_buffer(np.zeros(3 * SAMPLE_RATE), 0.0)).any()
    assert not temporal_audio_features(onset_locked_buffer(np.zeros(0), 0.0)).any()


def test_extraction_is_deterministic():
    x = np.random.default_rng(1).normal(size=3 * SAMPLE_RATE).astype(np.float32)
    assert np.array_equal(temporal_audio_features(onset_locked_buffer(x, 0.0)), temporal_audio_features(onset_locked_buffer(x, 0.0)))


def test_click_is_synchronised_to_its_100hz_bin():
    for t_ms in (0, 500, 1230, 2800):
        x = np.zeros(3 * SAMPLE_RATE, np.float32); s = t_ms * SAMPLE_RATE // 1000; x[s:s + 80] = 1.0
        f = temporal_audio_features(onset_locked_buffer(x, 0.0))
        assert abs(int(np.argmax(f[:, 0])) - t_ms // 10) <= 1           # envelope peak
        assert abs(int(np.argmax(f[:, 2])) - t_ms // 10) <= 2           # onset strength fires at the click


def test_stream_offset_shifts_features_and_short_audio_is_padded():
    x = np.zeros(SAMPLE_RATE, np.float32); x[:80] = 1.0
    late = temporal_audio_features(onset_locked_buffer(x, 0.5))
    assert abs(int(np.argmax(late[:, 0])) - 50) <= 1
    assert not late[200:].any()                                             # 1 s of audio starting at 0.5 s -> silence after 1.5 s + window
    buf = onset_locked_buffer(np.ones(10 * SAMPLE_RATE), 0.0); assert len(buf) == N_BINS * HOP + 400  # truncated


def test_pre_onset_audio_is_dropped_for_negative_offset():
    x = np.zeros(SAMPLE_RATE, np.float32); x[SAMPLE_RATE // 2:SAMPLE_RATE // 2 + 80] = 1.0
    f = temporal_audio_features(onset_locked_buffer(x, -0.25))
    assert abs(int(np.argmax(f[:, 0])) - 25) <= 1

# ---------------- grouped TRF ----------------

def test_lag_feature_major_is_causal_and_zero_padded():
    x = np.array([[1., 10.], [2., 20.], [3., 30.]], np.float32)
    X = lag_feature_major(x, 2)
    np.testing.assert_array_equal(X, [[1, 0, 10, 0], [2, 1, 20, 10], [3, 2, 30, 20]])
    np.testing.assert_array_equal(columns_for([1], 2), [2, 3])


def _synthetic(V=60, T=50, F=2, C=3, lags=4, seed=0, noise=.1):
    rng = np.random.default_rng(seed)
    stim = rng.normal(size=(V, T, F)).astype(np.float32)
    w = rng.normal(size=(F * lags, C)); erp = rng.normal(size=(T, C))
    y = np.stack([erp + lag_feature_major(s, lags) @ w for s in stim]) + noise * rng.normal(size=(V, T, C))
    return stim, y.astype(np.float32), w, erp


def test_statistics_are_additive_over_video_atoms():
    stim, y, *_ = _synthetic()
    whole = accumulate(stim, y, 4); parts = accumulate(stim[:25], y[:25], 4) + accumulate(stim[25:], y[25:], 4)
    for a, b in zip((whole.G, whole.Sx, whole.Sxy, whole.Sy), (parts.G, parts.Sx, parts.Sxy, parts.Sy)):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-4)
    assert (whole - accumulate(stim[25:], y[25:], 4)).n == 25 and empty_like(whole).n == 0


def test_fit_matches_explicit_centred_standardised_ridge():
    stim, y, *_ = _synthetic(V=30, T=20, lags=3)
    X = np.stack([lag_feature_major(s, 3) for s in stim]); n, T, P = X.shape
    xbar, ybar = X.mean(0), y.mean(0)
    Xc, Yc = (X - xbar).reshape(-1, P).astype(np.float64), (y - ybar).reshape(-1, y.shape[2]).astype(np.float64)
    sd = Xc.std(0); Xs = Xc / sd; lam = 0.1
    W_ref = np.linalg.solve(Xs.T @ Xs + lam * len(Xs) * np.eye(P), Xs.T @ Yc) / sd[:, None]
    model = fit(accumulate(stim, y, 3), np.arange(P), 3, np.array([lam]))
    np.testing.assert_allclose(model.W[0], W_ref, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(model.predict(stim, 0), ybar + (X - xbar) @ W_ref, rtol=1e-4, atol=1e-4)


def test_fit_recovers_trf_and_generalises_to_unseen_videos():
    stim, y, w, erp = _synthetic(V=80, noise=.05)
    model = fit(accumulate(stim[:60], y[:60], 4), np.arange(8), 4, np.array([1e-6]))
    np.testing.assert_allclose(model.W[0], w, atol=.05)
    resid = model.predict(stim[60:], 0) - y[60:]
    assert np.sqrt(np.mean(resid ** 2)) < .1


def test_fit_on_column_subset_ignores_other_features():
    stim, y, *_ = _synthetic()
    full = accumulate(stim, y, 4)
    sub = fit(full, columns_for([0], 4), 4, np.array([.01]))
    only = fit(accumulate(stim[:, :, :1], y, 4), np.arange(4), 4, np.array([.01]))
    np.testing.assert_allclose(sub.W, only.W, rtol=1e-4, atol=1e-6)


def test_fit_rejects_too_few_videos():
    stim, y, *_ = _synthetic(V=1)
    with pytest.raises(ValueError):
        fit(accumulate(stim, y, 4), np.arange(8), 4, np.array([1.0]))

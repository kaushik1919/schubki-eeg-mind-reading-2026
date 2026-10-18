"""Integrity tests for outputs/final_ensemble_v1 and the label-free assignment calibrations."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.assignment import hungarian_first, iterative_rowcol_z, rowz, sinkhorn_log
from src.data import CANDIDATE_IDS
from src.retrieval import rank_candidates, validate_rankings

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "final_ensemble_v1"
VARIANTS = ["S1/independent", "S1/sinkhorn_t0.5", "S3/independent", "S3/sinkhorn_t0.5"]
needs = pytest.mark.skipif(not (OUT / "MANIFEST.sha256").exists(), reason="final ensemble outputs absent")


def test_sinkhorn_is_doubly_stochastic_and_row_permutation_equivariant():
    rng = np.random.default_rng(0); s = rng.normal(size=(102, 102)); p = np.exp(sinkhorn_log(s, 0.5, 500))
    assert np.allclose(p.sum(1), 1, atol=1e-6) and np.allclose(p.sum(0), 1, atol=1e-3)
    perm = rng.permutation(102)
    assert np.allclose(sinkhorn_log(s[perm], 0.5), sinkhorn_log(s, 0.5)[perm], atol=1e-8)
    assert np.allclose(iterative_rowcol_z(s[perm]), iterative_rowcol_z(s)[perm], atol=1e-8)


def test_calibrated_scores_give_full_valid_rankings():
    rng = np.random.default_rng(1); s = rng.normal(size=(102, 102))
    for f in (lambda x: sinkhorn_log(x, 0.5), iterative_rowcol_z, hungarian_first):
        validate_rankings(rank_candidates(f(s), CANDIDATE_IDS))
    first = rank_candidates(hungarian_first(s), CANDIDATE_IDS)[:, 0]
    assert len(np.unique(first)) == 102                       # Hungarian puts a distinct candidate first per row


@needs
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("subject", ["05", "06"])
def test_versioned_submissions_are_valid(variant, subject):
    r = np.load(OUT / variant / f"sub-{subject}_ranked_candidate_list.npy", allow_pickle=False)
    assert r.shape == (102, 102) and np.issubdtype(r.dtype, np.integer)
    assert np.all(np.sort(r, axis=1) == CANDIDATE_IDS)


@needs
@pytest.mark.parametrize("subject", ["05", "06"])
def test_rankings_follow_saved_scores_in_file_order(subject):
    z = np.load(OUT / "scores" / f"sub-{subject}_scores.npz")
    assert np.array_equal(z["candidate_ids"], CANDIDATE_IDS)
    for k in ("decoder", "eeg_eeg", "model_c", "S1", "S3"): assert np.isfinite(z[k]).all() and z[k].shape == (102, 102)
    s3 = (z["decoder"] + z["eeg_eeg"] + rowz(z["model_c"])) / 3
    assert np.allclose(s3, z["S3"], atol=1e-6)
    assert np.array_equal(rank_candidates(z["S3"], CANDIDATE_IDS), np.load(OUT / "S3/independent" / f"sub-{subject}_ranked_candidate_list.npy"))
    mc = np.load(ROOT / "outputs/final_model_c/scores" / f"sub-{subject}_scores.npz")["fused"]
    assert np.array_equal(z["model_c"], mc)                    # same frozen Model C, same recording order


@needs
def test_manifest_matches():
    for line in (OUT / "MANIFEST.sha256").read_text().splitlines():
        h, n = line.split("  ", 1); assert hashlib.sha256((OUT / n).read_bytes()).hexdigest() == h, n

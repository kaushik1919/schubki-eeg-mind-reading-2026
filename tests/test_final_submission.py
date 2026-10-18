"""Integrity tests for the frozen Model C submission in outputs/final_model_c/.

Tests that need the real data or generated outputs skip when those are absent.
"""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from src.data import CANDIDATE_IDS, load_eeg, load_training_subject
from src.final_model import EXPECTED_WEIGHTS, FrozenModelC, load_frozen_config
from src.retrieval import rank_candidates, validate_rankings, weighted_score_ensemble

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "final_model_c"
DATA = ROOT / "eeg-mind-reading-challenge"
HIDDEN = ["05", "06"]
needs_outputs = pytest.mark.skipif(not (OUT / "metadata.json").exists() or not DATA.exists(), reason="final outputs or challenge data absent")


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def meta(): return json.loads((OUT / "metadata.json").read_text())


@pytest.fixture(scope="module")
def test_features():
    cfg = load_frozen_config(ROOT / "configs/frozen_balanced.json")
    return {c["name"]: np.load(ROOT / c["test_features"]) for c in cfg["constituents"]}


# ---------- synthetic (always run) ----------
def test_frozen_configs_all_encode_model_c():
    for role in ("best_top1", "best_top5", "balanced"):
        cfg = load_frozen_config(ROOT / f"configs/frozen_{role}.json")
        assert {c["name"]: c["weight"] for c in cfg["constituents"]} == EXPECTED_WEIGHTS


def test_synthetic_model_c_preserves_row_order_and_is_deterministic():
    rng = np.random.default_rng(1)
    ids = np.arange(1, 301); eeg = rng.normal(size=(300, 17, 1500)).astype(np.float32)
    feats = {n: rng.normal(size=(1000, 16)).astype(np.float32) for n in EXPECTED_WEIGHTS}
    cand = {n: rng.normal(size=(102, 16)).astype(np.float32) for n in EXPECTED_WEIGHTS}
    model = FrozenModelC(list(EXPECTED_WEIGHTS), list(EXPECTED_WEIGHTS.values())).fit(ids, eeg, feats)
    test = rng.normal(size=(102, 17, 1500)).astype(np.float32); perm = rng.permutation(102)
    r1, s1, _ = model.rank(test, cand, CANDIDATE_IDS); r2, s2, _ = model.rank(test[perm], cand, CANDIDATE_IDS)
    validate_rankings(r1)
    assert np.array_equal(r1[perm], r2) and np.array_equal(s1[perm], s2)  # row i depends only on EEG row i
    assert np.array_equal(r1, model.rank(test, cand, CANDIDATE_IDS)[0])


def test_model_rejects_test_ids_as_training_ids():
    rng = np.random.default_rng(2)
    feats = {n: np.zeros((1000, 4)) for n in EXPECTED_WEIGHTS}
    with pytest.raises(ValueError):
        FrozenModelC(list(EXPECTED_WEIGHTS), list(EXPECTED_WEIGHTS.values())).fit(np.array([1, 1001]), rng.normal(size=(2, 17, 1500)), feats)


# ---------- real outputs ----------
@needs_outputs
@pytest.mark.parametrize("subject", HIDDEN)
def test_submission_shape_dtype_permutations(subject):
    r = np.load(OUT / f"sub-{subject}_ranked_candidate_list.npy", allow_pickle=False)
    assert r.shape == (102, 102) and np.issubdtype(r.dtype, np.integer)
    assert np.all(np.sort(r, axis=1) == CANDIDATE_IDS)
    validate_rankings(r)


@needs_outputs
@pytest.mark.parametrize("subject", HIDDEN)
def test_scores_finite_and_rankings_follow_fused_scores(subject):
    s = np.load(OUT / "scores" / f"sub-{subject}_scores.npz")
    assert np.array_equal(s["candidate_ids"], CANDIDATE_IDS)
    for k in s.files:
        if k != "candidate_ids": assert np.isfinite(s[k]).all() and s[k].shape == (102, 102)
    fused = weighted_score_ensemble([s[f"scores_{n}"] for n in EXPECTED_WEIGHTS], np.array(list(EXPECTED_WEIGHTS.values())))
    assert np.array_equal(fused, s["fused"])  # saved fusion is exactly Model C's 0.45/0.45/0.10 row-z fusion
    r = np.load(OUT / f"sub-{subject}_ranked_candidate_list.npy")
    assert np.array_equal(rank_candidates(s["fused"], CANDIDATE_IDS), r)
    assert np.all(np.diff(np.take_along_axis(s["fused"], r - 1001, axis=1), axis=1) <= 0)  # best -> worst


@needs_outputs
@pytest.mark.parametrize("subject", HIDDEN)
def test_checkpoint_reproduces_submission_in_test_file_order(subject, test_features, meta):
    bundle = joblib.load(OUT / "models" / f"sub-{subject}_model_c.joblib")
    assert bundle["config_id"] == "C_final_block8_audio_45_45_10"
    m = bundle["model"]
    assert dict(zip(m.names, m.weights.tolist())) == pytest.approx(EXPECTED_WEIGHTS)
    tst = DATA / "tst" / f"sub-{subject}_tst.npy"
    assert sha(tst) == meta["subjects"][subject]["test_eeg_sha256"]
    eeg = load_eeg(tst)
    r, fused, _ = m.rank(eeg, test_features, CANDIDATE_IDS)
    assert np.array_equal(r, np.load(OUT / f"sub-{subject}_ranked_candidate_list.npy"))
    # Order preservation: reversing the input rows reverses the output rows exactly.
    r_rev, _, _ = m.rank(eeg[::-1], test_features, CANDIDATE_IDS)
    assert np.array_equal(r_rev[::-1], r)


@needs_outputs
@pytest.mark.parametrize("subject", HIDDEN)
def test_training_used_only_that_subjects_training_recordings(subject, meta):
    bundle = joblib.load(OUT / "models" / f"sub-{subject}_model_c.joblib")
    ids, _ = load_training_subject(DATA / "trn" / f"sub-{subject}_trn.npz")
    assert np.array_equal(bundle["training_ids"], ids) and np.array_equal(bundle["model"].training_ids, ids)
    assert ids.max() <= 1000 and not set(ids) & set(CANDIDATE_IDS)  # no test-video IDs in training
    assert meta["subjects"][subject]["training_video_ids"] == ids.tolist()
    assert bundle["train_npz_sha256"] == sha(DATA / "trn" / f"sub-{subject}_trn.npz")
    for n in EXPECTED_WEIGHTS:
        assert bundle["model"].models[n].x_pca.n_samples_ == len(ids)  # fitted on exactly the training rows


@needs_outputs
def test_feature_hashes_match_validated_caches_and_config(meta):
    validated = json.loads((ROOT / "experiments/nested_rerank/outer_results.json").read_text())["features"]
    key = {"final_clip": "final", "clip_block8": "block8", "audio_simple": "audio"}
    for n, rec in meta["features"].items():
        assert rec["train_sha256"] == validated[key[n]]["sha256"] == sha(ROOT / rec["train"])
        assert rec["test_sha256"] == sha(ROOT / rec["test"])


@needs_outputs
def test_manifest_hashes_match_files():
    lines = (OUT / "MANIFEST.sha256").read_text().splitlines()
    assert lines
    for line in lines:
        digest, name = line.split("  ", 1)
        assert sha(OUT / name) == digest, name


@needs_outputs
def test_hidden_test_eeg_is_not_training_data(meta):
    for s in HIDDEN:
        test = load_eeg(DATA / "tst" / f"sub-{s}_tst.npy")
        _, train = load_training_subject(DATA / "trn" / f"sub-{s}_trn.npz")
        # No test recording appears verbatim among the training recordings.
        train_keys = {hashlib.sha256(x.tobytes()).hexdigest() for x in train}
        assert not any(hashlib.sha256(x.tobytes()).hexdigest() in train_keys for x in test)
        assert meta["subjects"][s]["role"] == "hidden"


def test_missing_audio_maps_to_deterministic_silence_vector():
    from scripts.extract_audio_features import SAMPLE_RATE, features
    silent = features(np.zeros(3 * SAMPLE_RATE, dtype=np.float32))
    assert np.array_equal(silent, features(np.zeros(3 * SAMPLE_RATE, dtype=np.float32)))
    path = ROOT / "features/test_audio_simple_features.npy"
    if not path.exists():
        pytest.skip("test audio cache absent")
    cache = np.load(path)
    near = np.abs(cache).max(axis=1) < 2e-6
    # Clips without an audio stream map exactly to the silence vector; near-silent tracks differ by < 1e-6.
    assert np.all(cache == silent, axis=1).sum() >= 1
    assert np.abs(cache[near] - silent).max() < 1e-6


@needs_outputs
@pytest.mark.parametrize("subject", HIDDEN)
def test_model_c_checkpoint_reproduces_its_versioned_submission_in_file_order(subject, test_features):
    r = np.load(OUT / f"sub-{subject}_ranked_candidate_list.npy", allow_pickle=False)
    model = joblib.load(OUT / "models" / f"sub-{subject}_model_c.joblib")["model"]
    rerun, _, _ = model.rank(load_eeg(DATA / "tst" / f"sub-{subject}_tst.npy"), test_features, CANDIDATE_IDS)
    assert np.array_equal(rerun, r)  # row i == test recording i in original file order


@needs_outputs
@pytest.mark.parametrize("subject", HIDDEN)
def test_root_submission_matches_recorded_promoted_source(subject):
    root = ROOT / f"sub-{subject}_ranked_candidate_list.npy"
    record = ROOT / "outputs" / "ROOT_SUBMISSION.json"
    if not root.exists() or not record.exists():
        pytest.skip("root submission or promotion record absent")
    rec = json.loads(record.read_text()); name = root.name
    r = np.load(root, allow_pickle=False)
    assert r.shape == (102, 102) and np.issubdtype(r.dtype, np.integer)
    validate_rankings(r)
    assert sha(root) == rec["files"][name]["sha256"] == sha(ROOT / rec["source"] / name)

"""Synthetic tests for top-K reranking, rank diagnostics, and nested-split leakage guards."""
import numpy as np
import pytest

from src.controlled_cv import common_ids, disjoint_inner_folds, shared_holdout_ids
from src.data import CANDIDATE_IDS
from src.rank_analysis import summarize_positions, target_positions, transition_counts
from src.reranking import fused_rerank_scores, topk_rerank_order, topk_rerank_rankings
from src.retrieval import rank_candidates, validate_rankings

RNG = np.random.default_rng(0)


def test_topk_rerank_rows_are_valid_permutations_of_candidate_ids():
    base, rer = RNG.normal(size=(102, 102)), RNG.normal(size=(102, 102))
    for k in (1, 3, 5, 102):
        validate_rankings(topk_rerank_rankings(base, rer, CANDIDATE_IDS, k))


def test_topk_rerank_preserves_topk_set_and_tail_order():
    base, rer = RNG.normal(size=(50, 102)), RNG.normal(size=(50, 102))
    for k in (3, 5):
        a, d = rank_candidates(base), topk_rerank_rankings(base, rer, CANDIDATE_IDS, k)
        assert np.array_equal(np.sort(a[:, :k], 1), np.sort(d[:, :k], 1))
        assert np.array_equal(a[:, k:], d[:, k:])


def test_topk_rerank_never_changes_topj_accuracy_for_j_at_least_k():
    base, rer = RNG.normal(size=(200, 102)), RNG.normal(size=(200, 102))
    targets = CANDIDATE_IDS[RNG.integers(0, 102, 200)]
    pa = target_positions(rank_candidates(base), targets)
    for k in (3, 5):
        pd = target_positions(topk_rerank_rankings(base, rer, CANDIDATE_IDS, k), targets)
        for j in range(k, 103):
            assert np.mean(pa < j) == np.mean(pd < j)


def test_k1_is_identity_and_full_k_equals_rerank_ranking():
    base, rer = RNG.normal(size=(10, 102)), RNG.normal(size=(10, 102))
    assert np.array_equal(topk_rerank_rankings(base, rer, CANDIDATE_IDS, 1), rank_candidates(base))
    assert np.array_equal(topk_rerank_rankings(base, rer, CANDIDATE_IDS, 102), rank_candidates(rer))


def test_rerank_is_row_independent_no_joint_assignment():
    base, rer = RNG.normal(size=(20, 102)), RNG.normal(size=(20, 102))
    full = topk_rerank_order(base, rer, 5)
    perm = RNG.permutation(20)
    assert np.array_equal(topk_rerank_order(base[perm], rer[perm], 5), full[perm])
    for i in range(20):
        assert np.array_equal(topk_rerank_order(base[i:i + 1], rer[i:i + 1], 5)[0], full[i])
    # Duplicated winners across rows are allowed: no one-to-one constraint between recordings.
    same = np.tile(base[:1], (3, 1))
    assert len(set(topk_rerank_order(same, same, 5)[:, 0])) == 1


def test_rerank_respects_shuffled_candidate_column_map():
    ids = RNG.permutation(CANDIDATE_IDS)
    base = np.zeros((1, 102)); base[0, :3] = [3, 2, 1]
    rer = np.zeros((1, 102)); rer[0, 2] = 10
    out = topk_rerank_rankings(base, rer, ids, 3)
    assert out[0, 0] == ids[2] and list(out[0, 1:3]) == [ids[0], ids[1]]


def test_rerank_rejects_bad_inputs():
    with pytest.raises(ValueError): topk_rerank_order(np.zeros((2, 5)), np.zeros((2, 4)), 2)
    with pytest.raises(ValueError): topk_rerank_order(np.zeros((2, 5)), np.zeros((2, 5)), 0)
    with pytest.raises(ValueError): topk_rerank_rankings(np.zeros((2, 5)), np.zeros((2, 5)), np.array([1, 1, 2, 3, 4]), 2)


def test_fused_rerank_scores_is_label_free_and_row_local():
    a, b = RNG.normal(size=(4, 102)), RNG.normal(size=(4, 102))
    f = fused_rerank_scores([a, b], [.5, .5])
    g = fused_rerank_scores([a * 7 + 3, b], [.5, .5])  # per-row affine invariance
    assert np.allclose(f, g, atol=1e-5)
    assert np.allclose(fused_rerank_scores([a[:1], b[:1]], [.5, .5]), f[:1], atol=1e-6)


def test_target_positions_and_transitions():
    r = np.tile(CANDIDATE_IDS, (3, 1))
    p = target_positions(r, np.array([1001, 1003, 1010]))
    assert list(p) == [0, 2, 9]
    s = summarize_positions(p)
    assert s["top1"] == pytest.approx(1 / 3) and s["top5"] == pytest.approx(2 / 3)
    t = transition_counts(np.array([2, 0, 3, 7]), np.array([0, 6, 3, 1]))
    assert t["promoted_2to5_to_1"] == 1 and t["demoted_top5_to_outside"] == 1 and t["top1_wins"] == 1 and t["top1_losses"] == 1
    with pytest.raises(ValueError): target_positions(r, np.array([1, 1003, 1010]))


def test_inner_folds_are_disjoint_from_each_other_and_outer_holdout():
    subject_ids = {"01": np.arange(1, 801), "02": np.arange(1, 801)}
    pool = common_ids(subject_ids)
    outer = shared_holdout_ids(subject_ids, seed=2031, size=102)
    folds = disjoint_inner_folds(pool, exclude=outer, n_folds=5, size=102, seed=2032)
    flat = np.concatenate(folds)
    assert len(np.unique(flat)) == len(flat) and not set(flat) & set(outer)
    assert all(np.array_equal(a, b) for a, b in zip(folds, disjoint_inner_folds(pool, exclude=outer, n_folds=5, size=102, seed=2032)))
    with pytest.raises(ValueError): disjoint_inner_folds(pool, exclude=outer, n_folds=7, size=102, seed=1)


def test_saved_holdouts_are_mutually_disjoint_and_shared():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    conf, final = root / "manifests/confirmation_holdout_ids.npy", root / "manifests/final_holdout_ids.npy"
    if not (conf.exists() and final.exists()):
        pytest.skip("holdout manifests not present")
    c, f = np.load(conf), np.load(final)
    assert c.shape == f.shape == (102,) and len(np.unique(f)) == 102 and not set(c) & set(f)

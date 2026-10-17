"""Synthetic tests only: no challenge EEG/video data is included in this repository."""
import numpy as np
import pytest

from src.data import CANDIDATE_IDS, load_training_subject, validate_eeg
from src.controlled_cv import millisecond_slice, shared_holdout_ids
from src.evaluation import video_heldout_split
from src.retrieval import organizer_eeg_correlation_scores, rank_candidates, validate_rankings, weighted_score_ensemble


def test_eeg_shape_is_strict():
    assert validate_eeg(np.zeros((2, 17, 1500))).shape == (2, 17, 1500)
    with pytest.raises(ValueError): validate_eeg(np.zeros((2, 16, 1500)))


def test_ranking_maps_columns_to_supplied_candidate_ids():
    ids = CANDIDATE_IDS[::-1]
    scores = np.zeros((1, 102)); scores[0, 5] = 1
    assert rank_candidates(scores, ids)[0, 0] == ids[5]


def test_submission_rows_are_permutations():
    rankings = np.tile(CANDIDATE_IDS, (102, 1))
    assert np.array_equal(validate_rankings(rankings), rankings)
    rankings[0, 0] = rankings[0, 1]
    with pytest.raises(ValueError): validate_rankings(rankings)


def test_grouped_split_is_deterministic_and_video_disjoint():
    ids = np.repeat(np.arange(20), 3)
    first = video_heldout_split(ids, validation_fraction=0.25, seed=2026)
    second = video_heldout_split(ids, validation_fraction=0.25, seed=2026)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert not (set(ids[first[0]]) & set(ids[first[1]]))


def test_ranking_does_not_reorder_eeg_rows():
    scores = np.zeros((3, 102)); scores[np.arange(3), [7, 2, 99]] = 1
    assert np.array_equal(rank_candidates(scores)[:, 0], CANDIDATE_IDS[[7, 2, 99]])


def test_organizer_npz_is_sorted_by_numeric_stimulus_id(tmp_path):
    np.savez(tmp_path / "sub-01_trn.npz", **{"10": np.ones((17, 1500)), "2": np.zeros((17, 1500))})
    ids, eeg = load_training_subject(tmp_path / "sub-01_trn.npz")
    assert np.array_equal(ids, [2, 10]) and eeg[0].sum() == 0


def test_organizer_score_matches_aligned_trials_on_synthetic_data():
    rng = np.random.default_rng(2)
    eeg = rng.normal(size=(3, 17, 300))
    scores = organizer_eeg_correlation_scores(eeg, eeg)
    assert np.array_equal(scores.argmax(1), [0, 1, 2])


def test_weighted_score_ensemble_preserves_shared_best_candidate():
    first = np.array([[0.1, 0.8, 0.2], [0.9, 0.1, 0.2]])
    second = np.array([[0.2, 0.7, 0.3], [0.8, 0.2, 0.1]])
    ensemble = weighted_score_ensemble([first, second], np.array([0.9, 0.1]))
    assert np.array_equal(ensemble.argmax(axis=1), [1, 0])


def test_shared_holdout_is_deterministic_and_present_for_every_subject():
    subject_ids = {"01": np.arange(1, 151), "02": np.arange(25, 176), "03": np.arange(1, 151)}
    first = shared_holdout_ids(subject_ids, seed=7, size=50)
    second = shared_holdout_ids(subject_ids, seed=7, size=50)
    assert np.array_equal(first, second)
    assert all(set(first).issubset(ids) for ids in subject_ids.values())


def test_stimulus_locked_window_is_clipped_without_shifting_alignment():
    assert millisecond_slice(representation_start_ms=0, representation_stop_ms=3000, sampling_hz=100, window_ms=(50, 600)) == slice(5, 60)
    assert millisecond_slice(representation_start_ms=100, representation_stop_ms=800, sampling_hz=100, window_ms=(50, 600)) == slice(0, 50)

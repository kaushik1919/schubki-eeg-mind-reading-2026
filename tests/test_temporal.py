import numpy as np
from src.temporal import lagged_design, resample_trajectory, temporal_design


def test_causal_lags_are_zero_padded_and_ordered():
    values = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    np.testing.assert_array_equal(lagged_design(values, 2), [[1, 0, 0], [2, 1, 0], [3, 2, 1]])


def test_temporal_design_has_one_row_per_eeg_sample():
    trajectories = np.array([[[0.0], [1.0]], [[1.0], [0.0]]], dtype=np.float32)
    assert resample_trajectory(trajectories[0], 4).shape == (4, 1)
    assert temporal_design(trajectories, 4, 3).shape == (8, 4)

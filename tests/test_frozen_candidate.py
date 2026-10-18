"""Checks on the frozen contrastive candidate configuration and fusion."""
import json
from pathlib import Path

import numpy as np

from scripts.confirm_frozen_candidate import fuse
from scripts.train_contrastive_eeg import SUBJECTS
from src.retrieval import row_zscore

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs/frozen_contrastive_candidate.json").read_text())


def test_frozen_config_matches_exploratory_candidate_settings():
    base = json.loads((ROOT / "configs/contrastive_eeg.json").read_text())["base"]
    hp = CFG["decoder"]["hyperparameters"]
    assert all(hp[k] == base[k] for k in base if k != "seed")
    assert CFG["decoder"]["seeds"] == [2050, 2051, 2052, 2053, 2054] and CFG["fusion"]["weights"] == [0.5, 0.5]
    assert CFG["decoder"]["target_features"]["train"].endswith("all_train_clip_intermediate_block8.npy")


def test_subject_mapping_is_fixed_and_matches_training_code():
    assert list(CFG["decoder"]["subject_index"]) == SUBJECTS
    assert [CFG["decoder"]["subject_index"][s] for s in SUBJECTS] == list(range(6))


def test_fusion_is_row_local_equal_weight_and_shape_preserving():
    rng = np.random.default_rng(0); seeds = [rng.normal(size=(102, 102)) for _ in range(5)]; mc = rng.normal(size=(102, 102))
    ens, fused = fuse(seeds, mc)
    assert fused.shape == (102, 102)
    np.testing.assert_allclose(ens, np.mean([row_zscore(s) for s in seeds], axis=0), atol=1e-6)
    np.testing.assert_allclose(fused, 0.5 * row_zscore(ens) + 0.5 * row_zscore(mc), atol=1e-6)
    perm = rng.permutation(102); _, fp = fuse([s[perm] for s in seeds], mc[perm])
    np.testing.assert_allclose(fp, fused[perm], atol=1e-6)  # recording order is preserved row-by-row

#!/usr/bin/env bash
# Recreate the final validated score-level ensemble from saved hidden-subject scores.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
IDS="$ROOT_DIR/manifests/test_candidate_ids.npy"
cd "$ROOT_DIR"

for SUBJECT in sub-05 sub-06; do
  "$PYTHON" "$ROOT_DIR/scripts/ensemble_rankings.py" \
    --scores "$ROOT_DIR/experiments/${SUBJECT}_alexnet_scores.npy" \
             "$ROOT_DIR/experiments/${SUBJECT}_clip_scores.npy" \
             "$ROOT_DIR/experiments/${SUBJECT}_dinov2s_scores.npy" \
    --weights 0.36 0.04 0.60 \
    --candidate-ids "$IDS" \
    --output "$ROOT_DIR/${SUBJECT}_ranked_candidate_list.npy"
done

"$PYTHON" - <<'PY'
from pathlib import Path
import numpy as np
from src.retrieval import validate_rankings

root = Path.cwd()
for subject in ("sub-05", "sub-06"):
    path = root / f"{subject}_ranked_candidate_list.npy"
    rankings = np.load(path)
    validate_rankings(rankings)
    print(f"validated {path.name}: {rankings.shape}")
PY

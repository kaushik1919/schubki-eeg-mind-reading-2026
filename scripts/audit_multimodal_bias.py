#!/usr/bin/env python3
"""Missing-audio and tie audit for saved multimodal score matrices.

When several candidates lack audio, an audio-only stimulus head maps them to
the same embedding, so their scores tie exactly.  Stable-sort ranking breaks
ties by candidate-ID order (label-independent but arbitrary); a strict-">"
rank count breaks them in favour of the target (optimistic).  This script
reports, per experiment, MRR for held-out videos with and without audio under
(i) stable sort and (ii) pessimistic tie handling (ties rank ahead of the target).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.multimodal_stimulus import load_modality
from src.rank_analysis import target_positions
from src.retrieval import rank_candidates

SUBJECTS = ["01", "02", "03", "04", "05", "06"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--dirs", nargs="+", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    _, has = load_modality("clap"); res = {}
    for d in map(Path, a.dirs):
        stable, pess, audio, ties = [], [], [], []
        for o in range(6):
            for s in SUBJECTS:
                z = np.load(d / f"fold{o}_sub{s}.npz"); h, t, sc = z["heldout_ids"], z["target_ids"], z["scores"]
                stable.append(target_positions(rank_candidates(sc, h), t))
                tgt = sc[np.arange(len(sc)), np.searchsorted(h, t)][:, None]
                pess.append((sc >= tgt).sum(1) - 1); ties.append((sc == tgt).sum(1) - 1); audio.append(has[t - 1])
        stable, pess, audio, ties = map(np.concatenate, (stable, pess, audio, ties))
        mrr = lambda p, m: float(np.mean(1 / (p[m] + 1)))
        res[d.name] = {"n_with_audio": int(audio.sum()), "n_without_audio": int((~audio).sum()), "trials_with_exact_ties": int((ties > 0).sum()),
                       "stable": {"with_audio": mrr(stable, audio), "without_audio": mrr(stable, ~audio), "all_top1": float(np.mean(stable == 0))},
                       "pessimistic": {"with_audio": mrr(pess, audio), "without_audio": mrr(pess, ~audio), "all_top1": float(np.mean(pess == 0))}}
        print(d.name, res[d.name], flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

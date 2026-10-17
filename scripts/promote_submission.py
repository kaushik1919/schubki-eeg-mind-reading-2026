#!/usr/bin/env python3
"""Promote validated versioned submissions to the root paths (requires --confirm).

Existing root files are first copied to `outputs/previous_root_submissions/<sha-prefix>/`
so nothing is lost. The candidate files are re-validated and checked against
the versioned MANIFEST before copying.
"""
from __future__ import annotations
import argparse, hashlib, shutil, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.retrieval import validate_rankings


def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="outputs/final_model_c"); p.add_argument("--confirm", action="store_true")
    a = p.parse_args(); src = Path(a.source)
    manifest = dict(reversed(line.split("  ", 1)) for line in (src / "MANIFEST.sha256").read_text().splitlines())
    names = ["sub-05_ranked_candidate_list.npy", "sub-06_ranked_candidate_list.npy"]
    for n in names:
        r = np.load(src / n, allow_pickle=False); validate_rankings(r)
        if r.dtype.kind not in "iu" or sha(src / n) != manifest[n]: raise RuntimeError(f"{n} failed validation or manifest check")
    if not a.confirm:
        print("validated; re-run with --confirm to back up and replace root submissions"); return
    for n in names:
        root = Path(n)
        if root.exists():
            backup = Path("outputs/previous_root_submissions") / sha(root)[:12] / n
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists(): shutil.copy2(root, backup)
        shutil.copy2(src / n, root)
        assert sha(root) == manifest[n]
        print(f"promoted {n} ({manifest[n]})")


if __name__ == "__main__":
    main()

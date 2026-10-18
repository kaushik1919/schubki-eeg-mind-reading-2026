#!/usr/bin/env python3
"""Promote validated versioned submissions to the root paths (requires --confirm).

Existing root files are first copied to `outputs/previous_root_submissions/<sha-prefix>/`
so nothing is lost. The candidate files are re-validated and checked against
the versioned MANIFEST before copying. A record of the promoted source is
written to `outputs/ROOT_SUBMISSION.json`.

Examples:
  --source outputs/final_model_c
  --source outputs/final_ensemble_v1/S3/independent --manifest outputs/final_ensemble_v1/MANIFEST.sha256
"""
from __future__ import annotations
import argparse, hashlib, json, shutil, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.retrieval import validate_rankings


def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="outputs/final_model_c"); p.add_argument("--confirm", action="store_true")
    p.add_argument("--manifest", default=None, help="MANIFEST.sha256 (default: <source>/MANIFEST.sha256); entries may be relative to the manifest's directory")
    a = p.parse_args(); src = Path(a.source); mpath = Path(a.manifest) if a.manifest else src / "MANIFEST.sha256"
    manifest = {(mpath.parent / n).resolve(): h for h, n in (line.split("  ", 1) for line in mpath.read_text().splitlines())}
    names = ["sub-05_ranked_candidate_list.npy", "sub-06_ranked_candidate_list.npy"]
    expected = {}
    for n in names:
        f = (src / n).resolve(); r = np.load(f, allow_pickle=False); validate_rankings(r)
        if r.dtype.kind not in "iu" or f not in manifest or sha(f) != manifest[f]: raise RuntimeError(f"{n} failed validation or manifest check")
        expected[n] = manifest[f]
    if not a.confirm:
        print("validated; re-run with --confirm to back up and replace root submissions"); return
    record = {"source": str(src), "manifest": str(mpath), "files": {}}
    for n in names:
        root = Path(n)
        if root.exists():
            backup = Path("outputs/previous_root_submissions") / sha(root)[:12] / n
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists(): shutil.copy2(root, backup)
            record["files"].setdefault(n, {})["replaced_sha256"] = sha(root)
        shutil.copy2(src / n, root)
        assert sha(root) == expected[n]
        record["files"].setdefault(n, {})["sha256"] = expected[n]
        print(f"promoted {n} ({expected[n]})")
    Path("outputs/ROOT_SUBMISSION.json").write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()

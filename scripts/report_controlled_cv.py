#!/usr/bin/env python3
"""Summarize saved controlled-CV CSVs with subject-level paired bootstrap CIs."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np


def load(path: Path) -> tuple[str, dict[str, dict[str, float]], float]:
    rows=list(csv.DictReader(path.open())); name=rows[0]["experiment"]
    subjects=sorted({r["subject"] for r in rows}); result={}
    for subject in subjects:
        here=[r for r in rows if r["subject"]==subject]
        result[subject]={key:float(np.mean([float(r[key]) for r in here])) for key in ("top1","top5","seconds")}
    return name,result,float(np.mean([float(r["seconds"]) for r in rows]))


def paired_ci(baseline: dict[str, dict[str, float]], candidate: dict[str, dict[str, float]], metric: str) -> tuple[float,float,float]:
    keys=sorted(set(baseline)&set(candidate)); diff=np.array([candidate[k][metric]-baseline[k][metric] for k in keys]); rng=np.random.default_rng(20261017)
    sampled=np.array([diff[rng.integers(0,len(diff),len(diff))].mean() for _ in range(20000)])
    return float(diff.mean()),float(np.quantile(sampled,.025)),float(np.quantile(sampled,.975))


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--baseline",required=True);p.add_argument("--inputs",nargs="+",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    base_name,base,_=load(Path(a.baseline)); collected=[load(Path(x)) for x in a.inputs]
    lines=["# Controlled CLIP train-only results", "", "All figures use identical shared video-ID holdouts. Each reported subject value is the mean across its saved folds; confidence intervals bootstrap the six paired subject means rather than treating overlapping fold rows as independent.", "", "| Configuration | Top-1 | Delta top-1 (95% paired CI) | Top-5 | Delta top-5 (95% paired CI) | Mean runtime / subject-fold |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name,by_subject,runtime in collected:
        top1=np.mean([v["top1"] for v in by_subject.values()]);top5=np.mean([v["top5"] for v in by_subject.values()])
        d1,l1,h1=paired_ci(base,by_subject,"top1");d5,l5,h5=paired_ci(base,by_subject,"top5")
        lines.append(f"| {name} | {top1:.2%} | {d1:+.2%} ({l1:+.2%}, {h1:+.2%}) | {top5:.2%} | {d5:+.2%} ({l5:+.2%}, {h5:+.2%}) | {runtime:.2f}s |")
    lines.extend(["", "## Per-subject means", "", "| Configuration | " + " | ".join(sorted(base)) + " |", "| --- | " + " | ".join(["---:"]*len(base)) + " |"])
    for name,by_subject,_ in collected:
        lines.append("| " + name + " (top-1) | " + " | ".join(f"{by_subject[s]['top1']:.2%}" for s in sorted(base)) + " |")
        lines.append("| " + name + " (top-5) | " + " | ".join(f"{by_subject[s]['top5']:.2%}" for s in sorted(base)) + " |")
    lines.extend(["", "## Paired per-subject differences from baseline", "", "| Configuration | " + " | ".join(sorted(base)) + " |", "| --- | " + " | ".join(["---:"]*len(base)) + " |"])
    for name,by_subject,_ in collected:
        if name == base_name:
            continue
        lines.append("| " + name + " (top-1 delta) | " + " | ".join(f"{by_subject[s]['top1']-base[s]['top1']:+.2%}" for s in sorted(base)) + " |")
        lines.append("| " + name + " (top-5 delta) | " + " | ".join(f"{by_subject[s]['top5']-base[s]['top5']:+.2%}" for s in sorted(base)) + " |")
    Path(a.output).write_text("\n".join(lines)+"\n")

if __name__=="__main__":main()

"""End-to-end pipeline: OLD/NEW STEP -> object correspondences and change types.

1. read both STEP files and split them into objects (OLD 0..N-1, NEW 0..M-1)
2. extract exact B-Rep features of every object
3. for each OLD object (from index 0) keep only NEW objects whose AABB overlaps
   (no overlap = different design intent, even for an identical shape)
4. analyse every remaining (OLD, NEW) pair: transform estimation + scores
5. global 1:1 assignment maximising the total score (Hungarian algorithm)
6. matched pairs -> STAY / MODIFIED (+ detailed change types);
   unmatched OLD -> DELETION, unmatched NEW -> INSERTION
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .compare import PairAnalysis, aabb_overlap, analyse_pair, classify
from .config import Config
from .features import ObjectFeatures, extract_features
from .step_io import load_step


def _ref(f: ObjectFeatures) -> dict:
    return {"idx": f.idx, "name": f.name}


def extract_all(path: str | Path) -> list[ObjectFeatures]:
    return [extract_features(o.idx, o.name, o.shape) for o in load_step(path)]


def compare_objects(old: list[ObjectFeatures], new: list[ObjectFeatures], cfg: Config | None = None) -> dict:
    cfg = cfg or Config()
    t0 = time.perf_counter()

    # --- steps 3 + 4: AABB candidate filter, pair analysis
    pairs: dict[tuple[int, int], PairAnalysis] = {}
    candidates: dict[int, list[dict]] = {}
    for o in old:
        margin = cfg.aabb_margin + max(cfg.abs_tol, cfg.rel_pos_tol * o.diag)
        cands = []
        for n in new:
            if not aabb_overlap(o, n, margin):
                continue
            pa = analyse_pair(o, n, cfg)
            pairs[(o.idx, n.idx)] = pa
            cands.append(
                {
                    "new_idx": n.idx,
                    "new_name": n.name,
                    "alignment": pa.alignment.kind if pa.alignment else None,
                    "score": round(pa.score, 6),
                    "shape": round(pa.shape, 6),
                    "eligible": pa.eligible,
                }
            )
        candidates[o.idx] = sorted(cands, key=lambda c: -c["score"])

    # --- step 5: global 1:1 assignment
    big = 1e6
    cost = np.full((len(old), len(new)), big)
    for (i, j), pa in pairs.items():
        if pa.eligible:
            cost[i, j] = -pa.score
    matched: list[tuple[int, int]] = []
    if len(old) and len(new):
        rows, cols = linear_sum_assignment(cost)
        matched = [(int(i), int(j)) for i, j in zip(rows, cols) if cost[i, j] < big]

    # --- step 6: classification
    results = []
    for i, j in sorted(matched):
        pa = pairs[(i, j)]
        entry = {"old": _ref(old[i]), "new": _ref(new[j])}
        entry.update(classify(pa, cfg))
        results.append(entry)
    used_old = {i for i, _ in matched}
    used_new = {j for _, j in matched}
    for o in old:
        if o.idx not in used_old:
            results.append(
                {
                    "old": _ref(o),
                    "new": None,
                    "status": "DELETION",
                    "change_types": ["DELETION"],
                    "reason": "no overlapping NEW candidate" if not candidates[o.idx] else "no eligible / globally assigned candidate",
                    "volume": o.volume,
                    "centroid": o.centroid.tolist(),
                }
            )
    for n in new:
        if n.idx not in used_new:
            results.append(
                {
                    "old": None,
                    "new": _ref(n),
                    "status": "INSERTION",
                    "change_types": ["INSERTION"],
                    "volume": n.volume,
                    "centroid": n.centroid.tolist(),
                }
            )

    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    type_counts: dict[str, int] = {}
    for r in results:
        for t in r["change_types"]:
            type_counts[t] = type_counts.get(t, 0) + 1

    return {
        "summary": {
            "old_objects": len(old),
            "new_objects": len(new),
            "matched": len(matched),
            "status_counts": counts,
            "change_type_counts": type_counts,
            "elapsed_s": round(time.perf_counter() - t0, 3),
        },
        "old_objects": [o.summary() for o in old],
        "new_objects": [n.summary() for n in new],
        "candidates": {str(k): v for k, v in candidates.items()},
        "results": results,
        "config": cfg.to_dict(),
    }


def compare_step_files(old_path: str | Path, new_path: str | Path, cfg: Config | None = None) -> dict:
    report = compare_objects(extract_all(old_path), extract_all(new_path), cfg)
    report["files"] = {"old": str(old_path), "new": str(new_path)}
    return report

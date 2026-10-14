"""Pairwise comparison of one OLD object with one NEW candidate.

Produces (a) similarity scores used by the global assignment and (b) the
change classification of the pair:

* 1st level: ``STAY`` or ``MODIFIED``
* 2nd level (MODIFIED only): ``TRANSLATION``, ``ROTATION``,
  ``UNIFORM_SCALING`` (plus ``NON_UNIFORM_SCALING``, ``MIRROR``,
  ``SHAPE_CHANGE`` for cases outside a similarity transform)
* 3rd level: every component is detected and reported independently, so a
  composite change yields several types at once.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

import warnings

import numpy as np
from scipy.spatial.transform import Rotation

from .config import Config
from .features import ObjectFeatures
from .transform import (
    Alignment,
    estimate_nonuniform,
    estimate_partial_rigid,
    estimate_similarity,
    rotation_angle_deg,
)

_AXIS_NAMES = {0: "X", 1: "Y", 2: "Z"}


# --------------------------------------------------------------------------- #
# similarity measures (all invariant to position / orientation / scale)
# --------------------------------------------------------------------------- #
def _multiset_sim(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    return sum(min(a[k], b[k]) for k in keys) / sum(max(a[k], b[k]) for k in keys)


def _fraction_sim(a: dict, b: dict) -> float:
    ta, tb = sum(a.values()), sum(b.values())
    if ta <= 0 or tb <= 0:
        return 1.0 if ta == tb else 0.0
    keys = set(a) | set(b)
    return sum(min(a.get(k, 0) / ta, b.get(k, 0) / tb) for k in keys)


def _ratio(a: float, b: float) -> float:
    if a <= 0 and b <= 0:
        return 1.0
    return min(a, b) / max(a, b)


def topology_similarity(o: ObjectFeatures, n: ObjectFeatures) -> float:
    wl = np.mean([_multiset_sim(a, b) for a, b in zip(o.wl, n.wl)])
    counts = np.mean([_ratio(o.n_faces, n.n_faces), _ratio(o.n_edges, n.n_edges), _ratio(o.n_vertices, n.n_vertices)])
    euler = 1.0 if o.euler_characteristic == n.euler_characteristic else 0.0
    return float(0.6 * wl + 0.3 * counts + 0.1 * euler)


def geometry_similarity(o: ObjectFeatures, n: ObjectFeatures) -> float:
    def area_by_type(x):
        d: dict = {}
        for f in x.faces:
            d[f.surf_type] = d.get(f.surf_type, 0.0) + f.area
        return d

    def length_by_type(x):
        d: dict = {}
        for e in x.edges:
            d[e.curve_type] = d.get(e.curve_type, 0.0) + e.length
        return d

    def compactness(x):  # V / A^1.5, scale invariant
        return x.volume / x.area ** 1.5 if x.area > 0 else 0.0

    sims = [
        _fraction_sim(area_by_type(o), area_by_type(n)),
        _fraction_sim(length_by_type(o), length_by_type(n)),
        _ratio(compactness(o), compactness(n)) if o.is_solid and n.is_solid else 1.0,
    ]
    return float(np.mean(sims))


def aabb_overlap(o: ObjectFeatures, n: ObjectFeatures, margin: float) -> bool:
    return bool(np.all(o.bbox_min <= n.bbox_max + margin) and np.all(n.bbox_min <= o.bbox_max + margin))


def aabb_iou(o: ObjectFeatures, n: ObjectFeatures) -> float:
    lo = np.maximum(o.bbox_min, n.bbox_min)
    hi = np.minimum(o.bbox_max, n.bbox_max)
    inter = np.prod(np.clip(hi - lo, 0, None))
    vo, vn = np.prod(o.bbox_size), np.prod(n.bbox_size)
    union = vo + vn - inter
    return float(inter / union) if union > 0 else 1.0


def spatial_similarity(o: ObjectFeatures, n: ObjectFeatures) -> float:
    ref = 0.5 * (o.diag + n.diag) or 1.0
    d = np.linalg.norm(n.centroid - o.centroid)
    return float(0.5 * aabb_iou(o, n) + 0.5 * np.exp(-d / ref))


# --------------------------------------------------------------------------- #
# pair analysis
# --------------------------------------------------------------------------- #
@dataclass
class PairAnalysis:
    old: ObjectFeatures
    new: ObjectFeatures
    alignment: Optional[Alignment]
    topology: float
    geometry: float
    shape: float
    spatial: float
    size: float
    score: float
    eligible: bool  # may take part in the global assignment


def analyse_pair(o: ObjectFeatures, n: ObjectFeatures, cfg: Config) -> PairAnalysis:
    alignment = estimate_similarity(o, n, cfg)
    if alignment is None and cfg.detect_nonuniform:
        alignment = estimate_nonuniform(o, n, cfg)
    if alignment is None:
        alignment = estimate_partial_rigid(o, n, cfg)

    topo = topology_similarity(o, n)
    geom = geometry_similarity(o, n)
    if alignment is not None and alignment.kind == "SIMILARITY":
        shape = 1.0
    elif alignment is not None and alignment.kind == "NONUNIFORM":
        shape = 0.9
    else:
        partial = alignment.support if alignment is not None else 0.0
        shape = 0.5 * topo + 0.25 * geom + 0.25 * partial
    spatial = spatial_similarity(o, n)
    size = _ratio(o.volume, n.volume) if o.is_solid and n.is_solid else _ratio(o.area, n.area)
    score = cfg.w_shape * shape + cfg.w_spatial * spatial + cfg.w_size * size
    eligible = shape >= cfg.min_shape_similarity and score >= cfg.min_score
    return PairAnalysis(o, n, alignment, topo, geom, shape, spatial, size, score, eligible)


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
def _axis_label(axis: np.ndarray, cos_tol: float) -> str:
    k = int(np.argmax(np.abs(axis)))
    if abs(axis[k]) >= cos_tol:
        return ("+" if axis[k] > 0 else "-") + _AXIS_NAMES[k]
    return "oblique"


def _clean(v, nd=9):
    arr = np.round(np.asarray(v, dtype=float), nd) + 0.0  # also turns -0.0 into 0.0
    return arr.tolist()


def classify(pa: PairAnalysis, cfg: Config) -> dict:
    o, n, al = pa.old, pa.new, pa.alignment
    cos_tol = float(np.cos(np.radians(cfg.angle_tol_deg)))
    types: list[str] = []

    vol_ratio = n.volume / o.volume if o.volume > 0 else None
    shape_info = {
        "changed": al is None or al.kind == "PARTIAL",
        "volume_old": o.volume,
        "volume_new": n.volume,
        "volume_ratio": vol_ratio,
        "area_old": o.area,
        "area_new": n.area,
        "faces_old": o.n_faces,
        "faces_new": n.n_faces,
        "edges_old": o.n_edges,
        "edges_new": n.n_edges,
        "vertices_old": o.n_vertices,
        "vertices_new": n.n_vertices,
        "topology_similarity": round(pa.topology, 6),
        "geometry_similarity": round(pa.geometry, 6),
    }

    translation = {"detected": False}
    rotation = {"detected": False}
    scaling = {"detected": False, "type": None}
    mirror = {"detected": False}
    basis = "exact"
    matrix = None

    if al is None:
        basis = "centroid_only"
        d = n.centroid - o.centroid
        translation.update(
            {"basis": basis, "centroid_shift": _clean(d), "note": "shape changed; no rigid alignment found, value is the centroid shift only"}
        )
    else:
        if al.kind == "PARTIAL":
            basis = "partial_alignment"
            matched_o = {i for i, _ in al.face_pairs}
            matched_n = {j for _, j in al.face_pairs}
            shape_info.update(
                {
                    "unchanged_area_fraction": round(al.support, 6),
                    "unchanged_faces": len(al.face_pairs),
                    "removed_or_modified_old_faces": sorted(set(range(o.n_faces)) - matched_o),
                    "added_or_modified_new_faces": sorted(set(range(n.n_faces)) - matched_n),
                }
            )
        matrix = _clean(al.matrix)

        # --- translation (displacement of the reference point, global axes)
        t = al.translation
        dist = float(np.linalg.norm(t))
        translation = {
            "detected": dist > cfg.translation_eps,
            "dx": float(_clean(t[0])),
            "dy": float(_clean(t[1])),
            "dz": float(_clean(t[2])),
            "distance": float(_clean(dist)),
            "moved_axes": [_AXIS_NAMES[k] for k in range(3) if abs(t[k]) > cfg.translation_eps],
            "reference_point": "volume centroid" if al.kind != "PARTIAL" else "old volume centroid (carried by the unchanged faces)",
        }

        # --- rotation
        ang = rotation_angle_deg(al.rotation)
        rv = Rotation.from_matrix(al.rotation).as_rotvec()
        axis = rv / np.linalg.norm(rv) if np.linalg.norm(rv) > 1e-12 else np.array([0.0, 0.0, 1.0])
        with warnings.catch_warnings():  # gimbal lock is harmless: axis-angle is the primary output
            warnings.simplefilter("ignore")
            euler = Rotation.from_matrix(al.rotation).as_euler("xyz", degrees=True)
        rotation = {
            "detected": ang > cfg.rotation_eps_deg and al.determinacy != "none",
            "angle_deg": float(_clean(ang, 6)),
            "axis": _clean(axis),
            "axis_label": _axis_label(axis, cos_tol),
            "euler_xyz_deg": _clean(euler, 6),
            "center": _clean(al.c_old),
            "determinacy": al.determinacy,
            "equivalent_solutions": al.n_solutions,
        }
        if al.determinacy == "axisymmetric":
            rotation["symmetry_axis_old"] = _clean(al.symmetry_axis)
            rotation["note"] = (
                "object is rotationally symmetric about its own axis: the reported rotation is the exact "
                "minimal rotation of that axis; spin about the axis itself is not observable"
            )
        elif al.determinacy == "none":
            rotation["note"] = "object has no directional feature (e.g. sphere); rotation is not observable"
        elif al.n_solutions > 1:
            rotation["note"] = (
                f"object has {al.n_solutions} symmetry-equivalent orientations; the smallest rotation is reported"
            )

        # --- scaling
        if al.kind == "SIMILARITY":
            if abs(al.scale - 1.0) > cfg.scale_eps:
                scaling = {"detected": True, "type": "uniform", "factor": float(_clean(al.scale, 9))}
            else:
                scaling = {"detected": False, "type": None, "factor": 1.0}
        elif al.kind == "NONUNIFORM":
            scaling = {
                "detected": True,
                "type": "non_uniform",
                "axis_factors": _clean(al.scale_axes, 9),
                "axes_old_frame": _clean(al.scale_frame.T),
                "note": "factors differ per axis -> not a uniform scaling; factors along the listed axes of OLD",
            }

        # --- mirror
        if al.mirror:
            mirror = {
                "detected": True,
                "plane_normal": _clean(al.mirror_normal),
                "plane_label": _axis_label(al.mirror_normal, cos_tol),
                "plane_point": _clean(al.c_old),
                "note": "OLD is reflected in this plane, then the listed rotation (about the plane normal) is applied",
            }

    if translation.get("detected"):
        types.append("TRANSLATION")
    if rotation.get("detected"):
        types.append("ROTATION")
    if scaling.get("type") == "uniform":
        types.append("UNIFORM_SCALING")
    if scaling.get("type") == "non_uniform":
        types.append("NON_UNIFORM_SCALING")
    if mirror.get("detected"):
        types.append("MIRROR")
    if shape_info["changed"]:
        types.append("SHAPE_CHANGE")

    return {
        "status": "MODIFIED" if types else "STAY",
        "change_types": types,
        "basis": basis,
        "transformations": {
            "translation": translation,
            "rotation": rotation,
            "scaling": scaling,
            "mirror": mirror,
        },
        "shape": shape_info,
        "matrix_4x4": matrix,
        "scores": {
            "total": round(pa.score, 6),
            "shape": round(pa.shape, 6),
            "spatial": round(pa.spatial, 6),
            "size": round(pa.size, 6),
        },
    }

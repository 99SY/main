"""Tolerances and weights. All relative tolerances are w.r.t. the object size."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Config:
    # --- geometric matching tolerances (used to decide "same entity") ---
    rel_pos_tol: float = 1e-4  # position tolerance / object bbox diagonal
    rel_value_tol: float = 1e-4  # relative tolerance on areas, radii, volumes
    angle_tol_deg: float = 0.01  # tolerance on directions (normals, axes)
    abs_tol: float = 1e-6  # floor for every tolerance (model units)

    # --- reporting thresholds (what counts as an actual change) ---
    translation_eps: float = 1e-4  # model units; |d| below this = no move
    rotation_eps_deg: float = 1e-3
    scale_eps: float = 1e-5  # |s - 1| below this = no scaling

    # --- candidate filter ---
    aabb_margin: float = 0.0  # extra margin (model units) on the AABB overlap test

    # --- global assignment ---
    w_shape: float = 0.5
    w_spatial: float = 0.3
    w_size: float = 0.2
    min_shape_similarity: float = 0.35  # below this a pair cannot be matched
    min_score: float = 0.3

    # --- hypothesis enumeration limits (exhaustive, deterministic) ---
    max_anchor_faces: int = 4
    max_hypotheses: int = 4000
    partial_min_support: float = 0.2  # area fraction for a partial alignment

    # --- optional transform families ---
    detect_mirror: bool = True
    detect_nonuniform: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

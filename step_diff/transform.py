"""Deterministic estimation of the transform between two B-Rep objects.

The estimators work on *exact* B-Rep entities (faces, their analytic
parameters, area centroids, oriented plane normals, axes and vertices) and
never sample points. Face correspondences are not guessed: every hypothesis
comes from an exhaustive enumeration of the candidate correspondences of one
or two anchor faces, and each hypothesis is accepted only if it maps *every*
face and *every* vertex of OLD onto NEW within tolerance.

Three estimators are provided, tried in this order by :mod:`compare`:

1. :func:`estimate_similarity` - ``x' = c_new + s * Q (x - c_old)`` with a
   uniform scale ``s`` and orthogonal ``Q`` (rotation, or rotation + mirror).
2. :func:`estimate_nonuniform` - ``x' = c_new + R E D E^T (x - c_old)`` with a
   diagonal scale ``D`` in a frame ``E`` attached to the object's own planar
   faces (stretching one dimension of a part).
3. :func:`estimate_partial_rigid` - when the shape itself changed, the rigid
   motion ``x' = R x + t`` that carries the largest unchanged part of OLD onto
   NEW (translation voting over exact face centroids).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from . import occ
from .config import Config
from .features import ObjectFeatures

_REFLECT = np.diag([-1.0, 1.0, 1.0])  # fixed reference reflection for mirror search
_PRE_TOL = 0.02  # loose dot-product pre-filter; every hypothesis is verified exactly afterwards


# --------------------------------------------------------------------------- #
# small linear-algebra helpers
# --------------------------------------------------------------------------- #
def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _perp(v):
    """A deterministic unit vector perpendicular to ``v``."""
    a = np.eye(3)[int(np.argmin(np.abs(v)))]
    return _unit(np.cross(v, a))


def triad(a1, a2, b1, b2) -> np.ndarray:
    """Exact rotation taking the frame of (a1, a2) to the frame of (b1, b2)."""

    def frame(u, v):
        e1 = _unit(u)
        e2 = _unit(np.cross(u, v))
        return np.column_stack([e1, e2, np.cross(e1, e2)])

    return frame(b1, b2) @ frame(a1, a2).T


def min_rotation(a, b) -> np.ndarray:
    """Smallest rotation taking direction ``a`` to direction ``b``."""
    a, b = _unit(a), _unit(b)
    c = float(np.clip(a @ b, -1.0, 1.0))
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        return Rotation.from_rotvec(np.pi * _perp(a)).as_matrix()
    return Rotation.from_rotvec(axis / s * np.arctan2(s, c)).as_matrix()


def kabsch(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Closed-form optimal rotation for *known* vector correspondences a->b."""
    m = (b * w[:, None]).T @ a
    u, _, vt = np.linalg.svd(m)
    d = np.sign(np.linalg.det(u @ vt)) or 1.0
    return u @ np.diag([1.0, 1.0, d]) @ vt


def rotation_angle_deg(r: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(r).as_rotvec())))


def _same_rotation(r1, r2, tol_rad) -> bool:
    return np.linalg.norm(Rotation.from_matrix(r1.T @ r2).as_rotvec()) < tol_rad


def _rank_axis(vectors, cos_tol):
    """0, 1 (all parallel) or 2+ for a set of direction vectors."""
    ref = None
    for v in vectors:
        if ref is None:
            ref = _unit(v)
        elif abs(_unit(v) @ ref) < cos_tol:
            return 2, ref
    return (0, None) if ref is None else (1, ref)


# --------------------------------------------------------------------------- #
# vectorised face tables
# --------------------------------------------------------------------------- #
class _FaceTable:
    def __init__(self, obj: ObjectFeatures, center: np.ndarray, length: float, linear=None):
        f = obj.faces
        lin = np.eye(3) if linear is None else linear
        self.n = len(f)
        self.types = np.array([x.surf_type for x in f])
        self.topo = np.array([x.n_edges * 1000 + x.n_wires for x in f])
        self.area = np.array([x.area for x in f])
        self.centroid = np.array([x.centroid for x in f]).reshape(-1, 3)
        self.offset = ((self.centroid - center) / length) @ lin.T
        nan3 = np.full(3, np.nan)
        self.normal = np.array([x.normal if x.normal is not None else nan3 for x in f]).reshape(-1, 3) @ lin.T
        self.axis = np.array([x.axis if x.axis is not None else nan3 for x in f]).reshape(-1, 3) @ lin.T
        self.has_normal = ~np.isnan(self.normal[:, 0])
        self.has_axis = ~np.isnan(self.axis[:, 0])
        nan = np.nan
        self.radius = np.array([x.radius if x.radius is not None else nan for x in f])
        self.radius2 = np.array([x.radius2 if x.radius2 is not None else nan for x in f])
        self.half_angle = np.array([x.half_angle if x.half_angle is not None else nan for x in f])


def _close_or_both_nan(a, b, rel, absolute):
    both_nan = np.isnan(a)[:, None] & np.isnan(b)[None, :]
    with np.errstate(invalid="ignore"):
        ok = np.abs(a[:, None] - b[None, :]) <= rel * np.abs(a[:, None]) + absolute
    return both_nan | np.nan_to_num(ok, nan=False).astype(bool)


def _compat(to: _FaceTable, tn: _FaceTable, s: float, cfg: Config, pos_tol: Optional[float]):
    """Which NEW faces can be the image of each OLD face under scale ``s``."""
    rel = cfg.rel_value_tol
    c = (to.types[:, None] == tn.types[None, :]) & (to.topo[:, None] == tn.topo[None, :])
    c &= np.abs(to.area[:, None] * s * s - tn.area[None, :]) <= rel * to.area[:, None] * s * s + cfg.abs_tol
    c &= _close_or_both_nan(to.radius * s, tn.radius, rel, cfg.abs_tol)
    c &= _close_or_both_nan(to.radius2 * s, tn.radius2, rel, cfg.abs_tol)
    c &= _close_or_both_nan(to.half_angle, tn.half_angle, 0.0, np.radians(cfg.angle_tol_deg))
    if pos_tol is not None:  # |offset| is invariant under rotation / mirror
        no = np.linalg.norm(to.offset, axis=1)
        nn = np.linalg.norm(tn.offset, axis=1)
        c &= np.abs(no[:, None] - nn[None, :]) <= pos_tol
    return c


def _direction_ok(to: _FaceTable, tn: _FaceTable, r: np.ndarray, cos_tol: float):
    pn = to.normal @ r.T
    pa = to.axis @ r.T
    with np.errstate(invalid="ignore"):
        nd = np.nan_to_num(pn @ tn.normal.T, nan=-2.0)
        ad = np.nan_to_num(np.abs(pa @ tn.axis.T), nan=-2.0)
    ok_n = np.where(to.has_normal[:, None], nd >= cos_tol, True)
    ok_a = np.where(to.has_axis[:, None], ad >= cos_tol, True)
    return ok_n & ok_a


def _greedy_pairs(mask: np.ndarray, cost: np.ndarray):
    """One-to-one face pairs from an admissibility mask (closest first)."""
    ii, jj = np.nonzero(mask)
    order = np.argsort(cost[ii, jj], kind="stable")
    used_i, used_j, pairs = set(), set(), []
    for k in order:
        i, j = int(ii[k]), int(jj[k])
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        pairs.append((i, j))
    return pairs


def _axial_radial(v: np.ndarray, axis: np.ndarray) -> np.ndarray:
    z = v @ axis
    r = np.linalg.norm(v - np.outer(z, axis), axis=1)
    return np.column_stack([z, r])


def _vertex_match(vo: np.ndarray, vn: np.ndarray, tol: float) -> bool:
    if len(vo) != len(vn):
        return False
    if len(vo) == 0:
        return True
    dist, idx = cKDTree(vn).query(vo)
    return bool(np.all(dist <= tol)) and len(set(idx.tolist())) == len(vo)


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
@dataclass
class Alignment:
    """Transform x' = c_new + linear @ (x - c_old)  (c_* = reference points)."""

    kind: str  # SIMILARITY | NONUNIFORM | PARTIAL
    rotation: np.ndarray  # proper rotation part
    scale: float = 1.0  # uniform scale (SIMILARITY)
    scale_axes: Optional[np.ndarray] = None  # NONUNIFORM: diag factors
    scale_frame: Optional[np.ndarray] = None  # NONUNIFORM: frame E (old coords, columns)
    mirror_normal: Optional[np.ndarray] = None  # set when mirrored: plane normal (old coords)
    determinacy: str = "full"  # full | axisymmetric | none
    symmetry_axis: Optional[np.ndarray] = None  # old symmetry axis when axisymmetric
    n_solutions: int = 1  # equivalent transforms (object symmetry)
    c_old: np.ndarray = field(default_factory=lambda: np.zeros(3))
    c_new: np.ndarray = field(default_factory=lambda: np.zeros(3))
    face_pairs: list = field(default_factory=list)
    support: float = 1.0  # matched area fraction (1.0 = complete)

    @property
    def mirror(self) -> bool:
        return self.mirror_normal is not None

    @property
    def orthogonal(self) -> np.ndarray:
        """Orthogonal part Q = R, or Q = R @ H(n) (reflection in the plane with normal n)."""
        if not self.mirror:
            return self.rotation
        return self.rotation @ _householder(self.mirror_normal)

    @property
    def linear(self) -> np.ndarray:
        if self.kind == "NONUNIFORM":
            e = self.scale_frame
            return self.rotation @ e @ np.diag(self.scale_axes) @ e.T
        return self.scale * self.orthogonal

    @property
    def matrix(self) -> np.ndarray:
        """4x4 homogeneous matrix in global coordinates."""
        m = np.eye(4)
        m[:3, :3] = self.linear
        m[:3, 3] = self.c_new - self.linear @ self.c_old
        return m

    @property
    def translation(self) -> np.ndarray:
        return self.c_new - self.c_old


# --------------------------------------------------------------------------- #
# 1. similarity (rigid + uniform scale, optionally mirrored)
# --------------------------------------------------------------------------- #
def _similarity_hypotheses(to, tn, cand, cfg, pos_tol, cos_tol):
    """Enumerate every rotation consistent with the anchor correspondences."""
    order = sorted(range(to.n), key=lambda i: (int(cand[i].sum()), -to.area[i]))

    def vecs(t, i):
        out = []
        if t.has_normal[i]:
            out.append(("nrm", t.normal[i], True))
        if np.linalg.norm(t.offset[i]) > pos_tol:
            out.append(("off", t.offset[i], True))
        if t.has_axis[i]:
            out.append(("axs", t.axis[i], False))  # unsigned
        return out

    def counterpart(t, j, kind):
        for k, v, _ in vecs(t, j):
            if k == kind:
                return v
        return None

    def signs(signed):
        return (1.0,) if signed else (1.0, -1.0)

    all_vecs = [v for i in range(to.n) for _, v, _ in vecs(to, i)]
    rank, sym_axis = _rank_axis(all_vecs, cos_tol)
    hyps: list[np.ndarray] = []

    if rank == 0:
        return [np.eye(3)], "none", None
    if rank == 1:
        anchors = [i for i in order if vecs(to, i)]
        f1 = anchors[0]
        kind, a1, signed = vecs(to, f1)[0]
        for g in np.nonzero(cand[f1])[0]:
            b = counterpart(tn, g, kind)
            if b is None:
                continue
            for sg in signs(signed):
                hyps.append(min_rotation(a1, sg * b))
        return hyps, "axisymmetric", sym_axis

    anchors = [i for i in order if vecs(to, i)][: cfg.max_anchor_faces]
    for f1 in anchors:
        v1 = vecs(to, f1)
        k1, a1, s1 = v1[0]
        # (a) one face carrying two independent directions
        second = next(((k, v, s) for k, v, s in v1[1:] if abs(_unit(v) @ _unit(a1)) < cos_tol), None)
        if second is not None:
            k2, a2, s2 = second
            for g in np.nonzero(cand[f1])[0]:
                b1, b2 = counterpart(tn, g, k1), counterpart(tn, g, k2)
                if b1 is None or b2 is None:
                    continue
                for sg1 in signs(s1):
                    for sg2 in signs(s2):
                        if abs(_unit(a1) @ _unit(a2) - _unit(sg1 * b1) @ _unit(sg2 * b2)) < _PRE_TOL:
                            hyps.append(triad(a1, a2, sg1 * b1, sg2 * b2))
        else:
            # (b) two faces with non-parallel directions
            for f2 in order:
                if f2 == f1:
                    continue
                pick = next(((k, v, s) for k, v, s in vecs(to, f2) if abs(_unit(v) @ _unit(a1)) < cos_tol), None)
                if pick is None:
                    continue
                k2, a2, s2 = pick
                for g1 in np.nonzero(cand[f1])[0]:
                    b1 = counterpart(tn, g1, k1)
                    if b1 is None:
                        continue
                    for g2 in np.nonzero(cand[f2])[0]:
                        if g2 == g1:
                            continue
                        b2 = counterpart(tn, g2, k2)
                        if b2 is None:
                            continue
                        for sg1 in signs(s1):
                            for sg2 in signs(s2):
                                bb1, bb2 = sg1 * b1, sg2 * b2
                                if abs(_unit(a1) @ _unit(a2) - _unit(bb1) @ _unit(bb2)) < _PRE_TOL:
                                    hyps.append(triad(a1, a2, bb1, bb2))
                break
        if hyps:  # one complete anchor enumeration is exhaustive
            break
        if len(hyps) >= cfg.max_hypotheses:
            break
    return hyps[: cfg.max_hypotheses], "full", None


def _verify(to, tn, cand, r, pos_tol, cos_tol):
    pred = to.offset @ r.T
    dist = np.linalg.norm(pred[:, None, :] - tn.offset[None, :, :], axis=2)
    mask = cand & (dist <= pos_tol) & _direction_ok(to, tn, r, cos_tol)
    pairs = _greedy_pairs(mask, dist)
    return pairs


def estimate_similarity(o: ObjectFeatures, n: ObjectFeatures, cfg: Config) -> Optional[Alignment]:
    if o.n_faces != n.n_faces or o.n_vertices != n.n_vertices or o.n_edges != n.n_edges:
        return None
    if o.is_solid and n.is_solid:
        s = float(np.cbrt(n.volume / o.volume))
    else:
        s = float(np.sqrt(n.area / o.area))
    if abs(n.area - s * s * o.area) > cfg.rel_value_tol * n.area + cfg.abs_tol:
        return None
    lo = o.size_measure
    ln = lo * s
    pos_tol = max(cfg.rel_pos_tol * o.diag, cfg.abs_tol) / lo
    cos_tol = float(np.cos(np.radians(cfg.angle_tol_deg)))
    vn = (n.vertices - n.centroid) / ln
    tn = _FaceTable(n, n.centroid, ln)

    best: Optional[Alignment] = None
    for mirror in (False, True) if cfg.detect_mirror else (False,):
        lin = _REFLECT if mirror else None
        to = _FaceTable(o, o.centroid, lo, lin)
        vo = ((o.vertices - o.centroid) / lo) @ (_REFLECT.T if mirror else np.eye(3))
        cand = _compat(to, tn, s, cfg, pos_tol)
        if not cand.any(axis=1).all():
            continue
        hyps, determinacy, sym_axis = _similarity_hypotheses(to, tn, cand, cfg, pos_tol, cos_tol)
        valid = []
        for r in hyps:
            pairs = _verify(to, tn, cand, r, pos_tol, cos_tol)
            if determinacy == "axisymmetric":
                # spin about the own axis is unobservable (only seam vertices move):
                # compare vertices in cylindrical coordinates (axial, radial)
                vertices_ok = _vertex_match(_axial_radial(vo @ r.T, r @ sym_axis), _axial_radial(vn, r @ sym_axis), pos_tol)
            else:
                vertices_ok = _vertex_match(vo @ r.T, vn, pos_tol)
            if len(pairs) == to.n and vertices_ok:
                r = _refine(to, tn, pairs, r) if determinacy == "full" else r
                if not any(_same_rotation(r, q, np.radians(cfg.angle_tol_deg)) for q, _ in valid):
                    valid.append((r, pairs))
        if not valid:
            continue
        normal = None
        if mirror:  # Q = r @ REFLECT, re-expressed as Rot(n, psi) @ H(n)
            valid = [(_rotoreflection(r @ _REFLECT), p) for r, p in valid]
            r, pairs = min(valid, key=lambda rp: rotation_angle_deg(rp[0][1]))
            normal, r = r
        else:
            r, pairs = min(valid, key=lambda rp: rotation_angle_deg(rp[0]))
        best = Alignment(
            kind="SIMILARITY",
            rotation=r,
            scale=s,
            mirror_normal=normal,
            determinacy=determinacy,
            symmetry_axis=sym_axis if determinacy == "axisymmetric" else None,
            n_solutions=len(valid),
            c_old=o.centroid.copy(),
            c_new=n.centroid.copy(),
            face_pairs=pairs,
            support=1.0,
        )
        break  # a proper (non-mirrored) solution always wins
    return best


def _refine(to, tn, pairs, r):
    """Least-squares rotation over all verified correspondences (closed form)."""
    a, b, w = [], [], []
    for i, j in pairs:
        wt = to.area[i]
        if np.linalg.norm(to.offset[i]) > 0:
            a.append(to.offset[i]); b.append(tn.offset[j]); w.append(wt)
        if to.has_normal[i]:
            a.append(to.normal[i]); b.append(tn.normal[j]); w.append(wt)
        if to.has_axis[i]:
            sg = 1.0 if (r @ to.axis[i]) @ tn.axis[j] >= 0 else -1.0
            a.append(to.axis[i]); b.append(sg * tn.axis[j]); w.append(wt)
    if len(a) < 2:
        return r
    return kabsch(np.array(a), np.array(b), np.array(w))


def _householder(n: np.ndarray) -> np.ndarray:
    n = _unit(n)
    return np.eye(3) - 2.0 * np.outer(n, n)


def _rotoreflection(q: np.ndarray):
    """Decompose an improper Q into Rot(n, psi) @ H(n); returns (n, Rot(n, psi)).

    -Q is a proper rotation about n, and H(n) commutes with every rotation
    about n, so n is the axis of -Q and Rot = Q @ H(n). A pure mirror gives
    Rot = I.
    """
    rv = Rotation.from_matrix(-q).as_rotvec()
    phi = np.linalg.norm(rv)
    n = rv / phi if phi > 1e-12 else np.array([0.0, 0.0, 1.0])  # Q = -I: any axis
    if n[int(np.argmax(np.abs(n)))] < 0:  # canonical sign for the plane normal
        n = -n
    return n, q @ _householder(n)


# --------------------------------------------------------------------------- #
# 2. non-uniform (anisotropic) scaling
# --------------------------------------------------------------------------- #
def _unique_dirs(vectors, weights, cos_tol):
    out = []
    for v, w in sorted(zip(vectors, weights), key=lambda x: -x[1]):
        v = _unit(v)
        for k, (u, wu) in enumerate(out):
            if u @ v >= cos_tol:
                out[k] = (u, wu + w)
                break
        else:
            out.append((v, w))
    return [u for u, _ in out]


def _object_dirs(o: ObjectFeatures, cos_tol):
    vecs, wts = [], []
    for f in o.faces:
        if f.normal is not None:
            vecs.append(f.normal); wts.append(f.area)
        if f.axis is not None:
            vecs.append(f.axis); wts.append(f.area)
            vecs.append(-f.axis); wts.append(f.area)
    return _unique_dirs(vecs, wts, cos_tol)


def _canonical_frame(e: np.ndarray, d: np.ndarray, cos_tol: float):
    """If the scaling frame is axis aligned, reorder it to +X, +Y, +Z."""
    k = np.argmax(np.abs(e), axis=0)
    if sorted(k.tolist()) != [0, 1, 2] or np.any(np.abs(e[k, range(3)]) < cos_tol):
        return e, d
    order = np.argsort(k)
    return np.eye(3), d[order]


def _extents(shape, rot: np.ndarray):
    lo, hi = occ.optimal_bbox(occ.transform_shape(shape, rot, np.zeros(3)))
    return hi - lo


def estimate_nonuniform(o: ObjectFeatures, n: ObjectFeatures, cfg: Config) -> Optional[Alignment]:
    if not (o.is_solid and n.is_solid):
        return None
    if (o.n_faces, o.n_edges, o.n_vertices) != (n.n_faces, n.n_edges, n.n_vertices):
        return None
    if o.face_type_hist() != n.face_type_hist() or o.wl[1] != n.wl[1]:
        return None
    cos_tol = float(np.cos(np.radians(cfg.angle_tol_deg)))
    do, dn = _object_dirs(o, cos_tol), _object_dirs(n, cos_tol)
    if not do or not dn:
        return None
    a1 = do[0]
    # prefer a perpendicular second direction (typical for prismatic parts)
    a2 = next((x for x in do[1:] if abs(x @ a1) < 1e-6), None)
    if a2 is None:
        a2 = next((x for x in do[1:] if abs(x @ a1) < cos_tol), None)
    if a2 is None:  # axisymmetric: rotation about the own axis is unobservable
        a2 = _perp(a1)
        pairs = [(b, _perp(b)) for b in dn]
    else:
        target = a1 @ a2
        pairs = [(b1, b2) for b1 in dn for b2 in dn if abs(b1 @ b2 - target) < _PRE_TOL and abs(b1 @ b2) < cos_tol]
    e2 = _unit(a2 - (a2 @ a1) * a1)
    e = np.column_stack([a1, e2, np.cross(a1, e2)])

    hyps = []
    for b1, b2 in pairs:
        r = triad(a1, a2, b1, b2)
        if not any(_same_rotation(r, q, np.radians(cfg.angle_tol_deg)) for q in hyps):
            hyps.append(r)
    ext_o = _extents(o.shape, e.T)
    pos_tol = max(cfg.rel_pos_tol * max(o.diag, n.diag), cfg.abs_tol)
    vratio = n.volume / o.volume
    valid = []
    for r in sorted(hyps, key=rotation_angle_deg)[:64]:
        ext_n = _extents(n.shape, (r @ e).T)
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(ext_o > cfg.abs_tol, ext_n / ext_o, 1.0)
        if abs(np.prod(d) - vratio) > cfg.rel_value_tol * 10 * vratio:
            continue
        p = ((o.vertices - o.centroid) @ e) * d
        q = (n.vertices - n.centroid) @ r @ e
        if _vertex_match(p, q, pos_tol):
            valid.append((r, d))
    if not valid:
        return None
    # simplest explanation first: fewest stretched axes, then smallest rotation
    r, d = min(valid, key=lambda rd: (int(np.sum(np.abs(rd[1] - 1) > cfg.scale_eps)), rotation_angle_deg(rd[0])))
    e_out, d_out = _canonical_frame(e, d, cos_tol)
    return Alignment(
        kind="NONUNIFORM",
        rotation=r,
        scale_axes=d_out,
        scale_frame=e_out,
        n_solutions=len(valid),
        c_old=o.centroid.copy(),
        c_new=n.centroid.copy(),
        support=1.0,
    )


# --------------------------------------------------------------------------- #
# 3. partial rigid alignment (shape changed)
# --------------------------------------------------------------------------- #
def estimate_partial_rigid(o: ObjectFeatures, n: ObjectFeatures, cfg: Config) -> Optional[Alignment]:
    """Rigid motion of the unchanged faces when OLD and NEW differ in shape."""
    if not o.faces or not n.faces:
        return None
    cos_tol = float(np.cos(np.radians(cfg.angle_tol_deg)))
    to = _FaceTable(o, np.zeros(3), 1.0)
    tn = _FaceTable(n, np.zeros(3), 1.0)
    rel = cfg.rel_value_tol
    cand = (to.types[:, None] == tn.types[None, :])
    cand &= np.abs(to.area[:, None] - tn.area[None, :]) <= rel * to.area[:, None] + cfg.abs_tol
    cand &= _close_or_both_nan(to.radius, tn.radius, rel, cfg.abs_tol)
    cand &= _close_or_both_nan(to.radius2, tn.radius2, rel, cfg.abs_tol)
    cand &= _close_or_both_nan(to.half_angle, tn.half_angle, 0.0, np.radians(cfg.angle_tol_deg))
    if not cand.any():
        return None
    pos_tol = max(cfg.rel_pos_tol * max(o.diag, n.diag), cfg.abs_tol)

    # rotation hypotheses from oriented plane normals / axes of compatible faces
    hyps = [np.eye(3)]
    dir_faces = [i for i in np.argsort(-to.area) if cand[i].any() and (to.has_normal[i] or to.has_axis[i])]

    def d(t, i):
        return t.normal[i] if t.has_normal[i] else t.axis[i]

    if dir_faces:
        f1 = dir_faces[0]
        f2 = next((i for i in dir_faces[1:] if abs(d(to, i) @ d(to, f1)) < cos_tol), None)
        for g1 in np.nonzero(cand[f1])[0]:
            for sg1 in ((1.0,) if to.has_normal[f1] else (1.0, -1.0)):
                b1 = sg1 * d(tn, g1)
                if f2 is None:
                    hyps.append(min_rotation(d(to, f1), b1))
                    continue
                for g2 in np.nonzero(cand[f2])[0]:
                    if g2 == g1:
                        continue
                    for sg2 in ((1.0,) if to.has_normal[f2] else (1.0, -1.0)):
                        b2 = sg2 * d(tn, g2)
                        if abs(d(to, f1) @ d(to, f2) - b1 @ b2) < _PRE_TOL:
                            hyps.append(triad(d(to, f1), d(to, f2), b1, b2))
                if len(hyps) > cfg.max_hypotheses:
                    break
    uniq = []
    for r in hyps:
        if not any(_same_rotation(r, q, np.radians(cfg.angle_tol_deg)) for q in uniq):
            uniq.append(r)

    total = to.area.sum() + tn.area.sum()
    best = None
    for r in uniq:
        mask = cand & _direction_ok(to, tn, r, cos_tol)
        ii, jj = np.nonzero(mask)
        if len(ii) == 0:
            continue
        t_votes = tn.centroid[jj] - to.centroid[ii] @ r.T
        q = 10 * pos_tol
        keys = np.round(t_votes / q).astype(np.int64)
        support: dict[tuple, set] = {}
        for k, key in enumerate(map(tuple, keys)):
            support.setdefault(key, set()).add(int(ii[k]))
        ranked = sorted(support.items(), key=lambda kv: -to.area[list(kv[1])].sum())[:3]
        for key, _ in ranked:
            sel = np.all(keys == np.array(key), axis=1)
            t = np.average(t_votes[sel], axis=0, weights=to.area[ii[sel]])
            pred = to.centroid @ r.T + t
            dist = np.linalg.norm(pred[:, None, :] - tn.centroid[None, :, :], axis=2)
            pairs = _greedy_pairs(mask & (dist <= pos_tol), dist)
            if not pairs:
                continue
            area = sum(to.area[i] + tn.area[j] for i, j in pairs) / total
            score = (round(area, 9), -rotation_angle_deg(r), -np.linalg.norm(t))
            if best is None or score > best[0]:
                best = (score, r, t, pairs, area)
    if best is None or best[4] < cfg.partial_min_support:
        return None
    _, r, t, pairs, area = best
    # express about the old centroid: x' = (R c_old + t) + R (x - c_old)
    return Alignment(
        kind="PARTIAL",
        rotation=r,
        c_old=o.centroid.copy(),
        c_new=r @ o.centroid + t,
        face_pairs=pairs,
        support=float(area),
    )

"""B-Rep feature extraction.

Everything here is computed from the exact B-Rep stored in the STEP file:

* mass properties (volume, area, centroid) by exact Gauss integration,
* precise AABB from the underlying curves / surfaces,
* per-face surface type and analytic parameters (plane normal, cylinder
  radius / axis, cone half-angle, ...), area and centroid,
* per-edge curve type and length,
* the vertex positions,
* the face adjacency graph (faces sharing an edge) and a Weisfeiler-Lehman
  signature of that graph, used as a topology fingerprint.

No tessellation and no point sampling is involved.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from OCP.GeomAbs import GeomAbs_CurveType, GeomAbs_SurfaceType

from . import occ
from .occ import (
    BRep_Tool,
    BRepAdaptor_Curve,
    BRepAdaptor_Surface,
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_SHELL,
    TopAbs_SOLID,
    TopAbs_VERTEX,
    TopAbs_WIRE,
)

_SURF_NAMES = {
    GeomAbs_SurfaceType.GeomAbs_Plane: "PLANE",
    GeomAbs_SurfaceType.GeomAbs_Cylinder: "CYLINDER",
    GeomAbs_SurfaceType.GeomAbs_Cone: "CONE",
    GeomAbs_SurfaceType.GeomAbs_Sphere: "SPHERE",
    GeomAbs_SurfaceType.GeomAbs_Torus: "TORUS",
    GeomAbs_SurfaceType.GeomAbs_BezierSurface: "BEZIER",
    GeomAbs_SurfaceType.GeomAbs_BSplineSurface: "BSPLINE",
    GeomAbs_SurfaceType.GeomAbs_SurfaceOfRevolution: "REVOLUTION",
    GeomAbs_SurfaceType.GeomAbs_SurfaceOfExtrusion: "EXTRUSION",
    GeomAbs_SurfaceType.GeomAbs_OffsetSurface: "OFFSET",
    GeomAbs_SurfaceType.GeomAbs_OtherSurface: "OTHER",
}

_CURVE_NAMES = {
    GeomAbs_CurveType.GeomAbs_Line: "LINE",
    GeomAbs_CurveType.GeomAbs_Circle: "CIRCLE",
    GeomAbs_CurveType.GeomAbs_Ellipse: "ELLIPSE",
    GeomAbs_CurveType.GeomAbs_Hyperbola: "HYPERBOLA",
    GeomAbs_CurveType.GeomAbs_Parabola: "PARABOLA",
    GeomAbs_CurveType.GeomAbs_BezierCurve: "BEZIER",
    GeomAbs_CurveType.GeomAbs_BSplineCurve: "BSPLINE",
    GeomAbs_CurveType.GeomAbs_OffsetCurve: "OFFSET",
    GeomAbs_CurveType.GeomAbs_OtherCurve: "OTHER",
}

WL_ITERATIONS = 3


@dataclass
class FaceFeature:
    index: int
    surf_type: str
    area: float
    centroid: np.ndarray
    n_edges: int
    n_wires: int
    normal: Optional[np.ndarray] = None  # oriented outward normal (planes)
    axis: Optional[np.ndarray] = None  # unsigned axis (cylinder / cone / torus / revolution)
    radius: Optional[float] = None  # cylinder / sphere radius, torus major radius
    radius2: Optional[float] = None  # torus minor radius
    half_angle: Optional[float] = None  # cone semi-angle (scale invariant)
    neighbors: list[int] = field(default_factory=list)


@dataclass
class EdgeFeature:
    curve_type: str
    length: float


@dataclass
class ObjectFeatures:
    idx: int
    name: str
    shape: object
    is_solid: bool
    volume: float
    area: float
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    faces: list[FaceFeature]
    edges: list[EdgeFeature]
    vertices: np.ndarray  # (n, 3)
    n_shells: int
    wl: list[Counter]  # Weisfeiler-Lehman label multisets per iteration

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def bbox_size(self) -> np.ndarray:
        return self.bbox_max - self.bbox_min

    @property
    def diag(self) -> float:
        return float(np.linalg.norm(self.bbox_size))

    @property
    def size_measure(self) -> float:
        """Characteristic length: cbrt(volume) for solids, sqrt(area) otherwise."""
        if self.is_solid and self.volume > 0:
            return float(np.cbrt(self.volume))
        return float(np.sqrt(max(self.area, 1e-300)))

    @property
    def euler_characteristic(self) -> int:
        return self.n_vertices - self.n_edges + self.n_faces

    def face_type_hist(self) -> Counter:
        return Counter(f.surf_type for f in self.faces)

    def edge_type_hist(self) -> Counter:
        return Counter(e.curve_type for e in self.edges)

    def summary(self) -> dict:
        return {
            "idx": self.idx,
            "name": self.name,
            "solid": self.is_solid,
            "volume": self.volume,
            "area": self.area,
            "centroid": self.centroid.tolist(),
            "bbox_min": self.bbox_min.tolist(),
            "bbox_max": self.bbox_max.tolist(),
            "faces": self.n_faces,
            "edges": self.n_edges,
            "vertices": self.n_vertices,
            "face_types": dict(sorted(self.face_type_hist().items())),
            "edge_types": dict(sorted(self.edge_type_hist().items())),
        }


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _face_feature(index: int, face) -> FaceFeature:
    adaptor = BRepAdaptor_Surface(face, True)
    stype = adaptor.GetType()
    props = occ.surface_props(face)
    ff = FaceFeature(
        index=index,
        surf_type=_SURF_NAMES.get(stype, "OTHER"),
        area=props.Mass(),
        centroid=occ.pnt(props.CentreOfMass()),
        n_edges=sum(1 for e in occ.explore(face, TopAbs_EDGE) if not BRep_Tool.Degenerated_s(occ.as_edge(e))),
        n_wires=sum(1 for _ in occ.explore(face, TopAbs_WIRE)),
    )
    if stype == GeomAbs_SurfaceType.GeomAbs_Plane:
        pos = adaptor.Plane().Position()
        # D1U x D1V direction; valid for both direct and indirect frames
        n = np.cross(occ.dir_(pos.XDirection()), occ.dir_(pos.YDirection()))
        if occ.is_reversed(face):
            n = -n
        ff.normal = _unit(n)
    elif stype == GeomAbs_SurfaceType.GeomAbs_Cylinder:
        cyl = adaptor.Cylinder()
        ff.radius = cyl.Radius()
        ff.axis = _unit(occ.dir_(cyl.Axis().Direction()))
    elif stype == GeomAbs_SurfaceType.GeomAbs_Cone:
        cone = adaptor.Cone()
        ff.half_angle = abs(cone.SemiAngle())
        ff.axis = _unit(occ.dir_(cone.Axis().Direction()))
    elif stype == GeomAbs_SurfaceType.GeomAbs_Sphere:
        ff.radius = adaptor.Sphere().Radius()
    elif stype == GeomAbs_SurfaceType.GeomAbs_Torus:
        tor = adaptor.Torus()
        ff.radius = tor.MajorRadius()
        ff.radius2 = tor.MinorRadius()
        ff.axis = _unit(occ.dir_(tor.Axis().Direction()))
    elif stype == GeomAbs_SurfaceType.GeomAbs_SurfaceOfRevolution:
        ff.axis = _unit(occ.dir_(adaptor.AxeOfRevolution().Direction()))
    elif stype == GeomAbs_SurfaceType.GeomAbs_SurfaceOfExtrusion:
        ff.axis = _unit(occ.dir_(adaptor.Direction()))
    return ff


def _wl_signature(faces: list[FaceFeature]) -> list[Counter]:
    """Weisfeiler-Lehman refinement of the labelled face adjacency graph."""
    labels = [f"{f.surf_type}:{f.n_edges}:{f.n_wires}" for f in faces]
    history = [Counter(labels)]
    for _ in range(WL_ITERATIONS):
        new = []
        for f in faces:
            ctx = labels[f.index] + "|" + ",".join(sorted(labels[n] for n in f.neighbors))
            new.append(hashlib.md5(ctx.encode()).hexdigest()[:12])
        labels = new
        history.append(Counter(labels))
    return history


def extract_features(idx: int, name: str, shape) -> ObjectFeatures:
    face_map = occ.indexed_map(shape, TopAbs_FACE)
    faces = [_face_feature(i, occ.as_face(face_map.FindKey(i + 1))) for i in range(face_map.Size())]

    # face adjacency through shared (non-degenerate) edges
    edge_faces = occ.ancestor_map(shape, TopAbs_EDGE, TopAbs_FACE)
    edges: list[EdgeFeature] = []
    adjacency = [set() for _ in faces]
    for k in range(1, edge_faces.Size() + 1):
        edge = occ.as_edge(edge_faces.FindKey(k))
        if BRep_Tool.Degenerated_s(edge):
            continue
        ctype = _CURVE_NAMES.get(BRepAdaptor_Curve(edge).GetType(), "OTHER")
        edges.append(EdgeFeature(ctype, occ.edge_length(edge)))
        owners = sorted({face_map.FindIndex(f) - 1 for f in occ.iter_list(edge_faces.FindFromIndex(k))})
        for a in owners:
            for b in owners:
                if a != b:
                    adjacency[a].add(b)
    for f, nb in zip(faces, adjacency):
        f.neighbors = sorted(nb)

    vmap = occ.indexed_map(shape, TopAbs_VERTEX)
    vertices = np.array(
        [occ.pnt(BRep_Tool.Pnt_s(occ.as_vertex(vmap.FindKey(i + 1)))) for i in range(vmap.Size())],
        dtype=float,
    ).reshape(-1, 3)

    is_solid = shape.ShapeType() == TopAbs_SOLID
    sprops = occ.surface_props(shape)
    area = sprops.Mass()
    volume = 0.0
    centroid = occ.pnt(sprops.CentreOfMass())
    if is_solid:
        vprops = occ.volume_props(shape)
        volume = vprops.Mass()
        if volume < 0:  # inside-out solid
            volume = -volume
        if volume > 0:
            centroid = occ.pnt(vprops.CentreOfMass())

    bmin, bmax = occ.optimal_bbox(shape)
    return ObjectFeatures(
        idx=idx,
        name=name,
        shape=shape,
        is_solid=is_solid and volume > 0,
        volume=volume,
        area=area,
        centroid=centroid,
        bbox_min=bmin,
        bbox_max=bmax,
        faces=faces,
        edges=edges,
        vertices=vertices,
        n_shells=sum(1 for _ in occ.explore(shape, TopAbs_SHELL)),
        wl=_wl_signature(faces),
    )

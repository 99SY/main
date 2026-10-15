"""STEP B-Rep 기반 OLD/NEW 객체 대응 및 설계 변경 유형 판정 — 단일 파일 버전.

사용법: 아래 [입력 설정]의 OLD_STEP / NEW_STEP 경로를 수정한 뒤
VS Code에서 ▷ (Run Python File) 또는 F5 로 실행하면 결과가 터미널에 바로 출력됩니다.
(파일 저장 없음. 필요 패키지: pip install cadquery-ocp numpy scipy)

파이프라인
  1. OLD/NEW STEP 읽기, 객체(Solid) 분리, IDX 부여 (OLD 0..N-1, NEW 0..M-1)
  2. 객체별 B-Rep 특징 추출 (면/모서리 유형, 해석적 파라미터, 부피/면적/중심, 정점, 면 인접 그래프)
  3. OLD 0부터 AABB 생성 → AABB가 겹치는 NEW 객체만 후보
  4. 후보 쌍마다 변환 추정(앵커 면 대응 전수 열거 + 모든 면·정점 검증) 및 유사도 계산
  5. 헝가리안 알고리즘으로 전역 1:1 매칭
  6. STAY / MODIFIED(TRANSLATION, ROTATION, UNIFORM_SCALING, NON_UNIFORM_SCALING, MIRROR,
     SHAPE_CHANGE) / INSERTION / DELETION 판정
머신러닝, RANSAC, ICP, PCA, 점군 샘플링은 사용하지 않습니다.
"""
from __future__ import annotations

# =============================================================================
# [입력 설정] 비교할 STEP 파일 경로 (절대 경로 또는 이 파일 기준 상대 경로)
# =============================================================================
OLD_STEP = r"examples/old.step"
NEW_STEP = r"examples/new.step"
SHOW_DETAILS = True  # True: 객체별 상세 수치(이동/회전/스케일/형상)까지 출력

# =============================================================================
# imports
# =============================================================================
import hashlib
import sys
import time
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepGProp import BRepGProp
from OCP.Bnd import Bnd_Box
from OCP.GeomAbs import GeomAbs_CurveType, GeomAbs_SurfaceType
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.STEPControl import STEPControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import (
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_REVERSED,
    TopAbs_SHELL,
    TopAbs_SOLID,
    TopAbs_VERTEX,
    TopAbs_WIRE,
)
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Trsf
from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ShapeTool

try:  # OCP >= 7.8
    from OCP.collections import (
        IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher as _AncestorMap,
        IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher as _IndexedMap,
        Sequence_TDF_Label as LabelSequence,
    )
except ImportError:  # OCP <= 7.7
    from OCP.TDF import TDF_LabelSequence as LabelSequence  # type: ignore
    from OCP.TopTools import (  # type: ignore
        TopTools_IndexedDataMapOfShapeListOfShape as _AncestorMap,
        TopTools_IndexedMapOfShape as _IndexedMap,
    )

# =============================================================================
# ■ section: occ
# =============================================================================
def _downcast(name):
    fn = getattr(TopoDS, name + "_s", None) or getattr(TopoDS, name)
    return fn


as_face = _downcast("Face")
as_edge = _downcast("Edge")
as_vertex = _downcast("Vertex")


def indexed_map(shape, kind):
    """Map of unique sub-shapes of ``kind`` (1-based index, like OCC)."""
    m = _IndexedMap()
    TopExp.MapShapes_s(shape, kind, m)
    return m


def ancestor_map(shape, kind, ancestor_kind):
    m = _AncestorMap()
    TopExp.MapShapesAndAncestors_s(shape, kind, ancestor_kind, m)
    return m


def iter_list(lst):
    """Iterate an OCC ``TopTools_ListOfShape`` regardless of binding version."""
    try:
        return list(lst)
    except TypeError:  # very old bindings
        from OCP.TopTools import TopTools_ListIteratorOfListOfShape  # type: ignore

        out = []
        it = TopTools_ListIteratorOfListOfShape(lst)
        while it.More():
            out.append(it.Value())
            it.Next()
        return out


def explore(shape, kind):
    exp = TopExp_Explorer(shape, kind)
    while exp.More():
        yield exp.Current()
        exp.Next()


def volume_props(shape):
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return props


def surface_props(shape):
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, props)
    return props


def edge_length(edge) -> float:
    props = GProp_GProps()
    BRepGProp.LinearProperties_s(edge, props)
    return props.Mass()


def optimal_bbox(shape):
    """Precise AABB from the exact B-Rep geometry (no triangulation)."""
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, box, False, False)
    if box.IsVoid():
        raise ValueError("empty shape: bounding box is void")
    return pnt(box.CornerMin()), pnt(box.CornerMax())


def pnt(p) -> np.ndarray:
    return np.array([p.X(), p.Y(), p.Z()], dtype=float)


vec = pnt
dir_ = pnt


def is_reversed(shape) -> bool:
    return shape.Orientation() == TopAbs_REVERSED


def transform_shape(shape, matrix3: np.ndarray, translation: np.ndarray):
    """Apply an exact rigid transform (3x3 rotation + translation) to a shape."""
    t = gp_Trsf()
    m = matrix3
    t.SetValues(
        m[0, 0], m[0, 1], m[0, 2], translation[0],
        m[1, 0], m[1, 1], m[1, 2], translation[1],
        m[2, 0], m[2, 1], m[2, 2], translation[2],
    )
    return BRepBuilderAPI_Transform(shape, t, True).Shape()


def silence_occ():
    """Remove the default OCC console printer (transfer statistics spam)."""
    try:
        from OCP.Message import Message

        messenger = Message.DefaultMessenger_s()
        printers = messenger.Printers()
        for p in list(printers):
            messenger.RemovePrinter(p)
    except Exception:  # pragma: no cover - purely cosmetic
        pass


# re-exported for convenience in other modules


# =============================================================================
# ■ section: step_io
# =============================================================================
@dataclass
class CadObject:
    idx: int
    name: str
    shape: object  # TopoDS_Shape


def _label_name(label) -> str:
    attr = TDataStd_Name()
    if label.FindAttribute(TDataStd_Name.GetID_s(), attr):
        return attr.Get().ToExtString()
    return ""


def _split(shape):
    solids = list(explore(shape, TopAbs_SOLID))
    if solids:
        return solids
    shells = list(explore(shape, TopAbs_SHELL))
    return shells or [shape]


def _read_xcaf(path: str):
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    reader.SetColorMode(False)
    reader.SetLayerMode(False)
    if reader.ReadFile(path) != IFSelect_RetDone:
        raise IOError(f"cannot read STEP file: {path}")
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    if not reader.Transfer(doc):
        raise IOError(f"STEP transfer failed: {path}")
    tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())

    found: list[tuple[str, object]] = []

    def walk(label, loc: TopLoc_Location, inherited: str):
        name = _label_name(label) or inherited
        if XCAFDoc_ShapeTool.IsAssembly_s(label):
            comps = LabelSequence()
            XCAFDoc_ShapeTool.GetComponents_s(label, comps, False)
            for comp in comps:
                ref = TDF_Label()
                comp_loc = XCAFDoc_ShapeTool.GetLocation_s(comp)
                target = ref if XCAFDoc_ShapeTool.GetReferredShape_s(comp, ref) else comp
                walk(target, loc.Multiplied(comp_loc), _label_name(comp) or name)
            return
        shape = XCAFDoc_ShapeTool.GetShape_s(label)
        if shape.IsNull():
            return
        if not loc.IsIdentity():
            shape = shape.Moved(loc)
        parts = _split(shape)
        for k, part in enumerate(parts):
            found.append((name if len(parts) == 1 else f"{name}#{k}", part))

    roots = LabelSequence()
    tool.GetFreeShapes(roots)
    for root in roots:
        walk(root, TopLoc_Location(), "")
    return found


def _read_plain(path: str):
    reader = STEPControl_Reader()
    if reader.ReadFile(path) != IFSelect_RetDone:
        raise IOError(f"cannot read STEP file: {path}")
    reader.TransferRoots()
    return [("", s) for s in _split(reader.OneShape())]


def load_step(path: str | Path, verbose: bool = False) -> list[CadObject]:
    """Read a STEP file and return its objects with sequential indices."""
    path = str(path)
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    if not verbose:
        silence_occ()
    try:
        items = _read_xcaf(path)
    except Exception:
        items = []
    if not items:
        items = _read_plain(path)
    return [
        CadObject(idx=i, name=name or f"object_{i}", shape=shape)
        for i, (name, shape) in enumerate(items)
    ]


# =============================================================================
# ■ section: features
# =============================================================================
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
    props = surface_props(face)
    ff = FaceFeature(
        index=index,
        surf_type=_SURF_NAMES.get(stype, "OTHER"),
        area=props.Mass(),
        centroid=pnt(props.CentreOfMass()),
        n_edges=sum(1 for e in explore(face, TopAbs_EDGE) if not BRep_Tool.Degenerated_s(as_edge(e))),
        n_wires=sum(1 for _ in explore(face, TopAbs_WIRE)),
    )
    if stype == GeomAbs_SurfaceType.GeomAbs_Plane:
        pos = adaptor.Plane().Position()
        # D1U x D1V direction; valid for both direct and indirect frames
        n = np.cross(dir_(pos.XDirection()), dir_(pos.YDirection()))
        if is_reversed(face):
            n = -n
        ff.normal = _unit(n)
    elif stype == GeomAbs_SurfaceType.GeomAbs_Cylinder:
        cyl = adaptor.Cylinder()
        ff.radius = cyl.Radius()
        ff.axis = _unit(dir_(cyl.Axis().Direction()))
    elif stype == GeomAbs_SurfaceType.GeomAbs_Cone:
        cone = adaptor.Cone()
        ff.half_angle = abs(cone.SemiAngle())
        ff.axis = _unit(dir_(cone.Axis().Direction()))
    elif stype == GeomAbs_SurfaceType.GeomAbs_Sphere:
        ff.radius = adaptor.Sphere().Radius()
    elif stype == GeomAbs_SurfaceType.GeomAbs_Torus:
        tor = adaptor.Torus()
        ff.radius = tor.MajorRadius()
        ff.radius2 = tor.MinorRadius()
        ff.axis = _unit(dir_(tor.Axis().Direction()))
    elif stype == GeomAbs_SurfaceType.GeomAbs_SurfaceOfRevolution:
        ff.axis = _unit(dir_(adaptor.AxeOfRevolution().Direction()))
    elif stype == GeomAbs_SurfaceType.GeomAbs_SurfaceOfExtrusion:
        ff.axis = _unit(dir_(adaptor.Direction()))
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
    face_map = indexed_map(shape, TopAbs_FACE)
    faces = [_face_feature(i, as_face(face_map.FindKey(i + 1))) for i in range(face_map.Size())]

    # face adjacency through shared (non-degenerate) edges
    edge_faces = ancestor_map(shape, TopAbs_EDGE, TopAbs_FACE)
    edges: list[EdgeFeature] = []
    adjacency = [set() for _ in faces]
    for k in range(1, edge_faces.Size() + 1):
        edge = as_edge(edge_faces.FindKey(k))
        if BRep_Tool.Degenerated_s(edge):
            continue
        ctype = _CURVE_NAMES.get(BRepAdaptor_Curve(edge).GetType(), "OTHER")
        edges.append(EdgeFeature(ctype, edge_length(edge)))
        owners = sorted({face_map.FindIndex(f) - 1 for f in iter_list(edge_faces.FindFromIndex(k))})
        for a in owners:
            for b in owners:
                if a != b:
                    adjacency[a].add(b)
    for f, nb in zip(faces, adjacency):
        f.neighbors = sorted(nb)

    vmap = indexed_map(shape, TopAbs_VERTEX)
    vertices = np.array(
        [pnt(BRep_Tool.Pnt_s(as_vertex(vmap.FindKey(i + 1)))) for i in range(vmap.Size())],
        dtype=float,
    ).reshape(-1, 3)

    is_solid = shape.ShapeType() == TopAbs_SOLID
    sprops = surface_props(shape)
    area = sprops.Mass()
    volume = 0.0
    centroid = pnt(sprops.CentreOfMass())
    if is_solid:
        vprops = volume_props(shape)
        volume = vprops.Mass()
        if volume < 0:  # inside-out solid
            volume = -volume
        if volume > 0:
            centroid = pnt(vprops.CentreOfMass())

    bmin, bmax = optimal_bbox(shape)
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
        n_shells=sum(1 for _ in explore(shape, TopAbs_SHELL)),
        wl=_wl_signature(faces),
    )


# =============================================================================
# ■ section: config
# =============================================================================
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


# =============================================================================
# ■ section: transform
# =============================================================================
_REFLECT = np.diag([-1.0, 1.0, 1.0])  # fixed reference reflection for mirror search
_PRE_TOL = 0.02  # loose dot-product pre-filter; every hypothesis is verified exactly afterwards


# --------------------------------------------------------------------------- #
# small linear-algebra helpers
# --------------------------------------------------------------------------- #
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
    lo, hi = optimal_bbox(transform_shape(shape, rot, np.zeros(3)))
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


# =============================================================================
# ■ section: compare
# =============================================================================
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


# =============================================================================
# ■ section: pipeline
# =============================================================================
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


# =============================================================================
# ■ section: report
# =============================================================================
def _fmt(x, nd=4):
    return f"{x:+.{nd}f}"


def describe(entry: dict) -> str:
    st = entry["status"]
    if st in ("INSERTION", "DELETION"):
        return st
    if st == "STAY":
        return "STAY"
    tr = entry["transformations"]
    parts = []
    if tr["translation"].get("detected"):
        t = tr["translation"]
        parts.append(f"TRANSLATION(dx={_fmt(t['dx'])}, dy={_fmt(t['dy'])}, dz={_fmt(t['dz'])})")
    if tr["rotation"].get("detected"):
        r = tr["rotation"]
        axis = ", ".join(f"{a:+.4f}" for a in r["axis"])
        txt = f"ROTATION({r['angle_deg']:.4f} deg about [{axis}] {r['axis_label']}"
        if r["determinacy"] == "axisymmetric":
            txt += ", axisymmetric"
        parts.append(txt + ")")
    sc = tr["scaling"]
    if sc.get("type") == "uniform":
        parts.append(f"UNIFORM_SCALING(s={sc['factor']:.6f})")
    elif sc.get("type") == "non_uniform":
        f = ", ".join(f"{v:.6f}" for v in sc["axis_factors"])
        parts.append(f"NON_UNIFORM_SCALING([{f}])")
    if tr["mirror"].get("detected"):
        parts.append("MIRROR")
    if "SHAPE_CHANGE" in entry["change_types"]:
        sh = entry["shape"]
        vr = sh.get("volume_ratio")
        parts.append(f"SHAPE_CHANGE(vol x{vr:.4f}, faces {sh['faces_old']}->{sh['faces_new']})" if vr else "SHAPE_CHANGE")
    return "MODIFIED: " + " + ".join(parts)


def format_text(report: dict) -> str:
    s = report["summary"]
    lines = [
        f"OLD objects: {s['old_objects']}   NEW objects: {s['new_objects']}   matched: {s['matched']}",
        "status: " + ", ".join(f"{k}={v}" for k, v in sorted(s["status_counts"].items())),
        "change types: " + ", ".join(f"{k}={v}" for k, v in sorted(s["change_type_counts"].items())),
        "",
        f"{'OLD':<24} {'NEW':<24} RESULT",
        "-" * 100,
    ]
    for e in report["results"]:
        o = f"[{e['old']['idx']}] {e['old']['name']}" if e["old"] else "-"
        n = f"[{e['new']['idx']}] {e['new']['name']}" if e["new"] else "-"
        lines.append(f"{o:<24} {n:<24} {describe(e)}")
    return "\n".join(lines)


def format_details(report: dict) -> str:
    """Per-object breakdown of every detected change component."""
    out = []
    for e in report["results"]:
        o = f"OLD[{e['old']['idx']}] {e['old']['name']}" if e["old"] else "OLD -"
        n = f"NEW[{e['new']['idx']}] {e['new']['name']}" if e["new"] else "NEW -"
        out.append(f"\n■ {o}  ->  {n}")
        out.append(f"  status       : {e['status']}")
        out.append(f"  change types : {', '.join(e['change_types']) or '-'}")
        tr = e.get("transformations")
        if not tr:
            continue
        t, r, sc, mi = tr["translation"], tr["rotation"], tr["scaling"], tr["mirror"]
        if "dx" in t:
            out.append(
                f"  translation  : {'YES' if t['detected'] else 'no '}  "
                f"dx={t['dx']:+.6f}  dy={t['dy']:+.6f}  dz={t['dz']:+.6f}  (|d|={t['distance']:.6f})"
            )
        elif "centroid_shift" in t:
            c = t["centroid_shift"]
            out.append(f"  translation  : unknown (centroid shift {c[0]:+.6f}, {c[1]:+.6f}, {c[2]:+.6f})")
        if "angle_deg" in r:
            axis = ", ".join(f"{a:+.6f}" for a in r["axis"])
            out.append(
                f"  rotation     : {'YES' if r['detected'] else 'no '}  angle={r['angle_deg']:.6f} deg  "
                f"axis=[{axis}] ({r['axis_label']})  determinacy={r['determinacy']}"
            )
            if r.get("note"):
                out.append(f"                 note: {r['note']}")
        if sc.get("type") == "uniform":
            out.append(f"  scaling      : YES  uniform, factor={sc['factor']:.9f}")
        elif sc.get("type") == "non_uniform":
            f = ", ".join(f"{v:.9f}" for v in sc["axis_factors"])
            out.append(f"  scaling      : YES  non-uniform, axis factors=[{f}]")
        else:
            out.append("  scaling      : no ")
        if mi.get("detected"):
            nv = ", ".join(f"{a:+.6f}" for a in mi["plane_normal"])
            out.append(f"  mirror       : YES  plane normal=[{nv}] ({mi['plane_label']})")
        sh = e["shape"]
        vr = sh.get("volume_ratio")
        out.append(
            f"  shape        : {'CHANGED' if sh['changed'] else 'same'}  "
            f"faces {sh['faces_old']}->{sh['faces_new']}, edges {sh['edges_old']}->{sh['edges_new']}"
            + (f", volume x{vr:.6f}" if vr else "")
        )
        if "unchanged_area_fraction" in sh:
            out.append(f"                 unchanged area fraction={sh['unchanged_area_fraction']:.4f}")
    return "\n".join(out)



# =============================================================================
# ■ main: 결과를 터미널에 출력
# =============================================================================
def _resolve(p) -> Path:
    p = Path(p).expanduser()
    return p if p.is_absolute() else Path(__file__).resolve().parent / p


def main(old_step=OLD_STEP, new_step=NEW_STEP, show_details=SHOW_DETAILS, cfg: Config | None = None) -> dict:
    try:  # Windows 콘솔에서도 한글/기호가 깨지지 않도록
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    old, new = _resolve(old_step), _resolve(new_step)
    for p in (old, new):
        if not p.is_file():
            sys.exit(f"STEP 파일을 찾을 수 없습니다: {p}")

    print(f"OLD : {old}\nNEW : {new}\n")
    report = compare_step_files(old, new, cfg or Config())
    print("=" * 100)
    print(format_text(report))
    if show_details:
        print("\n" + "=" * 100 + "\n상세 결과")
        print(format_details(report))
    print("\n" + "=" * 100)
    print(f"완료 ({report['summary']['elapsed_s']} s)")
    return report


if __name__ == "__main__":
    main()

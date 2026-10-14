"""Thin compatibility layer over OCP (OpenCascade Python bindings).

Collection class names differ between OCP releases (``TopTools_*`` in older
builds, ``OCP.collections.*`` in newer ones). Everything OCC-specific that is
version sensitive is resolved here once so the rest of the package can stay
clean.
"""
from __future__ import annotations

import numpy as np

from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepGProp import BRepGProp
from OCP.Bnd import Bnd_Box
from OCP.GProp import GProp_GProps
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
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec

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

__all__ = [
    "LabelSequence",
    "indexed_map",
    "ancestor_map",
    "iter_list",
    "explore",
    "volume_props",
    "surface_props",
    "edge_length",
    "optimal_bbox",
    "pnt",
    "vec",
    "dir_",
    "is_reversed",
    "transform_shape",
    "silence_occ",
    "BRep_Tool",
    "BRepAdaptor_Curve",
    "BRepAdaptor_Surface",
    "as_face",
    "as_edge",
    "as_vertex",
    "TopAbs_EDGE",
    "TopAbs_FACE",
    "TopAbs_SHELL",
    "TopAbs_SOLID",
    "TopAbs_VERTEX",
    "TopAbs_WIRE",
]


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
_ = (gp_Ax1, gp_Dir, gp_Pnt, gp_Vec)

"""STEP reading / writing and separation of a file into individual CAD objects.

Each *object* is one B-Rep solid (or, for surface-only models, one shell).
Product / instance names stored in the STEP file are kept when available.
Assembly placements are applied, so every object lives in global coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPCAFControl import STEPCAFControl_Reader, STEPCAFControl_Writer
from OCP.STEPControl import STEPControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label
from OCP.TDocStd import TDocStd_Document
from OCP.TopLoc import TopLoc_Location
from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ShapeTool

from .occ import LabelSequence, TopAbs_SHELL, TopAbs_SOLID, explore, silence_occ


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


def write_step(path: str | Path, objects: list[tuple[str, object]], verbose: bool = False) -> None:
    """Write named shapes as separate products in one STEP file."""
    if not verbose:
        silence_occ()
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    for name, shape in objects:
        label = tool.AddShape(shape, False)
        TDataStd_Name.Set_s(label, TCollection_ExtendedString(name))
    writer = STEPCAFControl_Writer()
    writer.SetNameMode(True)
    if not writer.Transfer(doc):
        raise IOError("STEP transfer failed")
    if writer.Write(str(path)) != IFSelect_RetDone:
        raise IOError(f"cannot write STEP file: {path}")

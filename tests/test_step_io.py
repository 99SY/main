"""STEP reading: names, assemblies with placements, plain (unnamed) files."""
import numpy as np
from OCP.gp import gp_Trsf, gp_Vec
from OCP.STEPCAFControl import STEPCAFControl_Writer
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDocStd import TDocStd_Document
from OCP.TopLoc import TopLoc_Location
from OCP.XCAFDoc import XCAFDoc_DocumentTool

from step_diff.features import extract_features
from step_diff.samples import block, box
from step_diff.step_io import load_step, write_step


def test_names_roundtrip(tmp_path):
    p = tmp_path / "a.step"
    write_step(p, [("first", box(0, 0, 0, 1, 2, 3)), ("second", block(10))])
    objs = load_step(p)
    assert [o.name for o in objs] == ["first", "second"]
    assert [o.idx for o in objs] == [0, 1]


def test_assembly_instances_get_their_placement(tmp_path):
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    part = tool.AddShape(box(0, 0, 0, 10, 10, 10), False)
    TDataStd_Name.Set_s(part, TCollection_ExtendedString("cube"))
    asm = tool.NewShape()
    TDataStd_Name.Set_s(asm, TCollection_ExtendedString("asm"))
    for dx in (0.0, 50.0):
        t = gp_Trsf()
        t.SetTranslation(gp_Vec(dx, 0, 0))
        tool.AddComponent(asm, part, TopLoc_Location(t))
    tool.UpdateAssemblies()
    w = STEPCAFControl_Writer()
    w.Transfer(doc)
    p = tmp_path / "asm.step"
    w.Write(str(p))

    objs = load_step(p)
    assert len(objs) == 2
    cx = sorted(extract_features(o.idx, o.name, o.shape).centroid[0] for o in objs)
    assert np.allclose(cx, [5.0, 55.0], atol=1e-9)


def test_plain_step_without_names(tmp_path):
    w = STEPControl_Writer()
    w.Transfer(block(0), STEPControl_AsIs)
    p = tmp_path / "plain.step"
    w.Write(str(p))
    objs = load_step(p)
    assert len(objs) == 1 and objs[0].name

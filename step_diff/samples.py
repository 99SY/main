"""Synthetic OLD/NEW STEP pairs with known ground truth (for tests and demos)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere
from OCP.gp import gp_Ax1, gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec

from . import occ
from .step_io import write_step


# --------------------------------------------------------------------------- #
# primitive builders
# --------------------------------------------------------------------------- #
def box(x, y, z, dx, dy, dz):
    return BRepPrimAPI_MakeBox(gp_Pnt(x, y, z), dx, dy, dz).Shape()


def cylinder(p, d, r, h):
    return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*p), gp_Dir(*d)), r, h).Shape()


def sphere(p, r):
    return BRepPrimAPI_MakeSphere(gp_Pnt(*p), r).Shape()


def cut(a, *tools):
    for t in tools:
        a = BRepAlgoAPI_Cut(a, t).Shape()
    return _first_solid(a)


def fuse(a, *tools):
    for t in tools:
        a = BRepAlgoAPI_Fuse(a, t).Shape()
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain

    u = ShapeUpgrade_UnifySameDomain(a, True, True, True)
    u.Build()
    return _first_solid(u.Shape())


def _first_solid(shape):
    return next(occ.explore(shape, occ.TopAbs_SOLID))


def centroid(shape):
    return occ.pnt(occ.volume_props(shape).CentreOfMass())


def _apply(shape, trsf):
    return _first_solid(BRepBuilderAPI_Transform(shape, trsf, True).Shape())


def translated(shape, v):
    t = gp_Trsf()
    t.SetTranslation(gp_Vec(*v))
    return _apply(shape, t)


def rotated(shape, axis, deg, center=None):
    c = centroid(shape) if center is None else center
    t = gp_Trsf()
    t.SetRotation(gp_Ax1(gp_Pnt(*c), gp_Dir(*axis)), np.radians(deg))
    return _apply(shape, t)


def scaled(shape, s, center=None):
    c = centroid(shape) if center is None else center
    t = gp_Trsf()
    t.SetScale(gp_Pnt(*c), s)
    return _apply(shape, t)


def mirrored(shape, normal, center=None):
    c = centroid(shape) if center is None else center
    t = gp_Trsf()
    t.SetMirror(gp_Ax2(gp_Pnt(*c), gp_Dir(*normal)))
    return _apply(shape, t)


# --------------------------------------------------------------------------- #
# demo parts
# --------------------------------------------------------------------------- #
def plate(x0=0.0, y0=0.0):
    return cut(
        box(x0, y0, 0, 100, 60, 10),
        cylinder((x0 + 20, y0 + 20, -1), (0, 0, 1), 5, 12),
        cylinder((x0 + 75, y0 + 40, -1), (0, 0, 1), 7, 12),
    )


def bracket(x0):
    return fuse(box(x0, 0, 0, 40, 40, 10), box(x0, 0, 0, 40, 10, 40))


def block(x0):
    return cut(box(x0, 0, 0, 20, 30, 40), cylinder((x0 + 6, 9, -1), (0, 0, 1), 4, 42))


def housing(x0):
    return cut(box(x0, 0, 0, 40, 30, 20), box(x0 + 5, 5, 8, 22, 15, 13))


def chiral(x0):
    """Box with three different off-centre blind holes: has no mirror symmetry."""
    return cut(
        box(x0, 0, 0, 40, 30, 20),
        cylinder((x0 + 10, 8, 12), (0, 0, 1), 3, 9),
        cylinder((x0 - 1, 20, 6), (1, 0, 0), 2, 15),
        cylinder((x0 + 28, -1, 14), (0, 1, 0), 4, 12),
    )


def build_demo():
    """Return (old_objects, new_objects, expected) for the demo scenario."""
    S = 200.0  # spacing between parts along X
    old, new, expected = [], [], {}

    def add(name, o, n, exp):
        if o is not None:
            old.append((name, o))
        if n is not None:
            new.append((name if o is not None else name, n))
        expected[name] = exp

    p0 = plate()
    add("plate", p0, p0, {"status": "STAY"})

    b = bracket(1 * S)
    add("bracket", b, translated(b, (5, -3, 2)), {"status": "MODIFIED", "types": ["TRANSLATION"], "t": (5, -3, 2)})

    k = block(2 * S)
    add("block", k, rotated(k, (0, 0, 1), 30), {"status": "MODIFIED", "types": ["ROTATION"], "angle": 30, "axis": (0, 0, 1)})

    pin = cylinder((3 * S, 0, 0), (0, 0, 1), 5, 30)
    add("pin", pin, scaled(pin, 1.5), {"status": "MODIFIED", "types": ["UNIFORM_SCALING"], "scale": 1.5})

    h = housing(4 * S)
    axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    h_new = translated(scaled(rotated(h, axis, 45), 1.2), (3, 4, 5))
    add(
        "housing",
        h,
        h_new,
        {"status": "MODIFIED", "types": ["TRANSLATION", "ROTATION", "UNIFORM_SCALING"], "angle": 45, "axis": tuple(axis), "scale": 1.2, "t": (3, 4, 5)},
    )

    # one dimension changed: 10 -> 20 along X, centred on the same point
    rib = box(5 * S, 0, 0, 10, 50, 5)
    rib_new = box(5 * S - 5, 0, 0, 20, 50, 5)
    add("rib", rib, rib_new, {"status": "MODIFIED", "types": ["NON_UNIFORM_SCALING"], "factors": (2, 1, 1)})

    boss = box(6 * S, 0, 0, 30, 30, 30)
    boss_new = translated(cut(boss, cylinder((6 * S + 15, 15, -1), (0, 0, 1), 5, 32)), (2, 0, 0))
    add("boss", boss, boss_new, {"status": "MODIFIED", "types": ["TRANSLATION", "SHAPE_CHANGE"], "t": (2, 0, 0)})

    shaft = cylinder((7 * S, 0, 0), (0, 0, 1), 6, 50)
    add("shaft", shaft, rotated(shaft, (1, 0, 0), 20), {"status": "MODIFIED", "types": ["ROTATION"], "angle": 20, "axis": (1, 0, 0), "axisymmetric": True})

    add("old_only", box(8 * S, 0, 0, 15, 15, 15), None, {"status": "DELETION"})

    cube = box(9 * S, 0, 0, 20, 20, 20)
    add("cube", cube, rotated(cube, (0, 0, 1), 90), {"status": "STAY"})

    ch = chiral(10 * S)
    add("chiral", ch, mirrored(ch, (1, 0, 0)), {"status": "MODIFIED", "types": ["MIRROR"]})

    add("new_only", None, sphere((11 * S, 0, 0), 8), {"status": "INSERTION"})
    # identical to "plate" but far away: no AABB overlap -> a different object
    add("plate_copy", None, plate(0, 500), {"status": "INSERTION"})
    return old, new, expected


def write_demo(directory: str | Path) -> tuple[Path, Path, dict]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    old, new, expected = build_demo()
    po, pn = directory / "old.step", directory / "new.step"
    write_step(po, old)
    write_step(pn, new)
    return po, pn, expected

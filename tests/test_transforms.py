"""Precision and robustness of the transform estimation."""
import numpy as np
import pytest

from step_compare import Config
from samples import (
    block,
    box,
    bracket,
    chiral,
    cut,
    cylinder,
    housing,
    mirrored,
    rotated,
    scaled,
    sphere,
    translated,
)

AXES = [
    (0, 0, 1),
    (1, 0, 0),
    (0, 1, 0),
    (1, 2, 3),
    (-0.3, 0.9, 0.1),
]


@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("angle", [0.5, 17.3, 90.0, 133.7, 179.0])
def test_rotation_angle_and_axis_are_exact(compare, axis, angle):
    part = chiral(0)
    _, by_old, _ = compare([("p", part)], [("p", rotated(part, axis, angle))])
    r = by_old["p"]
    assert r["change_types"] == ["ROTATION"]
    rot = r["transformations"]["rotation"]
    assert rot["angle_deg"] == pytest.approx(angle, abs=1e-6)
    a = np.array(axis, float) / np.linalg.norm(axis)
    assert np.allclose(rot["axis"], a, atol=1e-6)
    assert rot["determinacy"] == "full" and rot["equivalent_solutions"] == 1


def test_symmetric_box_reports_smallest_rotation(compare):
    b = box(0, 0, 0, 20, 30, 40)
    _, by_old, _ = compare([("b", b)], [("b", rotated(b, (0, 0, 1), 30))])
    rot = by_old["b"]["transformations"]["rotation"]
    assert rot["angle_deg"] == pytest.approx(30, abs=1e-6)
    assert rot["equivalent_solutions"] == 4  # D2 symmetry of a rectangular box


def test_composite_transform_components_are_independent(compare):
    h = housing(0)
    new = translated(scaled(rotated(h, (0, 1, 1), 63.0), 0.75), (-12.5, 0.25, 7))
    _, by_old, _ = compare([("h", h)], [("h", new)])
    r = by_old["h"]
    assert sorted(r["change_types"]) == ["ROTATION", "TRANSLATION", "UNIFORM_SCALING"]
    tr = r["transformations"]
    assert tr["scaling"]["factor"] == pytest.approx(0.75, abs=1e-9)
    assert tr["rotation"]["angle_deg"] == pytest.approx(63.0, abs=1e-6)
    assert [tr["translation"][k] for k in ("dx", "dy", "dz")] == pytest.approx([-12.5, 0.25, 7], abs=1e-6)
    assert tr["translation"]["moved_axes"] == ["X", "Y", "Z"]


def test_matrix_maps_old_vertices_onto_new(compare):
    b = bracket(0)
    new = translated(rotated(b, (1, 1, 0), 40), (1, 2, 3))
    from step_compare import extract_features

    fo, fn = extract_features(0, "o", b), extract_features(0, "n", new)
    _, by_old, _ = compare([("b", b)], [("b", new)])
    m = np.array(by_old["b"]["matrix_4x4"])
    mapped = fo.vertices @ m[:3, :3].T + m[:3, 3]
    from scipy.spatial import cKDTree

    d, _ = cKDTree(fn.vertices).query(mapped)
    assert d.max() < 1e-6


def test_tiny_translation_is_detected(compare):
    b = block(0)
    _, by_old, _ = compare([("b", b)], [("b", translated(b, (0.001, 0, 0)))])
    r = by_old["b"]
    assert r["change_types"] == ["TRANSLATION"]
    assert r["transformations"]["translation"]["dx"] == pytest.approx(0.001, abs=1e-9)


def test_stay(compare):
    b = block(0)
    _, by_old, _ = compare([("b", b)], [("b", block(0))])
    assert by_old["b"]["status"] == "STAY"


def test_axisymmetric_cylinder_gives_axis(compare):
    c = cylinder((0, 0, 0), (0, 0, 1), 4, 40)
    _, by_old, _ = compare([("c", c)], [("c", rotated(c, (0, 1, 0), 35))])
    rot = by_old["c"]["transformations"]["rotation"]
    assert rot["determinacy"] == "axisymmetric"
    assert rot["angle_deg"] == pytest.approx(35, abs=1e-6)
    assert np.allclose(rot["axis"], [0, 1, 0], atol=1e-6)


def test_sphere_rotation_not_observable(compare):
    s = sphere((0, 0, 0), 5)
    _, by_old, _ = compare([("s", s)], [("s", translated(s, (1, 0, 0)))])
    r = by_old["s"]
    assert r["change_types"] == ["TRANSLATION"]
    assert r["transformations"]["rotation"]["determinacy"] == "none"


def test_mirror_with_rotation(compare):
    p = chiral(0)
    new = rotated(mirrored(p, (0, 1, 0)), (0, 1, 0), 25)
    _, by_old, _ = compare([("p", p)], [("p", new)])
    r = by_old["p"]
    assert "MIRROR" in r["change_types"] and "ROTATION" in r["change_types"]
    tr = r["transformations"]
    assert np.allclose(tr["mirror"]["plane_normal"], [0, 1, 0], atol=1e-6)
    assert tr["rotation"]["angle_deg"] == pytest.approx(25, abs=1e-6)


def test_mirror_detection_can_be_disabled(compare):
    p = chiral(0)
    _, by_old, _ = compare([("p", p)], [("p", mirrored(p, (1, 0, 0)))], Config(detect_mirror=False))
    assert "MIRROR" not in by_old["p"]["change_types"]


def test_shape_change_with_rotation_and_translation(compare):
    b = block(0)
    modified = cut(b, cylinder((15, 22, -1), (0, 0, 1), 2, 42))  # extra hole
    new = translated(rotated(modified, (0, 0, 1), 90, center=np.array([10.0, 15.0, 20.0])), (4, 0, 0))
    _, by_old, _ = compare([("b", b)], [("b", new)])
    r = by_old["b"]
    assert "SHAPE_CHANGE" in r["change_types"]
    assert r["basis"] == "partial_alignment"
    tr = r["transformations"]
    assert tr["rotation"]["angle_deg"] == pytest.approx(90, abs=1e-6)
    assert np.allclose(tr["rotation"]["axis"], [0, 0, 1], atol=1e-6)
    m = np.array(r["matrix_4x4"])
    # unchanged rigid motion: x' = Rz(90) (x - c) + c + (4, 0, 0) with c = (10, 15, 20)
    rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    c = np.array([10.0, 15.0, 20.0])
    assert np.allclose(m[:3, :3], rz, atol=1e-9)
    assert np.allclose(m[:3, 3], c - rz @ c + [4, 0, 0], atol=1e-6)
    assert r["shape"]["added_or_modified_new_faces"]


def test_nonuniform_with_rotation(compare):
    rib = box(0, 0, 0, 10, 50, 5)
    rib_long = box(0, -25, 0, 10, 100, 5)  # length doubled, same centroid
    new = rotated(rib_long, (0, 0, 1), 90, center=np.array([5.0, 25.0, 2.5]))
    _, by_old, _ = compare([("r", rib)], [("r", new)])
    r = by_old["r"]
    assert sorted(r["change_types"]) == ["NON_UNIFORM_SCALING", "ROTATION"]
    tr = r["transformations"]
    assert tr["rotation"]["angle_deg"] == pytest.approx(90, abs=1e-6)
    assert np.allclose(tr["scaling"]["axis_factors"], [1, 2, 1], atol=1e-9)


def test_axisymmetric_spin_about_own_axis_is_not_a_shape_change(compare):
    # spinning a cylinder about its own axis only moves the seam vertices
    c = cylinder((0, 0, 0), (0, 0, 1), 5, 30)
    new = rotated(rotated(c, (0, 0, 1), 40), (1, 0, 0), 15)
    _, by_old, _ = compare([("c", c)], [("c", new)])
    r = by_old["c"]
    assert r["change_types"] == ["ROTATION"]
    rot = r["transformations"]["rotation"]
    assert rot["determinacy"] == "axisymmetric"
    assert rot["angle_deg"] == pytest.approx(15, abs=1e-6)
    assert np.allclose(rot["axis"], [1, 0, 0], atol=1e-6)


def test_axisymmetric_pure_spin_is_stay(compare):
    c = cylinder((0, 0, 0), (0, 0, 1), 5, 30)
    _, by_old, _ = compare([("c", c)], [("c", rotated(c, (0, 0, 1), 73))])
    assert by_old["c"]["status"] == "STAY"

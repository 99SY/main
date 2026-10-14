"""End-to-end: write the demo OLD/NEW STEP files, read them back, check ground truth."""
import numpy as np
import pytest

from step_diff import compare_step_files
from step_diff.samples import write_demo


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    po, pn, expected = write_demo(tmp_path_factory.mktemp("demo"))
    report = compare_step_files(po, pn)
    return report, expected


def _result(report, name):
    for r in report["results"]:
        side = r["old"] or r["new"]
        if side["name"] == name:
            return r
    raise KeyError(name)


def test_every_object_has_expected_status(demo):
    report, expected = demo
    for name, exp in expected.items():
        r = _result(report, name)
        assert r["status"] == exp["status"], name
        if "types" in exp:
            assert sorted(r["change_types"]) == sorted(exp["types"]), name


def test_matching_is_one_to_one_by_name(demo):
    report, _ = demo
    for r in report["results"]:
        if r["old"] and r["new"]:
            assert r["old"]["name"] == r["new"]["name"]


def test_transform_values(demo):
    report, expected = demo
    for name, exp in expected.items():
        r = _result(report, name)
        tr = r.get("transformations")
        if not tr:
            continue
        if "t" in exp:
            t = tr["translation"]
            assert np.allclose([t["dx"], t["dy"], t["dz"]], exp["t"], atol=1e-6), name
        if "angle" in exp:
            assert tr["rotation"]["angle_deg"] == pytest.approx(exp["angle"], abs=1e-6), name
            assert np.allclose(tr["rotation"]["axis"], exp["axis"], atol=1e-6), name
        if "scale" in exp:
            assert tr["scaling"]["factor"] == pytest.approx(exp["scale"], abs=1e-9), name
        if "factors" in exp:
            assert np.allclose(tr["scaling"]["axis_factors"], exp["factors"], atol=1e-9), name
        if exp.get("axisymmetric"):
            assert tr["rotation"]["determinacy"] == "axisymmetric"


def test_identical_shape_without_aabb_overlap_is_insertion(demo):
    report, _ = demo
    r = _result(report, "plate_copy")
    assert r["status"] == "INSERTION"
    # the only candidate of OLD plate is the NEW plate in place
    plate_old = next(o for o in report["old_objects"] if o["name"] == "plate")
    cands = report["candidates"][str(plate_old["idx"])]
    assert [c["new_name"] for c in cands] == ["plate"]

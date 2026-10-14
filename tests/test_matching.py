"""Global 1:1 matching, AABB filtering, insertion / deletion."""
from step_diff.samples import block, box, bracket, cylinder, sphere, translated


def test_identical_neighbours_resolved_by_position(compare):
    # two identical blocks side by side; both move by +1 in X: each AABB overlaps
    # both NEW blocks, but the global assignment keeps the spatially consistent pairs
    a, b = block(0), block(15)
    old = [("a", a), ("b", b)]
    new = [("b", translated(b, (1, 0, 0))), ("a", translated(a, (1, 0, 0)))]
    _, by_old, _ = compare(old, new)
    for name in "ab":
        r = by_old[name]
        assert r["new"]["name"] == name
        assert r["change_types"] == ["TRANSLATION"]


def test_far_move_is_deletion_plus_insertion(compare):
    b = bracket(0)
    _, by_old, by_new = compare([("b", b)], [("b", translated(b, (500, 0, 0)))])
    assert by_old["b"]["status"] == "DELETION"
    assert by_new["b"]["status"] == "INSERTION"


def test_unrelated_shape_in_same_place_is_not_matched(compare):
    old = [("box", box(0, 0, 0, 20, 20, 20))]
    new = [("ball", sphere((10, 10, 10), 9))]
    _, by_old, by_new = compare(old, new)
    assert by_old["box"]["status"] == "DELETION"
    assert by_new["ball"]["status"] == "INSERTION"


def test_counts_and_empty_sides(compare):
    rep, _, _ = compare([("x", box(0, 0, 0, 1, 1, 1))], [])
    assert rep["summary"]["status_counts"] == {"DELETION": 1}
    rep, _, _ = compare([], [("y", cylinder((0, 0, 0), (0, 0, 1), 1, 2))])
    assert rep["summary"]["status_counts"] == {"INSERTION": 1}

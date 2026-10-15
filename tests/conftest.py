import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from step_compare import Config, compare_objects  # noqa: E402
from step_compare import extract_features  # noqa: E402


def feats(named_shapes):
    return [extract_features(i, name, shape) for i, (name, shape) in enumerate(named_shapes)]


def run(old, new, cfg=None):
    """Compare two lists of (name, shape) and index the results by names."""
    report = compare_objects(feats(old), feats(new), cfg or Config())
    by_old, by_new = {}, {}
    for r in report["results"]:
        if r["old"]:
            by_old[r["old"]["name"]] = r
        if r["new"]:
            by_new[r["new"]["name"]] = r
    return report, by_old, by_new


@pytest.fixture
def compare():
    return run

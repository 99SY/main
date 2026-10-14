"""Command line interface.

    python -m step_diff OLD.step NEW.step [-o report.json] [--csv report.csv]
"""
from __future__ import annotations

import argparse
import sys

from .config import Config
from .pipeline import compare_step_files
from .report import format_text, write_csv, write_json


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="step_diff", description="B-Rep based OLD/NEW STEP object matching and change detection")
    p.add_argument("old", help="STEP file before the design change")
    p.add_argument("new", help="STEP file after the design change")
    p.add_argument("-o", "--json", help="write the full report as JSON")
    p.add_argument("--csv", help="write a one-row-per-result CSV table")
    p.add_argument("--rel-pos-tol", type=float, default=Config.rel_pos_tol)
    p.add_argument("--rel-value-tol", type=float, default=Config.rel_value_tol)
    p.add_argument("--angle-tol-deg", type=float, default=Config.angle_tol_deg)
    p.add_argument("--translation-eps", type=float, default=Config.translation_eps)
    p.add_argument("--rotation-eps-deg", type=float, default=Config.rotation_eps_deg)
    p.add_argument("--scale-eps", type=float, default=Config.scale_eps)
    p.add_argument("--aabb-margin", type=float, default=Config.aabb_margin)
    p.add_argument("--no-mirror", action="store_true", help="do not test mirrored (reflected) copies")
    p.add_argument("--no-nonuniform", action="store_true", help="do not test non-uniform scaling")
    a = p.parse_args(argv)

    cfg = Config(
        rel_pos_tol=a.rel_pos_tol,
        rel_value_tol=a.rel_value_tol,
        angle_tol_deg=a.angle_tol_deg,
        translation_eps=a.translation_eps,
        rotation_eps_deg=a.rotation_eps_deg,
        scale_eps=a.scale_eps,
        aabb_margin=a.aabb_margin,
        detect_mirror=not a.no_mirror,
        detect_nonuniform=not a.no_nonuniform,
    )
    report = compare_step_files(a.old, a.new, cfg)
    print(format_text(report))
    if a.json:
        write_json(report, a.json)
        print(f"\nJSON report: {a.json}")
    if a.csv:
        write_csv(report, a.csv)
        print(f"CSV report:  {a.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Human readable rendering of a comparison report."""
from __future__ import annotations

import csv
import json
from pathlib import Path


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


def write_json(report: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2, ensure_ascii=False))


def write_csv(report: dict, path: str | Path) -> None:
    cols = [
        "old_idx", "old_name", "new_idx", "new_name", "status", "change_types",
        "dx", "dy", "dz", "rot_angle_deg", "rot_axis_x", "rot_axis_y", "rot_axis_z", "rot_determinacy",
        "scale_type", "scale_factor", "volume_ratio",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for e in report["results"]:
            row = {
                "old_idx": e["old"]["idx"] if e["old"] else "",
                "old_name": e["old"]["name"] if e["old"] else "",
                "new_idx": e["new"]["idx"] if e["new"] else "",
                "new_name": e["new"]["name"] if e["new"] else "",
                "status": e["status"],
                "change_types": "|".join(e["change_types"]),
            }
            tr = e.get("transformations")
            if tr:
                t, r, sc = tr["translation"], tr["rotation"], tr["scaling"]
                if "dx" in t:
                    row.update(dx=t["dx"], dy=t["dy"], dz=t["dz"])
                if "angle_deg" in r:
                    row.update(
                        rot_angle_deg=r["angle_deg"],
                        rot_axis_x=r["axis"][0], rot_axis_y=r["axis"][1], rot_axis_z=r["axis"][2],
                        rot_determinacy=r["determinacy"],
                    )
                row["scale_type"] = sc.get("type") or ""
                row["scale_factor"] = sc.get("factor", "")
                row["volume_ratio"] = e["shape"].get("volume_ratio")
            w.writerow(row)

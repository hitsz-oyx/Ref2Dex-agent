"""Count real canonical surface identities before qualifying v4 geometry."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np


def inspect_geometry(path, point_key):
    with np.load(path, allow_pickle=False) as values:
        points = values[point_key]
        normals = values["normals"]
    finite = bool(np.isfinite(points).all() and np.isfinite(normals).all())
    unit = bool(normals.shape == points.shape and
                np.allclose(np.linalg.norm(normals, axis=-1), 1., atol=1e-5))
    unique = len(np.unique(points, axis=0))
    return dict(path=str(path), stored_points=len(points), unique_points=unique,
                finite=finite, unit_normals=unit,
                qualifies_512=finite and unit and unique >= 512)


def audit(recipe):
    native = Path(recipe["native_root"])
    meta = json.loads((native / "manifest.json").read_text())
    objects = {row["key"]: row for row in meta["objects"]}
    overlay = {}
    if recipe.get("surface_geometry_manifest"):
        path = Path(recipe["surface_geometry_manifest"])
        manifest = json.loads(path.read_text())
        if manifest.get("schema") != "ref2dex.wam-v4-dense-surface.v1" or manifest.get("status") != "COMPLETED":
            raise ValueError("completed dense surface manifest required")
        overlay = {row["key"]: path.parent / row["path"] for row in manifest["objects"]}
    keys = defaultdict(set)
    for row in meta["sequences"]:
        if row["source_id"] in (4, 5, 6, 8, 9, 10):
            keys[str(row["source_id"])].add(row["objects"][row["target_slot"]])
    report = {}
    for sid, selected in sorted(keys.items(), key=lambda item: int(item[0])):
        rows = [dict(key=key, **inspect_geometry(overlay.get(key, native / objects[key]["path"]), "surface"))
                for key in sorted(selected)]
        report[sid] = rows
        print("native source", sid, "target geometries", len(rows), flush=True)
    main = Path(recipe["main3_root"])
    mixed = json.loads((main / "processed/manifest.json").read_text())
    for sid, source in enumerate(mixed["sources"]):
        canonical = Path(source["root"]) / "processed/canonical"
        rows = [dict(key=path.stem, **inspect_geometry(path, "points"))
                for path in sorted(canonical.glob("*.npz"))]
        if not rows:
            raise ValueError("no canonical geometry: " + str(canonical))
        report[str(sid)] = rows
        print("main3 source", sid, "canonical geometries", len(rows), flush=True)
    summary = {}
    for sid, rows in sorted(report.items(), key=lambda item: int(item[0])):
        summary[sid] = dict(geometries=len(rows),
            unique_points_histogram=dict(Counter(str(row["unique_points"]) for row in rows)),
            qualifies_512=sum(row["qualifies_512"] for row in rows),
            requires_repair=sum(not row["qualifies_512"] for row in rows))
    return dict(schema="ref2dex.wam-v4-surface-audit.v1", sources=summary, geometries=report,
                scope="native target geometries and all main3 canonical artifacts; not a window coverage audit",
                qualifies_512=all(row["qualifies_512"] for rows in report.values() for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(json.loads(args.recipe.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report["sources"], indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Resample underfilled native geometries from existing read-only CAD archives."""
import argparse
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time
import zipfile

import numpy as np
import trimesh

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / "src"))
from pose_mesh_data.dense_surface import SCHEMA, sample_mesh_surface


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--native-root", type=Path, required=True)
    p.add_argument("--unidex-archive", type=Path, required=True)
    p.add_argument("--taco-archive", type=Path, required=True)
    p.add_argument("--hot3d-assets", type=Path, required=True)
    p.add_argument("--dexycb-models", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=int, default=1800)
    args = p.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "geometry").mkdir()
    started = time.monotonic()
    audit = json.loads(args.audit.read_text())
    meta = json.loads((args.native_root / "manifest.json").read_text())
    objects = {row["key"]: row for row in meta["objects"]}
    needed = {row["key"]: sid for sid, rows in audit["geometries"].items()
              for row in rows if not row["qualifies_512"]}
    rows = []

    def emit(key, mesh, source, scale=1.):
        if time.monotonic() - started > args.seconds or shutil.disk_usage(root).free < 20 * 2**30:
            raise RuntimeError("dense surface time or free-space bound")
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        with np.load(args.native_root / objects[key]["path"], allow_pickle=False) as values:
            original = {name: values[name].copy() for name in values.files}
        values = sample_mesh_surface(mesh, original, key, scale)
        dest = "geometry/object-%05d.npz" % len(rows)
        np.savez_compressed(root / dest, **values)
        rows.append(dict(key=key, path=dest, mesh=source, mesh_scale=scale,
                         source_id=int(needed[key]), original_unique_points=len(np.unique(original["surface"], axis=0)),
                         unique_points=512, preserved_flow_queries=True))
        if len(rows) % 100 == 0:
            print("sampled", len(rows), "/", len(needed), "elapsed_s", round(time.monotonic()-started, 1), flush=True)

    by_member = {}
    for key, sid in needed.items():
        if sid == "4":
            row = objects[key]
            name = "meshdatav3_scaled/%s/coacd/decomposed_%03d.obj" % (row["mesh_id"], round(row["scale"] * 100))
            by_member[name] = key
    # Stream the compressed archive once; never unpack all CADs onto disk.
    with tarfile.open(args.unidex_archive, "r|xz") as archive:
        for member in archive:
            name = member.name.removeprefix("./") if hasattr(str, "removeprefix") else member.name.lstrip("./")
            if name in by_member:
                key = by_member.pop(name)
                mesh = trimesh.load(io.BytesIO(archive.extractfile(member).read()), file_type="obj", process=False)
                emit(key, mesh, str(args.unidex_archive) + "::" + member.name)
            if not by_member:
                break
    if by_member:
        raise ValueError("required UniDex CAD members missing: " + repr(list(by_member)[:5]))
    with zipfile.ZipFile(args.taco_archive) as archive:
        names = {Path(name).name: name for name in archive.namelist()}
        for key, sid in needed.items():
            if sid == "5":
                member = names[key.split("/")[-1] + "_cm.obj"]
                mesh = trimesh.load(io.BytesIO(archive.read(member)), file_type="obj", process=False)
                emit(key, mesh, str(args.taco_archive) + "::" + member, .01)
    for key, sid in needed.items():
        if sid == "6":
            path = args.hot3d_assets / (key + ".glb")
            emit(key, trimesh.load(path, process=False), str(path))
        elif sid == "8":
            # Native DexYCB ID 18 is the large marker in the standard YCB list.
            if key != "DexYCB/18":
                raise ValueError("unexpected DexYCB repair identity: " + key)
            path = args.dexycb_models / "040_large_marker/textured_simple.obj"
            emit(key, trimesh.load(path, process=False), str(path))
        elif sid not in ("4", "5"):
            raise ValueError("unsupported deficient source: " + sid)
    if {row["key"] for row in rows} != set(needed):
        raise ValueError("incomplete dense surface repair")
    result = dict(schema=SCHEMA, status="COMPLETED", task="3d-data-preparation", run_id=root.name,
                  source_native_root=str(args.native_root.resolve()), objects=rows,
                  method="32 original flow vertices plus 480 deterministic area-weighted triangle samples",
                  elapsed_s=time.monotonic()-started, training_allowed=False,
                  scope="derived canonical geometry overlay only; does not grant dataset admission")
    (root / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status="COMPLETED", geometries=len(rows), elapsed_s=result["elapsed_s"])), flush=True)


if __name__ == "__main__":
    main()

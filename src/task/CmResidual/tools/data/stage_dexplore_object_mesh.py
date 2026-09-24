"""Safely link a GRAB object mesh into the local DExplore runtime assets.

The raw GRAB tree is read-only. Existing runtime assets are never replaced.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re


OBJECT_NAME = re.compile(r"[a-z][a-z0-9]*\Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stage(*, object_name: str, raw_object_root: Path, dexplore_root: Path) -> dict:
    if not OBJECT_NAME.fullmatch(object_name):
        raise ValueError(f"invalid GRAB object name: {object_name!r}")
    raw = raw_object_root / object_name / "mesh.obj"
    mjcf = dexplore_root / "dexplore/data/assets/mjcf"
    urdf = mjcf / f"{object_name}.urdf"
    target = mjcf / "objects" / object_name / f"{object_name}.obj"
    if not raw.is_file() or not urdf.is_file():
        raise FileNotFoundError(f"missing source mesh or DExplore URDF: {raw}, {urdf}")
    expected = f'filename="objects/{object_name}/{object_name}.obj"'
    if expected not in urdf.read_text(encoding="utf-8"):
        raise ValueError(f"URDF does not reference expected mesh: {urdf}")
    raw_digest = sha256(raw)
    if target.is_symlink():
        if target.resolve() != raw.resolve():
            raise ValueError(f"refusing to replace existing symlink: {target}")
        created = False
    elif target.exists():
        if not target.is_file() or sha256(target) != raw_digest:
            raise ValueError(f"refusing to replace existing asset: {target}")
        created = False
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(raw.resolve())
        created = True
    return {"object": object_name, "raw_mesh": str(raw.resolve()),
            "runtime_mesh": str(target.absolute()), "sha256": raw_digest,
            "created": created}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("object_name")
    parser.add_argument("--raw-object-root", type=Path, required=True)
    parser.add_argument("--dexplore-root", type=Path, required=True)
    args = parser.parse_args()
    import json
    print(json.dumps(stage(object_name=args.object_name,
                           raw_object_root=args.raw_object_root,
                           dexplore_root=args.dexplore_root), sort_keys=True))


if __name__ == "__main__":
    main()

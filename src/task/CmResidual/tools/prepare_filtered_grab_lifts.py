"""Audit and stage the available single-right-hand GRAB lift motions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import torch


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "data/processed_data/inspire_geometric_dexplore"
RAW_MESHES = ROOT / "data/raw_data/GRAB/objects"
ASSETS = ROOT / "third_party/DExplore/dexplore/data/assets/mjcf/objects"
CORRECTED_SPEC = ROOT / "src/task/CmResidual/configs/multitrajectory_12_motion_probe.json"
ROUTE_CONFIG = ROOT / "src/task/CmResidual/configs/multitrajectory_object_router_probe.json"
PATTERN = re.compile(r"s\d+_([^_]+)_lift(?:_Retake)?$")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    source_manifest = SOURCE / "manifest.json"
    source = json.loads(source_manifest.read_text())
    if (source["num_kept"] != 660 or not source["filter"]["exclude_any_left_hand_contact"]):
        raise ValueError("filtered source contract changed")
    entries = []
    object_names = set()
    for motion in sorted(SOURCE.iterdir()):
        match = PATTERN.fullmatch(motion.name)
        if match is None or not motion.is_dir():
            continue
        obj = match.group(1)
        tensor_path = motion / "interaction_hand_inspire.pt"
        tensor = torch.load(tensor_path, map_location="cpu", weights_only=True)
        if tensor.ndim != 2 or tensor.shape[1] != 598 or not torch.isfinite(tensor).all():
            raise ValueError(f"invalid motion tensor: {motion.name}")
        left = int((tensor[:, 206:222] > 0).sum().item())
        right = int((tensor[:, 222:238] > 0).sum().item())
        vertical_range = float((tensor[:, 200].max() - tensor[:, 200].min()).item())
        if left or right == 0 or vertical_range < 0.03:
            raise ValueError(f"invalid grasp reference: {motion.name}")
        source_mesh = RAW_MESHES / obj / "mesh.obj"
        if not source_mesh.is_file():
            raise FileNotFoundError(source_mesh)
        object_names.add(obj)
        entries.append({"sequence": motion.name, "object": obj,
                        "motion_path": str(motion), "tensor_sha256": digest(tensor_path),
                        "frames": int(tensor.shape[0]), "left_contact_labels": left,
                        "right_contact_labels": right,
                        "object_vertical_range_m": vertical_range})
    if len(entries) != 59 or len(object_names) != 29:
        raise ValueError(f"lift pool changed: {len(entries)} motions, {len(object_names)} objects")
    meshes = {}
    for obj in sorted(object_names):
        source_mesh = RAW_MESHES / obj / "mesh.obj"
        asset = ASSETS / obj / f"{obj}.obj"
        if asset.exists():
            if digest(asset) != digest(source_mesh):
                raise ValueError(f"existing asset differs from raw GRAB mesh: {obj}")
        else:
            asset.parent.mkdir(parents=True, exist_ok=True)
            asset.symlink_to(source_mesh)
        meshes[obj] = {"asset": str(asset), "source": str(source_mesh),
                       "sha256": digest(source_mesh)}
    geometry_checks = []
    for relative in json.loads(CORRECTED_SPEC.read_text())["motions"]:
        corrected_path = ROOT / relative / "interaction_hand_inspire.pt"
        source_path = SOURCE / Path(relative).name / "interaction_hand_inspire.pt"
        corrected = torch.load(corrected_path, map_location="cpu", weights_only=True)
        geometric = torch.load(source_path, map_location="cpu", weights_only=True)
        if corrected.shape != geometric.shape:
            raise ValueError(f"reference shape differs: {source_path.parent.name}")
        relative_delta = ((corrected[:, 51:54] - corrected[:, 198:201]) -
                          (geometric[:, 51:54] - geometric[:, 198:201]))
        max_error_m = float(relative_delta.norm(dim=1).max().item())
        if max_error_m > 1e-4:
            raise ValueError(f"wrist/object geometry differs: {source_path.parent.name}")
        geometry_checks.append({"sequence": source_path.parent.name,
                                "max_relative_wrist_object_error_m": max_error_m,
                                "contact_label_differences": int((corrected[:, 205:238] !=
                                                                  geometric[:, 205:238]).sum().item())})
    output.mkdir(parents=True)
    motion_root = output / "motions"
    motion_root.mkdir()
    for row in entries:
        (motion_root / row["sequence"]).symlink_to(SOURCE / row["sequence"],
                                                       target_is_directory=True)
    spec = {"description": "All 59 filtered single-right-hand GRAB lift motions",
            "input_classification": "filtered_geometric_dexplore",
            "motions": [str(SOURCE.relative_to(ROOT) / row["sequence"]) for row in entries]}
    (output / "motion_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    source_expert = json.loads(ROUTE_CONFIG.read_text())["experts"]["source_e260"]
    smoke_route = {"description": "Frozen source actor on the 59-motion input gate",
                   "experts": {"source_e260": source_expert},
                   "object_route": {obj: "source_e260" for obj in sorted(object_names)}}
    (output / "smoke_route_config.json").write_text(json.dumps(smoke_route, indent=2) + "\n")
    inventory = {"experiment_id": "P-20260925-grab59-input-gate",
                 "source_manifest": str(source_manifest),
                 "source_manifest_sha256": digest(source_manifest),
                 "motion_count": len(entries), "object_count": len(object_names),
                 "motion_root": str(motion_root), "motions": entries, "meshes": meshes,
                 "corrected_12_geometry_checks": geometry_checks}
    (output / "input_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(json.dumps({"motion_count": len(entries), "object_count": len(object_names),
                      "spec": str(output / "motion_spec.json")}, sort_keys=True))


if __name__ == "__main__":
    main()

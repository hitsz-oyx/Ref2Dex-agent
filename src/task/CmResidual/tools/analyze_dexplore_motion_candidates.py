"""Compare DExplore ``interaction_hand_inspire.pt`` motion tensors.

The column ranges below are the ones consumed by
``dexplore/env/tasks/base_dexplore_task.py``.  The report is deliberately
JSON-only so that candidate selection remains auditable and easy to diff.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


FIELDS = {
    "right_wrist_position": slice(51, 54),
    "right_hand_dof": slice(54, 102),
    "key_body_position": slice(150, 198),
    "object_position": slice(198, 201),
    "object_rotation_xyzw": slice(201, 205),
    "contact": slice(205, 206),
    "contact_parts": slice(222, 238),
    "robot_dof": slice(373, 391),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_summary(value: torch.Tensor) -> dict:
    value = value.float()
    return {
        "min": value.amin(0).tolist(),
        "max": value.amax(0).tolist(),
        "mean": value.mean(0).tolist(),
        "first": value[0].tolist(),
        "last": value[-1].tolist(),
    }


def load(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != 598:
        raise ValueError(f"{path}: expected a [T,598] tensor, got {type(value)} {getattr(value, 'shape', None)}")
    return value


def describe(path: Path, value: torch.Tensor) -> dict:
    contact = value[:, 205]
    contact_ids = torch.where(contact >= 0.5)[0]
    object_z = value[:, 200]
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(torch.isfinite(value).all()),
        "contact": {
            "occupancy": float((contact >= 0.5).float().mean()),
            "first_frame": int(contact_ids[0]) if contact_ids.numel() else None,
            "last_frame": int(contact_ids[-1]) if contact_ids.numel() else None,
        },
        "reference_object_lift_m": float(object_z.max() - object_z[0]),
        "fields": {name: tensor_summary(value[:, columns]) for name, columns in FIELDS.items()},
    }


def difference(value: torch.Tensor, reference: torch.Tensor) -> dict:
    if value.shape != reference.shape:
        return {"shape_match": False}
    result = {"shape_match": True}
    for name, columns in {"all": slice(None), **FIELDS}.items():
        delta = value[:, columns].float() - reference[:, columns].float()
        result[name] = {
            "max_abs": float(delta.abs().max()),
            "rmse": float(delta.square().mean().sqrt()),
        }
    return result


def build_report(paths: list[Path], reference_path: Path | None) -> dict:
    tensors = {str(path): load(path) for path in paths}
    reference = load(reference_path) if reference_path else None
    report = {
        "schema": "dexplore_motion_candidate_comparison_v1",
        "field_columns": {name: [columns.start, columns.stop] for name, columns in FIELDS.items()},
        "reference": str(reference_path.resolve()) if reference_path else None,
        "candidates": [],
    }
    for path in paths:
        value = tensors[str(path)]
        item = describe(path, value)
        if reference is not None:
            item["difference_from_reference"] = difference(value, reference)
        report["candidates"].append(item)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = build_report(args.paths, args.reference)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()

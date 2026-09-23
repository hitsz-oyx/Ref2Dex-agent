"""Small frozen transfer probe for the mixed MANO/Inspire ObjectInteractionCm.

This is an input-contract and domain-gap probe, not a physical counterfactual test.
The oracle next-q hand flow is diagnostic only and must never be used online.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from src.task.CmResidual.dexplore_cm_geometry import (
    DExploreCmv2GeometryBridge, dexplore_root_pose, native_joint_limits,
    world_to_object_frame,
)
from src.task.CmResidual.tools.audit_original_cmv2_on_policy import ASSETS, TRANSITIONS, select
from src.task.ObjectInteractionCm.model import ObjectInteractionCmModel


ROOT = Path(__file__).resolve().parents[4]
CHECKPOINT = Path(
    "/home2/wyy/oyx_ws/Ref2Dex/outputs/objectinteractioncm/"
    "object_interaction_cm_dexplore_rl_v1_3_20260910_020856/checkpoints/best.pt"
)
CHECKPOINT_SHA256 = "3a3d6c0f88565b9e41f257e4f8b87a3ca5731fd356a98f4514091754036a7283"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def object_flow_target(points_local: torch.Tensor, current_pose: torch.Tensor,
                       next_pose: torch.Tensor) -> torch.Tensor:
    next_world = torch.einsum("bnj,bkj->bnk", points_local, next_pose[:, :3, :3])
    next_world = next_world + next_pose[:, None, :3, 3]
    return world_to_object_frame(next_world, current_pose, vector=False) - points_local


@torch.no_grad()
def evaluate(model, bridge, lower, upper, rows, batch_size: int) -> dict:
    result = {name: {"samples": 0, "valid": 0, "zero_mm": 0.0,
                     "nominal_mm": 0.0, "shuffled_mm": 0.0,
                     "oracle_mm": 0.0, "nominal_oracle_hand_rms_mm": 0.0,
                     "predicted_shuffle_change_mm": 0.0}
              for name in ("contact", "no_contact")}
    for start in range(0, len(rows["q"]), batch_size):
        part = {key: value[start:start + batch_size].to(bridge.device)
                for key, value in rows.items() if isinstance(value, torch.Tensor)}
        current = bridge.current(part["q"], part["object_state"])
        _, _, nominal_world = bridge.nominal_hand_sweep(
            part["q"], part["action"][:, None], lower, upper)
        _, _, shuffled_world = bridge.nominal_hand_sweep(
            part["q"], part["shuffled_action"][:, None], lower, upper)
        next_links = bridge.kinematics.forward(part["next_q"][:, None])[:, 0]
        next_hand_world, _ = bridge.geometry.hand(next_links)
        oracle_world = next_hand_world - current.hand_points
        pose = current.object_pose
        obj = world_to_object_frame(current.object_points, pose, vector=False)
        obj_normals = world_to_object_frame(current.object_normals, pose, vector=True)
        hand = world_to_object_frame(current.hand_points, pose, vector=False)
        hand_normals = world_to_object_frame(current.hand_normals, pose, vector=True)
        gt = object_flow_target(obj, pose, dexplore_root_pose(part["next_object_state"]))
        batch = {"obj_points": obj, "obj_normals": obj_normals,
                 "hand_points": hand, "hand_normals": hand_normals}

        def predict(flow_world):
            batch["hand_flow"] = world_to_object_frame(flow_world, pose, vector=True)
            return model(batch)

        nominal = predict(nominal_world[:, 0])
        shuffled = predict(shuffled_world[:, 0])
        oracle = predict(oracle_world)
        predictions = {"nominal_mm": nominal["pred_obj_flow"],
                       "shuffled_mm": shuffled["pred_obj_flow"],
                       "oracle_mm": oracle["pred_obj_flow"]}
        errors = {key: (value - gt).norm(dim=-1).mean(dim=-1) * 1000
                  for key, value in predictions.items()}
        errors["zero_mm"] = gt.norm(dim=-1).mean(dim=-1) * 1000
        errors["nominal_oracle_hand_rms_mm"] = (
            nominal_world[:, 0] - oracle_world).norm(dim=-1).square().mean(dim=-1).sqrt() * 1000
        errors["predicted_shuffle_change_mm"] = (
            nominal["pred_obj_flow"] - shuffled["pred_obj_flow"]
        ).norm(dim=-1).mean(dim=-1) * 1000
        for i, flag in enumerate(part["contact_group"].bool()):
            item = result["contact" if flag else "no_contact"]
            item["samples"] += 1
            item["valid"] += int(nominal["sample_valid"][i].item())
            for key, value in errors.items():
                item[key] += float(value[i].item())
    for item in result.values():
        count = item["samples"]
        for key in tuple(item):
            if key not in ("samples", "valid"):
                item[key] /= count
        item["valid_fraction"] = item["valid"] / count
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-group", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.samples_per_group < 1 or args.batch_size < 1:
        raise ValueError("new output and positive sample counts are required")
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("mixed Cm checkpoint SHA mismatch")
    torch.set_num_threads(2)
    device = torch.device("cpu")
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    meta = payload["config"]["meta"]
    model = ObjectInteractionCmModel(SimpleNamespace(
        meta=SimpleNamespace(**meta), work_version="V1.3"))
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False).to(device)
    bridge = DExploreCmv2GeometryBridge(
        hand_urdf=ASSETS / "inspire_hand_new/inspire_hand_right.urdf",
        object_urdf=ASSETS / "mjcf/airplane.urdf", device=device, seed=42)
    lower, upper = native_joint_limits(bridge.hand_urdf, device)
    reports = {}
    for seed, (path, expected) in TRANSITIONS.items():
        if sha256(path) != expected:
            raise ValueError(f"transition SHA mismatch: seed {seed}")
        transition = torch.load(path, map_location="cpu", weights_only=False)
        if transition.get("schema") != "ref2dex.cmlite_transition.v1":
            raise ValueError("transition schema mismatch")
        rows = select(transition, seed, args.samples_per_group)
        reports[str(seed)] = evaluate(model, bridge, lower, upper, rows, args.batch_size)
        print(json.dumps({"seed": seed, **reports[str(seed)]}, sort_keys=True), flush=True)
    output = {"schema": "ref2dex.mixed_cm_sim_transfer_probe.v1",
              "checkpoint": str(CHECKPOINT), "checkpoint_sha256": CHECKPOINT_SHA256,
              "samples_per_group_per_seed": args.samples_per_group,
              "sim_transitions_sha256": {str(seed): item[1] for seed, item in TRANSITIONS.items()},
              "reports": reports,
              "interpretation_limit": "frozen transfer only; oracle next-q input is offline diagnostic; shuffled actions are not physical counterfactuals"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

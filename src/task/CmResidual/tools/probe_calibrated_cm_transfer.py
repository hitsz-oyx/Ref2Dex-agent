"""Frozen mixed Cm transfer with causal calibrated hand flow versus oracle."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import torch

from src.task.CmResidual.dexplore_cm_geometry import (
    DExploreCmv2GeometryBridge, dexplore_action_to_native_targets,
    dexplore_root_pose, native_joint_limits, world_to_object_frame,
)
from src.task.CmResidual.tools.analyze_cmlite_on_policy import first_episode_mask
from src.task.CmResidual.tools.audit_original_cmv2_on_policy import ASSETS, TRANSITIONS
from src.task.CmResidual.tools.probe_mixed_cm_sim_adapt import object_only
from src.task.CmResidual.tools.probe_mixed_cm_sim_transfer import (
    CHECKPOINT, CHECKPOINT_SHA256, object_flow_target, sha256,
)
from src.task.ObjectInteractionCm.model import ObjectInteractionCmModel


ROOT = Path(__file__).resolve().parents[4]
CALIBRATION = ROOT / "outputs/CmResidual/agent_executed_handflow_64/report.json"
CALIBRATION_SHA256 = "76bf65d450e3cfdcb7c0a9f54d68d7324e569fab18e2b7392224bd363ede0b16"
ENV_COUNT = 64


def selected_rows(path, expected, seed, each):
    if sha256(path) != expected:
        raise ValueError(f"transition SHA drift: {path}")
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if raw.get("schema") != "ref2dex.cmlite_transition.v1":
        raise ValueError("transition schema mismatch")
    mask = first_episode_mask(raw["done"], ENV_COUNT)
    mask[:ENV_COUNT] = False
    mask &= ~torch.roll(raw["done"].reshape(-1).bool(), ENV_COUNT)
    rows = {key: value[mask] for key, value in raw.items()
            if isinstance(value, torch.Tensor)}
    previous = torch.roll(raw["q"], ENV_COUNT, dims=0)
    rows["q_prev"] = previous[mask]
    contact = (rows["hand_contact"].bool() & rows["object_contact"].bool()).reshape(-1)
    generator = torch.Generator().manual_seed(2410 + seed)
    ids = []
    for flag in (True, False):
        candidates = torch.where(contact == flag)[0]
        if len(candidates) < each:
            raise ValueError(f"seed {seed} lacks {each} contact={flag}")
        ids.append(candidates[torch.randperm(len(candidates), generator=generator)[:each]])
    selected = torch.cat(ids)
    result = {key: value[selected] for key, value in rows.items()}
    result["contact_group"] = contact[selected]
    return result


@torch.no_grad()
def hand_points_at(bridge, native_q):
    links = bridge.kinematics.forward(native_q[:, None])[:, 0]
    points, _ = bridge.geometry.hand(links)
    return points


@torch.no_grad()
def evaluate(model, bridge, lower, upper, calibration, rows, batch_size):
    flow_names = ("nominal", "calibrated", "oracle", "zero_hand")
    groups = {group: {name: {"epe_mm": 0.0, "gated_epe_mm": 0.0,
                             "valid_count": 0} for name in flow_names}
              for group in ("contact", "no_contact")}
    for group in groups.values():
        group["count"] = 0
        group["zero_object_epe_mm"] = 0.0
        group["calibrated_hand_epe_mm"] = 0.0
    for start in range(0, len(rows["q"]), batch_size):
        part = {key: value[start:start + batch_size] for key, value in rows.items()}
        current = bridge.current(part["q"], part["object_state"])
        nominal_world = bridge.nominal_hand_sweep(
            part["q"], part["action"][:, None], lower, upper)[2][:, 0]
        target = dexplore_action_to_native_targets(part["action"], part["q"], lower, upper)
        predicted_q = (part["q"] + calibration["alpha"] * (target - part["q"])
                       + calibration["beta"] * (part["q"] - part["q_prev"]))
        calibrated_world = hand_points_at(bridge, predicted_q) - current.hand_points
        oracle_world = hand_points_at(bridge, part["next_q"]) - current.hand_points
        flows = {"nominal": nominal_world, "calibrated": calibrated_world,
                 "oracle": oracle_world, "zero_hand": torch.zeros_like(nominal_world)}
        pose = current.object_pose
        obj = world_to_object_frame(current.object_points, pose, vector=False)
        batch = {
            "obj_points": obj,
            "obj_normals": world_to_object_frame(current.object_normals, pose, vector=True),
            "hand_points": world_to_object_frame(current.hand_points, pose, vector=False),
            "hand_normals": world_to_object_frame(current.hand_normals, pose, vector=True),
        }
        true_flow = object_flow_target(
            obj, pose, dexplore_root_pose(part["next_object_state"]))
        zero_epe = true_flow.norm(dim=-1).mean(dim=-1) * 1000
        hand_error = (calibrated_world - oracle_world).norm(dim=-1).mean(dim=-1) * 1000
        evaluations = {}
        for name, world_flow in flows.items():
            batch["hand_flow"] = world_to_object_frame(world_flow, pose, vector=True)
            prediction, valid = object_only(model, batch)
            if not torch.isfinite(prediction).all():
                raise FloatingPointError(f"non-finite {name} prediction")
            errors = (prediction - true_flow).norm(dim=-1).mean(dim=-1) * 1000
            gated = (prediction * valid[:, None, None] - true_flow).norm(dim=-1).mean(dim=-1) * 1000
            evaluations[name] = (errors, gated, valid)
        for i, flag in enumerate(part["contact_group"]):
            group = groups["contact" if flag else "no_contact"]
            group["count"] += 1
            group["zero_object_epe_mm"] += float(zero_epe[i])
            group["calibrated_hand_epe_mm"] += float(hand_error[i])
            for name, (error, gated, valid) in evaluations.items():
                item = group[name]
                item["epe_mm"] += float(error[i])
                item["gated_epe_mm"] += float(gated[i])
                item["valid_count"] += int(valid[i])
    for group in groups.values():
        count = group["count"]
        group["zero_object_epe_mm"] /= count
        group["calibrated_hand_epe_mm"] /= count
        for name in flow_names:
            item = group[name]
            item["epe_mm"] /= count
            item["gated_epe_mm"] /= count
            item["valid_fraction"] = item["valid_count"] / count
    return groups


def execute(args, manifest):
    torch.set_num_threads(2)
    torch.manual_seed(2410)
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256 or sha256(CALIBRATION) != CALIBRATION_SHA256:
        raise ValueError("Cm or calibration SHA drift")
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    meta = SimpleNamespace(**payload["config"]["meta"])
    model = ObjectInteractionCmModel(SimpleNamespace(meta=meta, work_version="V1.3"))
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)
    values = json.loads(CALIBRATION.read_text())["coefficients"]
    calibration = {key: torch.tensor(values[key], dtype=torch.float32)
                   for key in ("alpha", "beta")}
    bridge = DExploreCmv2GeometryBridge(
        hand_urdf=ASSETS / "inspire_hand_new/inspire_hand_right.urdf",
        object_urdf=ASSETS / "mjcf/airplane.urdf", device="cpu", seed=42)
    lower, upper = native_joint_limits(bridge.hand_urdf, "cpu")
    reports = {}
    for seed, (path, expected) in TRANSITIONS.items():
        rows = selected_rows(path, expected, seed, args.samples_per_group)
        reports[str(seed)] = evaluate(model, bridge, lower, upper, calibration,
                                      rows, args.batch_size)
        print(json.dumps({"seed": seed,
                          "contact": {key: value["epe_mm"] for key, value
                                      in reports[str(seed)]["contact"].items()
                                      if isinstance(value, dict)},
                          "zero_object_mm": reports[str(seed)]["contact"]["zero_object_epe_mm"]}),
              flush=True)
    report = {"schema": "ref2dex.calibrated_cm_transfer_probe.v1",
              "run_status": "COMPLETED", "git_commit": manifest["git_commit"],
              "checkpoint_sha256": CHECKPOINT_SHA256,
              "calibration_sha256": CALIBRATION_SHA256,
              "transition_sha256": manifest["transition_sha256"],
              "samples_per_group_per_seed": args.samples_per_group,
              "reports": reports,
              "interpretation_limit": "oracle next_q is diagnostic only; observational actions are not physical counterfactuals"}
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest.update(run_status="COMPLETED", completed_at=datetime.now(timezone.utc).isoformat(),
                    report_sha256=sha256(report_path))
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-group", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists() or min(args.samples_per_group, args.batch_size) < 1:
        raise ValueError("new output and positive counts required")
    args.output.mkdir(parents=True)
    manifest = {"run_status": "STARTED", "run_id": args.output.name,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       cwd=ROOT, text=True).strip(),
                "cpu_threads": 2, "gpu_count": 0, "wall_budget_minutes": 10,
                "output_budget_mb": 2,
                "stop_rule": "input SHA drift, nonfinite output or wall budget",
                "args": dict(vars(args), output=str(args.output)),
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "calibration_sha256": CALIBRATION_SHA256,
                "transition_sha256": {str(seed): item[1] for seed, item in TRANSITIONS.items()}}
    manifest_path = args.output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    try:
        execute(args, manifest)
    except BaseException as error:
        manifest.update(run_status="FAILED", completed_at=datetime.now(timezone.utc).isoformat(),
                        failure=f"{type(error).__name__}: {error}")
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        raise


if __name__ == "__main__":
    main()

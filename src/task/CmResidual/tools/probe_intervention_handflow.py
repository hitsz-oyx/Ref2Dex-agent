"""Fit online action-to-actual-hand calibration on randomized sim transitions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import torch

from src.task.CmResidual.dexplore_cm_geometry import (
    DExploreCmv2GeometryBridge, dexplore_action_to_native_targets,
    native_joint_limits,
)
from src.task.CmResidual.tools.audit_original_cmv2_on_policy import ASSETS
from src.task.CmResidual.tools.probe_mixed_cm_sim_transfer import sha256


ROOT = Path(__file__).resolve().parents[4]
INPUTS = {
    "train_s145_d03": (
        ROOT / "outputs/CmResidual/agent_randomized_wristz_s145_n64/transitions.pt",
        "9afa81895b060c4fa7b214cb8f48501a0b0f2b75bb9618a36336e80e712ee532"),
    "test_s146_d03": (
        ROOT / "outputs/CmResidual/agent_randomized_wristz_s146_n64/transitions.pt",
        "b3326a06b65f8fce4679bc0a008a2a22012a0b286039c3838a3821857dce38de"),
    "test_s147_d01": (
        ROOT / "outputs/CmResidual/agent_randomized_wristz_s147_d01_n64/transitions.pt",
        "ec1486fbe81773444097518cda3021693865d3db9d8b5ec5e579c4fe72231b99"),
    "test_s148_d01": (
        ROOT / "outputs/CmResidual/agent_randomized_wristz_s148_d01_n64/transitions.pt",
        "44db8a0a0300d371feeffc6e7dbdc4499b452a58ddb9787131a138925d0f7de2"),
}
OLD_CALIBRATION = ROOT / "outputs/CmResidual/agent_executed_handflow_64/report.json"
OLD_CALIBRATION_SHA256 = "76bf65d450e3cfdcb7c0a9f54d68d7324e569fab18e2b7392224bd363ede0b16"


def load_rows(path, expected_sha, lower, upper):
    if sha256(path) != expected_sha:
        raise ValueError(f"transition SHA drift: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "ref2dex.randomized_action_transitions.v1" or (
            payload.get("run_status") != "COMPLETED"):
        raise ValueError("randomized transition schema or status mismatch")
    data = payload["records"]
    selected = data["assignment"].reshape(-1) != 0
    if not data["pre_contact"][selected].all():
        raise ValueError("non-contact intervention in selected training data")
    rows = {key: value[selected].float() for key, value in data.items()
            if key in ("q", "dof_vel", "executed_action", "next_q", "assignment")}
    rows["pd_target"] = dexplore_action_to_native_targets(
        rows["executed_action"], rows["q"], lower, upper)
    if not all(torch.isfinite(value).all() for value in rows.values()):
        raise FloatingPointError("non-finite randomized hand transition")
    return rows


def fit_per_joint(rows, *, use_action=True, use_velocity=True, ridge=1e-5):
    if not use_action and not use_velocity:
        raise ValueError("fit requires an online signal")
    features = []
    if use_action:
        features.append((rows["pd_target"] - rows["q"]).double())
    if use_velocity:
        features.append(rows["dof_vel"].double())
    features.append(torch.ones_like(features[0]))
    x = torch.stack(features, dim=-1)
    y = (rows["next_q"] - rows["q"]).double()
    coefficients = []
    for joint in range(y.shape[1]):
        design = x[:, joint]
        gram = design.T @ design + ridge * torch.eye(design.shape[1], dtype=torch.double)
        coefficients.append(torch.linalg.solve(gram, design.T @ y[:, joint]))
    return torch.stack(coefficients).float()


def predict_linear(rows, coefficients, *, use_action=True, use_velocity=True):
    features = []
    if use_action:
        features.append(rows["pd_target"] - rows["q"])
    if use_velocity:
        features.append(rows["dof_vel"])
    features.append(torch.ones_like(features[0]))
    design = torch.stack(features, dim=-1)
    return rows["q"] + (design * coefficients[None]).sum(dim=-1)


@torch.no_grad()
def surface_points(bridge, q, batch_size=32):
    result = []
    for start in range(0, len(q), batch_size):
        links = bridge.kinematics.forward(q[start:start + batch_size, None])[:, 0]
        points, _ = bridge.geometry.hand(links)
        result.append(points)
    return torch.cat(result)


@torch.no_grad()
def evaluate(bridge, rows, fitted, old_gain):
    q = rows["q"]
    predictions = {
        "stationary": q,
        "pd_target": rows["pd_target"],
        "velocity_extrapolation": q + rows["dof_vel"] / 30,
        "old_observational_gain": q + old_gain * (rows["pd_target"] - q),
        "new_action_only": predict_linear(rows, fitted["action_only"],
                                           use_velocity=False),
        "new_velocity_only": predict_linear(rows, fitted["velocity_only"],
                                             use_action=False),
        "new_action_velocity": predict_linear(rows, fitted["action_velocity"]),
    }
    actual_surface = surface_points(bridge, rows["next_q"])
    current_surface = surface_points(bridge, q)
    actual_motion_mm = (actual_surface - current_surface).norm(dim=-1).mean() * 1000
    assignment = rows["assignment"].reshape(-1)
    plus, minus = assignment == 1, assignment == -1
    actual_wrist_contrast_mm = float(((rows["next_q"] - q)[plus, 2].mean() -
                                      (rows["next_q"] - q)[minus, 2].mean()) * 1000)
    result = {}
    for name, predicted_q in predictions.items():
        surface = surface_points(bridge, predicted_q)
        result[name] = {
            "hand_surface_epe_mm": float((surface - actual_surface).norm(dim=-1).mean() * 1000),
            "q_delta_mae": float((predicted_q - rows["next_q"]).abs().mean()),
            "wrist_z_mae_mm": float((predicted_q[:, 2] - rows["next_q"][:, 2]).abs().mean() * 1000),
            "predicted_wrist_z_plus_minus_mm": float(
                ((predicted_q - q)[plus, 2].mean() -
                 (predicted_q - q)[minus, 2].mean()) * 1000),
        }
    return {"samples": len(q), "actual_hand_surface_motion_mm": float(actual_motion_mm),
            "actual_wrist_z_plus_minus_mm": actual_wrist_contrast_mm,
            "models": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mix-small-dose", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    manifest_path = args.output / "run_manifest.json"
    manifest = {
        "run_status": "STARTED", "run_id": args.output.name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                               cwd=ROOT, text=True).strip(),
        "input_sha256": {name: item[1] for name, item in INPUTS.items()},
        "old_calibration_sha256": OLD_CALIBRATION_SHA256,
        "mix_small_dose": args.mix_small_dose,
        "cpu_threads": 2, "gpu_count": 0, "wall_budget_minutes": 20,
        "output_budget_mb": 5,
        "stop_rule": "input drift, non-finite prediction, wall or output budget",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    start = time.monotonic()
    try:
        torch.set_num_threads(2)
        bridge = DExploreCmv2GeometryBridge(
            hand_urdf=ASSETS / "inspire_hand_new/inspire_hand_right.urdf",
            object_urdf=ASSETS / "mjcf/airplane.urdf", device="cpu", seed=42)
        lower, upper = native_joint_limits(bridge.hand_urdf, "cpu")
        rows = {name: load_rows(*item, lower, upper) for name, item in INPUTS.items()}
        if sha256(OLD_CALIBRATION) != OLD_CALIBRATION_SHA256:
            raise ValueError("old calibration SHA drift")
        old_gain = torch.tensor(json.loads(OLD_CALIBRATION.read_text())
                                ["coefficients"]["per_joint"], dtype=torch.float32)
        train_keys = (["train_s145_d03", "test_s147_d01"] if args.mix_small_dose
                      else ["train_s145_d03"])
        test_keys = (["test_s146_d03", "test_s148_d01"] if args.mix_small_dose
                     else ["test_s146_d03", "test_s147_d01"])
        train = {key: torch.cat([rows[name][key] for name in train_keys])
                 for key in rows[train_keys[0]]}
        fitted = {
            "action_only": fit_per_joint(train, use_velocity=False),
            "velocity_only": fit_per_joint(train, use_action=False),
            "action_velocity": fit_per_joint(train),
        }
        reports = {name: evaluate(bridge, rows[name], fitted, old_gain)
                   for name in test_keys}
        report = {
            "schema": "ref2dex.intervention_handflow_probe.v1",
            "run_status": "COMPLETED", "git_commit": manifest["git_commit"],
            "train_samples": len(train["q"]), "train_keys": train_keys,
            "test_keys": test_keys, "input_sha256": manifest["input_sha256"],
            "coefficients": {key: value.tolist() for key, value in fitted.items()},
            "reports": reports, "elapsed_seconds": time.monotonic() - start,
            "limit": "one trajectory/object; separate train and test seeds; online features only",
        }
        report_path = args.output / "report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        manifest.update(run_status="COMPLETED", final_step=1,
                        completed_at=datetime.now(timezone.utc).isoformat(),
                        report_sha256=sha256(report_path))
        print(json.dumps({name: {model: item["hand_surface_epe_mm"]
                                 for model, item in value["models"].items()}
                          for name, value in reports.items()}, sort_keys=True), flush=True)
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}",
                        completed_at=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

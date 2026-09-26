"""Train a compact geometric Cm on randomized executed-action transitions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import torch
from torch import nn
import torch.nn.functional as F

from src.task.CmResidual.cmlite import local_to_world_translation, local_translation_target
from src.task.CmResidual.dexplore_cm_geometry import native_joint_limits
from src.task.CmResidual.tools.audit_original_cmv2_on_policy import ASSETS
from src.task.CmResidual.tools.probe_intervention_handflow import INPUTS, sha256
from src.task.CmResidual.tools.probe_local_geometric_cm import (
    SixRegionCm, _surface_geometry_class, extract_features,
)
from src.task.CmResidual.v118_planner import QUERY_LINKS, TorchInspireKinematics


ROOT = Path(__file__).resolve().parents[4]
CALIBRATION = ROOT / "outputs/CmResidual/agent_intervention_handflow_mix_s145147_train_s146148_test/report.json"
CALIBRATION_SHA256 = "d0342ff4e17c66bb1d2e999416eecf24a0754c891e35916822f10609b42bdde3"
TRAIN_KEYS = ("train_s145_d03", "test_s147_d01")
TEST_KEYS = ("test_s146_d03", "test_s148_d01")


def load_randomized_rows(key):
    path, digest = INPUTS[key]
    if sha256(path) != digest:
        raise ValueError(f"transition SHA drift: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "ref2dex.randomized_action_transitions.v1" or (
            payload.get("run_status") != "COMPLETED"):
        raise ValueError("randomized data schema/status mismatch")
    values = payload["records"]
    selected = values["assignment"].reshape(-1) != 0
    if not values["pre_contact"][selected].all():
        raise ValueError("intervention sample lacks pre-contact")
    rows = {"q": values["q"][selected].float(),
            "dof_vel": values["dof_vel"][selected].float(),
            "action": values["executed_action"][selected].float(),
            "base_action": values["base_action"][selected].float(),
            "object_state": values["object_state"][selected].float(),
            "next_object_state": values["next_object_state"][selected].float(),
            "assignment": values["assignment"][selected].reshape(-1),
            "delta_z_action": payload["delta_z_action"]}
    rows["target"] = local_translation_target(
        rows["object_state"], rows["next_object_state"])
    rows["category"] = (rows["target"].norm(dim=-1) <= .002).long()
    if not all(torch.isfinite(value.float()).all() for value in rows.values()
               if isinstance(value, torch.Tensor)):
        raise FloatingPointError("non-finite randomized Cm row")
    return rows


def merge_rows(parts):
    return {key: torch.cat([part[key] for part in parts])
            for key in parts[0] if isinstance(parts[0][key], torch.Tensor)}


def raw_inputs(rows):
    q = rows["q"].clone()
    q[:, :3] -= rows["object_state"][:, :3]
    return torch.cat((q, rows["action"], rows["dof_vel"],
                      rows["object_state"]), dim=-1)


class RawActionEffect(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std.clamp_min(.01))
        self.net = nn.Sequential(nn.Linear(len(mean), 64), nn.SiLU(),
                                 nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 3))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, raw):
        return self.net((raw - self.mean) / self.std) * .01


def predict(model, data, kind):
    if kind == "raw":
        return model(data["raw"])
    return model(data["region"], data["context"])["delta_local"]


def fit_model(model, kind, train_data, schedule, *, lr=5e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    final_loss = None
    for ids in schedule:
        model.train()
        output = predict(model, {key: value[ids] for key, value in train_data.items()}, kind)
        target = train_data["target"][ids]
        loss = F.smooth_l1_loss(output / .01, target / .01, beta=.5)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite randomized Cm loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    model.eval()
    return final_loss


@torch.no_grad()
def metrics(model, data, rows, kind):
    pred = predict(model, data, kind)
    error = (pred - data["target"]).norm(dim=-1) * 1000
    assignment = rows["assignment"]
    world = local_to_world_translation(rows["object_state"], pred)
    return {
        "epe_mm": float(error.mean()),
        "plus_epe_mm": float(error[assignment == 1].mean()),
        "minus_epe_mm": float(error[assignment == -1].mean()),
        "observed_group_dz_contrast_mm": float(
            (world[assignment == 1, 2].mean() -
             world[assignment == -1, 2].mean()) * 1000),
        "predicted_motion_mm": float(pred.norm(dim=-1).mean() * 1000),
    }


@torch.no_grad()
def model_intervention_contrast(model, plus_data, minus_data, rows, kind):
    plus = predict(model, plus_data, kind)
    minus = predict(model, minus_data, kind)
    effect = local_to_world_translation(rows["object_state"], plus - minus)[:, 2] * 1000
    return {"mean_dz_mm": float(effect.mean()),
            "median_dz_mm": float(effect.median()),
            "positive_fraction": float((effect > 0).float().mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=96)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.steps <= 1000 or not 16 <= args.batch_size <= 256:
        raise ValueError("new output, 1-1000 steps and batch size 16-256 required")
    args.output.mkdir(parents=True)
    manifest_path = args.output / "run_manifest.json"
    manifest = {
        "run_status": "STARTED", "run_id": args.output.name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                               cwd=ROOT, text=True).strip(),
        "input_sha256": {key: INPUTS[key][1] for key in (*TRAIN_KEYS, *TEST_KEYS)},
        "calibration_sha256": CALIBRATION_SHA256,
        "train_keys": TRAIN_KEYS, "test_keys": TEST_KEYS,
        "steps": args.steps, "batch_size": args.batch_size,
        "cpu_threads": 2, "gpu_count": 0, "wall_budget_minutes": 30,
        "output_budget_mb": 10,
        "stop_rule": "input drift, non-finite loss, train/test seed leak or wall budget",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    start = time.monotonic()
    try:
        torch.set_num_threads(2)
        torch.manual_seed(240924)
        if sha256(CALIBRATION) != CALIBRATION_SHA256:
            raise ValueError("calibration SHA drift")
        coefficients = torch.tensor(json.loads(CALIBRATION.read_text())
                                    ["coefficients"]["action_velocity"])
        calibration = {"alpha": coefficients[:, 0], "velocity": coefficients[:, 1],
                       "bias": coefficients[:, 2]}
        hand_urdf = ASSETS / "inspire_hand_new/inspire_hand_right.urdf"
        kinematics = TorchInspireKinematics(hand_urdf, "cpu")
        geometry = _surface_geometry_class()(
            hand_urdf=hand_urdf, object_urdf=ASSETS / "mjcf/airplane.urdf",
            query_links=QUERY_LINKS, object_count=256, hand_count=256,
            seed=42, device="cpu")
        lower, upper = native_joint_limits(hand_urdf, "cpu")
        row_groups = {key: load_randomized_rows(key) for key in (*TRAIN_KEYS, *TEST_KEYS)}
        train_rows = merge_rows([row_groups[key] for key in TRAIN_KEYS])
        train_geom = extract_features(train_rows, kinematics=kinematics,
                                      geometry=geometry, lower=lower, upper=upper,
                                      calibration=calibration, batch_size=32)
        raw_train = raw_inputs(train_rows)
        raw_mean, raw_std = raw_train.mean(0), raw_train.std(0).clamp_min(.01)
        train_data = {"region": train_geom["region"], "context": train_geom["context"],
                      "raw": raw_train, "target": train_rows["target"]}
        state_train = dict(train_data)
        state_train["region"] = train_geom["region"].clone()
        state_train["region"][..., 11:16] = 0
        torch.manual_seed(240925)
        initial = {key: value.detach().clone() for key, value in SixRegionCm().state_dict().items()}
        geometric = SixRegionCm()
        geometric.load_state_dict(initial)
        state_only = SixRegionCm()
        state_only.load_state_dict(initial)
        raw = RawActionEffect(raw_mean, raw_std)
        generator = torch.Generator().manual_seed(240926)
        schedule = [torch.randint(len(train_rows["q"]), (args.batch_size,), generator=generator)
                    for _ in range(args.steps)]
        losses = {"geometric": fit_model(geometric, "geometric", train_data, schedule),
                  "state_only": fit_model(state_only, "geometric", state_train, schedule),
                  "raw_action": fit_model(raw, "raw", train_data, schedule)}
        reports = {}
        for key in TEST_KEYS:
            rows = row_groups[key]
            actual = extract_features(rows, kinematics=kinematics, geometry=geometry,
                                      lower=lower, upper=upper, calibration=calibration,
                                      batch_size=32)
            data = {"region": actual["region"], "context": actual["context"],
                    "raw": raw_inputs(rows), "target": rows["target"]}
            state = dict(data)
            state["region"] = data["region"].clone()
            state["region"][..., 11:16] = 0
            delta = rows["delta_z_action"]
            counterfactual = {}
            for label, sign in (("plus", 1), ("minus", -1)):
                changed = dict(rows)
                changed["action"] = rows["base_action"].clone()
                changed["action"][:, 2] = (changed["action"][:, 2] + sign * delta).clamp(-1, 1)
                features = extract_features(changed, kinematics=kinematics,
                                            geometry=geometry, lower=lower, upper=upper,
                                            calibration=calibration, batch_size=32)
                counterfactual[label] = {
                    "region": features["region"], "context": features["context"],
                    "raw": raw_inputs(changed), "target": rows["target"]}
            models = {"geometric": (geometric, data, "geometric"),
                      "state_only": (state_only, state, "geometric"),
                      "raw_action": (raw, data, "raw")}
            model_metrics = {}
            for name, (model, model_data, kind) in models.items():
                if name == "state_only":
                    plus = dict(counterfactual["plus"])
                    minus = dict(counterfactual["minus"])
                    for part in (plus, minus):
                        part["region"] = part["region"].clone()
                        part["region"][..., 11:16] = 0
                else:
                    plus, minus = counterfactual["plus"], counterfactual["minus"]
                model_metrics[name] = metrics(model, model_data, rows, kind)
                model_metrics[name]["model_intervention"] = model_intervention_contrast(
                    model, plus, minus, rows, kind)
            reports[key] = {
                "samples": len(rows["q"]), "delta_z_action": delta,
                "zero_motion_epe_mm": float(rows["target"].norm(dim=-1).mean() * 1000),
                "actual_randomized_group_dz_mm": float(
                    ((rows["next_object_state"] - rows["object_state"])
                     [rows["assignment"] == 1, 2].mean() -
                     (rows["next_object_state"] - rows["object_state"])
                     [rows["assignment"] == -1, 2].mean()) * 1000),
                "models": model_metrics,
            }
            print(json.dumps({"key": key, "zero_mm": reports[key]["zero_motion_epe_mm"],
                              "models": {name: {"epe_mm": item["epe_mm"],
                                                "model_ate_mm": item["model_intervention"]["mean_dz_mm"]}
                                         for name, item in model_metrics.items()}}, sort_keys=True), flush=True)
        report = {"schema": "ref2dex.randomized_geometric_cm_probe.v1",
                  "run_status": "COMPLETED", "git_commit": manifest["git_commit"],
                  "train_samples": len(train_rows["q"]),
                  "parameter_count": {"geometric": sum(p.numel() for p in geometric.parameters()),
                                      "raw_action": sum(p.numel() for p in raw.parameters())},
                  "final_train_losses": losses, "reports": reports,
                  "elapsed_seconds": time.monotonic() - start,
                  "limit": "randomized population effects only; no individual physical counterfactual or policy utility"}
        report_path = args.output / "report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        torch.save({"model": geometric.state_dict(), "schema": "six_region_cm_randomized.v1"},
                   args.output / "geometric.pt")
        torch.save({"model": state_only.state_dict(), "schema": "six_region_cm_state_only.v1"},
                   args.output / "state_only.pt")
        torch.save({"model": raw.state_dict(), "schema": "raw_action_effect.v1"},
                   args.output / "raw_action.pt")
        manifest.update(run_status="COMPLETED", final_step=args.steps,
                        completed_at=datetime.now(timezone.utc).isoformat(),
                        report_sha256=sha256(report_path),
                        checkpoint_sha256={name: sha256(args.output / f"{name}.pt")
                                           for name in ("geometric", "state_only", "raw_action")})
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}",
                        completed_at=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

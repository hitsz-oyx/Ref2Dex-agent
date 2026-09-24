"""Probe shared geometric action Cm on expert-origin, object-disjoint sim data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.task.CmResidual.cmlite import local_to_world_translation, local_translation_target
from src.task.CmResidual.dexplore_cm_geometry import (
    dexplore_action_to_native_targets, native_joint_limits,
)
from src.task.CmResidual.tools.analyze_randomized_action import weighted_step_difference
from src.task.CmResidual.tools.audit_original_cmv2_on_policy import ASSETS
from src.task.CmResidual.tools.probe_intervention_handflow import fit_per_joint, sha256
from src.task.CmResidual.tools.probe_local_geometric_cm import (
    SixRegionCm, _surface_geometry_class, extract_features,
)
from src.task.CmResidual.tools.probe_randomized_geometric_cm import RawActionEffect, raw_inputs
from src.task.CmResidual.tools.probe_contact_aware_cm import ranked_effect
from src.task.CmResidual.v118_planner import QUERY_LINKS, TorchInspireKinematics


ROOT = Path(__file__).resolve().parents[4]
TRAIN = ROOT / "outputs/CmResidual/agent_expert_crossobject_randomized_train_s186_h5"
TEST = ROOT / "outputs/CmResidual/agent_expert_crossobject_randomized_apple_s187_h5"
EXPECTED = {"train": "83bad847843606ee4058b31d35ddb0c5db44b2e4ca2effba6e5de790cc597dc5",
            "heldout": "7150f7298303a1428e9363c8a1094115fe6ce0c4b03550cfa0a3949487527aa6"}
SPLIT_SHA = "35fffcb500f1f3db76fb8a59c940f5da0a113627bb0be9db0d68b0112fe7f516"
OFFICIAL_SHA = "8f6823db752288f1bddd6d042981d33514e29dac5a68e58726e76215fea6d553"


def load_rows(path: Path, partition: str) -> tuple[dict, dict]:
    manifest = json.loads((path / "run_manifest.json").read_text())
    source = path / "transitions.pt"
    if (manifest.get("run_status") != "COMPLETED" or
            manifest.get("source_partition") != partition or
            manifest.get("source_actor_role") != "official_data_collector" or
            manifest.get("checkpoint_sha256") != OFFICIAL_SHA or
            manifest.get("motion_manifest_sha256") != SPLIT_SHA or
            manifest.get("transitions_sha256") != EXPECTED[partition] or
            sha256(source) != EXPECTED[partition]):
        raise ValueError(f"pinned expert data provenance drift: {path}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if (payload.get("run_status") != "COMPLETED" or
            payload.get("schema") != "ref2dex.randomized_action_followup.v1" or
            payload.get("followup_horizon") != 5 or
            payload.get("intervention_axes") != [2] or
            payload.get("delta_z_action") != .1):
        raise ValueError("randomized source schema drift")
    full = payload["records"]
    take = full["assignment"].reshape(-1) != 0
    if not full["pre_contact"][take].all():
        raise ValueError("treated state without pre-contact")
    dose = full["executed_action"] - full["base_action"]
    expected = torch.zeros_like(dose)
    expected[:, 2] = full["assignment"] * .1
    if not torch.allclose(dose, expected, atol=1e-5):
        raise ValueError("actual randomized dose mismatch")
    rows = {name: value[take].clone() for name, value in full.items()}
    rows["env_id"] = torch.arange(len(take))[take] % 64
    rows["action"] = rows["executed_action"].float()
    rows["target"] = local_translation_target(
        rows["object_state"], rows["followup_object_state"])
    rows["category"] = torch.zeros(len(rows["q"]), dtype=torch.long)
    rows["contact_fraction"] = rows["followup_contact_count"].float() / 5
    rows["survived"] = ((rows["followup_progress"] - rows["progress"] == 5) &
                        (rows["followup_reset"] == 0))
    if not rows["survived"].all():
        raise ValueError("incomplete five-step followup")
    if not all(torch.isfinite(value.float()).all() for value in rows.values()):
        raise FloatingPointError("non-finite randomized Cm data")
    return rows, manifest


def calibrate(train: dict) -> dict:
    hand_urdf = ASSETS / "inspire_hand_new/inspire_hand_right.urdf"
    lower, upper = native_joint_limits(hand_urdf, "cpu")
    calibration_rows = {"q": train["q"].float(), "dof_vel": train["dof_vel"].float(),
                        "next_q": train["next_q"].float(),
                        "pd_target": dexplore_action_to_native_targets(
                            train["action"], train["q"], lower, upper)}
    coeff = fit_per_joint(calibration_rows)
    calibration = {"alpha": coeff[:, 0], "velocity": coeff[:, 1],
                   "bias": coeff[:, 2]}
    return {"calibration": calibration, "lower": lower, "upper": upper,
            "hand_urdf": hand_urdf, "coefficients": coeff}


@torch.no_grad()
def geometric_inputs(rows: dict, names: list[str], setup: dict) -> dict:
    n = len(rows["q"])
    region = torch.empty(n, 6, 19)
    context = torch.empty(n, 6)
    kinematics = TorchInspireKinematics(setup["hand_urdf"], "cpu")
    for motion_id, name in enumerate(names):
        selected = torch.where(rows["motion_id"] == motion_id)[0]
        if not len(selected):
            raise ValueError(f"no rows for {name}")
        part = {key: value[selected] for key, value in rows.items()
                if isinstance(value, torch.Tensor)}
        geometry = _surface_geometry_class()(
            hand_urdf=setup["hand_urdf"],
            object_urdf=ASSETS / f"mjcf/{name}.urdf",
            query_links=QUERY_LINKS, object_count=256, hand_count=256,
            seed=42, device="cpu")
        features = extract_features(
            part, kinematics=kinematics, geometry=geometry,
            lower=setup["lower"], upper=setup["upper"],
            calibration=setup["calibration"], batch_size=32)
        region[selected] = features["region"]
        context[selected] = features["context"]
    return {"region": region, "context": context}


def arm_inputs(rows: dict, names: list[str], setup: dict, sign: int) -> dict:
    candidate = dict(rows)
    candidate["action"] = rows["base_action"].clone()
    candidate["action"][:, 2] = (candidate["action"][:, 2] + sign * .1).clamp(-1, 1)
    return {**geometric_inputs(candidate, names, setup), "raw": raw_inputs(candidate)}


def fit(model, kind: str, data: dict, target: torch.Tensor,
        schedule: list[torch.Tensor]) -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    loss_value = float("nan")
    for ids in schedule:
        model.train()
        if kind == "raw":
            prediction = model(data["raw"][ids])
        else:
            prediction = model(data["region"][ids], data["context"][ids])["delta_local"]
        loss = F.smooth_l1_loss(prediction / .02, target[ids] / .02, beta=.5)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite Cm fit")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
    model.eval()
    return loss_value


@torch.no_grad()
def predict(model, kind: str, data: dict) -> torch.Tensor:
    if kind == "raw":
        return model(data["raw"])
    return model(data["region"], data["context"])["delta_local"]


def evaluate(model, kind: str, rows: dict, factual: dict, plus: dict,
             minus: dict, *, seed: int) -> dict:
    actual = rows["target"]
    factual_pred = predict(model, kind, factual)
    error = (factual_pred - actual).norm(dim=-1).mean() * 1000
    plus_pred = predict(model, kind, plus)
    minus_pred = predict(model, kind, minus)
    plus_z = local_to_world_translation(rows["object_state"], plus_pred)[:, 2]
    minus_z = local_to_world_translation(rows["object_state"], minus_pred)[:, 2]
    score = ((plus_z - minus_z) * 1000).numpy()
    observed = ((rows["followup_object_state"][:, 2] -
                 rows["object_state"][:, 2]) *
                rows["contact_fraction"] * 1000).numpy()
    assignment = rows["assignment"].numpy()
    stratum = (rows["global_step"] + 1000 * rows["motion_id"]).numpy()
    env_id = rows["env_id"].numpy()
    report = {"factual_epe_mm": float(error),
              "predicted_counterfactual_effect_mean_mm": float(score.mean()),
              "predicted_counterfactual_effect_std_mm": float(score.std()),
              "actual_randomized_effect_mm": weighted_step_difference(
                  observed, assignment, stratum)[0]}
    if float(np.quantile(score, .75) - np.quantile(score, .25)) >= 1e-6:
        report["ranked_uplift"] = ranked_effect(
            score, observed, assignment, stratum, env_id,
            bootstraps=500, seed=seed)
    else:
        report["ranked_uplift"] = {"status": "NO_SCORE_SPREAD"}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.steps <= 500:
        parser.error("new output and 1..500 bounded steps required")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    manifest_path = args.output / "run_manifest.json"
    manifest = {"run_status": "STARTED", "started_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       cwd=ROOT, text=True).strip(),
                "source_actor_role": "official_data_collector",
                "fit_objects": ["airplane", "mug", "toothpaste"],
                "heldout_objects": ["apple"], "split_sha256": SPLIT_SHA,
                "source_sha256": EXPECTED, "steps": args.steps,
                "gpu_count": 0, "cpu_threads": 2, "wall_budget_minutes": 60,
                "output_budget_mb": 20,
                "stop_rule": "source drift, leakage, non-finite output or wall budget"}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    try:
        torch.set_num_threads(2)
        torch.manual_seed(240924)
        train, train_manifest = load_rows(TRAIN, "train")
        test, test_manifest = load_rows(TEST, "heldout")
        train_names, test_names = train_manifest["source_objects"], test_manifest["source_objects"]
        if (train_names != ["airplane", "mug", "toothpaste"] or test_names != ["apple"] or
                set(train_names) & set(test_names)):
            raise ValueError("object identity leakage or split drift")
        setup = calibrate(train)
        train_input = {**geometric_inputs(train, train_names, setup),
                       "raw": raw_inputs(train)}
        test_input = {**geometric_inputs(test, test_names, setup),
                      "raw": raw_inputs(test)}
        test_plus = arm_inputs(test, test_names, setup, 1)
        test_minus = arm_inputs(test, test_names, setup, -1)
        mean = train_input["raw"].mean(0)
        std = train_input["raw"].std(0).clamp_min(.01)
        torch.manual_seed(240925)
        geometric = SixRegionCm()
        initial = {key: value.detach().clone() for key, value in geometric.state_dict().items()}
        state_only = SixRegionCm()
        state_only.load_state_dict(initial)
        raw = RawActionEffect(mean, std)
        state_train = dict(train_input)
        state_train["region"] = train_input["region"].clone()
        state_train["region"][..., 11:16] = 0
        state_test = dict(test_input)
        state_test["region"] = test_input["region"].clone()
        state_test["region"][..., 11:16] = 0
        state_plus = dict(test_plus)
        state_plus["region"] = test_plus["region"].clone()
        state_plus["region"][..., 11:16] = 0
        state_minus = dict(test_minus)
        state_minus["region"] = test_minus["region"].clone()
        state_minus["region"][..., 11:16] = 0
        generator = torch.Generator().manual_seed(240926)
        schedule = [torch.randint(len(train["q"]), (96,), generator=generator)
                    for _ in range(args.steps)]
        losses = {"geometric": fit(geometric, "geometric", train_input, train["target"], schedule),
                  "state_only": fit(state_only, "geometric", state_train, train["target"], schedule),
                  "raw_action": fit(raw, "raw", train_input, train["target"], schedule)}
        results = {"geometric": evaluate(geometric, "geometric", test, test_input,
                                           test_plus, test_minus, seed=240927),
                   "state_only": evaluate(state_only, "geometric", test, state_test,
                                            state_plus, state_minus, seed=240928),
                   "raw_action": evaluate(raw, "raw", test, test_input,
                                            test_plus, test_minus, seed=240929)}
        train_outcome = ((train["followup_object_state"][:, 2] -
                          train["object_state"][:, 2]) * train["contact_fraction"] * 1000).numpy()
        train_effect = weighted_step_difference(
            train_outcome, train["assignment"].numpy(),
            (train["global_step"] + train["motion_id"] * 1000).numpy())[0]
        report = {"schema": "ref2dex.expert_crossobject_cm_probe.v1",
                  "source_actor_role": "official_data_collector",
                  "train_count": len(train["q"]), "heldout_count": len(test["q"]),
                  "train_constant_action_effect_mm": train_effect,
                  "heldout_zero_epe_mm": float(test["target"].norm(dim=-1).mean() * 1000),
                  "handflow_train_q_mae": float((train["next_q"] - train["q"] -
                        setup["coefficients"][:, 0] *
                        (dexplore_action_to_native_targets(train["action"], train["q"],
                         setup["lower"], setup["upper"]) - train["q"]) -
                        setup["coefficients"][:, 1] * train["dof_vel"] -
                        setup["coefficients"][:, 2]).abs().mean()),
                  "losses": losses, "heldout": results,
                  "elapsed_seconds": time.monotonic() - started}
        (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        manifest.update(run_status="COMPLETED", report_sha256=sha256(args.output / "report.json"))
        print(json.dumps(report, sort_keys=True), flush=True)
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

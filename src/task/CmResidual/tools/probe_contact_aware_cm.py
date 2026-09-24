"""Train and test a contact-aware action Cm on randomized simulator followups."""
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

from src.task.CmResidual.cmlite import local_translation_target
from src.task.CmResidual.contact_aware_cm import ContactAwareCm, RawContactAwareCm
from src.task.CmResidual.dexplore_cm_geometry import native_joint_limits
from src.task.CmResidual.tools.analyze_randomized_action import weighted_step_difference
from src.task.CmResidual.tools.audit_original_cmv2_on_policy import ASSETS
from src.task.CmResidual.tools.probe_intervention_handflow import sha256
from src.task.CmResidual.tools.probe_local_geometric_cm import (
    _surface_geometry_class, extract_features,
)
from src.task.CmResidual.tools.probe_randomized_geometric_cm import (
    CALIBRATION, CALIBRATION_SHA256, raw_inputs,
)
from src.task.CmResidual.v118_planner import QUERY_LINKS, TorchInspireKinematics


ROOT = Path(__file__).resolve().parents[4]
DATA = {
    151: (ROOT / "outputs/CmResidual/agent_randomized_followup_s151_d01_h5_n64/transitions.pt",
          "151825821ba8c43762d71c40c4dd20886d83d4f66c1c9168564a002658ac2b05"),
    152: (ROOT / "outputs/CmResidual/agent_randomized_followup_s152_d01_h5_n64/transitions.pt",
          "2d34b3adb6de8c61da7ce817b2fcf4fa898d513bf9cadf6385214eb103826a15"),
    153: (ROOT / "outputs/CmResidual/agent_randomized_followup_s153_d01_h5_n64/transitions.pt",
          "9e2d96ac37b6bbafc393215274360038692511487a755899ce04e5dd81ee948b"),
    154: (ROOT / "outputs/CmResidual/agent_randomized_followup_s154_d01_h5_n64/transitions.pt",
          "9120d8e52e6b7b24bd4d66876c82938a1db3761fb399e5232a05947b42b9e210"),
}
TRAIN_SEEDS = (151, 152)
TEST_SEEDS = (153, 154)


def load_rows(seed):
    path, digest = DATA[seed]
    if sha256(path) != digest:
        raise ValueError(f"followup input SHA drift: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "ref2dex.randomized_action_followup.v1" or (
            payload.get("run_status") != "COMPLETED" or payload.get("followup_horizon") != 5):
        raise ValueError("followup schema/status/horizon mismatch")
    full = payload["records"]
    selected = full["assignment"].reshape(-1) != 0
    if not full["pre_contact"][selected].all():
        raise ValueError("non-contact intervention")
    rows = {"q": full["q"][selected].float(),
            "dof_vel": full["dof_vel"][selected].float(),
            "action": full["executed_action"][selected].float(),
            "base_action": full["base_action"][selected].float(),
            "object_state": full["object_state"][selected].float(),
            "step_target": local_translation_target(
                full["object_state"][selected], full["next_object_state"][selected]),
            "followup_target": local_translation_target(
                full["object_state"][selected], full["followup_object_state"][selected]),
            "contact_target": full["followup_contact_count"][selected].float() / 5,
            "assignment": full["assignment"][selected].reshape(-1),
            "step": full["global_step"][selected].reshape(-1),
            "env_id": torch.arange(len(selected))[selected] % 64}
    rows["target"] = rows["step_target"]
    rows["category"] = (rows["target"].norm(dim=-1) <= .002).long()
    if not all(torch.isfinite(value.float()).all() for value in rows.values()):
        raise FloatingPointError("non-finite followup row")
    return rows


def concatenate(rows):
    return {key: torch.cat([part[key] for part in rows]) for key in rows[0]}


def make_geometry():
    if sha256(CALIBRATION) != CALIBRATION_SHA256:
        raise ValueError("handflow calibration SHA drift")
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
    return {"kinematics": kinematics, "geometry": geometry,
            "lower": lower, "upper": upper, "calibration": calibration,
            "batch_size": 32}


def model_input(rows, geom_args):
    geo = extract_features(rows, **geom_args)
    return {"region": geo["region"], "context": geo["context"],
            "raw": raw_inputs(rows),
            "step": rows["step_target"],
            "followup": rows["followup_target"],
            "contact": rows["contact_target"]}


def predict(model, data, kind):
    return model(data["raw"]) if kind == "raw" else model(data["region"], data["context"])


def train(model, data, schedule, *, kind):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    last_loss = None
    for ids in schedule:
        model.train()
        output = predict(model, {key: value[ids] for key, value in data.items()}, kind)
        one = F.smooth_l1_loss(output["delta_local"] / .01,
                               data["step"][ids] / .01, beta=.5)
        future = F.smooth_l1_loss(output["followup_delta_local"] / .02,
                                  data["followup"][ids] / .02, beta=.5)
        contact = F.mse_loss(output["contact_fraction"], data["contact"][ids])
        loss = .5 * one + .5 * future + 2 * contact
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite contact-aware Cm loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())
    model.eval()
    return last_loss


@torch.no_grad()
def prediction_metrics(model, data, *, kind):
    output = predict(model, data, kind)
    return {"one_step_epe_mm": float((output["delta_local"] - data["step"]).norm(dim=-1).mean() * 1000),
            "followup_epe_mm": float((output["followup_delta_local"] - data["followup"]).norm(dim=-1).mean() * 1000),
            "contact_rmse": float(((output["contact_fraction"] - data["contact"]).square().mean()).sqrt()),
            "contact_mean_predicted": float(output["contact_fraction"].mean()),
            "contact_mean_actual": float(data["contact"].mean())}


@torch.no_grad()
def candidate_contact_score(model, rows, geom_args, *, kind):
    predictions = []
    for sign in (1, -1):
        candidate = dict(rows)
        candidate["action"] = rows["base_action"].clone()
        candidate["action"][:, 2] = (candidate["action"][:, 2] + sign * .1).clamp(-1, 1)
        data = model_input(candidate, geom_args)
        predictions.append(predict(model, data, kind)["contact_fraction"])
    return (predictions[0] - predictions[1]).numpy()


def ranked_effect(score, observed, assignment, stratum, env_id, *, bootstraps, seed):
    q25, q75 = np.quantile(score, [.25, .75])
    if q75 - q25 < 1e-6:
        raise ValueError("contact action-score spread too small")
    low, high = score <= q25, score >= q75
    if min(low.sum(), high.sum()) < 80:
        raise ValueError("sparse contact-score quartiles")
    def group_effect(indices, group):
        chosen = indices[group[indices]]
        return weighted_step_difference(observed[chosen], assignment[chosen],
                                        stratum[chosen])[0]
    full = np.arange(len(score))
    low_ate = group_effect(full, low)
    high_ate = group_effect(full, high)
    rng = np.random.default_rng(seed)
    clusters = [np.flatnonzero(env_id == identifier) for identifier in np.unique(env_id)]
    draws = []
    for _ in range(bootstraps):
        sampled = rng.integers(0, len(clusters), len(clusters))
        indices = np.concatenate([clusters[index] for index in sampled])
        try:
            draws.append(group_effect(indices, high) - group_effect(indices, low))
        except ValueError:
            continue
    if len(draws) < .9 * bootstraps:
        raise ValueError("too few effective cluster bootstraps")
    return {"score_q25": float(q25), "score_q75": float(q75),
            "low_count": int(low.sum()), "high_count": int(high.sum()),
            "actual_contact_ate_low": low_ate,
            "actual_contact_ate_high": high_ate,
            "actual_high_minus_low": high_ate - low_ate,
            "cluster_95ci": np.quantile(draws, [.025, .975]).tolist(),
            "effective_bootstraps": len(draws)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--bootstraps", type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.steps <= 1000 or not 100 <= args.bootstraps <= 5000:
        parser.error("new output, bounded steps and bootstraps required")
    args.output.mkdir(parents=True)
    manifest_path = args.output / "run_manifest.json"
    manifest = {"run_status": "STARTED", "run_id": args.output.name,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       cwd=ROOT, text=True).strip(),
                "train_seeds": TRAIN_SEEDS, "test_seeds": TEST_SEEDS,
                "input_sha256": {seed: DATA[seed][1] for seed in DATA},
                "calibration_sha256": CALIBRATION_SHA256,
                "steps": args.steps, "bootstraps": args.bootstraps,
                "cpu_threads": 2, "gpu_count": 0, "wall_budget_minutes": 60,
                "output_budget_mb": 20,
                "stop_rule": "input drift, leakage, non-finite output or wall budget"}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    started = time.monotonic()
    try:
        torch.set_num_threads(2)
        torch.manual_seed(240930)
        geom_args = make_geometry()
        rows = {seed: load_rows(seed) for seed in DATA}
        train_rows = concatenate([rows[seed] for seed in TRAIN_SEEDS])
        train_data = model_input(train_rows, geom_args)
        raw_mean, raw_std = train_data["raw"].mean(0), train_data["raw"].std(0).clamp_min(.01)
        torch.manual_seed(240931)
        geometric = ContactAwareCm()
        initial = {key: value.detach().clone() for key, value in geometric.state_dict().items()}
        state_only = ContactAwareCm()
        state_only.load_state_dict(initial)
        state_data = dict(train_data)
        state_data["region"] = train_data["region"].clone()
        state_data["region"][..., 11:16] = 0
        raw = RawContactAwareCm(raw_mean, raw_std)
        generator = torch.Generator().manual_seed(240932)
        schedule = [torch.randint(len(train_rows["q"]), (96,), generator=generator)
                    for _ in range(args.steps)]
        losses = {"geometric": train(geometric, train_data, schedule, kind="geometric"),
                  "state_only": train(state_only, state_data, schedule, kind="geometric"),
                  "raw_action": train(raw, train_data, schedule, kind="raw")}
        models = {"geometric": (geometric, "geometric"),
                  "state_only": (state_only, "geometric"),
                  "raw_action": (raw, "raw")}
        metrics = {}
        scoring = {"geometric": [], "raw_action": []}
        pooled = {"observed": [], "assignment": [], "stratum": [], "env_id": []}
        for seed_index, seed in enumerate(TEST_SEEDS):
            test = model_input(rows[seed], geom_args)
            no_action = dict(test)
            no_action["region"] = test["region"].clone()
            no_action["region"][..., 11:16] = 0
            metrics[seed] = {
                name: prediction_metrics(model, no_action if name == "state_only" else test,
                                         kind=kind)
                for name, (model, kind) in models.items()}
            for name in scoring:
                model, kind = models[name]
                scoring[name].append(candidate_contact_score(model, rows[seed], geom_args,
                                                              kind=kind))
            pooled["observed"].append(rows[seed]["contact_target"].numpy())
            pooled["assignment"].append(rows[seed]["assignment"].numpy())
            pooled["stratum"].append((rows[seed]["step"] + seed_index * 1000).numpy())
            pooled["env_id"].append((rows[seed]["env_id"] + seed_index * 64).numpy())
        pooled = {key: np.concatenate(parts) for key, parts in pooled.items()}
        rank = {name: ranked_effect(np.concatenate(scores),
                                    pooled["observed"], pooled["assignment"],
                                    pooled["stratum"], pooled["env_id"],
                                    bootstraps=args.bootstraps, seed=240933)
                for name, scores in scoring.items()}
        report = {"schema": "ref2dex.contact_aware_cm_probe.v1",
                  "run_status": "COMPLETED", "train_samples": len(train_rows["q"]),
                  "test_samples": {seed: len(rows[seed]["q"]) for seed in TEST_SEEDS},
                  "parameter_count": {name: sum(p.numel() for p in model.parameters())
                                      for name, (model, _) in models.items()},
                  "final_train_losses": losses, "test_prediction": metrics,
                  "contact_effect_ranking_pooled": rank,
                  "observed_contact_ate_pooled": weighted_step_difference(
                      pooled["observed"], pooled["assignment"], pooled["stratum"])[0],
                  "elapsed_seconds": time.monotonic() - started,
                  "limits": "Probe: sequentially randomized short-horizon effects, not individual counterfactuals or policy utility"}
        report_path = args.output / "report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        for name, (model, _) in models.items():
            torch.save({"schema": "ref2dex.contact_aware_cm.v1", "name": name,
                        "model": model.state_dict()}, args.output / f"{name}.pt")
        manifest.update(run_status="COMPLETED", final_step=args.steps,
                        completed_at=datetime.now(timezone.utc).isoformat(),
                        report_sha256=sha256(report_path),
                        checkpoint_sha256={name: sha256(args.output / f"{name}.pt")
                                           for name in models})
        print(json.dumps({"run_status": "COMPLETED", "rank": rank,
                          "test_prediction": metrics}, sort_keys=True), flush=True)
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}",
                        completed_at=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

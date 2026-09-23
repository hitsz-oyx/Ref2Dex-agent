"""CPU-only causal calibration of DExplore action to actually executed hand flow."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import torch

from src.task.CmResidual.dexplore_cm_geometry import (
    DExploreCmv2GeometryBridge, dexplore_action_to_native_targets, native_joint_limits,
)
from src.task.CmResidual.tools.analyze_cmlite_on_policy import first_episode_mask
from src.task.CmResidual.tools.audit_original_cmv2_on_policy import ASSETS, TRANSITIONS
from src.task.CmResidual.tools.probe_mixed_cm_sim_adapt import TRAIN
from src.task.CmResidual.tools.probe_mixed_cm_sim_transfer import sha256


ROOT = Path(__file__).resolve().parents[4]
ENV_COUNT = 64


def load_rows(path: Path, expected: str, lower: torch.Tensor, upper: torch.Tensor):
    if sha256(path) != expected:
        raise ValueError(f"transition SHA drift: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "ref2dex.cmlite_transition.v1":
        raise ValueError(f"transition schema mismatch: {path}")
    q, action, next_q = (payload[key].float() for key in ("q", "action", "next_q"))
    if q.shape != action.shape or q.shape != next_q.shape or q.shape[1] != 18:
        raise ValueError("q/action/next_q must match [N,18]")
    if len(q) % ENV_COUNT:
        raise ValueError("step-major transition length must divide num_envs")
    done = payload["done"].reshape(-1).bool()
    mask = first_episode_mask(payload["done"], ENV_COUNT)
    mask[:ENV_COUNT] = False  # no valid previous control-step state at t=0
    mask &= ~torch.roll(done, ENV_COUNT, dims=0)
    indices = torch.where(mask)[0]
    previous = torch.roll(q, ENV_COUNT, dims=0)
    contact = (payload["hand_contact"].bool() & payload["object_contact"].bool()).reshape(-1)
    rows = {"q": q[indices], "q_prev": previous[indices],
            "action": action[indices], "next_q": next_q[indices],
            "contact": contact[indices]}
    rows["target"] = dexplore_action_to_native_targets(rows["action"], rows["q"], lower, upper)
    for key, value in rows.items():
        if not torch.isfinite(value.float()).all():
            raise FloatingPointError(f"non-finite {key}")
    return rows


def fit(train):
    x = (train["target"] - train["q"]).double()
    v = (train["q"] - train["q_prev"]).double()
    y = (train["next_q"] - train["q"]).double()
    uniform = ((x * y).sum() / x.square().sum().clamp_min(1e-8)).clamp(0, 1)
    per_joint = ((x * y).sum(dim=0) / x.square().sum(dim=0).clamp_min(1e-8)).clamp(0, 1)
    ridge = 1e-5
    xx = x.square().sum(dim=0) + ridge
    vv = v.square().sum(dim=0) + ridge
    xv = (x * v).sum(dim=0)
    xy = (x * y).sum(dim=0)
    vy = (v * y).sum(dim=0)
    determinant = (xx * vv - xv.square()).clamp_min(1e-8)
    alpha = ((xy * vv - vy * xv) / determinant).clamp(0, 1)
    beta = ((vy * xx - xy * xv) / determinant).clamp(-1, 1)
    return {"uniform": uniform.float(), "per_joint": per_joint.float(),
            "alpha": alpha.float(), "beta": beta.float()}


def predict(rows, fitted):
    q = rows["q"]
    x = rows["target"] - q
    v = q - rows["q_prev"]
    return {"stationary": q,
            "pd_target": rows["target"],
            "uniform_gain": q + fitted["uniform"] * x,
            "per_joint_gain": q + fitted["per_joint"] * x,
            "gain_momentum": q + fitted["alpha"] * x + fitted["beta"] * v}


def sample(rows, seed, each):
    generator = torch.Generator().manual_seed(2400 + seed)
    ids = []
    for flag in (True, False):
        choices = torch.where(rows["contact"] == flag)[0]
        if len(choices) < each:
            raise ValueError(f"seed {seed} lacks {each} contact={flag}")
        ids.append(choices[torch.randperm(len(choices), generator=generator)[:each]])
    selected = torch.cat(ids)
    return {key: value[selected] for key, value in rows.items()}


@torch.no_grad()
def surface_points(bridge, q):
    links = bridge.kinematics.forward(q[:, None])[:, 0]
    points, _ = bridge.geometry.hand(links)
    return points


@torch.no_grad()
def evaluate(bridge, rows, fitted, lower, upper, batch_size):
    names = tuple(predict(rows, fitted))
    totals = {name: {group: {"count": 0, "hand_epe_mm": 0.0,
                            "q_delta_mae": 0.0} for group in ("contact", "no_contact")}
              for name in names}
    actual_motion = {group: {"count": 0, "hand_mean_mm": 0.0}
                     for group in ("contact", "no_contact")}
    action_change = {"count": 0, "mean_mm": 0.0}
    shuffled_actions = rows["action"].roll(shifts=1, dims=0)
    for start in range(0, len(rows["q"]), batch_size):
        stop = start + batch_size
        batch = {key: value[start:stop] for key, value in rows.items()}
        current = surface_points(bridge, batch["q"])
        actual = surface_points(bridge, batch["next_q"])
        actual_mm = (actual - current).norm(dim=-1).mean(dim=-1) * 1000
        predicted = predict(batch, fitted)
        for name, q_hat in predicted.items():
            points = surface_points(bridge, q_hat)
            errors = (points - actual).norm(dim=-1).mean(dim=-1) * 1000
            q_error = (q_hat - batch["next_q"]).abs().mean(dim=-1)
            for i, flag in enumerate(batch["contact"]):
                group = "contact" if flag else "no_contact"
                item = totals[name][group]
                item["count"] += 1
                item["hand_epe_mm"] += float(errors[i])
                item["q_delta_mae"] += float(q_error[i])
        shuffled = dict(batch)
        shuffled["action"] = shuffled_actions[start:stop]
        shuffled["target"] = dexplore_action_to_native_targets(
            shuffled["action"], shuffled["q"], lower, upper)
        shifted_points = surface_points(bridge, predict(shuffled, fitted)["gain_momentum"])
        original_points = surface_points(bridge, predicted["gain_momentum"])
        action_change["mean_mm"] += float(
            (shifted_points - original_points).norm(dim=-1).mean(dim=-1).sum() * 1000)
        action_change["count"] += len(batch["q"])
        for i, flag in enumerate(batch["contact"]):
            group = "contact" if flag else "no_contact"
            actual_motion[group]["count"] += 1
            actual_motion[group]["hand_mean_mm"] += float(actual_mm[i])
    for by_group in totals.values():
        for item in by_group.values():
            for key in ("hand_epe_mm", "q_delta_mae"):
                item[key] /= item["count"]
    for item in actual_motion.values():
        item["hand_mean_mm"] /= item["count"]
    action_change["mean_mm"] /= action_change["count"]
    return {"metrics": totals, "actual_motion": actual_motion,
            "predicted_action_change": action_change}


def execute(args, manifest):
    torch.set_num_threads(2)
    torch.manual_seed(2400)
    bridge = DExploreCmv2GeometryBridge(
        hand_urdf=ASSETS / "inspire_hand_new/inspire_hand_right.urdf",
        object_urdf=ASSETS / "mjcf/airplane.urdf", device="cpu", seed=42)
    lower, upper = native_joint_limits(bridge.hand_urdf, "cpu")
    train_parts = [load_rows(path, digest, lower, upper) for path, digest in TRAIN.values()]
    train = {key: torch.cat([item[key] for item in train_parts]) for key in train_parts[0]}
    fitted = fit(train)
    reports = {}
    for seed, (path, digest) in TRANSITIONS.items():
        rows = load_rows(path, digest, lower, upper)
        selected = sample(rows, seed, args.samples_per_group)
        reports[str(seed)] = evaluate(bridge, selected, fitted, lower, upper, args.batch_size)
        print(json.dumps({"seed": seed, "contact": {name: value["contact"]["hand_epe_mm"]
                    for name, value in reports[str(seed)]["metrics"].items()},
                    "actual_contact_motion_mm": reports[str(seed)]["actual_motion"]["contact"]["hand_mean_mm"]}),
              flush=True)
    result = {"schema": "ref2dex.executed_handflow_probe.v1",
              "run_status": "COMPLETED", "git_commit": manifest["git_commit"],
              "samples_per_group_per_seed": args.samples_per_group,
              "train_samples": len(train["q"]),
              "train_sha256": manifest["train_sha256"],
              "test_sha256": manifest["test_sha256"],
              "coefficients": {key: value.tolist() for key, value in fitted.items()},
              "reports": reports,
              "limit": "observational action data; shuffled actions are not physical counterfactuals"}
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest.update(run_status="COMPLETED", completed_at=datetime.now(timezone.utc).isoformat(),
                    report_sha256=sha256(report_path))
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-group", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists() or min(args.samples_per_group, args.batch_size) < 1:
        raise ValueError("new output and positive sample counts required")
    args.output.mkdir(parents=True)
    manifest = {"run_status": "STARTED", "run_id": args.output.name,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       cwd=ROOT, text=True).strip(),
                "cpu_threads": 2, "gpu_count": 0, "wall_budget_minutes": 15,
                "output_budget_mb": 2,
                "stop_rule": "nonfinite, input SHA drift, episode alignment failure or wall budget",
                "args": dict(vars(args), output=str(args.output)),
                "train_sha256": {str(seed): item[1] for seed, item in TRAIN.items()},
                "test_sha256": {str(seed): item[1] for seed, item in TRANSITIONS.items()}}
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

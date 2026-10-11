"""Replay the goal-conditioned residual execution contract independently."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch


TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
OLD = ROOT / "src/task/consequence-evaluator"
sys.path[:0] = [str(TASK / "src"), str(OLD / "src"), str(OLD / "tools/run"), str(ROOT)]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def max_error(actual, expected):
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--residual-checkpoint", type=Path, required=True)
    parser.add_argument("--residual-manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists() or ROOT / "outputs/trajectory-policy" not in args.output.parents:
        raise ValueError("fresh task-owned audit output required")

    manifest = json.loads((args.evaluation / "manifest.json").read_text())
    result = json.loads((args.evaluation / "result.json").read_text())
    if (manifest.get("status") != "COMPLETED"
            or manifest.get("schema") != "ref2dex.goal-conditioned-residual-execution.v1"):
        raise ValueError("completed residual execution required")
    for path, digest in manifest.get("input_sha256", {}).items():
        if sha(path) != digest:
            raise ValueError("execution provenance drift: " + path)

    fit_manifest = json.loads(args.residual_manifest.read_text())
    fit_result = json.loads((args.residual_checkpoint.parent / "result.json").read_text())
    if (fit_manifest.get("status") != "COMPLETED"
            or fit_manifest.get("schema") not in (
                "ref2dex.goal-conditioned-residual-bc-run.v1",
                "ref2dex.goal-conditioned-finger-residual-bc-run.v1",
            )
            or fit_result.get("status") != "PROMISING"):
        raise ValueError("promising residual BC provenance required")
    if sha(args.residual_checkpoint) != fit_result.get("checkpoint_sha256"):
        raise ValueError("residual checkpoint hash mismatch")
    for path, digest in fit_manifest.get("input_sha256", {}).items():
        if sha(path) != digest:
            raise ValueError("fit provenance drift: " + path)

    with np.load(args.evaluation / "trajectory.npz", allow_pickle=False) as stream:
        data = {key: stream[key].copy() for key in stream.files}
    required = {
        "q", "dq", "hand", "obj", "velocity", "surface_gap", "support_gap",
        "table_footprint", "features", "previous_residual", "latent",
        "bounded_residual", "nominal_q", "nominal_velocity", "target", "intended",
        "action", "applied", "pd_targets", "clipped",
    }
    if not required.issubset(data):
        raise ValueError("residual trajectory fields are incomplete")
    controls, envs = 542, 16
    if data["q"].shape != (controls + 1, envs, 18) or data["action"].shape != (controls, envs, 18):
        raise ValueError("unexpected trajectory shape")
    for name, value in data.items():
        if not np.isfinite(value).all():
            raise ValueError("nonfinite trajectory field: " + name)

    with args.reference.open("rb") as stream:
        packet = pickle.load(stream)
    goal_env = int(manifest["goal_env"])
    goal_hand = np.asarray(packet["hand_keypoints"][:, goal_env], np.float32)
    goal_obj = np.asarray(packet["object_pose"][:, goal_env], np.float32)
    if goal_hand.shape != (controls + 1, 11, 3) or goal_obj.shape != (controls + 1, 4, 4):
        raise ValueError("goal packet shape mismatch")
    with np.load(args.geometry / "geometry.npz", allow_pickle=False) as stream:
        geometry_q = stream["q"].copy()
    if geometry_q.shape != (controls + 1, 18):
        raise ValueError("geometry q shape mismatch")

    from consequence_evaluator.gate1 import episode_outcome
    from consequence_evaluator.reference_tracking import reference_velocity
    from trajectory_policy.goal_executor import GoalConditionedExecutor
    from trajectory_policy.goal_policy import (
        FINGER_RESIDUAL_INDICES, RESIDUAL_GOAL_HORIZON,
        GoalConditionedFingerResidualPolicy, GoalConditionedResidualPolicy,
        ResidualStandardizer, residual_goal_features,
    )

    index = np.minimum(
        np.arange(controls)[:, None] + 1 + np.arange(RESIDUAL_GOAL_HORIZON)[None, :],
        controls,
    )
    q = torch.from_numpy(data["q"][:-1].reshape(-1, 18))
    dq = torch.from_numpy(data["dq"][:-1].reshape(-1, 18))
    hand = torch.from_numpy(data["hand"][:-1].reshape(-1, 11, 3))
    obj = torch.from_numpy(data["obj"][:-1].reshape(-1, 4, 4))
    velocity = torch.from_numpy(data["velocity"][:-1].reshape(-1, 6))
    previous = torch.from_numpy(data["previous_residual"].reshape(-1, 12))
    goals_hand = torch.from_numpy(np.broadcast_to(
        goal_hand[index][:, None], (controls, envs, RESIDUAL_GOAL_HORIZON, 11, 3)
    ).reshape(-1, RESIDUAL_GOAL_HORIZON, 11, 3))
    goals_obj = torch.from_numpy(np.broadcast_to(
        goal_obj[index][:, None], (controls, envs, RESIDUAL_GOAL_HORIZON, 4, 4)
    ).reshape(-1, RESIDUAL_GOAL_HORIZON, 4, 4))
    with torch.inference_mode():
        expected_features = residual_goal_features(
            q, dq, hand, obj, velocity, previous, goals_hand, goals_obj
        ).numpy().reshape(controls, envs, -1)
    errors = {"features": max_error(data["features"], expected_features)}
    if errors["features"] > 2e-5:
        raise ValueError("feature replay mismatch: " + repr(errors))

    payload = torch.load(args.residual_checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") == GoalConditionedResidualPolicy.schema:
        finger_only = False
        policy_class = GoalConditionedResidualPolicy
    elif payload.get("schema") == GoalConditionedFingerResidualPolicy.schema:
        finger_only = True
        policy_class = GoalConditionedFingerResidualPolicy
    else:
        raise ValueError("residual checkpoint schema mismatch")
    policy = policy_class(width=int(payload["width"]))
    policy.load_state_dict(payload["state_dict"], strict=True)
    policy.eval()
    standardizer = ResidualStandardizer(
        np.asarray(payload["standardizer"]["mean"], np.float32),
        np.asarray(payload["standardizer"]["scale"], np.float32),
    )
    fixed_offset = np.asarray(manifest.get("fixed_residual_offset", np.zeros(12)), np.float32)
    if fixed_offset.shape != (12,) or not np.isfinite(fixed_offset).all():
        raise ValueError("invalid fixed residual offset provenance")
    with torch.inference_mode():
        raw_output = policy(torch.from_numpy(
            standardizer.encode(expected_features.reshape(-1, expected_features.shape[-1]))
        ))
        if finger_only:
            replay_latent = torch.zeros(raw_output.shape[0], 12, dtype=raw_output.dtype)
            replay_latent[:, list(FINGER_RESIDUAL_INDICES)] = raw_output
        else:
            replay_latent = raw_output
        replay_latent = replay_latent.numpy().reshape(controls, envs, 12) + fixed_offset
    errors["latent"] = max_error(data["latent"], replay_latent)
    errors["bounded_residual"] = max_error(data["bounded_residual"], np.tanh(replay_latent))
    if errors["latent"] > 2e-5 or errors["bounded_residual"] > 2e-5:
        raise ValueError("policy replay mismatch: " + repr(errors))

    control = manifest["native_controller"]
    offset = torch.as_tensor(control["offset"], dtype=torch.float32)
    scale = torch.as_tensor(control["scale"], dtype=torch.float32)
    ratio = torch.as_tensor(control["gain_ratio"], dtype=torch.float32)
    nominal_q = torch.from_numpy(np.broadcast_to(
        geometry_q[1:, None], (controls, envs, 18)
    ).copy())
    nominal_velocity = reference_velocity(torch.from_numpy(geometry_q), float(manifest["dt"]))[1:]
    nominal_velocity = nominal_velocity[:, None].expand(controls, envs, 18).contiguous()
    executor = GoalConditionedExecutor(offset, scale, ratio, control["residual_limits"], execute_steps=8)
    flat = executor.step(
        q, nominal_q.reshape(-1, 18), nominal_velocity.reshape(-1, 18),
        torch.from_numpy(replay_latent.reshape(-1, 12)),
    )
    for name, expected in (
            ("nominal_q", nominal_q.numpy()),
            ("nominal_velocity", nominal_velocity.numpy()),
            ("target", flat["target"].numpy().reshape(controls, envs, 18)),
            ("intended", flat["intended"].numpy().reshape(controls, envs, 18)),
            ("action", flat["command"].numpy().reshape(controls, envs, 18)),
            ("clipped", flat["clipped"].numpy().reshape(controls, envs)),):
        if expected.dtype == bool:
            errors[name] = int(np.count_nonzero(data[name] != expected))
        else:
            errors[name] = max_error(data[name], expected)
    errors["applied"] = max_error(data["action"], data["applied"])
    pd = data["action"].copy()
    pd[..., 6:] = (pd[..., 6:] + 1) / 2
    pd = np.asarray(control["offset"], np.float32) + np.asarray(control["scale"], np.float32) * pd
    pd[..., :6] += data["q"][:-1, :, :6]
    for distal, parent, factor in ((7, 6, 1.05), (9, 8, 1.05), (11, 10, 1.05),
                                   (13, 12, 1.05), (16, 15, .6), (17, 15, .8)):
        pd[..., distal] = pd[..., parent] * factor
    errors["pd_targets"] = max_error(data["pd_targets"], pd)
    if max(v if isinstance(v, float) else float(v) for v in errors.values()) > 2e-5:
        raise ValueError("execution replay mismatch: " + repr(errors))

    rows = result.get("outcomes", [])
    if len(rows) != envs:
        raise ValueError("outcome row count mismatch")
    for row in rows:
        env = int(row["env"])
        outcome = episode_outcome(dict(
            object_pose=data["obj"][:, env], object_velocity=data["velocity"][:, env],
            surface_gap=data["surface_gap"][:, env], support_gap=data["support_gap"][:, env],
            table_footprint=data["table_footprint"][:, env],
        ))
        for key, value in outcome.items():
            if isinstance(value, (float, np.floating)):
                if abs(float(value) - float(row[key])) > 1e-7:
                    raise ValueError("outcome mismatch: " + key)
            elif value != row[key]:
                raise ValueError("outcome mismatch: " + key)
    expected_status = ("PROMISING" if result["summary"]["long_held_terminal"] >= 12
                       and result["summary"]["clipping_rate"] < .01 else "UNPROMISING")
    if result.get("status") != expected_status:
        raise ValueError("execution gate mismatch")

    audit = dict(
        schema="ref2dex.goal-conditioned-residual-execution-audit.v1",
        status="PASS", probe_status=result["status"], errors=errors,
        controls=controls, envs=envs,
        privileged_nominal_reference=True,
        policy_mode="finger_only" if finger_only else "full_residual",
        fixed_residual_offset=fixed_offset.tolist(),
        audited=("source hashes, exact goal horizon, live-state features, previous bounded residual, "
                 "policy latent, tanh residual, executor target/intended/command/clipping, PD targets and outcomes"),
        claim=manifest["claim"],
    )
    args.output.mkdir(parents=True)
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    audit_files = [args.evaluation / name for name in (
        "manifest.json", "result.json", "trajectory.npz")]
    audit["input_sha256"] = {
        str(path.resolve()): sha(path) for path in audit_files + [Path(__file__)]
    }
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()

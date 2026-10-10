"""Audit the frozen fixed-offset tau residual-basin diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


BASIN_ROLES = ("baseline", "thumb_pitch_plus", "thumb_pitch_minus", "thumb_yaw_plus")
BASIN_DELTAS = {
    "baseline": (),
    "thumb_pitch_plus": ((11, 0.05),),
    "thumb_pitch_minus": ((11, -0.05),),
    "thumb_yaw_plus": ((10, 0.05),),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def consecutive(values):
    values = np.asarray(values, dtype=bool)
    result = np.zeros(values.shape, dtype=np.int32)
    if len(values):
        result[0] = values[0].astype(np.int32)
        for index in range(1, len(values)):
            result[index] = np.where(values[index], result[index - 1] + 1, 0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[5]
    output = args.output.resolve()
    output.relative_to(root / "outputs/consequence-evaluator")
    if output.exists():
        raise ValueError("fresh task-owned audit output required")

    evaluation = args.evaluation.resolve()
    manifest = json.loads((evaluation / "manifest.json").read_text())
    result = json.loads((evaluation / "result.json").read_text())
    if manifest.get("status") != "COMPLETED" or result.get("status") not in {
            "PROMISING", "UNPROMISING", "UNCLEAR"}:
        raise ValueError("completed residual-basin evaluation required")
    if (not manifest.get("residual_basin") or manifest.get("tau_only") is not True
            or manifest.get("num_envs") != 16 or manifest.get("planned_controls") != 542):
        raise ValueError("unexpected residual-basin protocol")

    controls = int(manifest["planned_controls"])
    envs = int(manifest["num_envs"])
    roles = np.asarray(manifest["roles"])
    if roles.shape != (envs,) or sorted(set(roles)) != sorted(BASIN_ROLES):
        raise ValueError("role assignment drift")
    if any(int((roles == role).sum()) != 4 for role in BASIN_ROLES):
        raise ValueError("residual-basin roles are not four rows each")
    if manifest.get("basin_deltas") != {
            role: [[int(index), float(value)] for index, value in BASIN_DELTAS[role]]
            for role in BASIN_ROLES}:
        raise ValueError("declared residual delta drift")

    input_drift = []
    for path, digest in manifest.get("input_sha256", {}).items():
        candidate = Path(path)
        if not candidate.exists():
            input_drift.append({"path": path, "reason": "missing"})
        elif sha(candidate) != digest:
            input_drift.append({"path": path, "reason": "hash_changed"})
    if input_drift:
        raise ValueError("evaluation provenance drift: " + repr(input_drift[:3]))

    with np.load(evaluation / "trajectory.npz", allow_pickle=False) as stream:
        data = {key: stream[key] for key in stream.files}
    required = {
        "object_pose", "hand_keypoints", "surface_gap", "support_gap", "table_footprint",
        "object_velocity", "action", "clipped", "dof_position", "dof_velocity", "latent",
        "pd_targets", "student_features", "actor_latent", "residual_delta", "intended",
    }
    if not required.issubset(data):
        raise ValueError("residual-basin trajectory fields are incomplete")
    for name, value in data.items():
        if not np.isfinite(value).all():
            raise ValueError("nonfinite trajectory field: " + name)
    if (data["action"].shape != (controls, envs, 18)
            or data["intended"].shape != (controls, envs, 18)
            or data["pd_targets"].shape != (controls, envs, 18)
            or data["latent"].shape != (controls, envs, 12)
            or data["actor_latent"].shape != (controls, envs, 12)
            or data["residual_delta"].shape != (controls, envs, 12)
            or data["student_features"].shape != (controls, envs, 897)
            or data["dof_position"].shape != (controls + 1, envs, 18)):
        raise ValueError("unexpected trajectory shape")

    expected_delta = np.zeros_like(data["residual_delta"])
    for env, role in enumerate(roles):
        for index, value in BASIN_DELTAS[role]:
            expected_delta[:, env, index] = value
    delta_error = float(np.abs(data["residual_delta"] - expected_delta).max())
    latent_error = float(np.abs(data["latent"] -
                                (data["actor_latent"] + data["residual_delta"])).max())
    applied_error = float(np.abs(data["action"] -
                                 np.clip(data["intended"], -1, 1)).max())
    clipped = np.max(np.abs(data["intended"] - data["action"]), axis=-1) > 1e-6
    clipping_flag_error = int(np.count_nonzero(clipped != data["clipped"]))
    if delta_error > 1e-7 or latent_error > 1e-6 or applied_error > 1e-6 or clipping_flag_error:
        raise ValueError("latent/action contract mismatch")

    controller = manifest["native_controller"]
    offset = np.asarray(controller["offset"], np.float32)
    scale = np.asarray(controller["scale"], np.float32)
    native_action = data["action"].copy()
    native_action[..., 6:] = (1 + native_action[..., 6:]) / 2
    target = offset + scale * native_action
    target[..., :6] += data["dof_position"][:-1, :, :6]
    for distal, parent, ratio in ((7, 6, 1.05), (9, 8, 1.05), (11, 10, 1.05),
                                  (13, 12, 1.05), (16, 15, .6), (17, 15, .8)):
        target[..., distal] = target[..., parent] * ratio
    pd_error = float(np.abs(target - data["pd_targets"]).max())
    if pd_error > 1e-6:
        raise ValueError("native PD target/action reconstruction mismatch")

    with args.reference.resolve().open("rb") as stream:
        packet = pickle.load(stream)
    reference_hand = np.asarray(packet["hand_keypoints"][:, 0], np.float32)
    geometry_file = Path(manifest["geometry_reference"]) / "geometry.npz"
    with np.load(geometry_file, allow_pickle=False) as stream:
        geometric_q = stream["q"].copy()
        if not np.array_equal(stream["target_points"], reference_hand):
            raise ValueError("tau geometry/reference mismatch")
    if reference_hand.shape != (controls + 1, 11, 3) or geometric_q.shape != (controls + 1, 18):
        raise ValueError("reference/geometry shape mismatch")
    q = data["dof_position"][:-1]
    dq = data["dof_velocity"][:-1]
    current_obj = data["object_pose"][:-1]
    rotation = current_obj[..., :3, :3]
    center = current_obj[..., :3, 3]
    local_hand = (data["hand_keypoints"][:-1] - center[..., None, :]) @ rotation
    window = np.minimum(np.arange(controls)[:, None, None] + np.arange(1, 25)[None, None], controls)
    window = np.broadcast_to(window, (controls, envs, 24))
    local_future = (reference_hand[window] - center[..., None, None, :]) @ rotation[..., None, :, :]
    indices = np.broadcast_to(np.arange(1, controls + 1)[:, None], (controls, envs))
    q_error = geometric_q[indices] - q
    q_error = q_error.copy()
    q_error[..., 3:6] = np.arctan2(np.sin(q_error[..., 3:6]), np.cos(q_error[..., 3:6]))
    previous = np.concatenate((np.zeros((1, envs, 12), np.float32), np.tanh(data["latent"][:-1])))
    expected_features = np.concatenate((q, dq / 8, local_hand.reshape(controls, envs, 33) / .1,
        local_future.reshape(controls, envs, 792) / .1, q_error, previous,
        data["object_velocity"][:-1] / 2), -1)
    expected_features = np.clip(expected_features, -20, 20)
    feature_error = float(np.abs(expected_features - data["student_features"]).max())
    if feature_error > 2e-5:
        raise ValueError("tau/live-state feature replay mismatch")

    height = data["object_pose"][:, :, 2, 3] - data["object_pose"][0, :, 2, 3]
    supported = data["table_footprint"] & (np.abs(data["support_gap"]) <= .02)
    held = (data["surface_gap"] <= .01) & ~supported & (height >= .03)
    held[0] = False
    held_run = consecutive(held)
    maximum_held = held_run.max(axis=0)
    terminal_held = np.asarray([bool(np.flatnonzero(held_run[:, env] >= 45)[-1] == controls)
                                if np.any(held_run[:, env] >= 45) else False
                                for env in range(envs)])
    rows = result["outcomes"]
    if len(rows) != envs:
        raise ValueError("outcome row count mismatch")
    for row in rows:
        env = int(row["env"])
        if row["role"] != roles[env] or int(row["maximum_held_frames"]) != int(maximum_held[env]) \
                or bool(row["terminal_held"]) != bool(terminal_held[env]):
            raise ValueError("outcome does not match trajectory-derived holding metrics")

    summary = {}
    for role in BASIN_ROLES:
        selected = [row for row in rows if row["role"] == role]
        count = sum(int(row["clipping_count"]) for row in selected)
        summary[role] = {
            "episodes": len(selected),
            "terminal_held_count": int(sum(terminal_held[int(row["env"])] for row in selected)),
            "long_held_terminal_count": int(sum(
                int(row["maximum_held_frames"]) >= 433 and bool(row["terminal_held"])
                for row in selected)),
            "clipping_count": count,
            "clipping_rate": count / float(len(selected) * controls),
            "full_success_count": int(sum(bool(row["success"]) for row in selected)),
        }
    baseline_ok = summary["baseline"]["long_held_terminal_count"] >= 3
    useful = any(summary[role]["long_held_terminal_count"] >= 3
                 and summary[role]["clipping_rate"] < .01
                 for role in BASIN_ROLES if role != "baseline")
    expected_probe_status = "PROMISING" if baseline_ok and useful else ("UNPROMISING" if baseline_ok else "UNCLEAR")
    if result["status"] != expected_probe_status:
        raise ValueError("probe status gate mismatch")

    audit = {
        "schema": "ref2dex.tau-residual-basin-audit.v1",
        "status": "PASS",
        "probe_status": result["status"],
        "evaluation": str(evaluation),
        "controls": controls,
        "envs": envs,
        "input_drift": input_drift,
        "maximum_delta_reconstruction_error": delta_error,
        "maximum_latent_reconstruction_error": latent_error,
        "maximum_applied_command_error": applied_error,
        "clipping_flag_mismatches": clipping_flag_error,
        "maximum_pd_reconstruction_error": pd_error,
        "maximum_feature_replay_error": feature_error,
        "summary": summary,
        "interpretation": ("Contract audit passed. PROMISING is a fixed-offset, single-motion descriptive "
                            "Probe result; it does not establish full placement success, causality, or generalization."),
        "input_sha256": {str(path.resolve()): sha(path) for path in (
            args.reference, evaluation / "manifest.json", evaluation / "result.json",
            evaluation / "trajectory.npz", Path(__file__))},
    }
    output.mkdir(parents=True)
    (output / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: audit[key] for key in (
        "status", "probe_status", "maximum_delta_reconstruction_error",
        "maximum_latent_reconstruction_error", "maximum_applied_command_error",
        "maximum_pd_reconstruction_error", "maximum_feature_replay_error", "summary")}, indent=2))


if __name__ == "__main__":
    main()

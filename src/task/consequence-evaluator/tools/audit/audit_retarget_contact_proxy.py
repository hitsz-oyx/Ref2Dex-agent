"""Audit contact/preload proxies in an existing same-CPU retarget packet.

This is a CPU-only attribution audit.  Older packets have only pair, surface
gap, support gap, object velocity, and q/dq proxies; newer packets may also
carry native hand/object force vectors.  Even saved net-force vectors are
diagnostics, not force recovery, impulses, or a preload mechanism.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / "src"))
from consequence_evaluator.data import sha


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rms(value):
    return float(np.sqrt(np.mean(np.square(value))))


def first_near_run(actual_surface_gap, threshold=.01):
    """Return the first sampled near-gap run and its first far tick.

    The execution packet has no R-side pair/force label.  This is consequently
    a geometry-only event, not a recovered contact-loss event.
    """
    actual_near = np.asarray(actual_surface_gap) <= threshold
    onset = np.flatnonzero(actual_near)
    if not len(onset):
        return None
    start = int(onset[0])
    end = start
    while end + 1 < len(actual_near) and actual_near[end + 1]:
        end += 1
    return dict(
        near_onset_tick=start,
        near_last_tick=end,
        first_far_tick_after_near=end + 1 if end + 1 < len(actual_near) else None,
    )


def state_metrics(actual, source, source_hand, env, event, actual_force_capture,
                  source_force_capture):
    hand_error_broadcast = actual["hand_keypoints"][:, env] - source_hand
    hand_error_same_env = actual["hand_keypoints"][:, env] - source["hand_keypoints"][:, env]
    out = dict(
        role_env=int(env),
        hand_rms_broadcast_m=rms(hand_error_broadcast),
        hand_rms_same_env_teacher_m=rms(hand_error_same_env),
        wrist_rms_broadcast_m=rms(hand_error_broadcast[:, 0]),
        finger_rms_broadcast_m=rms(hand_error_broadcast[:, 1:]),
        tip_rms_broadcast_m=rms(hand_error_broadcast[:, [2, 4, 6, 8, 10]]),
    )
    if event is None:
        out["geometry_near_run"] = None
        return out
    tick = event["first_far_tick_after_near"]
    out["geometry_near_onset_tick"] = event["near_onset_tick"]
    out["geometry_near_last_tick"] = event["near_last_tick"]
    out["first_far_tick_after_geometry_near"] = tick
    if tick is None:
        return out
    tick = int(tick)
    near_onset = int(event["near_onset_tick"])
    current = dict(
        source_pair_at_geometry_near_onset=bool(source["pair"][near_onset, env]),
        source_pair_at_geometry_near_exit=bool(source["pair"][tick, env]),
        surface_gap_m=float(actual["surface_gap"][tick, env]),
        support_gap_m=float(actual["support_gap"][tick, env]),
        object_linear_speed_m_s=float(np.linalg.norm(actual["object_velocity"][tick, env, :3])),
        object_angular_speed_rad_s=float(np.linalg.norm(actual["object_velocity"][tick, env, 3:])),
        hand_error_broadcast_at_geometry_near_exit_m=rms(hand_error_broadcast[tick]),
        hand_error_same_env_at_geometry_near_exit_m=rms(hand_error_same_env[tick]),
        wrist_error_broadcast_at_geometry_near_exit_m=rms(hand_error_broadcast[tick, 0]),
        finger_error_broadcast_at_geometry_near_exit_m=rms(hand_error_broadcast[tick, 1:]),
        tip_error_broadcast_at_geometry_near_exit_m=rms(hand_error_broadcast[tick, [2, 4, 6, 8, 10]]),
        q_error_same_env_teacher_at_geometry_near_exit_rad=rms(
            actual["dof_position"][tick, env, 3:] - source["dof_position"][tick, env, 3:]),
        wrist_q_error_same_env_teacher_at_geometry_near_exit_m=rms(
            actual["dof_position"][tick, env, :3] - source["dof_position"][tick, env, :3]),
        dq_error_same_env_teacher_at_geometry_near_exit=rms(
            actual["dof_velocity"][tick, env] - source["dof_velocity"][tick, env]),
    )
    if actual_force_capture:
        current.update(
            actual_pair_at_geometry_near_onset=bool(actual["pair"][near_onset, env]),
            actual_pair_at_geometry_near_exit=bool(actual["pair"][tick, env]),
            actual_pair_true_frames=int(actual["pair"][:, env].sum()),
            actual_hand_force_sum_n=float(np.linalg.norm(
                actual["native_contact_forces"][tick, env], axis=-1).sum()),
            actual_hand_force_max_n=float(np.linalg.norm(
                actual["native_contact_forces"][tick, env], axis=-1).max()),
            actual_object_force_norm_n=float(np.linalg.norm(
                actual["native_object_contact_forces"][tick, env])),
        )
    if source_force_capture:
        current.update(
            source_hand_force_sum_n=float(np.linalg.norm(
                source["native_contact_forces"][tick, env], axis=-1).sum()),
            source_hand_force_max_n=float(np.linalg.norm(
                source["native_contact_forces"][tick, env], axis=-1).max()),
            source_object_force_norm_n=float(np.linalg.norm(
                source["native_object_contact_forces"][tick, env])),
        )
    out.update(current)
    if tick < len(actual["requested_actions"]):
        out["requested_action_error_current_frame_same_env_teacher"] = rms(
            actual["requested_actions"][tick, env] - source["action"][tick, env])
    if tick > 0:
        out["requested_action_error_preceding_frame_same_env_teacher"] = rms(
            actual["requested_actions"][tick - 1, env] - source["action"][tick - 1, env])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.parent.resolve() != (ROOT / "outputs/consequence-evaluator").resolve():
        raise ValueError("fresh task-owned contact audit output required")
    source_path = args.source.resolve()
    execution_path = args.execution.resolve()
    import pickle
    with np.load(source_path, allow_pickle=False) as packed:
        source = {key: packed[key] for key in packed.files}
    packet = pickle.load(execution_path.open("rb"))
    actual = packet["arrays"]
    required_source = {"hand_keypoints", "pair", "surface_gap", "support_gap",
                       "table_footprint", "object_velocity", "dof_position", "dof_velocity",
                       "done", "clipped", "action", "length"}
    required_actual = (required_source - {"pair", "action", "length"}) | {"requested_actions",
                                                     "actions", "clipped", "table_footprint", "lengths"}
    if not required_source.issubset(source) or not required_actual.issubset(actual):
        raise ValueError("contact audit packet missing required arrays")
    state_shapes = {
        "hand_keypoints": (543, 4, 11, 3),
        "surface_gap": (543, 4), "support_gap": (543, 4),
        "object_velocity": (543, 4, 6), "dof_position": (543, 4, 18),
        "dof_velocity": (543, 4, 18), "table_footprint": (543, 4),
    }
    for key, shape in state_shapes.items():
        if source[key].shape != shape or actual[key].shape != shape:
            raise ValueError(f"unexpected {key} shape")
        if not np.isfinite(source[key]).all() or not np.isfinite(actual[key]).all():
            raise ValueError(f"nonfinite {key}")
    if source["pair"].shape != (543, 4) or source["pair"].dtype != np.bool_:
        raise ValueError("unexpected source pair shape or dtype")
    action_shapes = {"action": (542, 4, 18), "done": (542, 4), "clipped": (542, 4)}
    for key, shape in action_shapes.items():
        actual_key = "actions" if key == "action" else key
        if source[key].shape != shape or actual[actual_key].shape != shape:
            raise ValueError(f"unexpected source/execution {key} shape")
        if key != "done" and key != "clipped":
            if not np.isfinite(source[key]).all() or not np.isfinite(actual[actual_key]).all():
                raise ValueError(f"nonfinite {key}")
    if not np.array_equal(source["length"], np.full(4, 542)) or not np.array_equal(
            actual["lengths"], np.full(4, 542)):
        raise ValueError("unexpected episode lengths")
    if actual["requested_actions"].shape != (542, 4, 18) or not np.isfinite(
            actual["requested_actions"]).all():
        raise ValueError("unexpected requested action shape or values")
    if not np.array_equal(actual["actions"], actual["requested_actions"]):
        raise ValueError("requested and applied actions differ")
    if not np.array_equal(actual["clipped"], np.zeros_like(actual["clipped"])):
        raise ValueError("contact proxy audit requires zero-clipping execution")
    source_hand = np.asarray(packet["source_hand"], dtype="float32")
    if source_hand.shape != (543, 11, 3):
        raise ValueError("unexpected broadcast source hand shape")

    def validate_force_capture(arrays, label):
        keys = {"native_contact_forces", "native_object_contact_forces", "pair"}
        present = keys.intersection(arrays)
        if present and present != keys:
            raise ValueError(f"partial {label} force capture")
        if not present:
            return False
        hand_force = arrays["native_contact_forces"]
        object_force = arrays["native_object_contact_forces"]
        if (hand_force.shape[:2] != (543, 4) or hand_force.shape[-1] != 3
                or object_force.shape != (543, 4, 3)
                or arrays["pair"].shape != (543, 4)
                or arrays["pair"].dtype != np.bool_):
            raise ValueError(f"invalid {label} force capture shape")
        if not np.isfinite(hand_force).all() or not np.isfinite(object_force).all():
            raise ValueError(f"nonfinite {label} force capture")
        return True

    actual_force_capture = validate_force_capture(actual, "execution")
    source_force_capture = validate_force_capture(source, "source")
    roles = packet["manifest"].get("roles", ["teacher", "R1", "R2", "R3"])
    reports = []
    for env in range(1, 4):
        event = first_near_run(actual["surface_gap"][:, env])
        report = state_metrics(actual, source, source_hand, env, event,
                               actual_force_capture, source_force_capture)
        report["role"] = roles[env] if env < len(roles) else "retarget_env" + str(env)
        report["source_pair_true_frames"] = int(source["pair"][:, env].sum())
        report["surface_gap_near_frames_le_10mm"] = int((actual["surface_gap"][:, env] <= .01).sum())
        report["table_support_proxy_frames"] = int(
            (actual["table_footprint"][:, env] & (np.abs(actual["support_gap"][:, env]) <= .02)).sum())
        reports.append(report)

    source_packet_path = None
    source_backend = "unknown_source_packet"
    source_packet_consistency = None
    source_ref = packet["manifest"].get("source")
    if not isinstance(source_ref, str) or not Path(source_ref).exists():
        raise ValueError("execution manifest source packet is missing")
    if isinstance(source_ref, str) and Path(source_ref).exists():
        source_packet_path = Path(source_ref).resolve()
        with source_packet_path.open("rb") as handle:
            source_packet = pickle.load(handle)
        source_backend = source_packet.get("source_backend", "unknown_source_packet")
        packet_pairs = {
            "hand_keypoints": "hand_keypoints", "surface_gap": "surface_gap",
            "support_gap": "support_gap", "object_velocity": "object_velocity",
            "dof_position": "dof_position", "dof_velocity": "dof_velocity",
        }
        if source_force_capture:
            packet_pairs.update({"native_contact_forces": "native_contact_forces",
                                 "native_object_contact_forces": "native_object_contact_forces",
                                 "pair": "pair"})
        source_packet_consistency = {}
        for trajectory_key, packet_key in packet_pairs.items():
            source_packet_consistency[trajectory_key] = float(np.max(np.abs(
                source[trajectory_key] - np.asarray(source_packet[packet_key]))))
        source_packet_consistency["action"] = float(np.max(np.abs(
            source["action"] - np.asarray(source_packet["actions"]))))
        source_packet_consistency["broadcast_source_hand"] = float(np.max(np.abs(
            source_hand - source["hand_keypoints"][:, 0])))
        if any(value != 0.0 for value in source_packet_consistency.values()):
            raise ValueError("source trajectory does not match source packet")
    hashes = {str(source_path): sha(source_path), str(execution_path): sha(execution_path),
              str(Path(__file__).resolve()): sha(Path(__file__).resolve())}
    if source_packet_path is not None:
        hashes[str(source_packet_path)] = sha(source_packet_path)
    result = dict(
        status="COMPLETED",
        schema="ref2dex.retarget-contact-proxy-audit.v1",
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        input_sha256=hashes,
        execution_run_id=packet["manifest"].get("run_id"),
        source_backend=source_backend,
        roles=reports,
        source_hand_contract="broadcast env0 teacher hand; same-env teacher metrics are separate",
        source_packet_consistency_max_abs=source_packet_consistency,
        force_capture=dict(execution=actual_force_capture, source=source_force_capture,
                           semantics="native net-force vectors; no impulse/preload recovery"),
        contact_proxy_fields=["source_pair", "surface_gap", "table_support_proxy",
                              "object_velocity"] + (["native_contact_forces",
                              "native_object_contact_forces"]
                              if actual_force_capture else []),
        state_diagnostic_fields=["hand_error", "q_error", "dq_error", "action_error"],
        near_gap_definition=("first contiguous sampled surface-gap run <=10mm; no hysteresis "
                             "or minimum-duration claim"),
        state_diagnostic_units=("q wrist xyz=m, q fingers=rad; dq wrist=m/s, dq fingers=rad/s; "
                                "reported RMS mixes only within each named field"),
        action_frame_contract=("state[t] is after command[t-1] and before command[t]; preceding "
                               "action is transition-associated and current action is the state-t "
                               "policy output, but neither proves a contact mechanism"),
        limitation=(("Native hand/object net-force vectors are persisted, but they remain "
                     "diagnostics and do not recover impulse, preload, or a contact mechanism.")
                    if actual_force_capture else
                    ("No per-body contact force/impulse was persisted; these measurements cannot "
                     "recover preload or prove a contact-dynamics mechanism.")))
    output.mkdir(parents=True)
    write(output / "result.json", result)
    write(output / "manifest.json", result)
    print(json.dumps({"roles": reports, "limitation": result["limitation"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()

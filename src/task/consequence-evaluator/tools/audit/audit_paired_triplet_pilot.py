"""Audit deterministic control/+d/-d triplet prefixes from one rollout.

Triplets are an engineering data-collection contract.  They do not clone the
hidden PhysX solver state and their post-pulse trajectories are not causal
counterfactual evidence.
"""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
import sys
sys.path.insert(0, str(TASK / "src"))

from consequence_evaluator.data import sha


SCHEMA = "ref2dex.paired-triplet-pilot.v1"
FRAMES = 543
COMMANDS = 542
ACTIVE_FINGERS = np.asarray([6, 8, 10, 12, 14, 15], dtype=np.int64)
STATE_FIELDS = (
    "object_pose", "hand_keypoints", "surface_gap", "support_gap",
    "table_footprint", "object_velocity", "reference_object_pose", "pair",
    "native_contact_forces", "native_object_contact_forces", "dof_position",
    "dof_velocity", "actor_observation",
)
COMMAND_FIELDS = (
    "action", "actor_action", "structured_residual", "structured_mode",
    "native_pd_target", "clipped", "done",
)
GLOBAL_FIELDS = ("structured_phase",)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _load(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("status") != "COMPLETED" or manifest.get("mode") != "retarget"
            or manifest.get("structured_profile") != "paired-triplet"
            or manifest.get("structured_residual_schema") != "ref2dex.structured-residual.v1"):
        raise ValueError(f"completed paired-triplet rollout required: {root}")
    schedule = manifest.get("paired_schedule")
    if not isinstance(schedule, dict) or schedule.get("branch_order") != ["control", "plus", "minus"]:
        raise ValueError(f"paired schedule missing or malformed: {root}")
    if not manifest.get("actor_observation_saved"):
        raise ValueError(f"actor observations are required: {root}")
    with np.load(root / "trajectory.npz", allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    required = set(STATE_FIELDS) | set(COMMAND_FIELDS) | set(GLOBAL_FIELDS) | {
        "length", "paired_group", "paired_branch"
    }
    missing = required.difference(data)
    if missing:
        raise ValueError(f"paired fields missing: {sorted(missing)}")
    groups = int(schedule.get("groups", 0))
    envs = groups * 3
    if (groups < 1 or data["object_pose"].shape[:2] != (FRAMES, envs)
            or data["action"].shape != (COMMANDS, envs, 18)
            or data["native_pd_target"].shape != (COMMANDS, envs, 6)
            or data["length"].shape != (envs,)
            or not np.all(data["length"] == COMMANDS)
            or data["structured_phase"].shape != (COMMANDS,)
            or data["paired_group"].shape != (envs,)
            or data["paired_branch"].shape != (envs,)
            or data["paired_group"].tolist() != np.repeat(np.arange(groups), 3).tolist()
            or data["paired_branch"].tolist() != np.tile([0, 1, 2], groups).tolist()):
        raise ValueError(f"paired shape/length mismatch: {root}")
    if any(data[name].shape[:2] != (FRAMES, envs) for name in STATE_FIELDS):
        raise ValueError(f"paired state shape mismatch: {root}")
    if any(data[name].shape[:2] != (COMMANDS, envs) for name in COMMAND_FIELDS):
        raise ValueError(f"paired command shape mismatch: {root}")
    if not isinstance(manifest.get("input_sha256"), dict) or not manifest.get("actor_sha256"):
        raise ValueError(f"paired provenance hashes missing: {root}")
    for name in STATE_FIELDS + COMMAND_FIELDS:
        if data[name].dtype.kind in "fc" and not np.isfinite(data[name]).all():
            raise ValueError(f"nonfinite paired field: {root}/{name}")
    expected = np.clip(data["actor_action"] + data["structured_residual"], -1., 1.)
    if not np.allclose(expected, data["action"], atol=1e-7, rtol=0):
        raise ValueError(f"action composition mismatch: {root}")
    if int(data["clipped"].sum()) != 0:
        raise ValueError(f"paired rollout clipped a command: {root}")
    return data, manifest, {
        str(root / "manifest.json"): sha(root / "manifest.json"),
        str(root / "trajectory.npz"): sha(root / "trajectory.npz"),
    }


def _phase_contract(data, manifest):
    """Validate the time-global phase label separately from env-indexed fields."""
    phase = np.asarray(data["structured_phase"])
    boundaries = manifest.get("phase_boundaries", {})
    approach_end = int(boundaries.get("approach_end_tick", 120))
    contact_end = int(boundaries.get("contact_end_tick", 240))
    if not 0 < approach_end < contact_end < COMMANDS:
        raise ValueError("invalid phase boundaries")
    expected = np.full(COMMANDS, 2, dtype=np.int8)
    expected[:approach_end] = 0
    expected[approach_end:contact_end] = 1
    return dict(
        shape=list(phase.shape),
        finite=bool(np.isfinite(phase).all()),
        boundaries={"approach_end_tick": approach_end,
                    "contact_end_tick": contact_end},
        exact=bool(np.array_equal(phase, expected)),
        values=np.unique(phase).astype(int).tolist(),
    )


def _rms(value):
    value = np.asarray(value, dtype="float64")
    return float(np.sqrt(np.mean(np.square(value))))


def _prefix_equal(data, indices, tick):
    control, plus, minus = indices
    state = {}
    for name in STATE_FIELDS:
        source = data[name][:tick + 1]
        state[name] = bool(np.array_equal(source[:, control], source[:, plus])
                           and np.array_equal(source[:, control], source[:, minus]))
    command = {}
    for name in COMMAND_FIELDS:
        source = data[name][:tick]
        command[name] = bool(np.array_equal(source[:, control], source[:, plus])
                             and np.array_equal(source[:, control], source[:, minus]))
    return state, command


def _triplet_record(data, group, tick, finger_index, pulse_value):
    indices = (3 * group, 3 * group + 1, 3 * group + 2)
    control, plus, minus = indices
    state_prefix, command_prefix = _prefix_equal(data, indices, tick)
    residual = data["structured_residual"][tick]
    action = data["action"][tick]
    actor = data["actor_action"][tick]
    control_value = float(residual[control, finger_index])
    expected_values = np.asarray(
        [control_value, control_value + pulse_value, control_value - pulse_value],
        dtype="float32")
    expected_delta = np.zeros((3, 18), dtype="float32")
    expected_delta[1, finger_index] = pulse_value
    expected_delta[2, finger_index] = -pulse_value
    action_delta = action[list(indices)] - action[control]
    residual_delta = residual[list(indices)] - residual[control]
    actor_equal = bool(np.array_equal(actor[control], actor[plus])
                       and np.array_equal(actor[control], actor[minus]))
    residual_exact = bool(np.array_equal(residual[list(indices), finger_index], expected_values)
                          and np.allclose(residual_delta, expected_delta, atol=1e-7, rtol=0))
    action_delta_exact = bool(np.allclose(action_delta, expected_delta, atol=1e-7, rtol=0))
    command_exact = bool(residual_exact and action_delta_exact and actor_equal)
    active_delta = float(abs(action[plus, finger_index] - action[minus, finger_index]))
    pair = data["pair"][tick, list(indices)]
    hand_delta = _rms(data["hand_keypoints"][tick, plus] - data["hand_keypoints"][tick, control])
    hand_delta_minus = _rms(data["hand_keypoints"][tick, minus] - data["hand_keypoints"][tick, control])
    q_delta = _rms(data["dof_position"][tick, plus] - data["dof_position"][tick, control])
    q_delta_minus = _rms(data["dof_position"][tick, minus] - data["dof_position"][tick, control])
    dq_delta = _rms(data["dof_velocity"][tick, plus] - data["dof_velocity"][tick, control])
    dq_delta_minus = _rms(data["dof_velocity"][tick, minus] - data["dof_velocity"][tick, control])
    visible_match = bool(max(hand_delta, hand_delta_minus) <= .003
                         and max(q_delta, q_delta_minus) <= .03
                         and max(dq_delta, dq_delta_minus) <= .10)
    return dict(
        group=int(group), tick=int(tick), indices=list(indices),
        prefix_bitwise=bool(all(state_prefix.values()) and all(command_prefix.values())),
        state_prefix_bitwise=state_prefix, command_prefix_bitwise=command_prefix,
        command_exact=command_exact, residual_exact=residual_exact,
        action_delta_exact=action_delta_exact, actor_equal_at_pulse=actor_equal,
        pair=list(bool(x) for x in pair), same_pair=bool(np.all(pair == pair[0])),
        control_pair=bool(pair[0]), hand_rms_m=hand_delta, hand_rms_m_minus=hand_delta_minus,
        q_rms=q_delta, q_rms_minus=q_delta_minus, dq_rms=dq_delta, dq_rms_minus=dq_delta_minus,
        visible_match=visible_match,
        eligible=(bool(all(state_prefix.values()) and all(command_prefix.values()))
                  and visible_match and bool(np.all(pair == pair[0]))),
        active_coordinate_delta=active_delta,
        nontrivial=(bool(all(state_prefix.values()) and all(command_prefix.values()))
                    and visible_match and bool(np.all(pair == pair[0]))
                    and active_delta > .05),
        pd_delta_plus_max=float(np.max(np.abs(data["native_pd_target"][tick, plus]
                                               - data["native_pd_target"][tick, control]))),
        pd_delta_minus_max=float(np.max(np.abs(data["native_pd_target"][tick, minus]
                                                - data["native_pd_target"][tick, control]))),
    )


def audit(root):
    data, manifest, hashes = _load(root)
    phase_contract = _phase_contract(data, manifest)
    schedule = manifest["paired_schedule"]
    phase = schedule["phase"]
    ticks = [int(x) for x in schedule["pulse_ticks"]]
    groups = int(schedule["groups"])
    finger_index = int(schedule["finger_index"])
    pulse_value = float(schedule["pulse_value"])
    if finger_index not in set(int(index) for index in ACTIVE_FINGERS):
        raise ValueError("paired schedule finger index is not independently commanded")
    if not np.isfinite(pulse_value) or abs(pulse_value) > .120001:
        raise ValueError("paired schedule pulse exceeds the registered bound")
    if len(ticks) != groups or len(set(ticks)) != groups:
        raise ValueError("paired pulse schedule must have one unique tick per group")
    if phase not in {"contact", "hold"}:
        raise ValueError(f"unknown paired pulse phase: {phase}")
    lower, upper = (120, 239) if phase == "contact" else (240, COMMANDS - 1)
    if any(tick < lower or tick > upper for tick in ticks):
        raise ValueError(f"paired pulse ticks fall outside {phase} phase")
    records = [_triplet_record(data, group, ticks[group], finger_index, pulse_value)
               for group in range(groups)]
    engineering = dict(
        finite=True,
        zero_clipping=int(data["clipped"].sum()) == 0,
        phase_contract=phase_contract["exact"],
        all_prefix_bitwise=all(record["prefix_bitwise"] for record in records),
        all_command_exact=all(record["command_exact"] for record in records),
        all_residual_exact=all(record["residual_exact"] for record in records),
        all_action_delta_exact=all(record["action_delta_exact"] for record in records),
        all_actor_equal_at_pulse=all(record["actor_equal_at_pulse"] for record in records),
        all_same_pair=all(record["same_pair"] for record in records),
    )
    matched = [record for record in records if record["control_pair"] and record["eligible"]]
    nontrivial = [record for record in matched if record["nontrivial"]]
    screen = dict(phase=phase, matched_rows=len(matched), nontrivial_rows=len(nontrivial),
                  minimum_matched_rows=20, minimum_nontrivial_rows=10,
                  screen_gate=(len(matched) >= 20 and len(nontrivial) >= 10))
    status = ("PROMISING" if all(engineering.values()) and screen["screen_gate"]
              else "UNCLEAR" if all(engineering.values()) else "INVALID_IMPLEMENTATION")
    return dict(
        schema=SCHEMA, status=status, seed=int(manifest["seed"]),
        phase=phase, groups=groups, finger_index=finger_index,
        pulse_value=pulse_value, engineering=engineering, screen=screen,
        phase_contract=phase_contract,
        records=records, matched_rows=matched,
        pair_rate=float(data["pair"][:-1].mean()),
        input_sha256=manifest.get("input_sha256"), source_hashes=hashes,
        scope=("triplet prefix/command contract and visible matched-state coverage only; "
               "post-pulse state and force changes are not causal evidence"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not str(output).startswith(str(ROOT / "outputs/consequence-evaluator")):
        raise ValueError("fresh task-owned output required")
    result = audit(args.input)
    output.mkdir(parents=True)
    manifest = dict(schema=SCHEMA, status="COMPLETED",
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       cwd=ROOT, text=True).strip(),
                    source=str(args.input.resolve()), result_sha256=None,
                    scope=result["scope"])
    write(output / "result.json", result)
    manifest["result_sha256"] = sha(output / "result.json")
    write(output / "manifest.json", manifest)
    print(json.dumps({key: result[key] for key in
                      ("status", "phase", "engineering", "screen", "pair_rate")},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()

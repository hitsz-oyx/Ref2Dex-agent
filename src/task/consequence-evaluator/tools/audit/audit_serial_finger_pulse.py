"""Audit a serial zero/pulse contact-stage capture contract.

The comparison is deliberately limited to a deterministic prefix and to
command/state bookkeeping.  It is not a PhysX fork, counterfactual causal
estimate, decoder fit, or native success evaluation.
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


SCHEMA = "ref2dex.serial-finger-pulse-audit.v1"
FRAMES = 543
COMMANDS = 542
STATE_FIELDS = (
    "object_pose", "hand_keypoints", "surface_gap", "support_gap",
    "table_footprint", "object_velocity", "reference_object_pose", "pair",
    "native_contact_forces", "native_object_contact_forces", "dof_position",
    "dof_velocity", "done", "actor_observation",
)
COMMAND_FIELDS = (
    "action", "actor_action", "structured_residual", "structured_mode",
    "structured_phase", "native_pd_target", "clipped",
)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _load(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("status") != "COMPLETED" or manifest.get("mode") != "retarget"
            or manifest.get("structured_residual_schema") != "ref2dex.structured-residual.v1"):
        raise ValueError(f"completed structured retarget rollout required: {root}")
    if not manifest.get("actor_observation_saved"):
        raise ValueError(f"actor observations are required for prefix audit: {root}")
    with np.load(root / "trajectory.npz", allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    required = set(STATE_FIELDS) | set(COMMAND_FIELDS) | {"length"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"serial audit fields missing: {sorted(missing)}")
    if (data["object_pose"].shape[:2] != (FRAMES, 1)
            or data["action"].shape != (COMMANDS, 1, 18)
            or data["native_pd_target"].shape[0:2] != (COMMANDS, 1)
            or data["native_pd_target"].shape[-1] != 6
            or data["length"].tolist() != [COMMANDS]):
        raise ValueError(f"serial audit shape/length mismatch: {root}")
    for name in STATE_FIELDS + COMMAND_FIELDS:
        if data[name].dtype.kind in "fc" and not np.isfinite(data[name]).all():
            raise ValueError(f"nonfinite serial audit field: {root}/{name}")
    expected = np.clip(data["actor_action"] + data["structured_residual"], -1., 1.)
    if not np.allclose(expected, data["action"], atol=1e-7, rtol=0):
        raise ValueError(f"captured action composition mismatch: {root}")
    if int(data["clipped"].sum()) != 0:
        raise ValueError(f"clipping is not allowed in serial pulse audit: {root}")
    if data["structured_phase"].tolist() != [0] * 120 + [1] * 120 + [2] * 302:
        raise ValueError(f"structured phase boundaries mismatch: {root}")
    return data, manifest, {
        str(root / "manifest.json"): sha(root / "manifest.json"),
        str(root / "trajectory.npz"): sha(root / "trajectory.npz"),
    }


def _max_abs(left, right):
    if left.dtype.kind in "b" or right.dtype.kind in "b":
        return float(np.any(left != right))
    return float(np.max(np.abs(left.astype("float64") - right.astype("float64"))))


def _pair_rates(data):
    pair = data["pair"][:-1, 0]
    return {
        "onset_40_80": float(pair[40:80].mean()),
        "contact_120_240": float(pair[120:240].mean()),
        "hold_240_542": float(pair[240:542].mean()),
    }


def compare(control, perturbed):
    control_data, control_manifest, control_hashes = _load(control)
    perturbed_data, perturbed_manifest, perturbed_hashes = _load(perturbed)
    if control_manifest.get("structured_profile") != "zero":
        raise ValueError("control rollout must use structured-profile=zero")
    if (control_manifest.get("seed") != perturbed_manifest.get("seed")
            or control_manifest.get("num_envs") != 1
            or perturbed_manifest.get("num_envs") != 1
            or control_manifest.get("actor_sha256") != perturbed_manifest.get("actor_sha256")
            or control_manifest.get("input_sha256") != perturbed_manifest.get("input_sha256")):
        raise ValueError("paired rollouts do not share seed, actor, or frozen inputs")
    profile = perturbed_manifest.get("structured_profile")
    if profile not in ("zero", "finger-pulse"):
        raise ValueError("perturbed rollout must use zero or finger-pulse profile")

    if profile == "finger-pulse":
        pulse = perturbed_manifest.get("structured_pulse")
        if not isinstance(pulse, dict):
            raise ValueError("finger-pulse manifest is missing its pulse specification")
        start = int(pulse["start_tick"]); end = int(pulse["end_tick"])
        index = int(pulse["finger_index"]); value = float(pulse["value"])
        if not 0 <= start < end <= COMMANDS:
            raise ValueError("pulse interval is outside the command sequence")
        if not np.isfinite(value) or abs(value) > .120001:
            raise ValueError("pulse value is outside the registered bound")
    else:
        pulse = None
        start = COMMANDS
        end = COMMANDS
        index = None
        value = 0.

    state_prefix = {name: np.array_equal(control_data[name][:start + 1],
                                         perturbed_data[name][:start + 1])
                    for name in STATE_FIELDS}
    command_prefix = {name: np.array_equal(control_data[name][:start],
                                            perturbed_data[name][:start])
                      for name in COMMAND_FIELDS}
    prefix_bitwise = all(state_prefix.values()) and all(command_prefix.values())

    control_residual = control_data["structured_residual"]
    perturbed_residual = perturbed_data["structured_residual"]
    if profile == "zero":
        pulse_exact = bool(np.array_equal(control_data["action"], perturbed_data["action"])
                           and np.array_equal(control_residual, perturbed_residual))
        pulse_delta = dict(max_abs=0., active_max_abs=0., active_min_abs=0.,
                           native_pd_max_abs=0.)
    else:
        expected = np.zeros_like(perturbed_residual[start:end])
        expected[:, :, index] = value
        residual_exact = np.array_equal(perturbed_residual[start:end], expected)
        before_actor_equal = np.array_equal(control_data["actor_action"][start:end],
                                            perturbed_data["actor_action"][start:end])
        action_delta = perturbed_data["action"][start:end] - control_data["action"][start:end]
        pd_delta = perturbed_data["native_pd_target"][start:end] - control_data["native_pd_target"][start:end]
        pulse_exact = bool(residual_exact and before_actor_equal
                           and np.allclose(action_delta, expected, atol=1e-7, rtol=0))
        active_delta = np.abs(action_delta[:, :, index])
        pulse_delta = dict(
            max_abs=float(np.max(np.abs(action_delta))),
            active_max_abs=float(np.max(active_delta)),
            active_min_abs=float(np.min(active_delta)),
            native_pd_max_abs=float(np.max(np.abs(pd_delta))),
            action_rms=float(np.sqrt(np.mean(np.square(action_delta)))),
            residual_exact=bool(residual_exact),
            actor_equal_at_pulse=bool(before_actor_equal),
        )

    post_fields = ("hand_keypoints", "object_pose", "dof_position", "dof_velocity",
                   "surface_gap", "native_contact_forces", "native_object_contact_forces")
    post_end = min(end + 5, FRAMES)
    post_response = ({name: _max_abs(control_data[name][start + 1:post_end],
                                     perturbed_data[name][start + 1:post_end])
                      for name in post_fields}
                     if start + 1 < post_end else {})
    gates = dict(
        finite=True,
        zero_clipping=(int(control_data["clipped"].sum()) == 0
                       and int(perturbed_data["clipped"].sum()) == 0),
        prefix_bitwise=bool(prefix_bitwise),
        command_profile_exact=bool(pulse_exact),
        native_pd_target_present=True,
    )
    status = "PROMISING" if all(gates.values()) else "INVALID_IMPLEMENTATION"
    return dict(
        schema=SCHEMA,
        status=status,
        control_profile=control_manifest.get("structured_profile"),
        perturbed_profile=profile,
        seed=int(control_manifest["seed"]),
        pulse=pulse,
        gates=gates,
        state_prefix_bitwise=state_prefix,
        command_prefix_bitwise=command_prefix,
        pulse_delta=pulse_delta,
        post_response_max_abs=post_response,
        pair_rates=dict(control=_pair_rates(control_data), perturbed=_pair_rates(perturbed_data)),
        source_hashes=dict(control=control_hashes, perturbed=perturbed_hashes),
        scope=("serial deterministic prefix and requested/applied command/PD-target audit; "
               "post-pulse state and force changes are diagnostics, not counterfactual or causal evidence"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--perturbed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not str(output).startswith(str(ROOT / "outputs/consequence-evaluator")):
        raise ValueError("fresh task-owned output required")
    result = compare(args.control, args.perturbed)
    output.mkdir(parents=True)
    manifest = dict(
        schema=SCHEMA, status="COMPLETED",
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        control=str(args.control.resolve()), perturbed=str(args.perturbed.resolve()),
        result_sha256=None, scope=result["scope"],
    )
    write(output / "result.json", result)
    manifest["result_sha256"] = sha(output / "result.json")
    write(output / "manifest.json", manifest)
    print(json.dumps({key: result[key] for key in
                      ("status", "gates", "pulse_delta", "post_response_max_abs", "pair_rates")},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()

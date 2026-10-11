"""Audit a serial exact-prefix candidate bank and frozen U32 labels.

The collector runs one seed and one environment per candidate.  This audit
checks the deterministic prefix, the bounded residual schedule, and the
post-pulse trajectory/label screen.  It is an engineering/data Probe only;
serial replay is not a hidden-PhysX fork or a causal counterfactual.
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
from consequence_evaluator.old_utility import teacher


SCHEMA = "ref2dex.serial-exact-h-candidate-bank-audit.v1"
CANDIDATES = (
    (0, None, 0.0),
    (1, 6, 0.08),
    (2, 6, -0.08),
    (3, 8, 0.08),
    (4, 8, -0.08),
    (5, 10, 0.08),
    (6, 10, -0.08),
)
PREFIX_TICK = 120
PULSE_END_TICK = 152
U32_HORIZON = 32

STATE_FIELDS = (
    "actor_observation",
    "hand_keypoints",
    "object_pose",
    "object_velocity",
    "dof_position",
    "dof_velocity",
)
COMMAND_FIELDS = ("action", "native_pd_target")
REQUIRED_FIELDS = set(STATE_FIELDS + COMMAND_FIELDS + (
    "actor_action",
    "structured_residual",
    "clipped",
    "pair",
    "surface_gap",
    "support_gap",
    "table_footprint",
    "reference_object_pose",
    "object_velocity",
    "length",
    "done",
    "interventions",
))


def write(path, value):
    def encode(item):
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError("unsupported JSON value: %r" % (type(item),))
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False,
                                      default=encode) + "\n")


def finite_arrays(data):
    for key, value in data.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("nonfinite field: " + key)


def load_candidate(root, candidate):
    path = root / ("candidate-%d" % candidate)
    manifest_path = path / "manifest.json"
    trajectory_path = path / "trajectory.npz"
    if not manifest_path.is_file() or not trajectory_path.is_file():
        raise ValueError("candidate %d is missing manifest or trajectory" % candidate)
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("status") != "COMPLETED" or manifest.get("mode") != "retarget"
            or int(manifest.get("seed", -1)) != 421
            or int(manifest.get("num_envs", -1)) != 1
            or int(manifest.get("physical_gpu", -1)) != 2
            or not manifest.get("actor_observation_saved")):
        raise ValueError("candidate %d runtime contract mismatch" % candidate)
    with np.load(trajectory_path, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    missing = REQUIRED_FIELDS - set(data)
    if missing:
        raise ValueError("candidate %d missing fields: %s" % (candidate, sorted(missing)))
    if data["actor_observation"].shape != (543, 1, 1442):
        raise ValueError("candidate %d actor observation shape mismatch" % candidate)
    for key in ("hand_keypoints", "object_velocity", "dof_position", "dof_velocity",
                "pair", "surface_gap", "support_gap", "table_footprint",
                "reference_object_pose", "object_pose"):
        if data[key].shape[0:2] != (543, 1):
            raise ValueError("candidate %d state shape mismatch for %s" % (candidate, key))
    if data["hand_keypoints"].shape != (543, 1, 11, 3):
        raise ValueError("candidate %d hand shape mismatch" % candidate)
    if data["object_pose"].shape != (543, 1, 4, 4):
        raise ValueError("candidate %d object pose shape mismatch" % candidate)
    if data["object_velocity"].shape != (543, 1, 6):
        raise ValueError("candidate %d object velocity shape mismatch" % candidate)
    for key in ("dof_position", "dof_velocity"):
        if data[key].shape != (543, 1, 18):
            raise ValueError("candidate %d q/dq shape mismatch" % candidate)
    if data["action"].shape != (542, 1, 18) or data["actor_action"].shape != (542, 1, 18):
        raise ValueError("candidate %d action shape mismatch" % candidate)
    if data["native_pd_target"].shape != (542, 1, 6):
        raise ValueError("candidate %d native PD target shape mismatch" % candidate)
    if data["structured_residual"].shape != (542, 1, 18):
        raise ValueError("candidate %d structured residual shape mismatch" % candidate)
    if data["clipped"].shape != (542, 1) or data["done"].shape != (542, 1):
        raise ValueError("candidate %d completion field shape mismatch" % candidate)
    if not np.array_equal(data["length"], np.array([542])):
        raise ValueError("candidate %d incomplete length" % candidate)
    if not np.array_equal(data["interventions"], np.array([0])):
        raise ValueError("candidate %d unexpected intervention count" % candidate)
    if int(data["clipped"].sum()) != 0:
        raise ValueError("candidate %d has clipped actions" % candidate)
    finite_arrays(data)
    return path, manifest, data


def expected_residual_report(data, candidate, finger, value):
    residual = data["structured_residual"][:, 0]
    expected = np.zeros_like(residual)
    if finger is not None:
        expected[PREFIX_TICK:PULSE_END_TICK, finger] = value
    if not np.array_equal(residual, expected):
        raise ValueError("candidate %d residual schedule mismatch" % candidate)
    if not np.array_equal(data["action"],
                          data["actor_action"] + data["structured_residual"]):
        raise ValueError("candidate %d action/residual contract mismatch" % candidate)
    return {
        "finger_index": finger,
        "value": float(value),
        "pulse_start_tick": PREFIX_TICK,
        "pulse_end_tick": PULSE_END_TICK,
        "nonzero_ticks": int(np.count_nonzero(np.max(np.abs(residual), axis=1) > 1e-8)),
        "max_abs_residual": float(np.max(np.abs(residual))),
        "clipped_steps": int(data["clipped"].sum()),
    }


def prefix_report(items):
    result = {}
    exact = True
    for key in STATE_FIELDS + COMMAND_FIELDS:
        reference = items[0][2][key]
        end = PREFIX_TICK + 1 if key in STATE_FIELDS else PREFIX_TICK
        rows = []
        for candidate, (_, _, data) in enumerate(items):
            equal = bool(np.array_equal(data[key][:end], reference[:end]))
            rows.append({"candidate": candidate, "bitwise_equal": equal,
                         "prefix_rows": int(end)})
            exact &= equal
        result[key] = rows
    result["exact"] = bool(exact)
    return result


def pairwise_rms(values):
    values = np.asarray(values)
    difference = values[:, None] - values[None, :]
    return np.sqrt(np.mean(np.square(difference), axis=tuple(range(2, difference.ndim))))


def spread_report(items):
    result = {}
    distinct = {}
    for key in ("hand_keypoints", "object_pose", "object_velocity", "dof_position"):
        values = np.stack([data[key][PREFIX_TICK + 1:, 0] for _, _, data in items])
        if key == "object_pose":
            values = values[..., :3, 3]
        distances = pairwise_rms(values)
        off_diagonal = distances[np.triu_indices(len(values), 1)]
        result[key] = {
            "pairwise_rms_mean": float(np.mean(off_diagonal)),
            "pairwise_rms_max": float(np.max(off_diagonal)),
            "nonzero_pairs": int(np.count_nonzero(off_diagonal > 1e-8)),
        }
        hashes = [sha_bytes(value.tobytes()) for value in values]
        distinct[key] = int(len(set(hashes)))
    result["distinct_trajectory_counts"] = distinct
    result["distinct_hand_trajectories"] = distinct["hand_keypoints"]
    return result


def sha_bytes(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()


def u32_report(items):
    poses = [data["object_pose"][:, 0] for _, _, data in items]
    pairs = [data["pair"][:, 0] for _, _, data in items]
    references = [data["reference_object_pose"][:, 0] for _, _, data in items]
    current_height = np.asarray([pose[PREFIX_TICK, 2, 3] for pose in poses], dtype=np.float32)
    current_pair = np.asarray([pair[PREFIX_TICK] for pair in pairs], dtype=bool)
    height = np.stack([pose[PREFIX_TICK + 1:PREFIX_TICK + 1 + U32_HORIZON, 2, 3]
                       for pose in poses])
    pair = np.stack([contact[PREFIX_TICK + 1:PREFIX_TICK + 1 + U32_HORIZON]
                     for contact in pairs])
    rest = np.asarray([reference[0, 2, 3] for reference in references], dtype=np.float32)
    y, utility = teacher(current_height, current_pair, height, pair, rest)
    y = np.asarray(y, dtype=np.float32)
    utility = np.asarray(utility, dtype=np.float32)
    if y.shape != (len(items), 8) or utility.shape != (len(items),):
        raise ValueError("unexpected U32 output shape")
    if not np.isfinite(y).all() or not np.isfinite(utility).all():
        raise ValueError("nonfinite U32 labels")
    delta = np.abs(utility[:, None] - utility[None, :])
    strict_pairs = int(np.count_nonzero(delta[np.triu_indices(len(items), 1)] > .02))
    informative = int(strict_pairs > 0)
    return {
        "query_tick": PREFIX_TICK,
        "horizon": U32_HORIZON,
        "current_pair_true": int(current_pair.sum()),
        "current_height_range": [float(current_height.min()), float(current_height.max())],
        "utility": utility.tolist(),
        "utility_range": float(np.ptp(utility)),
        "utility_unique_rounded_1e-6": int(len(np.unique(np.round(utility, 6)))),
        "strict_pair_threshold": .02,
        "strict_pairs": strict_pairs,
        "informative_label_rows": informative,
        "y_components": y.tolist(),
    }


def contact_report(items):
    result = []
    for candidate, (_, _, data) in enumerate(items):
        window = slice(PREFIX_TICK + 1, PULSE_END_TICK + 1)
        pair = data["pair"][window, 0]
        result.append({
            "candidate": candidate,
            "pair_true_frames_post_pulse": int(pair.sum()),
            "pair_any_post_pulse": bool(pair.any()),
            "surface_gap_min_m": float(np.min(data["surface_gap"][window, 0])),
            "surface_gap_max_m": float(np.max(data["surface_gap"][window, 0])),
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_root.resolve()
    output = args.output.resolve()
    expected_parent = (ROOT / "outputs/consequence-evaluator").resolve()
    if output.exists() or output.parent != expected_parent:
        raise ValueError("fresh task-owned audit output required")
    items = [load_candidate(root, candidate) for candidate, _, _ in CANDIDATES]
    manifests = [manifest for _, manifest, _ in items]
    hashes = [manifest.get("input_sha256") for manifest in manifests]
    if any(not value for value in hashes) or any(value != hashes[0] for value in hashes[1:]):
        raise ValueError("candidate input hashes differ")
    residuals = [expected_residual_report(data, candidate, finger, value)
                 for (candidate, finger, value), (_, _, data) in zip(CANDIDATES, items)]
    prefix = prefix_report(items)
    spread = spread_report(items)
    labels = u32_report(items)
    contacts = contact_report(items)
    trajectory_distinct = spread["distinct_hand_trajectories"]
    screen = {
        "exact_prefix": bool(prefix["exact"]),
        "minimum_distinct_hand_trajectories": 4,
        "distinct_hand_trajectories": trajectory_distinct,
        "minimum_informative_label_rows": 2,
        "informative_label_rows": labels["informative_label_rows"],
        "passed": bool(prefix["exact"] and trajectory_distinct >= 4
                       and labels["informative_label_rows"] >= 2),
    }
    result = {
        "status": "COMPLETED",
        "conclusion": "PROMISING" if screen["passed"] else "UNPROMISING",
        "schema": SCHEMA,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                               text=True).strip(),
        "scope": "one seed, seven serial one-env exact-prefix trajectories; engineering/data Probe only",
        "run_ids": [manifest["run_id"] for manifest in manifests],
        "candidate_contract": [
            {"candidate": candidate, "finger_index": finger, "value": value}
            for candidate, finger, value in CANDIDATES
        ],
        "input_sha256": hashes[0],
        "trajectory_sha256": {
            str(candidate): sha(root / ("candidate-%d/trajectory.npz" % candidate))
            for candidate, _, _ in CANDIDATES
        },
        "manifest_sha256": {
            str(candidate): sha(root / ("candidate-%d/manifest.json" % candidate))
            for candidate, _, _ in CANDIDATES
        },
        "residuals": residuals,
        "prefix": prefix,
        "spread": spread,
        "contact_proxy": contacts,
        "u32": labels,
        "screen": screen,
        "limitations": [
            "one seed and one reset; no episode-split panel or evaluator fit",
            "serial replay establishes visible-prefix determinism only, not hidden PhysX-state identity",
            "U32 is the frozen short-horizon outcome proxy and all labels are tied when the query is not at risk",
            "trajectory spread is an engineering diversity screen, not a causal effect estimate",
        ],
    }
    output.mkdir(parents=True)
    write(output / "result.json", result)
    write(output / "manifest.json", result)
    print(json.dumps({"conclusion": result["conclusion"], "screen": screen,
                      "u32": {key: labels[key] for key in
                              ("utility", "utility_range", "strict_pairs", "informative_label_rows")}},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()

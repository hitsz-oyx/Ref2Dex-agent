"""Audit diversity and consequence coverage in frozen structured rollouts.

This is deliberately an offline screen.  The structured rollout packet does not
contain the actor observation/history at every query, so it cannot establish an
H->tau evaluator or selector result.  It only asks whether the already captured
episodes contain reset-matched, locally similar states with different future
hand trajectories and different frozen-teacher outcomes.
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


SCHEMA = "ref2dex.structured-candidate-bank-audit.v1"
DEFAULT_TICKS = (8, 16, 24, 32)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rms(value, axis):
    return np.sqrt(np.mean(np.square(value), axis=axis))


def pairwise_rms(value):
    """Return the symmetric RMS distance matrix for a batch of vectors."""
    value = np.asarray(value)
    if value.ndim < 2:
        raise ValueError("pairwise RMS requires a batch and feature dimensions")
    difference = value[:, None] - value[None, :]
    return rms(difference, tuple(range(2, difference.ndim)))


def current_frame_hand_future(object_pose, hand_keypoints, tick):
    """Express t+1:t+24 hand points in each row's current object frame."""
    pose = np.asarray(object_pose)
    hand = np.asarray(hand_keypoints)
    if pose.ndim != 4 or hand.ndim != 4:
        raise ValueError("object pose and hand arrays must retain time and episode axes")
    if tick < 0 or tick + 24 >= len(pose):
        raise ValueError("24-step future is not available at this tick")
    center = pose[tick, :, :3, 3]
    rotation = pose[tick, :, :3, :3]
    world_relative = hand[tick + 1:tick + 25] - center[None, :, None, :]
    return np.einsum("teki,eij->tekj", world_relative, rotation)


def frozen_teacher_labels(object_pose, pair, reference_object_pose, tick):
    """Compute the frozen U32 teacher for all episodes at one query tick."""
    pose = np.asarray(object_pose)
    pair = np.asarray(pair)
    reference = np.asarray(reference_object_pose)
    if tick < 0 or tick + 32 >= len(pose):
        raise ValueError("32-step teacher future is not available at this tick")
    current_height = pose[tick, :, 2, 3]
    current_pair = pair[tick]
    future_height = pose[tick + 1:tick + 33, :, 2, 3].T
    future_pair = pair[tick + 1:tick + 33].T
    rest = reference[0, :, 2, 3]
    _, utility = teacher(current_height, current_pair, future_height, future_pair, rest)
    utility = np.asarray(utility, dtype=np.float32)
    if utility.shape != (pose.shape[1],) or not np.isfinite(utility).all():
        raise ValueError("frozen teacher returned an invalid utility vector")
    return utility


def reset_report(data):
    """Report whether the reset rows are bitwise identical across episodes."""
    fields = {
        "object_pose": data["object_pose"][0],
        "hand_keypoints": data["hand_keypoints"][0],
        "dof_position": data["dof_position"][0],
        "dof_velocity": data["dof_velocity"][0],
        "pair": data["pair"][0],
    }
    result = {}
    exact = True
    for name, value in fields.items():
        value = np.asarray(value)
        reference = value[0]
        difference = value.astype(np.float64) - reference.astype(np.float64)
        max_abs = float(np.max(np.abs(difference))) if value.size else 0.0
        bitwise = bool(np.array_equal(value, np.broadcast_to(reference, value.shape)))
        result[name] = {"bitwise_equal": bitwise, "max_abs_difference": max_abs}
        exact &= bitwise
    result["exact"] = bool(exact)
    return result


def validate_rollout(root):
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    trajectory_path = root / "trajectory.npz"
    if not manifest_path.is_file() or not trajectory_path.is_file():
        raise ValueError(f"incomplete structured rollout: {root}")
    manifest = json.loads(manifest_path.read_text())
    required = {
        "status": "COMPLETED",
        "mode": "retarget",
        "structured_residual_schema": "ref2dex.structured-residual.v1",
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError(f"structured rollout contract mismatch: {root}")
    with np.load(trajectory_path, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    required_arrays = {
        "object_pose", "hand_keypoints", "reference_object_pose", "pair",
        "dof_position", "dof_velocity", "action", "clipped", "length",
    }
    if not required_arrays.issubset(data):
        raise ValueError(f"structured rollout fields missing: {root}")
    frames, episodes = data["hand_keypoints"].shape[:2]
    if data["object_pose"].shape != (frames, episodes, 4, 4):
        raise ValueError("object pose shape mismatch")
    if data["hand_keypoints"].shape != (frames, episodes, 11, 3):
        raise ValueError("hand keypoint shape mismatch")
    if data["dof_position"].shape != (frames, episodes, 18):
        raise ValueError("dof position shape mismatch")
    if data["dof_velocity"].shape != (frames, episodes, 18):
        raise ValueError("dof velocity shape mismatch")
    if data["pair"].shape != (frames, episodes):
        raise ValueError("pair shape mismatch")
    if data["action"].shape != (frames - 1, episodes, 18):
        raise ValueError("action shape mismatch")
    if data["clipped"].shape != (frames - 1, episodes):
        raise ValueError("clipping shape mismatch")
    if len(data["length"]) != episodes or not np.all(data["length"] == frames - 1):
        raise ValueError("incomplete or unequal episode lengths")
    for name, value in data.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"nonfinite structured rollout field: {name}")
    return root, manifest, data


def query_report(data, tick, hand_threshold_m=0.003, q_threshold=0.03,
                 label_threshold=0.02):
    hand = np.asarray(data["hand_keypoints"])
    pose = np.asarray(data["object_pose"])
    q = np.asarray(data["dof_position"])
    labels = frozen_teacher_labels(pose, data["pair"], data["reference_object_pose"], tick)
    tau = current_frame_hand_future(pose, hand, tick)

    current_hand_spread = pairwise_rms(hand[tick])
    current_q_spread = pairwise_rms(q[tick])
    current_object_spread = pairwise_rms(pose[tick, :, :3, 3])
    hand_to_episode0_mm = rms(hand[tick] - hand[tick, :1], (1, 2)) * 1000.0
    q_to_episode0 = rms(q[tick] - q[tick, :1], (1,))
    object_to_episode0_mm = rms(
        pose[tick, :, :3, 3] - pose[tick, :1, :3, 3], (1,)) * 1000.0
    tau_to_episode0_mm = rms(tau - tau[:, :1], (0, 2, 3)) * 1000.0

    row, col = np.triu_indices(len(labels), 1)
    near = ((current_hand_spread[row, col] <= hand_threshold_m)
            & (current_q_spread[row, col] <= q_threshold))
    near_tau_mm = rms(tau[:, row] - tau[:, col], (0, 2, 3)) * 1000.0
    near_tau_mm = near_tau_mm[near]
    near_label_delta = np.abs(labels[row] - labels[col])[near]
    strict = near_label_delta > label_threshold

    def summary(value):
        value = np.asarray(value)
        if not len(value):
            return {"count": 0, "mean": None, "max": None}
        return {"count": int(len(value)), "mean": float(np.mean(value)),
                "max": float(np.max(value))}

    return {
        "tick": int(tick),
        "episodes": int(len(labels)),
        "candidate_tau_spread_mm": summary(tau_to_episode0_mm),
        "current_hand_spread_to_episode0_mm": summary(hand_to_episode0_mm),
        "current_q_spread_to_episode0": summary(q_to_episode0),
        "current_object_translation_spread_to_episode0_mm": summary(object_to_episode0_mm),
        "all_pairwise_current_hand_spread_mm": summary(current_hand_spread[row, col] * 1000.0),
        "all_pairwise_current_q_spread": summary(current_q_spread[row, col]),
        "teacher_label": {
            "min": float(labels.min()), "max": float(labels.max()),
            "mean": float(labels.mean()), "std": float(labels.std()),
            "unique_rounded_1e-6": int(len(np.unique(np.round(labels, 6)))),
        },
        "near_pair_rule": {
            "hand_rms_mm": float(hand_threshold_m * 1000.0),
            "q_rms": float(q_threshold),
            "label_delta_strict": float(label_threshold),
        },
        "near_pairs": int(near.sum()),
        "near_pair_episodes": int(np.unique(np.concatenate((row[near], col[near]))).size)
        if np.any(near) else 0,
        "near_pair_tau_spread_mm": summary(near_tau_mm),
        "near_pair_label_delta": summary(near_label_delta),
        "near_pair_strict_label_pairs": int(strict.sum()),
        "near_pair_max_label_delta": float(near_label_delta.max()) if len(near_label_delta) else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    expected_parent = (ROOT / "outputs/consequence-evaluator").resolve()
    if output.exists() or output.parent != expected_parent:
        raise ValueError("fresh task-owned audit output required")

    paths = {name: path.resolve() for name, path in (
        ("train", args.train), ("val", args.val), ("test", args.test))}
    loaded = {name: validate_rollout(path) for name, path in paths.items()}
    actor_hashes = {value[1].get("actor_sha256") for value in loaded.values()}
    if len(actor_hashes) != 1 or None in actor_hashes:
        raise ValueError("structured splits must use one frozen actor")
    if not all(value[1].get("pipeline") == "cpu" and value[1].get("physics_gpu") is True
               for value in loaded.values()):
        raise ValueError("structured split runtime identity is not the frozen CPU/GPU-PhysX route")

    reports = {}
    for name, (_, manifest, data) in loaded.items():
        reports[name] = {
            "run_id": manifest["run_id"],
            "seed": int(manifest["seed"]),
            "episodes": int(data["hand_keypoints"].shape[1]),
            "steps": int(data["action"].shape[0]),
            "clipped_steps": int(np.asarray(data["clipped"]).sum()),
            "reset": reset_report(data),
            "runtime": {
                "pipeline": manifest.get("pipeline"),
                "physics_gpu": bool(manifest.get("physics_gpu")),
                "actor_sha256": manifest.get("actor_sha256"),
            },
            "ticks": {str(tick): query_report(data, tick) for tick in DEFAULT_TICKS},
        }

    reset_exact = all(report["reset"]["exact"] for report in reports.values())
    screen_rows = []
    for split, report in reports.items():
        for tick, row in report["ticks"].items():
            screen_rows.append({
                "split": split, "tick": int(tick),
                "reset_exact": report["reset"]["exact"],
                "candidate_tau_spread_mean_mm": row["candidate_tau_spread_mm"]["mean"],
                "near_pairs": row["near_pairs"],
                "near_pair_strict_label_pairs": row["near_pair_strict_label_pairs"],
            })
    positive_rows = [row for row in screen_rows if row["reset_exact"]
                     and (row["candidate_tau_spread_mean_mm"] or 0.) > 5.
                     and row["near_pairs"] >= 20
                     and row["near_pair_strict_label_pairs"] >= 10]

    hashes = {}
    for split, path in paths.items():
        hashes[str((path / "manifest.json").resolve())] = sha(path / "manifest.json")
        hashes[str((path / "trajectory.npz").resolve())] = sha(path / "trajectory.npz")
    hashes[str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    result = {
        "status": "COMPLETED",
        "conclusion": "UNCLEAR",
        "schema": SCHEMA,
        "git_commit": git_commit,
        "scope": "offline structured-rollout candidate-bank diversity and frozen U32 label coverage",
        "input_sha256": hashes,
        "splits": reports,
        "screen": {
            "engineering_rule": "reset exact, mean tau spread >5 mm, >=20 near-current pairs, >=10 strict label differences",
            "reset_exact_all_splits": reset_exact,
            "positive_split_ticks": positive_rows,
            "positive_split_tick_count": len(positive_rows),
            "positive_for_candidate_diversity": bool(positive_rows),
        },
        "decision": {
            "history_contract_available": False,
            "next_action_if_continuing": "collect a bounded panel while saving exact actor history/observation at each query",
            "not_authorized_by_this_audit": ["evaluator ranking", "online selector", "PointWorld control", "native R gate"],
        },
        "limitations": [
            "structured packets do not save the actor observation/history used by H",
            "near-current matching uses hand keypoints and q only, not a full H equivalence contract",
            "teacher labels are frozen U32 outcome proxies, not a new causal preference validation",
        ],
    }
    output.mkdir(parents=True)
    write(output / "result.json", result)
    write(output / "manifest.json", result)
    print(json.dumps({"conclusion": result["conclusion"], "screen": result["screen"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()

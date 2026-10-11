"""Audit full actor-observation matching in a history-preserving rollout."""

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


SCHEMA = "ref2dex.history-candidate-bank-audit.v1"
TICKS = (8, 16, 24, 32, 40)
H_THRESHOLDS = (0.01, 0.03, 0.05)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def pairwise_rms(value):
    value = np.asarray(value)
    difference = value[:, None] - value[None, :]
    return np.sqrt(np.mean(np.square(difference), axis=tuple(range(2, difference.ndim))))


def teacher_labels(data, tick):
    pose = data["object_pose"]
    pair = data["pair"]
    reference = data["reference_object_pose"]
    height = pose[tick + 1:tick + 33, :, 2, 3].T
    _, utility = teacher(pose[tick, :, 2, 3], pair[tick], height,
                         pair[tick + 1:tick + 33].T, reference[0, :, 2, 3])
    utility = np.asarray(utility, dtype=np.float32)
    if utility.shape != (pose.shape[1],) or not np.isfinite(utility).all():
        raise ValueError("invalid frozen U32 labels")
    return utility


def summary(values):
    values = np.asarray(values)
    if values.size == 0:
        return {"count": 0, "mean": None, "max": None}
    return {"count": int(values.size), "mean": float(np.mean(values)),
            "max": float(np.max(values))}


def validate(root):
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    trajectory_path = root / "trajectory.npz"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "COMPLETED" or manifest.get("mode") != "retarget":
        raise ValueError("completed retarget rollout required")
    with np.load(trajectory_path, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    required = {"actor_observation", "hand_keypoints", "object_pose", "pair",
                "reference_object_pose", "dof_position", "action", "clipped", "length"}
    if not required.issubset(data):
        raise ValueError("history-preserving rollout is missing required fields")
    frames, episodes, history_dim = data["actor_observation"].shape
    if history_dim != 1442 or data["actor_observation"].shape != (frames, episodes, history_dim):
        raise ValueError("actor observation shape mismatch")
    if data["hand_keypoints"].shape != (frames, episodes, 11, 3):
        raise ValueError("hand shape mismatch")
    if data["object_pose"].shape != (frames, episodes, 4, 4):
        raise ValueError("object pose shape mismatch")
    if data["dof_position"].shape != (frames, episodes, 18):
        raise ValueError("q shape mismatch")
    if data["action"].shape != (frames - 1, episodes, 18):
        raise ValueError("action shape mismatch")
    if data["clipped"].shape != (frames - 1, episodes):
        raise ValueError("clipping shape mismatch")
    if not np.all(data["length"] == frames - 1):
        raise ValueError("incomplete episode")
    for key, value in data.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("nonfinite field: " + key)
    if not manifest.get("actor_observation_saved"):
        raise ValueError("manifest does not declare actor observation capture")
    return root, manifest, data


def reset_report(data):
    fields = {
        "actor_observation": data["actor_observation"][0],
        "object_pose": data["object_pose"][0],
        "hand_keypoints": data["hand_keypoints"][0],
        "dof_position": data["dof_position"][0],
        "pair": data["pair"][0],
    }
    result = {}
    exact = True
    for key, value in fields.items():
        equal = bool(np.array_equal(value, np.broadcast_to(value[0], value.shape)))
        difference = value.astype(np.float64) - value[0].astype(np.float64)
        result[key] = {"bitwise_equal": equal,
                       "max_abs_difference": float(np.max(np.abs(difference)))}
        exact &= equal
    result["exact"] = bool(exact)
    return result


def query_report(data, tick):
    history = data["actor_observation"][tick]
    hand = data["hand_keypoints"][tick]
    q = data["dof_position"][tick]
    h_distance = pairwise_rms(history)
    hand_distance = pairwise_rms(hand)
    q_distance = pairwise_rms(q)
    row, col = np.triu_indices(len(history), 1)
    labels = teacher_labels(data, tick)
    label_delta = np.abs(labels[row] - labels[col])
    state_near = ((hand_distance[row, col] * 1000. <= 3.)
                  & (q_distance[row, col] <= .03))
    result = {
        "tick": int(tick),
        "episodes": int(len(history)),
        "history_rms_all_pairs": summary(h_distance[row, col]),
        "state_near_rule": {"hand_rms_mm": 3., "q_rms": .03},
        "teacher_label": {"min": float(labels.min()), "max": float(labels.max()),
                           "mean": float(labels.mean()), "std": float(labels.std()),
                           "unique_rounded_1e-6": int(len(np.unique(np.round(labels, 6))))},
        "thresholds": {},
    }
    for threshold in H_THRESHOLDS:
        near = (h_distance[row, col] <= threshold) & state_near
        tau_near = np.zeros((0,), dtype=np.float32)
        if np.any(near):
            pose = data["object_pose"]
            hand_all = data["hand_keypoints"]
            center = pose[tick, :, :3, 3]
            rotation = pose[tick, :, :3, :3]
            world_relative = hand_all[tick + 1:tick + 25] - center[None, :, None, :]
            tau = np.einsum("teki,eij->tekj", world_relative, rotation)
            tau_near = np.sqrt(np.mean(np.square(tau[:, row[near]] - tau[:, col[near]]),
                                       axis=(0, 2, 3))) * 1000.
        strict = label_delta[near] > .02
        result["thresholds"][str(threshold)] = {
            "near_pairs": int(near.sum()),
            "near_pair_episodes": int(np.unique(np.concatenate((row[near], col[near]))).size)
            if np.any(near) else 0,
            "strict_label_pairs": int(strict.sum()),
            "near_pair_tau_spread_mm": summary(tau_near),
            "near_pair_label_delta": summary(label_delta[near]),
            "max_label_delta": float(label_delta[near].max()) if np.any(near) else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    expected_parent = (ROOT / "outputs/consequence-evaluator").resolve()
    if output.exists() or output.parent != expected_parent:
        raise ValueError("fresh task-owned audit output required")
    root, manifest, data = validate(args.input)
    if any(tick + 32 >= len(data["object_pose"]) for tick in TICKS):
        raise ValueError("query tick exceeds retained future")
    reports = {str(tick): query_report(data, tick) for tick in TICKS}
    positive = []
    for tick, report in reports.items():
        row = report["thresholds"]["0.03"]
        if row["near_pairs"] >= 20 and row["strict_label_pairs"] >= 10:
            positive.append({"tick": int(tick), **row})
    hashes = {str((root / name).resolve()): sha(root / name)
              for name in ("manifest.json", "trajectory.npz")}
    hashes[str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    result = {
        "status": "COMPLETED",
        "conclusion": "PROMISING" if positive else "UNPROMISING",
        "schema": SCHEMA,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scope": "one bounded history-preserving structured rollout; candidate-bank data screen only",
        "input_sha256": hashes,
        "run_id": manifest["run_id"],
        "episodes": int(data["actor_observation"].shape[1]),
        "frames": int(data["actor_observation"].shape[0]),
        "history_dim": int(data["actor_observation"].shape[2]),
        "reset": reset_report(data),
        "clipped_steps": int(data["clipped"].sum()),
        "ticks": reports,
        "screen": {
            "history_threshold": 0.03,
            "state_near_rule": "hand RMS <=3 mm and q RMS <=.03",
            "strict_label_delta": 0.02,
            "positive_ticks": positive,
        },
        "limitations": [
            "single 64-env launch and one seed; no episode-split evaluator fit",
            "actor observation is the feed-forward policy input, not proof of hidden PhysX-state equivalence",
            "U32 labels are frozen outcome proxies and the thresholds are an engineering screen",
        ],
    }
    output.mkdir(parents=True)
    write(output / "result.json", result)
    write(output / "manifest.json", result)
    print(json.dumps({"conclusion": result["conclusion"], "positive_ticks": positive}, indent=2), flush=True)


if __name__ == "__main__":
    main()

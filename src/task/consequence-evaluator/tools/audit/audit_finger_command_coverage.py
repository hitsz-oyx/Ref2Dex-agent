"""Audit commanded-finger label coverage in structured retarget rollouts."""

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


SCHEMA = "ref2dex.finger-command-coverage.v1"
PHASES = (("approach", 0, 120), ("contact", 120, 240), ("hold", 240, 542))
FINGER_SLICE = slice(6, 18)
HAND_RMS_MAX = 0.003
Q_RMS_MAX = 0.03
DQ_RMS_MAX = 0.10
ACTION_RMS_STRICT = 0.05


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _rms(value, axis):
    return np.sqrt(np.mean(np.square(value), axis=axis))


def near_pair_stats(hand, q, dq, action, pair, mode, phase):
    """Count near current-state pairs and nontrivial finger command differences."""
    hand = np.asarray(hand)
    q = np.asarray(q)
    dq = np.asarray(dq)
    action = np.asarray(action)
    pair = np.asarray(pair)
    mode = np.asarray(mode)
    if (hand.shape[0] != q.shape[0] or q.shape != dq.shape
            or action.shape[:2] != q.shape[:2] or pair.shape != q.shape[:2]
            or mode.shape != q.shape[:2]):
        raise ValueError("near-pair arrays must share [ticks, episodes]")
    if hand.shape[-2:] != (11, 3) or q.shape[-1] != 18 or action.shape[-1] != 18:
        raise ValueError("unexpected near-pair feature shape")
    row, col = np.triu_indices(q.shape[1], 1)
    result = {name: dict(near=0, nontrivial=0, same_mode=0,
                         same_mode_nontrivial=0, action_rms_sum=0.0)
              for name, _, _ in PHASES}
    phase = np.asarray(phase)
    for tick in range(q.shape[0]):
        phase_name = next((name for name, start, end in PHASES
                           if start <= tick < end), None)
        if phase_name is None:
            raise ValueError("tick outside registered phase boundaries")
        hand_rms = _rms(hand[tick, row] - hand[tick, col], (1, 2))
        q_rms = _rms(q[tick, row] - q[tick, col], 1)
        dq_rms = _rms(dq[tick, row] - dq[tick, col], 1)
        finger_rms = _rms(action[tick, row, FINGER_SLICE]
                          - action[tick, col, FINGER_SLICE], 1)
        near = ((hand_rms <= HAND_RMS_MAX) & (q_rms <= Q_RMS_MAX)
                & (dq_rms <= DQ_RMS_MAX)
                & (pair[tick, row] == pair[tick, col]))
        same_mode = mode[tick, row] == mode[tick, col]
        record = result[phase_name]
        record["near"] += int(near.sum())
        record["nontrivial"] += int((near & (finger_rms > ACTION_RMS_STRICT)).sum())
        record["same_mode"] += int((near & same_mode).sum())
        record["same_mode_nontrivial"] += int(
            (near & same_mode & (finger_rms > ACTION_RMS_STRICT)).sum())
        record["action_rms_sum"] += float(finger_rms[near].sum())
    for record in result.values():
        record["near_action_rms_mean"] = (
            record["action_rms_sum"] / record["near"] if record["near"] else 0.0)
        del record["action_rms_sum"]
    return result


def _load(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("status") != "COMPLETED" or manifest.get("mode") != "retarget"
            or manifest.get("structured_residual_schema") != "ref2dex.structured-residual.v1"):
        raise ValueError(f"completed structured retarget rollout required: {root}")
    with np.load(root / "trajectory.npz", allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    required = {"action", "actor_action", "structured_residual", "structured_mode",
                "structured_phase", "dof_position", "dof_velocity", "hand_keypoints",
                "object_pose", "pair", "native_contact_forces",
                "native_object_contact_forces", "clipped", "length"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"structured rollout fields missing: {sorted(missing)}")
    frames, episodes = data["hand_keypoints"].shape[:2]
    if (frames != 543 or data["action"].shape != (542, episodes, 18)
            or data["actor_action"].shape != data["action"].shape
            or data["structured_residual"].shape != data["action"].shape
            or data["structured_mode"].shape != (542, episodes)
            or data["structured_phase"].shape != (542,)
            or data["dof_position"].shape != (frames, episodes, 18)
            or data["dof_velocity"].shape != data["dof_position"].shape
            or data["object_pose"].shape != (frames, episodes, 4, 4)
            or data["pair"].shape != (frames, episodes)
            or data["clipped"].shape != (542, episodes)
            or data["length"].shape != (episodes,)
            or not np.all(data["length"] == 542)):
        raise ValueError(f"structured rollout shape/length mismatch: {root}")
    finite_fields = ("action", "actor_action", "structured_residual", "dof_position",
                     "dof_velocity", "hand_keypoints", "object_pose",
                     "native_contact_forces", "native_object_contact_forces")
    if any(not np.isfinite(data[name]).all() for name in finite_fields):
        raise ValueError(f"nonfinite structured rollout field: {root}")
    expected = np.clip(data["actor_action"] + data["structured_residual"], -1., 1.)
    if not np.allclose(expected, data["action"], atol=1e-7, rtol=0):
        raise ValueError(f"captured action composition mismatch: {root}")
    force_pair = ((np.linalg.norm(data["native_contact_forces"][:-1], axis=-1) > .1).any(-1)
                  & (np.linalg.norm(data["native_object_contact_forces"][:-1], axis=-1) > .1))
    if not np.array_equal(force_pair, data["pair"][:-1]):
        raise ValueError(f"saved pair does not match native force proxy: {root}")
    if data["structured_phase"].tolist() != [0] * 120 + [1] * 120 + [2] * 302:
        raise ValueError(f"structured phase boundaries mismatch: {root}")
    return data, manifest, {
        str(root / "manifest.json"): sha(root / "manifest.json"),
        str(root / "trajectory.npz"): sha(root / "trajectory.npz"),
    }


def audit(paths):
    splits = {}
    hashes = {}
    actor_hashes = set()
    for name, path in paths.items():
        splits[name], manifest, frozen = _load(path)
        hashes.update(frozen)
        actor_hashes.add(manifest.get("actor_sha256"))
    if len(actor_hashes) != 1 or None in actor_hashes:
        raise ValueError("all splits must use one actor hash")
    rows = {}
    near = {}
    for name, data in splits.items():
        action = data["action"]
        residual = data["structured_residual"]
        phase = data["structured_phase"]
        q = data["dof_position"][:-1]
        dq = data["dof_velocity"][:-1]
        pose = data["object_pose"][:-1]
        hand = data["hand_keypoints"][:-1]
        hand_object = (hand - pose[:, :, None, :3, 3]) @ pose[:, :, :3, :3]
        pair = data["pair"][:-1]
        split_rows = {}
        for phase_name, start, end in PHASES:
            selector = np.arange(start, end)
            finger = action[start:end, ..., FINGER_SLICE].reshape(-1, 12)
            residual_finger = residual[start:end, ..., FINGER_SLICE].reshape(-1, 12)
            mode_values = data["structured_mode"][start:end].reshape(-1)
            unique, counts = np.unique(mode_values, return_counts=True)
            split_rows[phase_name] = dict(
                rows=int(finger.shape[0]),
                action_finger_std_mean=float(finger.std(0).mean()),
                residual_finger_std_mean=float(residual_finger.std(0).mean()),
                action_finger_p01_p50_p99=np.quantile(finger, [0.01, .5, .99]).tolist(),
                pair_rate=float(pair[start:end].mean()),
                mode_counts={str(int(k)): int(v) for k, v in zip(unique, counts)},
                clipped_steps=int(data["clipped"][start:end].sum()),
            )
        rows[name] = split_rows
        near[name] = near_pair_stats(hand_object, q, dq, action, pair,
                                      data["structured_mode"], phase)
    coverage = all(rows[name][phase]["rows"] >= 1000
                   and rows[name][phase]["action_finger_std_mean"] >= .02
                   for name in rows for phase, _, _ in PHASES)
    ambiguity = all(near[name][phase]["near"] >= 50
                    and near[name][phase]["nontrivial"] >= 10
                    for name in near for phase in ("contact", "hold"))
    return dict(schema=SCHEMA, status="PROMISING" if coverage and ambiguity else "UNCLEAR",
                coverage_screen=bool(coverage), ambiguity_screen=bool(ambiguity),
                thresholds=dict(hand_rms_m=HAND_RMS_MAX, q_rms=Q_RMS_MAX,
                                dq_rms=DQ_RMS_MAX, action_finger_rms=ACTION_RMS_STRICT,
                                minimum_near_pairs=50, minimum_nontrivial_pairs=10),
                actor_sha256=next(iter(actor_hashes)), splits=rows, near_pairs=near,
                input_sha256=hashes,
                scope="label coverage and state ambiguity only; no decoder or native claim")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "val", "test"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not str(output).startswith(str(ROOT / "outputs/consequence-evaluator")):
        raise ValueError("fresh task-owned output required")
    result = audit({name: getattr(args, name) for name in ("train", "val", "test")})
    output.mkdir(parents=True)
    manifest = dict(schema=SCHEMA, status="COMPLETED", git_commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_splits={name: str(getattr(args, name).resolve())
                      for name in ("train", "val", "test")},
        episode_split=True, actor_sha256=result["actor_sha256"],
        input_sha256=result["input_sha256"],
        scope=result["scope"])
    write(output / "manifest.json", manifest)
    write(output / "result.json", result)
    print(json.dumps({key: result[key] for key in
                      ("status", "coverage_screen", "ambiguity_screen", "splits", "near_pairs")},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()

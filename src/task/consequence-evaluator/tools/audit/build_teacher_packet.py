"""Convert one complete baseline rollout into the minimal teacher packet contract."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / "src"))

from consequence_evaluator.gate1 import episode_outcome


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rollout = args.rollout.resolve(); output = args.output.resolve()
    if output.exists() or output.parent.resolve() != (ROOT / "outputs/consequence-evaluator").resolve():
        raise ValueError("fresh task-owned packet output required")
    manifest = json.loads((rollout / "manifest.json").read_text())
    if manifest.get("status") != "COMPLETED" or manifest.get("mode") != "baseline":
        raise ValueError("completed baseline rollout required")
    with np.load(rollout / "trajectory.npz", allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    required = {"object_pose", "hand_keypoints", "surface_gap", "support_gap",
                "table_footprint", "object_velocity", "dof_position", "dof_velocity",
                "done", "action", "length"}
    if not required.issubset(arrays):
        raise ValueError("baseline rollout is missing teacher fields")
    n = int(arrays["object_pose"].shape[1])
    if (arrays["object_pose"].shape[0] != 543 or arrays["hand_keypoints"].shape != (543, n, 11, 3)
            or arrays["dof_position"].shape != (543, n, 18)
            or arrays["dof_velocity"].shape != (543, n, 18)
            or arrays["action"].shape != (542, n, 18)
            or arrays["done"].shape != (542, n) or not np.all(arrays["length"] == 542)):
        raise ValueError("complete 542-step baseline tensor contract required")
    if not all(np.isfinite(arrays[key]).all() for key in
               ("object_pose", "hand_keypoints", "dof_position", "dof_velocity", "action")):
        raise ValueError("nonfinite baseline tensor")
    outcomes = {}
    for env in range(n):
        packet = {key: arrays[key][:, env] for key in
                  ("object_pose", "surface_gap", "support_gap", "table_footprint", "object_velocity")}
        outcomes["reactive_teacher" if env == 0 else "baseline_%d" % env] = episode_outcome(packet)
    backend = dict(name="cpu_pipeline", pipeline="cpu", sim_device="cuda:0", tensor_device="cuda:0",
                   physx_use_gpu=True)
    replay_identity = dict(
        backend=backend,
        actor_execution=dict(layout="environment_rows_direct", copies=1, total_rows=n),
        environment=dict(input_sha256=manifest.get("input_sha256", {})),
        rollout_manifest_sha256=sha(rollout / "manifest.json"))
    packet = dict(
        object_pose=arrays["object_pose"].astype("float32"),
        hand_keypoints=arrays["hand_keypoints"].astype("float32"),
        surface_gap=arrays["surface_gap"].astype("float32"),
        support_gap=arrays["support_gap"].astype("float32"),
        table_footprint=arrays["table_footprint"].astype(bool),
        object_velocity=arrays["object_velocity"].astype("float32"),
        dof_position=arrays["dof_position"].astype("float32"),
        dof_velocity=arrays["dof_velocity"].astype("float32"),
        actions=arrays["action"].astype("float32"),
        done=np.concatenate((np.zeros((1, n), dtype=bool), arrays["done"]), axis=0),
        seed=int(manifest["seed"]), role_names=["reactive_teacher"] + ["baseline_%d" % i for i in range(1, n)],
        role_outcomes=outcomes, engineering_only=True, training_allowed=False,
        group_mode="synchronous_same_cpu_baseline_teacher_packet", group_envs=n,
        source_backend=backend, replay_identity=replay_identity,
        source_rollout=str(rollout), source_rollout_sha256=sha(rollout / "trajectory.npz"),
        target_semantics="captured actor native action under CPU tensor pipeline",
        provenance_note="teacher packet for source/backend isolation; not a new independent seed")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump(packet, stream, protocol=4)
    (output.with_suffix(".json")).write_text(json.dumps(
        dict(status="COMPLETED", schema="ref2dex.teacher-packet.v1", packet=str(output),
             packet_sha256=sha(output), source_rollout=str(rollout),
             source_rollout_sha256=sha(rollout / "trajectory.npz"),
             source_backend=backend, role_outcomes=outcomes), indent=2) + "\n")
    print(json.dumps(dict(packet=str(output), packet_sha256=sha(output), role_outcomes=outcomes), indent=2))


if __name__ == "__main__":
    main()

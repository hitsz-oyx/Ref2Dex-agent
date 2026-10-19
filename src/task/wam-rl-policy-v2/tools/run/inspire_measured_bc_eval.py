"""Bounded closed-loop evaluation of a measured Inspire BC checkpoint.

This runner deliberately has no WAM prediction or PPO update.  It validates
the measured347/task10 observation boundary against the native physics host;
Scheduled686 remains a separate online-conditioning decision.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


TASK = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("REF2DEX_RUN_ROOT", TASK.parents[2])).resolve()
sys.path[:0] = [str(TASK / "src"), str(ROOT / "src/task/trajectory-policy/src")]


def write(path, value):
    path = Path(path)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(value, indent=2) + "\n")
    part.replace(path)


def pose_from_object(values):
    import numpy as np
    from scipy.spatial.transform import Rotation

    values = np.asarray(values, dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = Rotation.from_quat(values[3:7]).as_matrix()
    pose[:3, 3] = values[:3]
    return pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--controls", type=int, default=60)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--state-transport", choices=("json", "packed"), default="json")
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    args.socket = args.socket.resolve()
    if args.output.exists() or ROOT / "outputs/wam-rl-policy-v2" not in args.output.parents:
        raise ValueError("fresh bounded task output required")
    if not 1 <= args.controls <= 1200 or not 30 <= args.seconds <= 1800:
        raise ValueError("bounded measured BC evaluation required")
    if ROOT / "tmp" not in args.socket.parents or args.socket.exists():
        raise ValueError("private project socket required")
    args.output.mkdir(parents=True)
    args.socket.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(args.socket.parent, 0o700)
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        TMPDIR=str(args.socket.parent),
        OMP_NUM_THREADS="2",
        PYTHONDONTWRITEBYTECODE="1",
    )
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("measured BC deadline")))
    signal.alarm(args.seconds)

    import numpy as np
    import torch
    from trajectory_policy.wam_ipc import peer_pid, receive, send
    from wam_rl_policy_v2.inspire_state_codec import unpack_state
    from wam_rl_policy_v2.inspire_contract import expand_action
    from wam_rl_policy_v2.inspire_shadow import current_state
    from wam_rl_policy_v2.measured_policy import MeasuredInspirePolicy
    from wam_rl_policy_v2.measured_reward_bridge import MeasuredRewardTracker, reward_features
    from wam_rl_policy_v2.robot_adapter import InspireAdapter
    from wam_rl_policy_v2.task_runtime import TaskController
    from wam_rl_policy_v2.teacher_dataset import TeacherRolloutCorpus

    torch.set_num_threads(2)
    packet = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if packet.get("schema") != "ref2dex.inspire-measured-current-bc.v1":
        raise ValueError("Current421 measured BC checkpoint required")
    model = MeasuredInspirePolicy(with_tau=False).cuda().eval()
    model.load_state_dict(packet["model"], strict=True)
    corpus = TeacherRolloutCorpus(args.manifest, "train")
    by_sequence = {row["sequence"]: row for row in corpus.records}
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.settimeout(300)
    listener.bind(str(args.socket))
    listener.listen(1)
    state = dict(
        schema="ref2dex.inspire-measured-bc-closed-loop.v1",
        status="WAITING_NATIVE",
        git_commit=os.environ.get("REF2DEX_CODE_COMMIT")
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        checkpoint=str(args.checkpoint),
        source_manifest=str(args.manifest),
        gpu=args.gpu,
        controls_requested=args.controls,
        state_transport=args.state_transport,
        observation_contract="measured347+task10+shape64=Current421",
        policy_updates=0,
    )
    write(args.output / "manifest.json", state)
    print(json.dumps(dict(status="READY", pid=os.getpid())), flush=True)
    connection = None
    started = time.monotonic()
    try:
        connection, _ = listener.accept()
        connection.settimeout(300)
        hello = receive(connection)
        if hello.get("schema") != "ref2dex.inspire-shadow-rpc.v1":
            raise ValueError("wrong native protocol")
        if hello["audit"].get("state_transport", "json") != args.state_transport:
            raise ValueError("native state transport mismatch")
        if not hello["audit"].get("extended_state"):
            raise ValueError("measured BC requires native --extended-state")
        state["native_peer_pid"] = peer_pid(connection)
        native = unpack_state(hello["state"])
        sequences = list(hello["state"]["sequences"])
        if len(set(sequences)) != len(sequences) or any(sequence not in by_sequence for sequence in sequences):
            raise ValueError("native sequence is outside the audited TRAIN corpus")
        contexts = []
        context_points = []
        for sequence in sequences:
            row = by_sequence[sequence]
            sidecar = args.manifest.parent / row["sidecar"]
            spec = json.loads((sidecar / "task.json").read_text())
            from wam_rl_policy_v2.task_spec import TaskSpec

            task_spec = TaskSpec.from_packet(spec)
            source = Path(row["source"])
            with np.load(source / "canonical.npz") as geometry:
                points = geometry["points"].astype(np.float32, copy=True)
            contexts.append(TaskController(task_spec, points))
            context_points.append(points)
        previous = np.zeros((len(sequences), 12), np.float32)
        decision_count = np.zeros(len(sequences), np.int64)
        force_valid_count = 0
        latest_decisions = [None] * len(sequences)
        terminal_records = {}
        reward_trackers = [MeasuredRewardTracker() for _ in sequences]
        reward_totals = np.zeros(len(sequences), np.float64)
        reward_last_parts = [None] * len(sequences)

        def observe_and_predict(snapshot):
            nonlocal force_valid_count
            snapshot = unpack_state(snapshot)
            n = len(sequences)
            measured, tasks, points = [], [], []
            for index in range(n):
                pose = pose_from_object(snapshot["objects"][index])
                hand = InspireAdapter.canonical_hand(np.asarray(snapshot["hand"][index]))
                dof_valid = bool(np.asarray(snapshot["dof_force_valid"])[index])
                contact_valid = bool(np.asarray(snapshot["contact_force_valid"])[index])
                contact = np.asarray(snapshot["contact_forces"], dtype=np.float32)[index]
                force = np.asarray(snapshot["dof_force"], dtype=np.float32)[index]
                velocity = np.asarray(snapshot["objects"], dtype=np.float32)[index, 7:13]
                tick = int(np.asarray(snapshot["tick"])[index])
                result = contexts[index].observe(
                    pose,
                    hand,
                    velocity,
                    tick / 30.0,
                    contact_forces=contact,
                    force_valid=contact_valid,
                    supported=None,
                )
                decision = result["decision"]
                features = reward_features(
                    pose, hand, context_points[index], decision, velocity, contact
                )
                reward_value = 0.0
                if reward_trackers[index].previous_decision is None:
                    reward_trackers[index].initialize(decision, features)
                elif not reward_trackers[index].reward.terminal:
                    reward_value, reward_last_parts[index] = reward_trackers[index].step(
                        decision, features, previous[index]
                    )
                    reward_totals[index] += reward_value
                latest_decisions[index] = dict(
                    phase=int(decision.phase), path_progress=float(decision.path_progress),
                    task_complete=bool(decision.task_complete), terminated=bool(decision.terminated),
                    failure_reason=decision.failure_reason,
                    reward=float(reward_value),
                )
                if decision.terminated and index not in terminal_records:
                    terminal_records[index] = latest_decisions[index]
                # The audited corpus uses the immutable task terminal endpoint
                # in base312; the moving local waypoint is represented only by
                # the explicit task10 block.
                endpoint = contexts[index].scheduler.spec.object_path[-1, :3, 3]
                measured_base = current_state(
                    np.asarray(snapshot["q"])[index : index + 1],
                    np.asarray(snapshot["dq"])[index : index + 1],
                    np.asarray(snapshot["physical"])[index : index + 1],
                    np.asarray(snapshot["objects"])[index : index + 1],
                    endpoint[None],
                    previous[index : index + 1],
                )[0]
                value = np.concatenate(
                    (
                        measured_base,
                        np.where(dof_valid, force, 0.0),
                        np.where(contact_valid, contact.reshape(-1), 0.0),
                        np.array([dof_valid, contact_valid], np.float32),
                    )
                )
                if value.shape != (347,) or not np.isfinite(value).all():
                    raise ValueError("finite measured347 state required")
                measured.append(value)
                tasks.append(result["task"])
                points.append(context_points[index])
                decision_count[index] += 1
                force_valid_count += int(contact_valid)
            with torch.inference_mode():
                output = model(
                    torch.as_tensor(np.asarray(measured), device="cuda"),
                    torch.as_tensor(np.asarray(tasks), device="cuda"),
                    torch.as_tensor(np.asarray(points), device="cuda"),
                )
            action = output.clamp(-1, 1).cpu().numpy().astype(np.float32)
            previous[:] = action
            return action, snapshot

        action, native = observe_and_predict(native)
        for control in range(args.controls):
            send(
                connection,
                dict(op="step", action=expand_action(action).tolist(), episodes=native["episodes"]),
            )
            reply = receive(connection)
            if "state" not in reply:
                raise ValueError("native step response lacks state")
            action, native = observe_and_predict(reply["state"])
            state.update(
                status="RUNNING",
                controls_completed=control + 1,
                force_valid_observations=int(force_valid_count),
                scheduler_decisions=decision_count.tolist(),
                latest_decisions=latest_decisions, terminal_records=terminal_records,
                reward_totals=reward_totals.tolist(), reward_last_parts=reward_last_parts,
                elapsed_s=time.monotonic() - started,
            )
            write(args.output / "progress.json", state)
        send(connection, dict(op="close"))
        receive(connection)
        state.update(
            status="COMPLETED",
            controls_completed=args.controls,
            force_valid_observations=int(force_valid_count),
            scheduler_decisions=decision_count.tolist(),
            latest_decisions=latest_decisions, terminal_records=terminal_records,
            reward_totals=reward_totals.tolist(), reward_last_parts=reward_last_parts,
            elapsed_s=time.monotonic() - started,
            task_success_claim=False,
            scope="bounded Current421 physical closed-loop input contract only",
        )
    except BaseException as error:
        state.update(status="FAILED", error=repr(error), elapsed_s=time.monotonic() - started)
        raise
    finally:
        signal.alarm(0)
        if connection is not None:
            connection.close()
        listener.close()
        args.socket.unlink(missing_ok=True)
        write(args.output / "manifest.json", state)


if __name__ == "__main__":
    main()

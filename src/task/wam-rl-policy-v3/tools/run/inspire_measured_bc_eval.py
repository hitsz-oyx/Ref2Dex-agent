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
    parser.add_argument("--seed", type=int, default=246)
    parser.add_argument("--controls", type=int, default=60)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--state-transport", choices=("json", "packed"), default="json")
    parser.add_argument("--ppo-collector-smoke", action="store_true",
                        help="record a bounded measured PPO transition batch and update once")
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    args.socket = args.socket.resolve()
    if args.output.exists() or ROOT / "outputs/wam-rl-policy-v3" not in args.output.parents:
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
    from wam_rl_policy_v3.inspire_state_codec import unpack_state
    from wam_rl_policy_v3.inspire_contract import expand_action
    from wam_rl_policy_v3.inspire_shadow import current_state
    from wam_rl_policy_v3.measured_policy import MeasuredInspirePolicy
    from wam_rl_policy_v3.measured_ppo import MeasuredCritic, MeasuredGaussianActor, update_measured
    from wam_rl_policy_v3.measured_reward_bridge import MeasuredRewardTracker, reward_features
    from wam_rl_policy_v3.robot_adapter import InspireAdapter
    from wam_rl_policy_v3.task_runtime import TaskController
    from wam_rl_policy_v3.teacher_dataset import TeacherRolloutCorpus
    from wam_rl_policy_v3.measured_execution import online_support, run_single_episode_batch, record_bc_decision, save_bc_trace

    torch.set_num_threads(2)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    packet = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if packet.get("schema") != "ref2dex.inspire-measured-current-bc.v1":
        raise ValueError("Current421 measured BC checkpoint required")
    model = MeasuredInspirePolicy(with_tau=False).cuda().eval()
    model.load_state_dict(packet["model"], strict=True)
    ppo_actor = MeasuredGaussianActor.from_bc_packet(packet, with_tau=False).cuda().eval() if args.ppo_collector_smoke else None
    ppo_critic = MeasuredCritic(421).cuda().eval() if args.ppo_collector_smoke else None
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
        np.savez_compressed(args.output / 'initial-state.npz', **{
            key: np.asarray(native[key]) for key in ('q','dq','physical','objects','hand','tick','table_pose')
        })
        sequences = list(hello["state"]["sequences"])
        if len(set(sequences)) != len(sequences) or any(sequence not in by_sequence for sequence in sequences):
            raise ValueError("native sequence is outside the audited TRAIN corpus")
        contexts = []
        context_points = []
        table_bounds = []
        for sequence in sequences:
            row = by_sequence[sequence]
            sidecar = args.manifest.parent / row["sidecar"]
            spec = json.loads((sidecar / "task.json").read_text())
            from wam_rl_policy_v3.task_spec import TaskSpec

            task_spec = TaskSpec.from_packet(spec)
            source = Path(row["source"])
            with np.load(source / "canonical.npz") as geometry:
                points = geometry["points"].astype(np.float32, copy=True)
            contexts.append(TaskController(task_spec, points))
            context_points.append(points)
            with np.load(sidecar / "admission.npz") as masks:
                table_bounds.append(masks["table_bounds"].copy())
        previous = np.zeros((len(sequences), 12), np.float32)
        decision_count = np.zeros(len(sequences), np.int64)
        force_valid_count = 0
        latest_decisions = [None] * len(sequences)
        terminal_records = {}
        reward_trackers = [MeasuredRewardTracker() for _ in sequences]
        state["gamma"] = reward_trackers[0].gamma
        state["seed"] = args.seed
        reward_totals = np.zeros(len(sequences), np.float64)
        reward_last_parts = [None] * len(sequences)
        transitions = [[] for _ in sequences]
        pending = [None] * len(sequences)
        decision_trace = []

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
                    supported=online_support(snapshot, index, pose, context_points[index], table_bounds[index]),
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
                measured_tensor = torch.as_tensor(np.asarray(measured), device="cuda")
                task_tensor = torch.as_tensor(np.asarray(tasks), device="cuda")
                points_tensor = torch.as_tensor(np.asarray(points), device="cuda")
                if ppo_actor is None:
                    output = model(measured_tensor, task_tensor, points_tensor)
                else:
                    shape = model.shape(points_tensor)
                    robot = torch.cat((measured_tensor[..., :345], shape, measured_tensor[..., 345:]), -1)
                    observation = torch.cat((robot, task_tensor), -1)
                    output, logp, behavior_mean, behavior_std = ppo_actor.sample(observation)
                    values = ppo_critic(observation)
                    for index in range(len(sequences)):
                        if pending[index] is not None:
                            old = pending[index]
                            transitions[index].append(dict(
                                observation=old["observation"].detach().cpu(),
                                action=old["action"].detach().cpu(),
                                logp=old["logp"].detach().cpu(),
                                mean=old["mean"].detach().cpu(),
                                std=old["std"].detach().cpu(),
                                value=old["value"].detach().cpu(),
                                reward=float(latest_decisions[index]["reward"]),
                                done=bool(latest_decisions[index]["terminated"]),
                            ))
                        pending[index] = dict(observation=observation[index], action=output[index],
                                              logp=logp[index], mean=behavior_mean[index],
                                              std=behavior_std[index], value=values[index])
            action = output.clamp(-1, 1).cpu().numpy().astype(np.float32)
            record_bc_decision(decision_trace, snapshot, measured, tasks, action)
            previous[:] = action
            return action, snapshot

        action, native = observe_and_predict(native)

        def record_progress(completed):
            state.update(
                status="RUNNING",
                controls_completed=completed,
                force_valid_observations=int(force_valid_count),
                scheduler_decisions=decision_count.tolist(),
                latest_decisions=latest_decisions, terminal_records=terminal_records,
                reward_totals=reward_totals.tolist(), reward_last_parts=reward_last_parts,
                elapsed_s=time.monotonic() - started,
            )
            write(args.output / "progress.json", state)

        completed = run_single_episode_batch(
            connection, action, native, args.controls,
            observe=observe_and_predict, on_step=record_progress,
            is_terminal=lambda: any(d["terminated"] for d in latest_decisions),
            send=send, receive=receive, expand_action=expand_action,
        )
        state["termination_policy"] = "stop_batch_at_first_task_terminal"
        save_bc_trace(args.output / 'first100-decisions.npz', decision_trace, completed)
        state["stop_reason"] = "task_terminal" if terminal_records else "control_budget"
        send(connection, dict(op="close"))
        receive(connection)
        collector_result = None
        if ppo_actor is not None and completed:
            if any(len(row) != completed for row in transitions):
                raise ValueError("online PPO collector did not produce one transition per control")
            batch = {}
            for key in ("observation", "action", "logp", "mean", "std", "value"):
                batch[key] = torch.stack([torch.stack([row[key] for row in transitions[index]])
                                           for index in range(len(sequences))], dim=1).cuda()
            batch["reward"] = torch.as_tensor(np.asarray([[row["reward"] for row in transitions[index]]
                                                          for index in range(len(sequences))], np.float32).T, device="cuda")
            batch["done"] = torch.as_tensor(np.asarray([[row["done"] for row in transitions[index]]
                                                        for index in range(len(sequences))], bool).T, device="cuda")
            batch["bootstrap"] = torch.stack([pending[index]["value"] for index in range(len(sequences))])
            # Keep the unclipped behavior and the actual native command separately.
            np.savez_compressed(args.output / "collector-actions.npz",
                action=batch["action"].cpu().numpy(), command=batch["action"].clamp(-1, 1).cpu().numpy(),
                mean=batch["mean"].cpu().numpy(), std=batch["std"].cpu().numpy(),
                logp=batch["logp"].cpu().numpy(), done=batch["done"].cpu().numpy(),
                reward=batch["reward"].cpu().numpy(), bootstrap=batch["bootstrap"].cpu().numpy())
            ppo_actor.train(); ppo_critic.train()
            actor_optimizer = torch.optim.Adam(ppo_actor.parameters(), lr=1e-5)
            critic_optimizer = torch.optim.Adam(ppo_critic.parameters(), lr=1e-4)
            collector_result = update_measured(ppo_actor, ppo_critic, batch, actor_optimizer, critic_optimizer, gamma=reward_trackers[0].gamma, epochs=1)
            state["ppo_collector"] = dict(transition_count=[len(row) for row in transitions], update=collector_result,
                                            bootstrap_source="actual final measured observation")
        state.update(
            status="COMPLETED",
            controls_completed=completed,
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

"""Bounded online spatial-tau evaluation for the Scheduled686 BC checkpoint."""

import argparse
import json
import os
from collections import deque
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("REF2DEX_RUN_ROOT", TASK.parents[2])).resolve()
sys.path[:0] = [str(TASK / "src"), str(ROOT / "src/task/trajectory-policy/src")]

from inspire_measured_bc_eval import pose_from_object, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--geometry-cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--socket", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=6)
    p.add_argument("--seed", type=int, default=246)
    p.add_argument("--controls", type=int, default=60)
    p.add_argument("--seconds", type=int, default=1800)
    p.add_argument("--state-transport", choices=("json", "packed"), default="json")
    p.add_argument("--ppo-collector-smoke", action="store_true",
                   help="record a bounded measured PPO transition batch and update once")
    a = p.parse_args()
    a.manifest, a.checkpoint = a.manifest.resolve(), a.checkpoint.resolve()
    a.geometry_cache = a.geometry_cache.resolve()
    a.output, a.socket = a.output.resolve(), a.socket.resolve()
    if a.output.exists() or ROOT / "outputs/wam-rl-policy-v3" not in a.output.parents:
        raise ValueError("fresh bounded task output required")
    if not 1 <= a.controls <= 1200 or not 30 <= a.seconds <= 1800:
        raise ValueError("bounded scheduled BC evaluation required")
    if ROOT / "tmp" not in a.socket.parents or a.socket.exists():
        raise ValueError("private project socket required")
    a.output.mkdir(parents=True); a.socket.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(a.socket.parent, 0o700)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), TMPDIR=str(a.socket.parent),
                      OMP_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1", REF2DEX_DEEP_OPT="all")
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("scheduled BC deadline")))
    signal.alarm(a.seconds)

    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    from trajectory_policy.wam_ipc import peer_pid, receive, send
    from wam_rl_policy_v3.inspire_contract import expand_action
    from wam_rl_policy_v3.inspire_shadow import current_state, tau_world
    from wam_rl_policy_v3.inspire_state_codec import unpack_state
    from wam_rl_policy_v3.measured_policy import MeasuredInspirePolicy
    from wam_rl_policy_v3.measured_ppo import MeasuredCritic, MeasuredGaussianActor, update_measured
    from wam_rl_policy_v3.measured_reward_bridge import MeasuredRewardTracker, reward_features
    from wam_rl_policy_v3.robot_adapter import InspireAdapter
    from wam_rl_policy_v3.task_runtime import TaskController
    from wam_rl_policy_v3.teacher_dataset import TeacherRolloutCorpus
    from wam_rl_policy_v3.measured_execution import online_support, run_single_episode_batch
    from wam_rl_policy_v3.goal_scheduler import rigid_goal_delta
    from wam_rl_policy_v3.measured_execution import environment_geometry, GOAL_CONTRACT, task_wam_goal
    from wam_rl_policy_v3.task_spec import TaskSpec
    from wam_rl_policy_v3.wam_v4_runtime import V4Runtime, collate_v4, v4_online_samples

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    packet = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    if packet.get("schema") != "ref2dex.inspire-measured-scheduled-bc.v2":
        raise ValueError("Scheduled686 measured BC checkpoint required")
    if packet.get("config", {}).get("goal_contract") != GOAL_CONTRACT:
        raise ValueError("checkpoint needs shared measured TaskController conditioning")
    model = MeasuredInspirePolicy(with_tau=True).cuda().eval()
    model.load_state_dict(packet["model"], strict=True)
    ppo_actor = MeasuredGaussianActor.from_bc_packet(packet, with_tau=True).cuda().eval() if a.ppo_collector_smoke else None
    ppo_critic = MeasuredCritic(686).cuda().eval() if a.ppo_collector_smoke else None
    corpus = TeacherRolloutCorpus(a.manifest, "train")
    by_sequence = {row["sequence"]: row for row in corpus.records}
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); listener.settimeout(300)
    listener.bind(str(a.socket)); listener.listen(1)
    state = dict(schema="ref2dex.inspire-measured-scheduled-bc-closed-loop.v1", status="WAITING_NATIVE",
                 git_commit=os.environ.get("REF2DEX_CODE_COMMIT") or subprocess.check_output(
                     ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                 checkpoint=str(a.checkpoint), source_manifest=str(a.manifest), geometry_cache=str(a.geometry_cache),
                 gpu=a.gpu, controls_requested=a.controls, state_transport=a.state_transport,
                 observation_contract="measured347+task10+tau264+valid1+shape64=Scheduled686", policy_updates=0,
                 event_conditioning=False)
    write(a.output / "manifest.json", state); print(json.dumps(dict(status="READY", pid=os.getpid())), flush=True)
    connection = None; started = time.monotonic()
    try:
        connection, _ = listener.accept(); connection.settimeout(300)
        hello = receive(connection)
        if hello.get("schema") != "ref2dex.inspire-shadow-rpc.v1": raise ValueError("wrong native protocol")
        if hello["audit"].get("state_transport", "json") != a.state_transport: raise ValueError("native state transport mismatch")
        if not hello["audit"].get("extended_state"): raise ValueError("scheduled BC requires --extended-state")
        state["native_peer_pid"] = peer_pid(connection)
        native = unpack_state(hello["state"]); sequences = list(native["sequences"])
        if len(set(sequences)) != len(sequences) or any(sequence not in by_sequence for sequence in sequences):
            raise ValueError("native sequence is outside audited TRAIN corpus")
        objects = [dict(object_code="dexplore/" + sequence.split("_")[1], scale=1) for sequence in sequences]
        runtime = V4Runtime(packet["config"]["wam"], objects, a.geometry_cache,
                            dense_manifest=a.geometry_cache / "surface-manifest.json",
                            allow_geometry_replacement=True, device="cuda:0")
        contexts, points, geometry, table_bounds = [], [], [], []
        for env_index, sequence in enumerate(sequences):
            row = by_sequence[sequence]; sidecar = a.manifest.parent / row["sidecar"]
            spec = TaskSpec.from_packet(json.loads((sidecar / "task.json").read_text()))
            source = Path(row["source"])
            with np.load(source / "canonical.npz") as values: surface = values["points"].astype(np.float32, copy=True)
            contexts.append(TaskController(spec, surface)); points.append(surface)
            geometry.append(environment_geometry(runtime, env_index, "dexplore/" + row["object"]))
            with np.load(sidecar / "admission.npz") as masks:
                table_bounds.append(masks["table_bounds"].copy())
        histories = [deque(maxlen=4) for _ in sequences]; hand_histories = [deque(maxlen=4) for _ in sequences]
        previous = np.zeros((len(sequences), 12), np.float32); decision_count = np.zeros(len(sequences), np.int64)
        tau_valid_count = 0; force_valid_count = 0; max_goal_task_error = 0.
        latest_decisions = [None] * len(sequences); terminal_records = {}
        reward_trackers = [MeasuredRewardTracker() for _ in sequences]
        state["gamma"] = reward_trackers[0].gamma
        state["seed"] = a.seed
        reward_totals = np.zeros(len(sequences), np.float64); reward_last_parts = [None] * len(sequences)
        transitions = [[] for _ in sequences]; pending = [None] * len(sequences)

        def observe_and_predict(snapshot):
            nonlocal tau_valid_count, force_valid_count, max_goal_task_error
            snapshot = unpack_state(snapshot); n = len(sequences)
            measured, tasks, shape_points, taus, tau_valid = [], [], [], [], []
            for index in range(n):
                pose = pose_from_object(snapshot["objects"][index]); hand = InspireAdapter.canonical_hand(np.asarray(snapshot["hand"][index]))
                dof_valid = bool(np.asarray(snapshot["dof_force_valid"])[index]); contact_valid = bool(np.asarray(snapshot["contact_force_valid"])[index])
                force = np.asarray(snapshot["dof_force"], dtype=np.float32)[index]
                contact = np.asarray(snapshot["contact_forces"], dtype=np.float32)[index]
                tick = int(np.asarray(snapshot["tick"])[index]); now = tick / 30.0
                result = contexts[index].observe(pose, hand, np.asarray(snapshot["objects"], dtype=np.float32)[index, 7:13], now,
                    contact_forces=contact, force_valid=contact_valid,
                    supported=online_support(snapshot, index, pose, points[index], table_bounds[index]))
                decision = result["decision"]
                features = reward_features(
                    pose, hand, points[index], decision,
                    np.asarray(snapshot["objects"], dtype=np.float32)[index, 7:13], contact
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
                histories[index].append(np.r_[pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat()])
                hand_histories[index].append(hand.copy())
                tau = np.zeros((8, 11, 3), np.float32); valid = False
                if len(histories[index]) == 4 and not decision.terminated:
                    objects4 = np.asarray(histories[index], np.float32)[None]
                    hands4 = np.asarray(hand_histories[index], np.float32)[None]
                    delta, error = task_wam_goal(pose, decision.local_goal_pose, result["task"], geometry[index][2])
                    max_goal_task_error = max(max_goal_task_error, error)
                    goal = delta[None]
                    sample = v4_online_samples(objects4, hands4, [geometry[index]], goal_delta=goal)[0]
                    if runtime.workspace_valid([sample])[0]:
                        batch = {key: value.to(runtime.device, non_blocking=True) for key, value in collate_v4([sample]).items()}
                        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
                            trajectory = runtime.predict_hand(batch)["trajectory"].float().cpu().numpy()
                        tau = tau_world(trajectory, pose[None, :3, :3])[0]; valid = True; tau_valid_count += 1
                endpoint = contexts[index].scheduler.spec.object_path[-1, :3, 3]
                base = current_state(np.asarray(snapshot["q"])[index:index + 1], np.asarray(snapshot["dq"])[index:index + 1],
                    np.asarray(snapshot["physical"])[index:index + 1], np.asarray(snapshot["objects"])[index:index + 1],
                    endpoint[None], previous[index:index + 1])[0]
                value = np.concatenate((base, np.where(dof_valid, force, 0.), np.where(contact_valid, contact.reshape(-1), 0.),
                                        np.array([dof_valid, contact_valid], np.float32)))
                if value.shape != (347,) or not np.isfinite(value).all(): raise ValueError("finite measured347 state required")
                measured.append(value); tasks.append(result["task"]); shape_points.append(points[index]); taus.append(tau); tau_valid.append(valid)
                decision_count[index] += 1; force_valid_count += int(contact_valid)
            with torch.inference_mode():
                measured_tensor = torch.as_tensor(np.asarray(measured), device="cuda")
                task_tensor = torch.as_tensor(np.asarray(tasks), device="cuda")
                shape_tensor = torch.as_tensor(np.asarray(shape_points), device="cuda")
                tau_tensor = torch.as_tensor(np.asarray(taus), device="cuda")
                tau_valid_tensor = torch.as_tensor(np.asarray(tau_valid), device="cuda")
                if ppo_actor is None:
                    output = model(measured_tensor, task_tensor, shape_tensor, tau=tau_tensor, tau_valid=tau_valid_tensor)
                else:
                    shape = model.shape(shape_tensor)
                    robot = torch.cat((measured_tensor[..., :345], shape, measured_tensor[..., 345:]), -1)
                    observation = torch.cat((robot, task_tensor, tau_tensor.flatten(1),
                                             tau_valid_tensor.to(measured_tensor.dtype)[:, None]), -1)
                    output, logp, behavior_mean, behavior_std = ppo_actor.sample(observation)
                    values = ppo_critic(observation)
                    for index in range(len(sequences)):
                        if pending[index] is not None:
                            old = pending[index]
                            transitions[index].append(dict(
                                observation=old["observation"].detach().cpu(), action=old["action"].detach().cpu(),
                                logp=old["logp"].detach().cpu(), mean=old["mean"].detach().cpu(),
                                std=old["std"].detach().cpu(), value=old["value"].detach().cpu(),
                                reward=float(latest_decisions[index]["reward"]),
                                done=bool(latest_decisions[index]["terminated"]),
                            ))
                        pending[index] = dict(observation=observation[index], action=output[index],
                                              logp=logp[index], mean=behavior_mean[index],
                                              std=behavior_std[index], value=values[index])
            action = output.clamp(-1, 1).cpu().numpy().astype(np.float32); previous[:] = action
            return action, snapshot

        action, native = observe_and_predict(native)

        def record_progress(completed):
            state.update(status="RUNNING", controls_completed=completed, tau_valid_observations=int(tau_valid_count),
                         force_valid_observations=int(force_valid_count), scheduler_decisions=decision_count.tolist(),
                         latest_decisions=latest_decisions, terminal_records=terminal_records, elapsed_s=time.monotonic() - started)
            state["reward_totals"] = reward_totals.tolist(); state["reward_last_parts"] = reward_last_parts
            write(a.output / "progress.json", state)

        completed = run_single_episode_batch(
            connection, action, native, a.controls,
            observe=observe_and_predict, on_step=record_progress,
            is_terminal=lambda: any(d["terminated"] for d in latest_decisions),
            send=send, receive=receive, expand_action=expand_action,
        )
        state["max_goal_task_error_m"] = max_goal_task_error
        state["termination_policy"] = "stop_batch_at_first_task_terminal"
        state["stop_reason"] = "task_terminal" if terminal_records else "control_budget"
        send(connection, dict(op="close")); receive(connection)
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
            np.savez_compressed(a.output / "collector-actions.npz",
                action=batch["action"].cpu().numpy(), command=batch["action"].clamp(-1, 1).cpu().numpy(),
                mean=batch["mean"].cpu().numpy(), std=batch["std"].cpu().numpy(),
                logp=batch["logp"].cpu().numpy(), done=batch["done"].cpu().numpy(),
                reward=batch["reward"].cpu().numpy(), bootstrap=batch["bootstrap"].cpu().numpy())
            ppo_actor.train(); ppo_critic.train()
            actor_optimizer = torch.optim.Adam(ppo_actor.parameters(), lr=1e-5)
            critic_optimizer = torch.optim.Adam(ppo_critic.parameters(), lr=1e-4)
            state["ppo_collector"] = dict(transition_count=[len(row) for row in transitions],
                                           update=update_measured(ppo_actor, ppo_critic, batch,
                                                                  actor_optimizer, critic_optimizer, gamma=reward_trackers[0].gamma, epochs=1),
                                           bootstrap_source="actual final measured observation")
        state.update(status="COMPLETED", controls_completed=completed, tau_valid_observations=int(tau_valid_count),
                     force_valid_observations=int(force_valid_count), scheduler_decisions=decision_count.tolist(),
                     latest_decisions=latest_decisions, terminal_records=terminal_records, elapsed_s=time.monotonic() - started,
                     reward_totals=reward_totals.tolist(), reward_last_parts=reward_last_parts,
                     task_success_claim=False, scope="bounded Scheduled686 spatial-tau online adapter only")
    except BaseException as error:
        state.update(status="FAILED", error=repr(error), elapsed_s=time.monotonic() - started); raise
    finally:
        signal.alarm(0)
        if connection is not None: connection.close()
        listener.close(); a.socket.unlink(missing_ok=True); write(a.output / "manifest.json", state)


if __name__ == "__main__": main()

"""Bounded Stage 1 goal-conditioned low-level PPO continuation in DExplore."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time

import numpy as np


TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
OLD = ROOT / "src/task/consequence-evaluator"
sys.path[:0] = [str(TASK / "src"), str(OLD / "src"), str(OLD / "tools/run"),
               str(ROOT), str(ROOT / "third_party/DExplore/dexplore"),
               str(ROOT / "src/task/cm-interaction-oracle/src")]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_state(gpu):
    row = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu),
        "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits",
    ], text=True)
    utilization, used, total = (int(value.strip()) for value in row.strip().split(","))
    processes = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    return dict(utilization=utilization, used_mib=used, total_mib=total, processes=processes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("inputs", "reference", "geometry", "bc-checkpoint", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=313)
    parser.add_argument("--updates", type=int, default=24)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--seconds", type=int, default=480)
    parser.add_argument("--actor-lr", type=float, default=3e-5)
    parser.add_argument("--goal-env", type=int, default=0)
    parser.add_argument("--engineering-smoke", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists() or ROOT / "outputs/trajectory-policy" not in args.output.parents:
        raise ValueError("fresh trajectory-policy output required")
    if not 1 <= args.updates <= 24 or not 2 <= args.steps <= 128 or not 1 <= args.seconds <= 480:
        raise ValueError("bounded goal PPO budget required")
    if args.engineering_smoke and not (args.updates == 2 and args.steps == 2 and args.seed < 100):
        raise ValueError("engineering smoke is two updates/two controls/debug seed")
    episode_controls = 14 if args.engineering_smoke else 542
    before = gpu_state(args.gpu)
    if before["utilization"] > 10 or before["used_mib"] > 512 or before["total_mib"] - before["used_mib"] < 20000:
        raise ValueError("GPU is not idle: " + repr(before))
    initial_pids = {int(line.split(",")[0]) for line in before["processes"].splitlines() if line.strip()}
    scratch = ROOT / "tmp/trajectory-policy"
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), TMPDIR=str(scratch),
                      TORCH_EXTENSIONS_DIR=str(scratch / "torch-extensions"),
                      TRITON_CACHE_DIR=str(scratch / "triton"), OMP_NUM_THREADS="2",
                      PYTHONDONTWRITEBYTECODE="1")

    from isaacgym import gymtorch  # noqa: F401, must precede torch
    import torch
    from consequence_evaluator.data import sha as data_sha
    from consequence_evaluator.contracts import HAND_LINKS
    from consequence_evaluator.native_reset import install_reset_patch
    from consequence_evaluator.physical_geometry import PhysicalGeometry, poses
    from consequence_evaluator.value_geometry import TableSupport
    from trajectory_policy.goal_policy import (
        ACTIVE_ACTION, GoalConditionedActor, GoalConditionedValue, GoalConditionedPolicy,
        Standardizer, goal_features,
    )
    from trajectory_policy.goal_ppo import goal_reward, update as ppo_update
    from trajectory_policy.inputs import load_geometry
    from oracle_y_utility import align_native_reference_tables
    import evaluate as native
    from env.tasks.base_dexplore_task import DexploreTask

    if torch.__version__ != "2.4.1+cu121":
        raise ValueError("pinned Graspenv Torch 2.4.1+cu121 required")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    urdf = ROOT / "third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf"
    geometry_data, initial, _, geometry_hashes = load_geometry(args.geometry, args.reference, urdf)
    with args.reference.open("rb") as stream:
        reference_packet = pickle.load(stream)
    if reference_packet.get("engineering_only") is not True or reference_packet.get("training_allowed") is not False:
        raise ValueError("goal reference must remain goal-only engineering packet")
    goal_hand = torch.as_tensor(reference_packet["hand_keypoints"][:, args.goal_env], device=device)
    goal_obj = torch.as_tensor(reference_packet["object_pose"][:, args.goal_env], device=device)
    if goal_hand.shape != (543, 11, 3) or goal_obj.shape != (543, 4, 4):
        raise ValueError("543-frame goal reference required")
    del reference_packet

    cfg = json.loads(args.inputs.read_text())
    essential = [cfg["actor"], cfg["cfg_env"], cfg["cfg_train"]]
    essential += [str(ROOT / "third_party/DExplore/dexplore" / name) for name in (
        "evaluate.py", "env/tasks/base_dexplore_task.py", "env/tasks/dexplore_inspire.py")]
    essential += [path for path in cfg["input_sha256"] if path.endswith("interaction_hand_inspire.pt")]
    for path in essential:
        if data_sha(path) != cfg["input_sha256"][path]:
            raise ValueError("native input drift: " + path)
    payload = torch.load(args.bc_checkpoint, map_location=device, weights_only=False)
    if payload.get("schema") != GoalConditionedPolicy.schema:
        raise ValueError("goal-conditioned BC checkpoint required")
    standardizer = Standardizer(np.asarray(payload["standardizer"]["mean"], np.float32),
                                np.asarray(payload["standardizer"]["scale"], np.float32))
    mean = torch.as_tensor(standardizer.mean, device=device)
    scale = torch.as_tensor(standardizer.scale, device=device)
    actor = GoalConditionedActor(width=int(payload["width"])).to(device)
    actor.network.load_state_dict({key[len("network."):]: value for key, value in payload["state_dict"].items()
                                   if key.startswith("network.")}, strict=True)
    value_net = GoalConditionedValue(standardizer, width=int(payload["width"])).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=3e-4)
    actor_initial = {key: value.detach().clone() for key, value in actor.state_dict().items()}
    hashes = dict(geometry_hashes)
    for path in [args.inputs, args.reference, args.bc_checkpoint, urdf, Path(__file__),
                 TASK / "src/trajectory_policy/goal_policy.py", TASK / "src/trajectory_policy/goal_ppo.py"]:
        hashes[str(path.resolve())] = sha(path)
    hashes.update({str(Path(path).resolve()): cfg["input_sha256"][path] for path in essential})
    args.output.mkdir(parents=True)
    manifest = dict(
        schema="ref2dex.goal-conditioned-ppo-run.v1", status="RUNNING", task="trajectory-policy",
        run_id=args.output.name, git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        physical_gpu=args.gpu, gpu_before=before, seed=args.seed, input_sha256=hashes,
        updates=args.updates, steps=args.steps, envs=16, episode_controls=episode_controls,
        engineering_smoke=args.engineering_smoke, gamma=.99, gae_lambda=1.,
        actor_lr=args.actor_lr, value_lr=3e-4, target_joint_kl=.02,
        inference_contract="live q/dq/hand/object/velocity + previous native command + 8 future hand/object goals; no future q",
        reward_contract="measured proximity/lift/held/loss/control + current hand/object goal tracking; native reference reward ignored",
        goal_authorization="engineering_only=true, training_allowed=false packet used only for hand/object goals",
        claim="bounded Stage 1 goal-conditioned PPO Probe; no high-level H or Cm benefit",
    )
    write(args.output / "manifest.json", manifest)
    install_reset_patch()
    reset = DexploreTask._reset_ref_state_init

    def aligned(task, ids):
        align_native_reference_tables(task)
        return reset(task, ids)

    DexploreTask._reset_ref_state_init = aligned
    loader = native.torch_ext.load_checkpoint

    def load(path):
        value = loader(path)
        compiled = [key.startswith("_orig_mod.") for key in value["model"]]
        if any(compiled) and not all(compiled):
            raise ValueError("mixed compiled native actor keys")
        if all(compiled):
            value = dict(value, model={key[10:]: val for key, val in value["model"].items()})
        return value

    native.torch_ext.load_checkpoint = load
    started = time.monotonic()

    class Player(native.EvalPlayer):
        def run(self):
            task = self.env.task
            n = task.num_envs
            if (n != 16 or self.is_rnn or len(task.motion_file) != 1 or str(task.device) != "cpu"
                    or not task.gym.get_sim_params(task.sim).physx.use_gpu):
                raise ValueError("fixed sixteen-env CPU exchange/GPU PhysX contract required")
            task._enable_early_termination = False
            task._adaptive_kappa_enabled = False
            task._state_init = DexploreTask.StateInit.Start
            task._hybrid_init_prob = 1.
            torch.set_float32_matmul_precision("highest")
            manifest["matmul_precision"] = torch.get_float32_matmul_precision()
            self.model.eval()
            obs = self.env_reset(torch.arange(n, device=task.device))
            self.get_batch_size(obs["obs"], 1)
            if (task.start_times != 0).any() or not np.array_equal(initial["dof_position"], task._dof_pos[0].numpy()):
                raise ValueError("frame0 reset mismatch")
            names = task.gym.get_actor_rigid_body_names(task.envs[0], task.humanoid_handles[0])
            key_ids = [names.index(name) for name in HAND_LINKS]
            capture = pd_target = None
            original_pre, original_pd = task.pre_physics_step, task._action_to_pd_targets

            def pre(action):
                nonlocal capture
                capture = action.detach().clone().to(device)
                return original_pre(action)

            def pd(action):
                nonlocal pd_target
                pd_target = original_pd(action)
                return pd_target

            task.pre_physics_step, task._action_to_pd_targets = pre, pd
            physical = PhysicalGeometry(task, ROOT / "third_party/DExplore/dexplore/data/assets", distance_device=device)
            support = TableSupport(ROOT / "third_party/DExplore/dexplore/data/assets", task.device)
            previous_action = torch.zeros(n, 18, device=device)
            previous_held = torch.zeros(n, dtype=torch.bool, device=device)
            held_run = torch.zeros(n, dtype=torch.long, device=device)
            maximum_held = held_run.clone()

            def measure():
                bodies = task._rigid_body_state.view(n, -1, 13)
                return dict(q=task._dof_pos.to(device), dq=task._dof_vel.to(device),
                            hand=bodies[:, key_ids, :3].to(device), obj=poses(task._target_states).to(device),
                            velocity=task._target_states[:, 7:13].to(device))

            current = measure()
            initial_z = current["obj"][:, 2, 3].clone()
            episode_tick = episode_index = control_index = 0

            def observation():
                index = (episode_tick + 1 + torch.arange(8, device=device)).clamp_max(542)
                hand_goal = goal_hand[index][None].expand(n, -1, -1, -1)
                obj_goal = goal_obj[index][None].expand(n, -1, -1, -1)
                raw = goal_features(current["q"], current["dq"], current["hand"], current["obj"],
                                    current["velocity"], previous_action, hand_goal, obj_goal)
                return raw, (raw - mean) / scale

            def checkpoint(update_index):
                return dict(schema=actor.schema, width=int(payload["width"]),
                            actor_state_dict={key: value.detach().cpu().clone() for key, value in actor.state_dict().items()},
                            value_state_dict={key: value.detach().cpu().clone() for key, value in value_net.state_dict().items()},
                            standardizer=standardizer.as_dict(), update=update_index,
                            inference_contract=manifest["inference_contract"], input_sha256=hashes,
                            git_commit=manifest["git_commit"])

            records, episodes, metrics = [], [], []
            work_started = time.monotonic()
            for update_index in range(args.updates):
                batch = {key: [] for key in ("features", "raw_features", "latent", "logp", "mean", "std", "value", "reward", "done")}
                for _ in range(args.steps):
                    with torch.no_grad():
                        raw, encoded = observation()
                        distribution = actor.distribution(encoded)
                        latent = distribution.sample()
                        command = actor.action(latent)
                        logp = distribution.log_prob(latent)[:, list(ACTIVE_ACTION)].sum(-1)
                        v = value_net(raw)
                        before_state = {key: value.detach().cpu().numpy().copy() for key, value in current.items()}
                        obs, _, raw_done, _ = self.env_step(self.env, command.clone())
                        if capture is None or not torch.equal(capture, command):
                            raise ValueError("requested/applied command mismatch")
                        after = measure()
                        _, gap = physical.measure(task)
                        support_gap, footprint = support.measure(task, physical)
                        gap, support_gap, footprint = (value.to(device) for value in (gap, support_gap, footprint))
                        supported = footprint & (support_gap.abs() <= .02)
                        lift = after["obj"][:, 2, 3] - initial_z
                        clipped = (command.abs() > 1).any(-1)
                        goal_index = min(episode_tick + 1, 542)
                        r, held, terms = goal_reward(after["hand"], after["obj"],
                                                    goal_hand[goal_index][None].expand(n, -1, -1),
                                                    goal_obj[goal_index][None].expand(n, -1, -1),
                                                    gap, lift, supported, previous_held, clipped,
                                                    command - previous_action)
                        terminal = episode_tick + 1 == episode_controls
                        if raw_done.any() and not terminal:
                            raise ValueError("unexpected early native termination")
                        done = torch.full((n,), terminal, dtype=torch.bool, device=device)
                        for key, value in dict(features=encoded, raw_features=raw, latent=latent, logp=logp,
                                               mean=distribution.loc, std=distribution.scale,
                                               value=v, reward=r, done=done).items():
                            batch[key].append(value.detach())
                        record = dict(before_state, update=update_index, episode_tick=episode_tick,
                                      features=raw.cpu().numpy().copy(), encoded=encoded.cpu().numpy().copy(),
                                      latent=latent.cpu().numpy().copy(), mean=distribution.loc.cpu().numpy().copy(),
                                      std=distribution.scale.cpu().numpy().copy(), logp=logp.cpu().numpy().copy(),
                                      value=v.cpu().numpy().copy(), reward=r.cpu().numpy().copy(), done=done.cpu().numpy().copy(),
                                      action=command.cpu().numpy().copy(), applied=capture.cpu().numpy().copy(),
                                      pd_targets=pd_target.cpu().numpy().copy(), surface_gap=gap.cpu().numpy().copy(),
                                      support_gap=support_gap.cpu().numpy().copy(), table_footprint=footprint.cpu().numpy().copy(),
                                      held=held.cpu().numpy().copy(), **{"reward_" + key: value.cpu().numpy().copy() for key, value in terms.items()})
                        records.append(record)
                        held_run = torch.where(held, held_run + 1, torch.zeros_like(held_run))
                        maximum_held = torch.maximum(maximum_held, held_run)
                        previous_action, previous_held, current = command, held, after
                        control_index += 1
                        episode_tick += 1
                        if terminal:
                            episodes.append(dict(episode=episode_index, maximum_held=maximum_held.cpu().tolist(),
                                                 terminal_held=held.cpu().tolist(), control_end=control_index))
                            obs = self.env_reset(torch.arange(n, device=task.device))
                            current = measure()
                            if (task.start_times != 0).any():
                                raise ValueError("reset start frame drift")
                            initial_z = current["obj"][:, 2, 3].clone()
                            previous_action.zero_()
                            previous_held.zero_()
                            held_run.zero_()
                            maximum_held.zero_()
                            episode_tick = 0
                            episode_index += 1
                stacked = {key: torch.stack(value) for key, value in batch.items()}
                with torch.no_grad():
                    raw, _ = observation()
                    bootstrap = value_net(raw)
                with torch.enable_grad():
                    result = ppo_update(actor, value_net, stacked, bootstrap,
                                        actor_optimizer, value_optimizer, gae_lambda=1.)
                with torch.no_grad():
                    actor.log_std.clamp_(-5., -1.2)
                state = gpu_state(args.gpu)
                for line in state["processes"].splitlines():
                    if line.strip():
                        pid, memory = (int(value.strip()) for value in line.split(","))
                        if pid not in initial_pids | {os.getpid()} and memory > 512:
                            raise RuntimeError("foreign GPU compute detected")
                monitor = dict(update=update_index + 1, controls=control_index,
                               environment_interactions=control_index * n,
                               mean_reward=float(stacked["reward"].mean()), held_steps=int(stacked["reward"].shape[0]),
                               actor_steps=result["actor_steps"], joint_action_kl=result["joint_action_kl"],
                               behavior_logprob_error=result["behavior_logprob_error"],
                               policy_loss=result["policy_loss"], value_loss=result["value_loss"], actor_lr=result["actor_lr"],
                               elapsed_s=time.monotonic() - started,
                               eta_s=(time.monotonic() - work_started) / (update_index + 1) * (args.updates - update_index - 1),
                               gpu=state)
                metrics.append(monitor)
                print(json.dumps(monitor), flush=True)
                with (args.output / "monitor.jsonl").open("a") as stream:
                    stream.write(json.dumps(monitor) + "\n")
            if any(sha(path) != digest for path, digest in hashes.items()):
                raise ValueError("source/input drift")
            torch.save(checkpoint(args.updates), args.output / "final.pt")
            np.savez_compressed(args.output / "low.npz", **{
                key: np.stack([row[key] for row in records]) for key in records[0]
            })
            parameter_change = float(torch.sqrt(sum((value - actor_initial[key]).square().sum()
                                                     for key, value in actor.state_dict().items()
                                                     if key.startswith("network."))))
            report = dict(status="UNCLEAR", engineering_pass=True, actor_parameter_change=parameter_change,
                          updates=args.updates, controls=control_index, environment_interactions=control_index * n,
                          completed_episodes=episodes, mean_reward=float(np.mean([row["mean_reward"] for row in metrics])),
                          final_checkpoint_sha256=sha(args.output / "final.pt"),
                          claim="actual PPO updates and physical traces; independent frozen evaluation required")
            write(args.output / "result.json", report)
            manifest.update(status="COMPLETED", elapsed_s=time.monotonic() - started,
                            torch_peak_bytes=torch.cuda.max_memory_allocated(), final_checkpoint_sha256=report["final_checkpoint_sha256"])
            write(args.output / "manifest.json", manifest)

    native.EvalPlayer = Player
    sys.argv = [sys.argv[0], "--task", "Dexplore_Inspire", "--cfg_env", cfg["cfg_env"],
                "--cfg_train", cfg["cfg_train"], "--checkpoint", cfg["actor"], "--motion_file", cfg["motions"],
                "--headless", "--num_envs", "16", "--seed", str(args.seed), "--sim_device", "cuda:0", "--rl_device", "cuda:0",
                "--pipeline", "cpu", "--graphics_device_id", "0", "--num_threads", "1", "--disable-early-termination",
                "--output", str(args.output / "native-unused.json"), "--output_path", str(args.output / "native")]
    cwd = Path.cwd()

    def deadline(signum, frame):
        raise TimeoutError("goal-conditioned PPO exceeded declared wall-time cap")

    handler = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.seconds)
    try:
        os.chdir(ROOT / "third_party/DExplore")
        native.main()
    except BaseException as error:
        manifest.update(status="FAILED", elapsed_s=time.monotonic() - started, error=repr(error))
        write(args.output / "manifest.json", manifest)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, handler)
        os.chdir(cwd)


if __name__ == "__main__":
    main()

"""Execute a frozen Stage 1 goal-conditioned BC policy in DExplore."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
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


def gpu_state(gpu):
    row = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu),
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ], text=True)
    util, used, total = (int(value.strip()) for value in row.strip().split(","))
    processes = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu),
        "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    return dict(utilization=util, used_mib=used, total_mib=total,
                processes=processes)


def sha(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("inputs", "reference", "geometry", "bc-checkpoint", "bc-manifest", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--goal-env", type=int, default=0)
    parser.add_argument("--seed", type=int, default=312)
    return parser.parse_args()


def main():
    args = _parse()
    args.output = args.output.resolve()
    for path in (args.inputs, args.reference, args.geometry, args.bc_checkpoint, args.bc_manifest):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output.exists() or ROOT / "outputs/trajectory-policy" not in args.output.parents:
        raise ValueError("fresh trajectory-policy output required")

    before = gpu_state(args.gpu)
    if before["utilization"] > 10 or before["used_mib"] > 512 or before["total_mib"] - before["used_mib"] < 20000:
        raise ValueError("GPU is not idle: " + repr(before))
    initial_pids = {
        int(line.split(",")[0]) for line in before["processes"].splitlines() if line.strip()
    }
    scratch = ROOT / "tmp/trajectory-policy"
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        TMPDIR=str(scratch),
        TORCH_EXTENSIONS_DIR=str(scratch / "torch-extensions"),
        TRITON_CACHE_DIR=str(scratch / "triton"),
        OMP_NUM_THREADS="2",
        PYTHONDONTWRITEBYTECODE="1",
    )

    from isaacgym import gymtorch  # noqa: F401, must precede torch
    import torch
    from consequence_evaluator.data import sha as data_sha
    from consequence_evaluator.contracts import HAND_LINKS
    from consequence_evaluator.native_reset import install_reset_patch
    from consequence_evaluator.physical_geometry import PhysicalGeometry, poses
    from consequence_evaluator.value_geometry import TableSupport
    from consequence_evaluator.gate1 import episode_outcome
    from trajectory_policy.goal_policy import (GoalConditionedPolicy, Standardizer,
                                                goal_features)
    from trajectory_policy.inputs import load_geometry
    from oracle_y_utility import align_native_reference_tables
    import evaluate as native
    from env.tasks.base_dexplore_task import DexploreTask

    if torch.__version__ != "2.4.1+cu121":
        raise ValueError("pinned Graspenv Torch 2.4.1+cu121 required")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    urdf = ROOT / "third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf"

    cfg = json.loads(args.inputs.read_text())
    essential = [cfg["actor"], cfg["cfg_env"], cfg["cfg_train"]]
    essential += [str(ROOT / "third_party/DExplore/dexplore" / file) for file in (
        "evaluate.py", "env/tasks/base_dexplore_task.py", "env/tasks/dexplore_inspire.py")]
    essential += [path for path in cfg["input_sha256"] if path.endswith("interaction_hand_inspire.pt")]
    for path in essential:
        if data_sha(path) != cfg["input_sha256"][path]:
            raise ValueError("native input drift: " + path)

    geometry_data, initial, _, geometry_hashes = load_geometry(args.geometry, args.reference, urdf)
    bc_manifest = json.loads(args.bc_manifest.read_text())
    if bc_manifest.get("status") != "COMPLETED" or bc_manifest.get("schema") != "ref2dex.goal-conditioned-bc.v1":
        raise ValueError("completed goal-conditioned BC manifest required")
    bc_result = json.loads((args.bc_checkpoint.parent / "result.json").read_text())
    if bc_result.get("status") != "PROMISING" or not bc_result.get("offline_only"):
        raise ValueError("offline BC gate did not pass")
    if sha(args.bc_checkpoint) != bc_result.get("checkpoint_sha256"):
        raise ValueError("BC checkpoint hash mismatch")
    for path, digest in bc_manifest.get("input_sha256", {}).items():
        if sha(path) != digest:
            raise ValueError("BC source/input drift: " + path)
    payload = torch.load(args.bc_checkpoint, map_location=device, weights_only=False)
    if payload.get("schema") != GoalConditionedPolicy.schema:
        raise ValueError("BC checkpoint schema mismatch")
    if payload.get("inference_contract", "").find("no future q") < 0:
        raise ValueError("BC checkpoint does not declare the no-future-q contract")
    policy = GoalConditionedPolicy(width=int(payload["width"])).to(device)
    policy.load_state_dict(payload["state_dict"], strict=True)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    standardizer = Standardizer(
        np.asarray(payload["standardizer"]["mean"], np.float32),
        np.asarray(payload["standardizer"]["scale"], np.float32),
    )
    mean = torch.as_tensor(standardizer.mean, device=device)
    scale = torch.as_tensor(standardizer.scale, device=device)
    with args.reference.open("rb") as stream:
        import pickle
        goal_packet = pickle.load(stream)
    if goal_packet.get("engineering_only") is not True or goal_packet.get("training_allowed") is not False:
        raise ValueError("goal reference must remain goal-only engineering packet")
    if args.goal_env < 0 or args.goal_env >= int(goal_packet["hand_keypoints"].shape[1]):
        raise ValueError("goal-env outside reference packet")
    goal_hand = torch.as_tensor(goal_packet["hand_keypoints"][:, args.goal_env], device=device)
    goal_obj = torch.as_tensor(goal_packet["object_pose"][:, args.goal_env], device=device)
    if goal_hand.shape != (543, 11, 3) or goal_obj.shape != (543, 4, 4):
        raise ValueError("goal reference must contain 543 hand/object frames")
    if not torch.isfinite(goal_hand).all() or not torch.isfinite(goal_obj).all():
        raise ValueError("nonfinite goal reference")

    hashes = dict(geometry_hashes)
    hashes.update({str(Path(path).resolve()): digest for path, digest in cfg["input_sha256"].items()})
    hashes[str(args.inputs.resolve())] = sha(args.inputs)
    hashes[str(args.reference.resolve())] = sha(args.reference)
    hashes[str(args.geometry.resolve() / "manifest.json")] = sha(args.geometry / "manifest.json")
    hashes[str(args.geometry.resolve() / "geometry.npz")] = sha(args.geometry / "geometry.npz")
    hashes[str(args.bc_checkpoint.resolve())] = sha(args.bc_checkpoint)
    hashes[str(args.bc_manifest.resolve())] = sha(args.bc_manifest)
    hashes[str((args.bc_checkpoint.parent / "result.json").resolve())] = sha(args.bc_checkpoint.parent / "result.json")
    for path in (Path(__file__), TASK / "src/trajectory_policy/goal_policy.py",
                 TASK / "src/trajectory_policy/goal_policy_data.py"):
        hashes[str(path.resolve())] = sha(path)

    args.output.mkdir(parents=True)
    roles = ["goal_bc"] * 16
    manifest = dict(
        schema="ref2dex.goal-conditioned-execution.v1", status="RUNNING",
        task="trajectory-policy", run_id=args.output.name,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        physical_gpu=args.gpu, seed=args.seed, roles=roles, envs=16, controls=542, dt=1 / 30,
        goal_env=args.goal_env, input_sha256=hashes,
        inference_contract=payload["inference_contract"],
        action_contract="frozen MLP native action with active-coordinate mask; DExplore performs PD conversion",
        goal_authorization="engineering_only=true, training_allowed=false packet used only for hand/object goals",
        claim="single-motion Stage 1 physical execution Probe; no PPO, high-level H, or Cm claim",
        gate=dict(long_held_terminal_min=12, clipping_rate_max=.01),
        gpu_before=before,
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
            raise ValueError("mixed compiled keys")
        if all(compiled):
            value = dict(value, model={key[10:]: val for key, val in value["model"].items()})
        return value

    native.torch_ext.load_checkpoint = load
    started = time.monotonic()

    class Player(native.EvalPlayer):
        def run(self):
            task = self.env.task
            n = task.num_envs
            if (n != 16 or self.is_rnn or len(task.motion_file) != 1
                    or str(task.device) != "cpu" or not task.gym.get_sim_params(task.sim).physx.use_gpu):
                raise ValueError("sixteen-env single-motion CPU exchange/GPU PhysX required")
            task._enable_early_termination = False
            task._adaptive_kappa_enabled = False
            task._state_init = DexploreTask.StateInit.Start
            task._hybrid_init_prob = 1.
            self.model.eval()
            obs = self.env_reset(torch.arange(n, device=task.device))
            self.get_batch_size(obs["obs"], 1)
            if (task.start_times != 0).any() or not np.array_equal(initial["dof_position"], task._dof_pos[0].numpy()):
                raise ValueError("actual frame0 initialization mismatch")
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
            arrays = {key: [] for key in (
                "q", "dq", "hand", "obj", "velocity", "surface_gap", "support_gap",
                "table_footprint", "features", "action", "applied", "pd_targets", "clipped")}
            previous = torch.zeros(n, 18, device=device)

            def measure():
                bodies = task._rigid_body_state.view(n, -1, 13)
                return dict(q=task._dof_pos.to(device), dq=task._dof_vel.to(device),
                            hand=bodies[:, key_ids, :3].to(device),
                            obj=poses(task._target_states).to(device),
                            velocity=task._target_states[:, 7:13].to(device))

            def record():
                current = measure()
                _, gap = physical.measure(task)
                sup, foot = support.measure(task, physical)
                current.update(surface_gap=gap, support_gap=sup, table_footprint=foot)
                for key, value in current.items():
                    arrays[key].append(value.detach().cpu().numpy().copy())

            record()
            work_started = time.monotonic()
            for tick in range(542):
                with torch.no_grad():
                    current = measure()
                    index = (tick + 1 + torch.arange(8, device=device)).clamp_max(542)
                    hand_goal = goal_hand[index][None].expand(n, -1, -1, -1)
                    obj_goal = goal_obj[index][None].expand(n, -1, -1, -1)
                    raw = goal_features(current["q"], current["dq"], current["hand"], current["obj"],
                                        current["velocity"], previous, hand_goal, obj_goal)
                    encoded = (raw - mean) / scale
                    command = policy(encoded).clamp(-1, 1)
                    if not all(torch.isfinite(value).all() for value in (raw, encoded, command)):
                        raise FloatingPointError("nonfinite goal-policy execution chain")
                    obs, _, done, _ = self.env_step(self.env, command.clone())
                    if capture is None or not torch.equal(capture, command):
                        raise ValueError("requested/applied command mismatch")
                    values = dict(features=raw, action=command, applied=capture,
                                  pd_targets=pd_target,
                                  clipped=(command.abs() > 1).any(-1))
                    for key, value in values.items():
                        arrays[key].append(value.detach().cpu().numpy().copy())
                    previous = command
                    record()
                    if tick < 541 and done.any():
                        raise ValueError("unexpected early reset/termination")
                if tick == 0 or (tick + 1) % 128 == 0:
                    state = gpu_state(args.gpu)
                    for line in state["processes"].splitlines():
                        if line.strip():
                            pid, memory = (int(value.strip()) for value in line.split(","))
                            if pid not in initial_pids | {os.getpid()} and memory > 512:
                                raise RuntimeError("foreign GPU compute detected")
                    eta = (time.monotonic() - work_started) / (tick + 1) * (542 - tick - 1)
                    monitor = dict(control=tick + 1, eta_s=eta, gpu=state)
                    print(json.dumps(monitor), flush=True)
                    with (args.output / "monitor.jsonl").open("a") as stream:
                        stream.write(json.dumps(monitor) + "\n")
            packed = {key: np.stack(value) for key, value in arrays.items()}
            np.savez_compressed(args.output / "trajectory.npz", **packed)
            outcomes = []
            for env, role in enumerate(roles):
                outcome = episode_outcome(dict(
                    object_pose=packed["obj"][:, env], object_velocity=packed["velocity"][:, env],
                    surface_gap=packed["surface_gap"][:, env], support_gap=packed["support_gap"][:, env],
                    table_footprint=packed["table_footprint"][:, env]))
                terminal_held = (
                    packed["surface_gap"][-1, env].max() <= .01
                    and not (packed["table_footprint"][-1, env].any()
                             and abs(packed["support_gap"][-1, env].min()) <= .02)
                    and packed["obj"][-1, env, 2, 3] - packed["obj"][0, env, 2, 3] >= .03
                )
                outcome.update(role=role, env=env, terminal_held=bool(terminal_held),
                               clipping_count=int(packed["clipped"][:, env].sum()))
                outcomes.append(outcome)
            summary = dict(
                episodes=len(outcomes),
                long_held_terminal=sum(row["maximum_held_frames"] >= 433 and row["terminal_held"] for row in outcomes),
                hold45=sum(row["maximum_held_frames"] >= 45 for row in outcomes),
                terminal=sum(row["terminal_held"] for row in outcomes),
                median_held=float(np.median([row["maximum_held_frames"] for row in outcomes])),
                clipping_rate=float(sum(row["clipping_count"] for row in outcomes) / (len(outcomes) * 542)),
            )
            status = ("PROMISING" if summary["long_held_terminal"] >= 12 and summary["clipping_rate"] < .01
                      else "UNPROMISING")
            write(args.output / "result.json", dict(status=status, summary=summary, outcomes=outcomes,
                                                     audit_required=True, claim=manifest["claim"]))
            manifest.update(status="COMPLETED", elapsed_s=time.monotonic() - started,
                            torch_peak_bytes=torch.cuda.max_memory_allocated())
            write(args.output / "manifest.json", manifest)

    native.EvalPlayer = Player
    sys.argv = [sys.argv[0], "--task", "Dexplore_Inspire", "--cfg_env", cfg["cfg_env"],
                "--cfg_train", cfg["cfg_train"], "--checkpoint", cfg["actor"],
                "--motion_file", cfg["motions"], "--headless", "--num_envs", "16",
                "--seed", str(args.seed), "--sim_device", "cuda:0", "--rl_device", "cuda:0",
                "--pipeline", "cpu", "--graphics_device_id", "0", "--num_threads", "1",
                "--disable-early-termination", "--output", str(args.output / "native-unused.json"),
                "--output_path", str(args.output / "native")]
    cwd = Path.cwd()

    def deadline(signum, frame):
        raise TimeoutError("goal-conditioned execution exceeded 300s")

    handler = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(300)
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


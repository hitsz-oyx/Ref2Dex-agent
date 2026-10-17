"""Evaluate the frozen direct tau executor after learner-state DAgger."""

from __future__ import annotations

import argparse
import hashlib
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
RUNTIME = Path("/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train")
OFFICIAL = Path("/home2/wyy/oyx_ws/_external/dexplore_official_v120/checkpoint/inspire.pth")
OFFICIAL_SHA = "8f6823db752288f1bddd6d042981d33514e29dac5a68e58726e76215fea6d553"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def gpu_state(gpu):
    row = subprocess.check_output(["nvidia-smi", "-i", str(gpu),
        "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], text=True)
    utilization, memory = map(int, row.strip().split(","))
    rows = subprocess.check_output(["nvidia-smi", "-i", str(gpu),
        "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], text=True).splitlines()
    return dict(utilization=utilization, used_mib=memory,
                process_mib={int(r.split(",")[0]): int(r.split(",")[1]) for r in rows if r.strip()})


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True,
                   help="completed unseen-sequence teacher trajectory providing GT τ and G")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--idle-context-pid", type=int, action="append", default=[])
    p.add_argument("--seed", type=int, default=418)
    return p.parse_args()


def summarize(packed, roles, goal):
    sys.path.insert(0, str(OLD / "src"))
    from consequence_evaluator.gate1 import episode_outcome
    outcomes = []
    for env, role in enumerate(roles):
        row = episode_outcome(dict(object_pose=packed["obj"][:, env],
            object_velocity=packed["velocity"][:, env], surface_gap=packed["surface_gap"][:, env],
            support_gap=packed["support_gap"][:, env], table_footprint=packed["table_footprint"][:, env]))
        row.update(env=env, role=role, clipping_count=int(packed["clipped"][:, env].sum()))
        outcomes.append(row)
    summary = {}
    for role in sorted(set(roles)):
        ids = [i for i, x in enumerate(roles) if x == role]
        rows = [outcomes[i] for i in ids]
        summary[role] = dict(episodes=len(ids), hold45=sum(x["maximum_held_frames"] >= 45 for x in rows),
            success=sum(x["success"] for x in rows), median_held=float(np.median([x["maximum_held_frames"] for x in rows])),
            clipping_rate=float(packed["clipped"][:, ids].mean()),
            endpoint_position_error_m=float(np.linalg.norm(
                packed["obj"][-1, ids, :3, 3] - goal[ids, :3, 3], axis=-1).mean()))
    direct = summary["direct_tau_dagger"]
    teacher = summary["teacher"]
    status = "PROMISING" if (teacher["hold45"] >= 6 and teacher["success"] >= 6
        and direct["hold45"] >= 6 and direct["success"] >= 6
        and direct["clipping_rate"] < .01) else "UNPROMISING"
    return dict(status=status, summary=summary, outcomes=outcomes,
                claim="reference-frame-tau DAgger executor evaluation; no teacher intervention, PPO or C claim")


def main():
    args = parse()
    args.output = args.output.resolve()
    for path in (args.run_dir, args.source, args.checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output.exists() or ROOT / "outputs/trajectory-policy" not in args.output.parents:
        raise ValueError("fresh trajectory-policy output required")
    if Path(sys.prefix).resolve() != RUNTIME.resolve():
        raise RuntimeError("native probe must run in archived DExplore runtime")
    before = gpu_state(args.gpu)
    allowed = set(args.idle_context_pid)
    foreign = set(before["process_mib"]) - allowed
    if foreign or before["utilization"] > 20 or any(before["process_mib"][pid] > 512 for pid in allowed):
        raise RuntimeError("GPU occupied: " + repr(before))
    scratch = ROOT / "tmp/trajectory-policy/direct-tau-execution"
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), TMPDIR=str(scratch),
        TORCH_EXTENSIONS_DIR=str(ROOT / "tmp/consequence-official-generator/torch201-extensions"),
        OMP_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    sys.dont_write_bytecode = True
    native_root = ROOT / "third_party/DExplore/dexplore"
    sys.path[:0] = [str(Path("/home2/wyy/isaac-gym/isaacgym/python")), str(TASK / "src"),
        str(OLD / "src"), str(OLD / "tools/audit"), str(ROOT), str(native_root),
        str(ROOT / "src/task/CmResidual/tools")]
    sys.path.append("/home2/wyy/.local/lib/python3.8/site-packages")
    from isaacgym import gymtorch  # noqa: F401
    import dexplore_ddp_rank_bootstrap  # noqa: F401
    import torch
    import evaluate as native
    from learning import common_player
    from rl_games.common import player as base_player
    from probe_official_generator import install_legacy_player_compat, model_state
    from consequence_evaluator.native_reset import install_reset_patch
    from consequence_evaluator.physical_geometry import PhysicalGeometry, poses
    from consequence_evaluator.value_geometry import TableSupport
    from consequence_evaluator.contracts import HAND_LINKS
    from env.tasks.base_dexplore_task import DexploreTask
    from trajectory_policy.retargeter_data import features, reference_tau
    from trajectory_policy.tau_executor import TemporalTauExecutorPolicy, expand_active_action
    if torch.__version__ != "2.0.1+cu118" or sha(OFFICIAL) != OFFICIAL_SHA:
        raise RuntimeError("pinned DExplore runtime/checkpoint identity mismatch")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda:0")
    cfg = json.loads((args.run_dir / "config.json").read_text())
    with np.load(args.source / "trajectory.npz", allow_pickle=False) as f:
        source = {k: f[k].copy() for k in f.files}
    source_manifest = json.loads((args.source / "manifest.json").read_text())
    if source_manifest.get("status") != "COMPLETED" or source_manifest.get("training_allowed") is not True:
        raise ValueError("completed training-eligible source required")
    controls = int(source["action"].shape[0])
    if source["q"].shape[0] != controls + 1 or controls < 45:
        raise ValueError("source state/action alignment or hold length invalid")
    payload = torch.load(args.checkpoint, map_location=device)
    if payload.get("schema") != TemporalTauExecutorPolicy.schema:
        raise ValueError("direct executor checkpoint schema mismatch")
    policy = TemporalTauExecutorPolicy(horizon=int(payload["horizon"]), width=int(payload["width"])).to(device)
    policy.load_state_dict(payload["state_dict"], strict=True)
    policy.eval()
    for value in policy.parameters():
        value.requires_grad_(False)
    mean = np.asarray(payload["mean"], dtype=np.float32)
    scale = np.asarray(payload["scale"], dtype=np.float32)
    if mean.shape != (363,) or scale.shape != (363,):
        raise ValueError("unexpected direct executor standardizer")
    donor = np.arange(24, 32)
    roles = ["teacher"] * 8 + ["direct_tau_dagger"] * 8
    direct_ids = np.arange(8, 16)
    goal_np = source["goal"][donor]
    full_goal_np = np.concatenate((goal_np, goal_np), axis=0)
    files = [Path(__file__), args.run_dir / "config.json", args.source / "manifest.json",
             args.source / "trajectory.npz", args.checkpoint, TASK / "src/trajectory_policy/tau_executor.py",
             TASK / "src/trajectory_policy/retargeter_data.py", OFFICIAL,
             Path(cfg["cfg_env"]), native_root / "data/cfg/train/rlg/inspire.yaml"]
    hashes = {str(Path(p).resolve()): sha(p) for p in files}
    args.output.mkdir(parents=True)
    manifest = dict(schema="ref2dex.direct-tau-dagger-execution.v1", status="RUNNING",
        task="trajectory-policy", experiment_id="P-20261018-direct-tau-dagger", run_id=args.output.name,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        physical_gpu=args.gpu, seed=args.seed, roles=roles, controls=controls, dt=1/30,
        checkpoint=str(OFFICIAL), checkpoint_sha256=OFFICIAL_SHA, direct_checkpoint=str(args.checkpoint.resolve()),
        source=str(args.source.resolve()), source_envs=donor.tolist(), input_sha256=hashes,
        inference_contract=payload["inference_contract"], action_contract="direct active native action12; passive zeros; DExplore applies PD conversion once",
        evaluation="unseen corrected s1 sequence; frozen DAgger executor, teacher is calibration only", tau_contract=payload["tau_contract"], gpu_before=before)
    write(args.output / "manifest.json", manifest)
    install_legacy_player_compat(native, common_player, base_player)
    install_reset_patch()
    started = time.monotonic()

    class Player(native.EvalPlayer):
        def restore(self, filename):
            value = native.torch_ext.load_checkpoint(filename)
            self.model.load_state_dict(model_state(value["model"]), strict=True)
            if self.normalize_input:
                self.running_mean_std.load_state_dict(value["running_mean_std"], strict=True)

        def run(self):
            task = self.env.task
            n = task.num_envs
            if n != 16 or self.is_rnn or str(task.device) != "cuda:0":
                raise ValueError("fixed sixteen-env GPU-pipeline run required")
            task._state_init = DexploreTask.StateInit.Start
            task._hybrid_init_prob = 1.
            task._adaptive_kappa_enabled = False
            task._enable_early_termination = False
            obs = self.env_reset(torch.arange(n, device=task.device))
            self.get_batch_size(obs["obs"], 1)
            if task.start_times.ne(0).any():
                raise ValueError("real frame0 reset required")
            names = task.gym.get_actor_rigid_body_names(task.envs[0], task.humanoid_handles[0])
            key_ids = [names.index(name) for name in HAND_LINKS]
            capture = actual_pd = None
            original_pre, original_pd = task.pre_physics_step, task._action_to_pd_targets
            def pre(action):
                nonlocal capture
                capture = action.clone()
                return original_pre(action)
            def pd(action):
                nonlocal actual_pd
                actual_pd = original_pd(action).clone()
                return actual_pd
            task.pre_physics_step, task._action_to_pd_targets = pre, pd
            geometry = PhysicalGeometry(task, native_root / "data/assets", distance_device=task.device)
            support = TableSupport(native_root / "data/assets", task.device)
            logs = {k: [] for k in ("q", "dq", "hand", "obj", "velocity", "surface_gap", "support_gap",
                "table_footprint", "action", "applied", "pd_targets", "clipped", "state_features", "tau_features")}
            previous = torch.zeros(8, 12, device=device)
            def measure():
                bodies = task._rigid_body_state.view(n, -1, 13)
                return dict(q=task._dof_pos.clone(), dq=task._dof_vel.clone(), hand=bodies[:, key_ids, :3].clone(),
                    obj=poses(task._target_states), velocity=task._target_states[:, 7:13].clone())
            def record():
                hand, gap = geometry.measure(task)
                sup, foot = support.measure(task, geometry)
                cur = measure(); cur.update(hand=hand, surface_gap=gap, support_gap=sup, table_footprint=foot)
                for k, v in cur.items(): logs[k].append(v.cpu().numpy().copy())
                return cur
            current = record()
            work_started = time.monotonic()
            for tick in range(controls):
                with torch.no_grad():
                    raw = self.get_action(dict(obs, obs=obs["obs"]), True)[:8].clamp(-1, 1).clone()
                    command = raw.new_zeros((n, 18)); command[:8] = raw
                    index = np.minimum(tick + 1 + np.arange(int(payload["horizon"])), controls)
                    future = np.moveaxis(source["hand"][index][:, donor], 0, 1)
                    current_values = tuple(current[k][direct_ids].cpu().numpy()
                                           for k in ("q", "dq", "hand", "obj", "velocity"))
                    s, _ = features(*current_values, future)
                    tau = reference_tau(source["hand"][tick, donor], source["obj"][tick, donor], future)
                    flat = np.concatenate((s, tau.reshape(8, -1), previous.cpu().numpy()), axis=1)
                    off = 0; state = (flat[:, off:off+87] - mean[off:off+87]) / scale[off:off+87]; off += 87
                    plan = (flat[:, off:off+264] - mean[off:off+264]) / scale[off:off+264]; off += 264
                    prev = (flat[:, off:off+12] - mean[off:off+12]) / scale[off:off+12]
                    active = policy(torch.as_tensor(state, device=device),
                                    torch.as_tensor(plan.reshape(8, int(payload["horizon"]), 33), device=device),
                                    torch.as_tensor(prev, device=device))
                    command[direct_ids] = expand_active_action(active)
                    if not torch.isfinite(command).all():
                        raise FloatingPointError("nonfinite direct command")
                    obs, _, done, _ = self.env_step(self.env, command.clone())
                    if not isinstance(obs, dict): obs = {"obs": obs}
                    if capture is None or not torch.equal(capture, command):
                        raise ValueError("requested/applied action mismatch")
                    for k, v in dict(action=command, applied=capture, pd_targets=actual_pd,
                        clipped=torch.zeros(n, dtype=torch.bool, device=device), state_features=torch.as_tensor(s, device=device),
                        tau_features=torch.as_tensor(tau, device=device)).items():
                        logs[k].append(v.cpu().numpy().copy())
                    previous = active
                    current = record()
                    if tick < controls - 1 and done.any():
                        raise ValueError("early termination/reset invalidates closed-loop trace")
                    if tick == 0 or (tick + 1) % 128 == 0:
                        state_gpu = gpu_state(args.gpu)
                        foreign_now = set(state_gpu["process_mib"]) - allowed - {os.getpid()}
                        if foreign_now or any(state_gpu["process_mib"][pid] > 512 for pid in allowed if pid in state_gpu["process_mib"]):
                            raise RuntimeError("foreign GPU process: " + repr(state_gpu))
                        elapsed = time.monotonic() - work_started
                        monitor = dict(control=tick+1, elapsed_s=elapsed, eta_s=elapsed/(tick+1)*(controls-tick-1), gpu=state_gpu)
                        print(json.dumps(monitor), flush=True)
                        with (args.output / "monitor.jsonl").open("a") as stream: stream.write(json.dumps(monitor) + "\n")
            packed = {k: np.stack(v) for k, v in logs.items()}
            packed["goal"] = full_goal_np
            np.savez_compressed(args.output / "trajectory.npz", **packed)
            result = summarize(packed, roles, full_goal_np)
            write(args.output / "result.json", result)
            manifest.update(status="COMPLETED", trajectory_sha256=sha(args.output / "trajectory.npz"),
                elapsed_s=time.monotonic()-started, torch_peak_bytes=torch.cuda.max_memory_allocated())
            write(args.output / "manifest.json", manifest)
            print(json.dumps(result["summary"]), flush=True)

    native.EvalPlayer = Player
    sys.argv = [sys.argv[0], "--task", "Dexplore_Inspire", "--cfg_env", cfg["cfg_env"],
        "--cfg_train", str(native_root / "data/cfg/train/rlg/inspire.yaml"), "--motion_file", cfg["motion_root"],
        "--checkpoint", str(OFFICIAL), "--headless", "--num_envs", "16", "--seed", str(args.seed),
        "--sim_device", "cuda:0", "--rl_device", "cuda:0", "--pipeline", "gpu", "--graphics_device_id", "0",
        "--num_threads", "1", "--disable-early-termination", "--output", str(args.output / "unused.json"),
        "--output_path", str(args.output / "native")]
    cwd = Path.cwd()
    def deadline(signum, frame): raise TimeoutError("direct executor native run exceeded 300s")
    handler = signal.signal(signal.SIGALRM, deadline); signal.alarm(300)
    try:
        os.chdir(ROOT / "third_party/DExplore"); native.main()
    except BaseException as error:
        manifest.update(status="FAILED", error=repr(error), elapsed_s=time.monotonic()-started)
        write(args.output / "manifest.json", manifest); raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, handler); os.chdir(cwd)


if __name__ == "__main__": main()

"""Native three-row simulation: teacher, GT tau/R, autonomous shared-WAM/R."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / "src"))
from ptv3_shared_wam.paths import add_dependencies

add_dependencies()
from ptv3_shared_wam.execution import ActionBuffer
from trajectory_policy.wam_ipc import send, receive, peer_pid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--socket", type=Path, required=True)
    p.add_argument("--server-pid", type=int, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--controls", type=int, default=542)
    p.add_argument("--execution-steps", type=int, choices=(1, 4, 8), default=4)
    p.add_argument("--seed", type=int, default=292)
    p.add_argument("--deadline-seconds", type=int, default=600)
    p.add_argument("--engineering-only", action="store_true")
    a = p.parse_args()
    a.output, a.checkpoint = a.output.resolve(), a.checkpoint.resolve()
    runtime = ROOT.parents[1] / ".runtime_envs/dexplore_v120_train"
    if (Path(sys.prefix).resolve() != runtime.resolve() or a.output.exists()
            or ROOT / "outputs/ptv3-shared-wam" not in a.output.parents or not 1 <= a.controls <= 542):
        raise ValueError("pinned runtime, fresh Task output and bounded episode required")
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), TMPDIR=str(a.socket.parent),
                      TORCH_EXTENSIONS_DIR=str(ROOT / "tmp/consequence-official-generator/torch201-extensions"),
                      OMP_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    old = ROOT / "src/task/consequence-evaluator"
    native_root = ROOT / "third_party/DExplore/dexplore"
    trajectory = ROOT / "src/task/trajectory-policy"
    sys.path[:0] = ["/home2/wyy/isaac-gym/isaacgym/python", str(old / "src"),
        str(old / "tools/audit"), str(trajectory / "tools/run"), str(ROOT), str(native_root),
        str(ROOT / "src/task/CmResidual/tools")]
    sys.path.append("/home2/wyy/.local/lib/python3.8/site-packages")
    from isaacgym import gymtorch  # native runtime must import Isaac before torch
    import dexplore_ddp_rank_bootstrap
    import torch
    import evaluate as native
    from learning import common_player
    from rl_games.common import player as base_player
    from probe_official_generator import install_legacy_player_compat, model_state
    from consequence_evaluator.native_reset import install_reset_patch
    from consequence_evaluator.physical_geometry import PhysicalGeometry, poses
    from consequence_evaluator.value_geometry import TableSupport
    from consequence_evaluator.gate1 import episode_outcome
    from env.tasks.base_dexplore_task import DexploreTask
    from trajectory_policy.retargeter_data import features, ACTIVE
    from trajectory_policy.chunk_retargeter import ChunkRetargeter
    from trajectory_policy.pd_retargeter import pd_from_action
    from collect_competent_pd import OFFICIAL, OFFICIAL_SHA, sha, gpu_state
    from execute_wam_native_bridge import partial_grasp_outcome
    from ptv3_shared_wam.checkpoint import SCHEMA, atomic_json, load_checkpoint
    if torch.__version__ != "2.0.1+cu118" or sha(OFFICIAL) != OFFICIAL_SHA:
        raise ValueError("pinned native runtime and official calibration teacher required")
    torch.set_num_threads(2); torch.manual_seed(a.seed)
    packet = load_checkpoint(a.checkpoint)
    r = ChunkRetargeter(128).cuda().eval()
    r.load_state_dict({key[len("retargeter."):]: value for key, value in packet["model"].items()
                       if key.startswith("retargeter.")}, strict=True)
    source = Path(packet["provenance"]["source"])
    with np.load(source / "trajectory.npz", allow_pickle=False) as stream:
        oracle = stream["hand"][:, 24].copy()
    with np.load(packet["provenance"]["reference"], allow_pickle=False) as stream:
        reference = stream["obj"].copy()
    cfg = json.loads((ROOT / "outputs/consequence-evaluator/baseline-transfer-s3-20261007-r2/airplane_base/config.json").read_text())
    a.output.mkdir()
    connection = socket.socket(socket.AF_UNIX)
    connection.settimeout(120); connection.connect(str(a.socket))
    if peer_pid(connection) != a.server_pid:
        raise ValueError("native peer PID differs")
    send(connection, dict(schema=SCHEMA, pid=os.getpid()))
    hello = receive(connection)
    if hello["checkpoint"] != str(a.checkpoint) or hello["schema"] != SCHEMA:
        raise ValueError("server and native checkpoints differ")
    manifest = dict(schema=SCHEMA, status="RUNNING", controls=a.controls, seed=a.seed,
                    execution_steps=a.execution_steps, checkpoint=str(a.checkpoint),
                    roles=["teacher", "GT_tau_R", "shared_WAM_R"], training_allowed=False,
                    engineering_only=a.engineering_only,
                    no_teacher_intervention=True, native_controller=packet["config"]["native_controller"])
    atomic_json(a.output / "manifest.json", manifest)
    install_legacy_player_compat(native, common_player, base_player); install_reset_patch()
    logs, executed = {}, 0
    started = time.monotonic()

    class Player(native.EvalPlayer):
        def restore(self, filename):
            state = native.torch_ext.load_checkpoint(filename)
            self.model.load_state_dict(model_state(state["model"]), strict=True)
            if self.normalize_input:
                self.running_mean_std.load_state_dict(state["running_mean_std"], strict=True)
            if self._normalize_amp_input:
                self._amp_input_mean_std.load_state_dict(state["amp_input_mean_std"], strict=True)

        def run(self):
            nonlocal executed
            task = self.env.task
            if task.num_envs != 3 or self.is_rnn:
                raise ValueError("three non-recurrent calibration/actor rows required")
            task._state_init = DexploreTask.StateInit.Start
            task._hybrid_init_prob = 1.; task._adaptive_kappa_enabled = False
            task._enable_early_termination = False
            obs = self.env_reset(torch.arange(3, device=task.device))
            self.get_batch_size(obs["obs"], 1)
            if (task.start_times != 0).any():
                raise ValueError("frame0 reset required")
            geometry = PhysicalGeometry(task, native_root / "data/assets", distance_device=task.device)
            support = TableSupport(native_root / "data/assets", task.device)
            offset, scale = task._pd_action_offset.clone(), task._pd_action_scale.clone()
            controller = dict(offset=offset.tolist(), scale=scale.tolist())
            if controller != manifest["native_controller"] or hello["native_controller"] != controller:
                raise ValueError("native PD units differ from robot training")
            captured, actual_pd = None, None
            original_pre, original_pd = task.pre_physics_step, task._action_to_pd_targets
            def pre(action):
                nonlocal captured
                captured = action.clone()
                return original_pre(action)
            def pd(action):
                nonlocal actual_pd
                actual_pd = original_pd(action).clone()
                return actual_pd
            task.pre_physics_step, task._action_to_pd_targets = pre, pd
            keys = ("q", "dq", "hand", "obj", "velocity", "surface_gap", "support_gap", "table_footprint",
                    "applied", "intended", "pd_targets", "clipped", "query_tick", "chunk_index",
                    "wam_object_history", "wam_hand_history", "wam_goal", "R_state", "R_tau")
            logs.update({key: [] for key in keys})
            def record():
                hand, gap = geometry.measure(task)
                sup, foot = support.measure(task, geometry)
                current = dict(q=task._dof_pos.clone(), dq=task._dof_vel.clone(), hand=hand,
                    obj=poses(task._target_states), velocity=task._target_states[:, 7:13].clone(),
                    surface_gap=gap, support_gap=sup, table_footprint=foot)
                for key, value in current.items():
                    logs[key].append(value.cpu().numpy().copy())
                return current
            current = record()
            buffers = [ActionBuffer(a.execution_steps), ActionBuffer(a.execution_steps)]
            own_state = own_tau = None
            for tick in range(a.controls):
                with torch.no_grad():
                    # Teacher is used only on row0; actor inputs are its own row2.
                    padded = obs["obs"].repeat(22, 1)[:64]
                    action = self.get_action(dict(obs, obs=padded), True)[:3].clamp(-1, 1).clone()
                    intended = action.clone()
                    if buffers[1].needs_query:
                        past = np.maximum(tick + np.arange(-3, 1), 0)
                        oh = np.stack([logs["obj"][i][2] for i in past])
                        hh = np.stack([logs["hand"][i][2] for i in past])
                        goal = reference[min(tick + 24, len(reference) - 1)]
                        own_state, _ = features(*(current[k][2].cpu().numpy() for k in ("q", "dq", "hand", "obj", "velocity")),
                            np.broadcast_to(current["hand"][2].cpu().numpy(), (24, 11, 3)))
                        send(connection, dict(tick=tick, object_history=oh.tolist(), hand_history=hh.tolist(),
                            planned_object_goal=goal.tolist(), native_state=own_state.tolist()))
                        reply = receive(connection)
                        if reply["tick"] != tick:
                            raise ValueError("server query clock differs")
                        own_tau = np.asarray(reply["native_tau"], np.float32)
                        buffers[1].refill(reply["action_chunk"], tick)
                        future = oracle[np.minimum(tick + np.arange(1, 25), len(oracle) - 1)]
                        state, tau = features(*(current[k][1].cpu().numpy() for k in ("q", "dq", "hand", "obj", "velocity")), future)
                        gt = r(torch.as_tensor(state, device=task.device)[None], torch.as_tensor(tau, device=task.device)[None])[0]
                        buffers[0].refill(gt.cpu().numpy(), tick)
                    receipts = []
                    for env, buffer in enumerate(buffers, start=1):
                        raw, receipt = buffer.pop(); receipts.append(receipt)
                        intended[env].zero_()
                        intended[env, list(ACTIVE)] = torch.as_tensor(raw, device=task.device)
                        action[env] = intended[env].clamp(-1, 1)
                    expected = pd_from_action(action, current["q"], offset, scale)
                    obs, _, done, _ = self.env_step(self.env, action.clone()); executed += 1
                    if not isinstance(obs, dict):
                        obs = dict(obs=obs)
                    if captured is None or not torch.equal(captured, action) or (actual_pd - expected).abs().max() > 1e-6:
                        raise ValueError("actual native command/PD differs")
                    if tick < a.controls - 1 and done.any():
                        raise ValueError("unexpected early reset")
                    values = dict(applied=captured, intended=intended, pd_targets=actual_pd,
                        clipped=(intended - captured).abs().amax(-1) > 1e-6,
                        query_tick=np.array([r["query_tick"] for r in receipts]),
                        chunk_index=np.array([r["chunk_index"] for r in receipts]),
                        wam_object_history=oh, wam_hand_history=hh, wam_goal=goal, R_state=own_state, R_tau=own_tau)
                    for key, value in values.items():
                        logs[key].append(value.cpu().numpy().copy() if isinstance(value, torch.Tensor) else np.asarray(value).copy())
                    current = record()
                if tick == 0 or (tick + 1) % 64 == 0:
                    gpu = gpu_state(a.gpu)
                    if set(gpu["pids"]) - {os.getpid(), a.server_pid}:
                        raise RuntimeError("foreign process on simulation GPU")
                    elapsed = time.monotonic() - started
                    row = dict(control=tick + 1, elapsed_s=elapsed,
                               eta_s=elapsed / (tick + 1) * (a.controls - tick - 1), gpu=gpu)
                    print(json.dumps(row), flush=True)
                    with (a.output / "monitor.jsonl").open("a") as stream:
                        stream.write(json.dumps(row) + "\n")
            arrays = {key: np.stack(value) for key, value in logs.items()}
            np.savez_compressed(a.output / "trajectory.npz", **arrays)
            outcomes = []
            for env, role in enumerate(manifest["roles"]):
                values = dict(object_pose=arrays["obj"][:, env], object_velocity=arrays["velocity"][:, env],
                              surface_gap=arrays["surface_gap"][:, env], support_gap=arrays["support_gap"][:, env],
                              table_footprint=arrays["table_footprint"][:, env])
                result = episode_outcome(values) if a.controls == 542 else partial_grasp_outcome(values)
                result.update(role=role, clipping=float(arrays["clipped"][:, env].mean()))
                outcomes.append(result)
            atomic_json(a.output / "result.json", dict(status="COMPLETED", outcomes=outcomes,
                        scope="engineering_only" if a.engineering_only else "single-reset Probe; no Cm attribution"))
            manifest["status"] = "COMPLETED"
            atomic_json(a.output / "manifest.json", manifest)
    native.EvalPlayer = Player
    sys.argv = [sys.argv[0], "--task", "Dexplore_Inspire", "--cfg_env", cfg["cfg_env"],
        "--cfg_train", str(native_root / "data/cfg/train/rlg/inspire.yaml"), "--motion_file", cfg["motion_root"],
        "--checkpoint", str(OFFICIAL), "--headless", "--num_envs", "3", "--seed", str(a.seed),
        "--sim_device", "cuda:0", "--rl_device", "cuda:0", "--pipeline", "gpu", "--graphics_device_id", "0",
        "--num_threads", "1", "--disable-early-termination", "--output", str(a.output / "unused.json"),
        "--output_path", str(a.output / "native")]
    previous_cwd = Path.cwd()
    def deadline(*_):
        raise TimeoutError("bounded native execution deadline")
    signal.signal(signal.SIGALRM, deadline); signal.alarm(a.deadline_seconds)
    try:
        os.chdir(ROOT / "third_party/DExplore"); native.main()
        send(connection, dict(stop=True))
    except BaseException as error:
        manifest.update(status="FAILED", error=repr(error), executed_controls=executed)
        atomic_json(a.output / "manifest.json", manifest)
        if logs:
            np.savez_compressed(a.output / "partial.npz", **{k: np.stack(v) for k, v in logs.items() if v})
        raise
    finally:
        signal.alarm(0); os.chdir(previous_cwd); connection.close()


if __name__ == "__main__":
    main()

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
ROOT = Path(os.environ.get('REF2DEX_RUN_ROOT', TASK.parents[2])).resolve()
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
    p.add_argument("--sequence")
    p.add_argument("--retargeter", type=Path)
    p.add_argument("--model-version", choices=("v1", "v2"), default="v1")
    p.add_argument('--shared-actor', action='store_true')
    p.add_argument('--diagnostic-hybrid', action='store_true')
    a = p.parse_args()
    execution_task = TASK if a.model_version == "v1" else TASK.parent / "ptv3-shared-wam-v2"
    if a.model_version == "v2":
        sys.path.insert(0, str(execution_task / "src"))
    a.output, a.checkpoint = a.output.resolve(), a.checkpoint.resolve()
    runtime = Path("/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train")
    if (Path(sys.prefix).resolve() != runtime.resolve() or a.output.exists()
            or ROOT / "outputs" / execution_task.name not in a.output.parents or not 1 <= a.controls <= 542):
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
    if a.model_version == "v2":
        from ptv3_shared_wam_v2.native_execution import load_robot_checkpoint as load_checkpoint
        from ptv3_shared_wam_v2.paths import SCHEMA
    if a.shared_actor and (a.model_version != 'v2' or a.retargeter):
        raise ValueError('shared bridge requires v2 without an R override')
    if a.diagnostic_hybrid and (not a.shared_actor or a.execution_steps != 1):
        raise ValueError('Diagnostic teacher components require shared bridge and K=1')
    packet = load_checkpoint(a.checkpoint, allow_shared=True) if a.shared_actor else load_checkpoint(a.checkpoint)
    from ptv3_shared_wam.native_inputs import resolve_native_inputs, load_native_arrays
    selected = resolve_native_inputs(packet, a.sequence)
    arrays = load_native_arrays(selected)
    oracle, reference = arrays["oracle"], arrays["reference"]
    if selected["kind"] == "rl660" and a.controls + 24 > len(oracle):
        raise ValueError("RL660 requires complete future windows; no padded oracle tail")
    from ptv3_shared_wam.native_inputs import load_execution_retargeter
    if a.shared_actor:
        from types import SimpleNamespace
        r = SimpleNamespace(action_representation=packet['config']['action_representation'], first_only=False)
    else:
        r = load_execution_retargeter(packet, a.retargeter, "cuda")
    from ptv3_shared_wam.action_targets import freeze_chunk,consume_target
    representation=r.action_representation
    if r.first_only and a.execution_steps != 1:
        raise ValueError("R-only control candidate requires first-only execution")
    cfg = json.loads((ROOT / "outputs/consequence-evaluator/baseline-transfer-s3-20261007-r2/airplane_base/config.json").read_text())
    a.output.mkdir()
    native_cwd = ROOT / "third_party/DExplore"
    if selected["kind"] == "rl660":
        motion_root = a.output / "motions"
        motion_root.mkdir()
        (motion_root / a.sequence).symlink_to(selected["motion"], target_is_directory=True)
        cfg["motion_root"] = str(motion_root)
        cfg["cfg_env"] = str((native_cwd / cfg["cfg_env"]).resolve())
        native_cwd = a.output / "gym_runtime"
        assets = native_cwd / "dexplore/data/assets"
        assets.parent.mkdir(parents=True)
        full_assets = Path("/home2/wyy/oyx_ws/dexplore/dexplore/data/assets")
        if not full_assets.is_dir():
            raise FileNotFoundError(full_assets)
        assets.symlink_to(full_assets, target_is_directory=True)
    measurement_assets = (full_assets if selected["kind"] == "rl660" else native_root / "data/assets")
    connection = socket.socket(socket.AF_UNIX)
    connection.settimeout(120); connection.connect(str(a.socket))
    if peer_pid(connection) != a.server_pid:
        raise ValueError("native peer PID differs")
    send(connection, dict(schema=SCHEMA, pid=os.getpid()))
    hello = receive(connection)
    if hello["checkpoint"] != str(a.checkpoint) or hello["schema"] != SCHEMA:
        raise ValueError("server and native checkpoints differ")
    if hello.get("native_input") != selected:
        raise ValueError("server and native selected different motion records")
    override = str(a.retargeter.resolve()) if a.retargeter else None
    if hello.get("retargeter_override") != override:
        raise ValueError("server and native loaded different R candidates")
    if hello.get('action_representation')!=representation:
        raise ValueError('server/native action representation differs')
    if bool(hello.get('shared_actor')) != a.shared_actor:
        raise ValueError('server/native shared interface differs')
    manifest = dict(schema=SCHEMA, status="RUNNING", controls=a.controls, seed=a.seed,
                    execution_steps=a.execution_steps, checkpoint=str(a.checkpoint),
                    roles=["teacher", "GT_tau_R", "shared_WAM_R"], training_allowed=False,
                    engineering_only=a.engineering_only,
                    no_teacher_intervention=not a.diagnostic_hybrid, native_controller=packet["config"]["native_controller"])
    manifest["measurement_assets"] = str(measurement_assets.resolve())
    manifest["native_input"] = selected
    manifest["retargeter_override"] = override
    manifest['action_representation']=representation
    manifest['shared_actor'] = a.shared_actor
    if a.shared_actor:
        manifest['roles'] = ['teacher', 'shared_actor_repeat', 'shared_actor']
    manifest['diagnostic_hybrid'] = a.diagnostic_hybrid
    if a.diagnostic_hybrid:
        manifest['roles'] = ['teacher', 'teacher_wrist_student_fingers', 'student_wrist_teacher_fingers']
        manifest['scope'] = 'Oracle component attribution only; teacher-assisted rows are not student policy successes'
    atomic_json(a.output / "manifest.json", manifest)
    install_legacy_player_compat(native, common_player, base_player); install_reset_patch()
    from ptv3_shared_wam.native_inputs import install_selected_motion_contract
    install_selected_motion_contract(DexploreTask, selected)
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
            if selected["kind"] == "rl660":
                names = [Path(task.motion_file[int(i)]).name for i in task.data_id.cpu().tolist()]
                if names != [a.sequence] * 3 or any(n != selected["object"] for n in task.object_name):
                    raise ValueError("native reset motion/object differs from selected RL660 record")
            geometry = PhysicalGeometry(task, measurement_assets, distance_device=task.device)
            support = TableSupport(measurement_assets, task.device)
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
                    "wam_object_history", "wam_hand_history", "wam_goal", "R_state", "R_tau",
                    "R_output_chunk", "R_query_q")
            logs.update({key: [] for key in keys})
            if a.shared_actor:
                logs.update({key: [] for key in ('repeat_hand_history','repeat_object_history','repeat_R_state')})
            if a.diagnostic_hybrid:
                logs.update({key: [] for key in ('student_intended','teacher_actions','teacher_obs')})
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
            from ptv3_shared_wam.action_targets import action_dimension
            buffers = [ActionBuffer(a.execution_steps,action_dimension(representation)) for _ in range(2)]
            own_state = own_tau = None
            for tick in range(a.controls):
                with torch.no_grad():
                    # Students query their own measurements; optional component oracles are mixed below.
                    teacher_obs = obs['obs'][:3].clone() if a.diagnostic_hybrid else None
                    padded = obs["obs"].repeat(22, 1)[:64]
                    action = self.get_action(dict(obs, obs=padded), True)[:3].clamp(-1, 1).clone()
                    teacher_actions = action.clone() if a.diagnostic_hybrid else None
                    intended = action.clone()
                    if buffers[1].needs_query:
                        past = np.maximum(tick + np.arange(-3, 1), 0)
                        oh = np.stack([logs["obj"][i][2] for i in past])
                        hh = np.stack([logs["hand"][i][2] for i in past])
                        goal = reference[min(tick + 24, len(reference) - 1)]
                        own_state, _ = features(*(current[k][2].cpu().numpy() for k in ("q", "dq", "hand", "obj", "velocity")),
                            np.broadcast_to(current["hand"][2].cpu().numpy(), (24, 11, 3)))
                        request = dict(tick=tick, object_history=oh.tolist(), hand_history=hh.tolist(),
                            planned_object_goal=goal.tolist(), native_state=own_state.tolist())
                        if a.shared_actor:
                            repeat_state, _ = features(*(current[k][1].cpu().numpy() for k in ('q','dq','hand','obj','velocity')),
                                np.broadcast_to(current['hand'][1].cpu().numpy(), (24,11,3)))
                            repeat_oh = np.stack([logs['obj'][i][1] for i in past])
                            repeat_hh = np.stack([logs['hand'][i][1] for i in past])
                            request['repeat'] = dict(object_history=repeat_oh.tolist(),
                                hand_history=repeat_hh.tolist(),
                                planned_object_goal=goal.tolist(), native_state=repeat_state.tolist())
                        send(connection, request)
                        reply = receive(connection)
                        if reply["tick"] != tick:
                            raise ValueError("server query clock differs")
                        if reply.get('action_representation')!=representation:
                            raise ValueError('query action units differ from loaded R')
                        own_tau = np.asarray(reply["native_tau"], np.float32)
                        raw_chunks=[None,torch.tensor(reply['action_chunk'],device=task.device)[None]]
                        query_q=current['q'][1:].clone()
                        buffers[1].refill(freeze_chunk(raw_chunks[1],query_q[1:2],representation)[0].cpu().numpy(), tick)
                        if a.shared_actor:
                            gt = torch.tensor(reply['repeat_action_chunk'], device=task.device)
                        else:
                            future = oracle[np.minimum(tick + np.arange(1, 25), len(oracle) - 1)]
                            state, tau = features(*(current[k][1].cpu().numpy() for k in ('q','dq','hand','obj','velocity')), future)
                            gt = r(torch.as_tensor(state, device=task.device)[None], torch.as_tensor(tau, device=task.device)[None])[0]
                        raw_chunks[0]=gt[None]
                        buffers[0].refill(freeze_chunk(gt[None],query_q[:1],representation)[0].cpu().numpy(), tick)
                    receipts = []
                    for env, buffer in enumerate(buffers, start=1):
                        raw, receipt = buffer.pop(); receipts.append(receipt)
                        intended[env],action[env],_=consume_target(torch.as_tensor(raw,device=task.device),
                              current['q'][env],representation,packet['config']['native_controller'])
                    if a.diagnostic_hybrid:
                        from ptv3_shared_wam_v2.native_execution import mix_teacher_components
                        student_intended = intended.clone()
                        intended = mix_teacher_components(student_intended, teacher_actions)
                        action = intended.clamp(-1,1)
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
                    values.update(R_output_chunk=torch.cat(raw_chunks),R_query_q=query_q)
                    if a.shared_actor:
                        values.update(repeat_hand_history=repeat_hh, repeat_object_history=repeat_oh,
                                      repeat_R_state=repeat_state)
                    if a.diagnostic_hybrid:
                        values.update(student_intended=student_intended,teacher_actions=teacher_actions,
                                      teacher_obs=teacher_obs)
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
        os.chdir(native_cwd); native.main()
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

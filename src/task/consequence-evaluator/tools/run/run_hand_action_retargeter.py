"""Execute a frozen full-action retargeter with privileged GT hand futures."""

import argparse
import hashlib
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
sys.path[:0] = [str(TASK / "src"), str(ROOT), str(ROOT / "third_party/DExplore/dexplore"),
               str(ROOT / "src/task/cm-interaction-oracle/src")]

from consequence_evaluator.contracts import is_within
from consequence_evaluator.gate1 import episode_outcome

HORIZON = 24
SCHEMA = "ref2dex.hand-action-retargeter.v1"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_source(path):
    with Path(path).open("rb") as stream:
        packet = pickle.load(stream)
    if (packet.get("engineering_only") is not True or packet.get("seed") != 282
            or packet.get("role_names", [None])[0] != "reactive_teacher"
            or packet["hand_keypoints"].shape != (543, 4, 11, 3)
            or packet["dof_position"].shape != (543, 4, 18)
            or packet["dof_velocity"].shape != (543, 4, 18)
            or packet["actions"].shape != (542, 4, 18)
            or packet["done"][:-1].any()):
        raise ValueError("complete held-out teacher source required")
    if packet["role_outcomes"]["reactive_teacher"]["maximum_held_frames"] < 45:
        raise ValueError("source teacher does not pass the sustained-hold behavior screen")
    return packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=282)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--seconds", type=int, default=300)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not is_within(output, ROOT / "outputs/consequence-evaluator"):
        raise ValueError("fresh task-owned execution output required")
    if args.envs != 4 or not 1 <= args.seconds <= 900:
        raise ValueError("the fixed four-role upper-bound contract is required")
    occupied = subprocess.check_output([
        "nvidia-smi", "-i", str(args.gpu), "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True).strip()
    if occupied:
        raise RuntimeError("GPU occupied: " + occupied)
    cfg = json.loads(args.inputs.read_text())
    source = load_source(args.source.resolve())
    scratch = ROOT / "tmp/hand-action-retarget"
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), TMPDIR=str(scratch),
                      TORCH_EXTENSIONS_DIR=str(scratch / "torch-extensions"),
                      TRITON_CACHE_DIR=str(scratch / "triton"), PYTHONDONTWRITEBYTECODE="1",
                      OMP_NUM_THREADS="2")
    from isaacgym import gymtorch  # noqa: F401  # imports before torch/native modules
    import torch
    from consequence_evaluator.hand_action_retargeter import (
        HandActionRetargeter, Standardizer, trajectory_input)
    if torch.__version__ != "2.4.1+cu121":
        raise ValueError("pinned graspenv runtime required")
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA or payload.get("manifest", {}).get("target_contract") != (
            "captured full native action A_t:t+24, no actor action at inference"):
        raise ValueError("full-action retarget checkpoint contract mismatch")
    model = HandActionRetargeter(payload["width"]).to(device).eval()
    model.load_state_dict(payload["state_dict"], strict=True)
    stats = {key: Standardizer(**value) for key, value in payload["statistics"].items()}
    source_hand = np.asarray(source["hand_keypoints"][:, 0], dtype="float32")
    source_sha = sha(args.source.resolve()); checkpoint_sha = sha(args.checkpoint.resolve())

    import evaluate as native
    from env.tasks.base_dexplore_task import DexploreTask
    from consequence_evaluator.native_reset import install_reset_patch
    from consequence_evaluator.physical_geometry import PhysicalGeometry, poses
    from consequence_evaluator.value_geometry import TableSupport
    from oracle_y_utility import align_native_reference_tables
    from src.task.CmResidual.paired_evaluation import fingerprint

    torch.set_num_threads(2); install_reset_patch()
    original_reset = DexploreTask._reset_ref_state_init
    def aligned_reset(task, ids):
        align_native_reference_tables(task)
        return original_reset(task, ids)
    DexploreTask._reset_ref_state_init = aligned_reset
    loader = native.torch_ext.load_checkpoint
    def load(path):
        state = loader(path); keys = list(state["model"])
        compiled = [key.startswith("_orig_mod.") for key in keys]
        if any(compiled) and not all(compiled):
            raise ValueError("mixed compiled actor keys")
        if all(compiled):
            state = dict(state, model={key[10:]: value for key, value in state["model"].items()})
        return state
    native.torch_ext.load_checkpoint = load

    frozen = {str(args.inputs.resolve()): sha(args.inputs.resolve()),
              str(args.source.resolve()): source_sha,
              str(args.checkpoint.resolve()): checkpoint_sha,
              str(Path(__file__).resolve()): sha(Path(__file__).resolve()),
              str((TASK / "src/consequence_evaluator/hand_action_retargeter.py").resolve()):
                  sha(TASK / "src/consequence_evaluator/hand_action_retargeter.py")}
    frozen.update(cfg["input_sha256"])
    if any(sha(path) != expected for path, expected in frozen.items()):
        raise ValueError("frozen retarget execution input drift")
    output.mkdir(parents=True)
    started = time.monotonic()
    manifest = dict(schema="ref2dex.hand-action-retarget-behavior.v1", status="RUNNING",
                    run_id=output.name, task="consequence-evaluator", route="ref7_2",
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    seed=args.seed, num_envs=args.envs, physical_gpu=args.gpu, budget_s=args.seconds,
                    source=str(args.source.resolve()), source_sha256=source_sha,
                    checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_sha,
                    input_sha256=frozen, query_period=HORIZON,
                    roles=["reactive_teacher", "retarget_gt_hand", "retarget_gt_hand_repeat",
                           "retarget_gt_hand_repeat2"],
                    privileged_input="frozen source teacher future hand geometry; current q,dq and live hand",
                    claim="GT-hand execution upper-bound Probe; no deployability or Cm claim")
    write(output / "manifest.json", manifest)

    class Player(native.EvalPlayer):
        @torch.no_grad()
        def run(self):
            task = self.env.task; n = task.num_envs
            if n != args.envs or self.is_rnn or len(task.motion_file) != 1:
                raise ValueError("fixed four-env feedforward full-reference contract required")
            task._enable_early_termination = False; task._adaptive_kappa_enabled = False
            task._state_init = DexploreTask.StateInit.Start; task._hybrid_init_prob = 1.
            self.model.eval(); ids = torch.arange(n, device=task.device)
            obs = self.env_reset(ids); self.get_batch_size(obs["obs"], 1)
            geometry = PhysicalGeometry(task, ROOT / "third_party/DExplore/dexplore/data/assets",
                                        distance_device=self.device)
            # The native route uses GPU PhysX with CPU tensor exchange; table
            # support must follow the task tensor device, while dense hand
            # geometry may still use the player's CUDA device internally.
            support = TableSupport(ROOT / "third_party/DExplore/dexplore/data/assets", task.device)
            active = np.ones(n, dtype=bool); lengths = np.zeros(n, dtype=np.int64)
            actions = []; clips = []; queries = []; chunks = []; inputs = []
            logs = {key: [] for key in ("object_pose", "hand_keypoints", "surface_gap", "support_gap",
                                        "table_footprint", "object_velocity", "dof_position", "dof_velocity", "done")}
            predicted = np.zeros((n, HORIZON, 18), dtype=np.float32)
            capture = None; original_pre = task.pre_physics_step
            def pre(value):
                nonlocal capture
                capture = value.detach().cpu().numpy().copy()
                return original_pre(value)
            task.pre_physics_step = pre

            def measure():
                hand, gap = geometry.measure(task); sup, foot = support.measure(task, geometry)
                value = dict(object_pose=poses(task._target_states), hand_keypoints=hand, surface_gap=gap,
                             support_gap=sup, table_footprint=foot,
                             object_velocity=task._target_states[:, 7:13],
                             dof_position=task._dof_pos, dof_velocity=task._dof_vel)
                for key, item in value.items(): logs[key].append(item.detach().cpu().numpy().copy())
                return value

            measured = measure()
            for tick in range(542):
                if time.monotonic() - started > args.seconds:
                    raise TimeoutError("retarget execution deadline")
                base = self.get_action(obs, True).clamp(-1, 1).to(self.device)
                if tick % HORIZON == 0:
                    future = source_hand[np.minimum(tick + np.arange(1, HORIZON + 1), 542)]
                    current = measured["hand_keypoints"].detach().cpu().numpy()
                    q = task._dof_pos.detach().cpu().numpy()
                    dq = task._dof_vel.detach().cpu().numpy()
                    hand = trajectory_input(current, np.broadcast_to(future, (n, HORIZON, 11, 3)))
                    state = np.concatenate((q, dq), axis=-1).astype("float32")
                    with torch.no_grad():
                        output = model(
                            torch.as_tensor(stats["hand"].encode(hand), device=self.device),
                            torch.as_tensor(stats["state"].encode(state), device=self.device))
                        predicted = stats["action"].decode(output.cpu().numpy())
                    queries.append(tick); chunks.append(predicted.copy())
                    inputs.append(dict(current_hand=current.copy(), future_hand=future.copy(), state=state.copy()))
                offset = tick % HORIZON
                command = base.detach().cpu().numpy().copy()
                command[1:] = predicted[1:, offset]
                intended = command.copy(); command = np.clip(command, -1, 1); command[~active] = 0
                obs, _, done, info = self.env_step(self.env, torch.as_tensor(command, device=self.device))
                if not isinstance(obs, dict): obs = {"obs": obs}
                if capture is None or not np.allclose(capture, command, atol=1e-7, rtol=0):
                    raise ValueError("native executed action differs from requested action")
                actions.append(capture.copy()); clips.append((np.abs(intended - command) > 1e-7).any(-1))
                logs["done"].append(done.detach().cpu().numpy().reshape(-1).astype(bool))
                measured = measure(); self._post_step(info)
                lengths[active] += 1; active &= ~done.detach().cpu().numpy().reshape(-1).astype(bool)
                if not active.any(): break
            if active.any():
                raise ValueError("retarget episode ended before 542 steps")
            arrays = {key: np.stack(value) for key, value in logs.items()}
            arrays.update(actions=np.asarray(actions), clipped=np.asarray(clips), lengths=lengths)
            outcomes = {}
            for env, name in enumerate(manifest["roles"]):
                packet = {key: arrays[key][:, env] for key in ("object_pose", "surface_gap", "support_gap",
                                                                  "table_footprint", "object_velocity")}
                outcomes[name] = episode_outcome(packet)
            hand_error = np.sqrt(np.mean((arrays["hand_keypoints"] - source_hand[:, None]) ** 2, axis=(0, 2, 3))) * 1000
            teacher = outcomes["reactive_teacher"]["maximum_held_frames"]
            gate = {name: bool(teacher >= 45 and outcomes[name]["maximum_held_frames"] >= .9 * teacher
                               and hand_error[index] < 40.) for index, name in enumerate(manifest["roles"])}
            manifest.update(status="COMPLETED", elapsed_s=time.monotonic() - started,
                            action_max_abs_error=float(np.max(np.abs(arrays["actions"][0] - arrays["actions"][0]))),
                            clipping_counts=np.sum(arrays["clipped"], axis=0).tolist(),
                            outcomes=outcomes, hand_coordinate_rmse_mm=hand_error.tolist(), gate=gate,
                            query_ticks=queries, teacher_source_outcome=source["role_outcomes"]["reactive_teacher"])
            with (output / "execution.pkl").open("wb") as stream:
                pickle.dump(dict(schema=manifest["schema"], manifest=manifest, arrays=arrays,
                                 source_hand=source_hand, query_ticks=np.asarray(queries),
                                 predicted_chunks=np.asarray(chunks), retarget_inputs=inputs), stream, protocol=4)
            write(output / "manifest.json", manifest)
            write(output / "result.json", dict(status="PROMISING" if all(gate.values()) else "UNCLEAR",
                                                outcomes=outcomes, hand_coordinate_rmse_mm=hand_error.tolist(),
                                                gate=gate, clipping_counts=np.sum(arrays["clipped"], axis=0).tolist(),
                                                claim=manifest["claim"]))

    native.EvalPlayer = Player
    argv = ["--task", "Dexplore_Inspire", "--cfg_env", cfg["cfg_env"], "--cfg_train", cfg["cfg_train"],
            "--checkpoint", cfg["actor"], "--motion_file", cfg["motions"], "--headless", "--num_envs", str(args.envs),
            "--seed", str(args.seed), "--sim_device", "cuda:0", "--rl_device", "cuda:0", "--pipeline", "cpu",
            "--graphics_device_id", "0", "--num_threads", "1", "--disable-early-termination",
            "--output", str(output / "native-unused.json"), "--output_path", str(output / "native")]
    sys.argv = [sys.argv[0], *argv]
    cwd = Path.cwd()
    def deadline(signum, frame):
        raise KeyboardInterrupt("owned retarget execution deadline")
    previous = signal.signal(signal.SIGALRM, deadline); signal.alarm(args.seconds)
    try:
        os.chdir(ROOT / "third_party/DExplore"); native.main()
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous); os.chdir(cwd)


if __name__ == "__main__":
    main()

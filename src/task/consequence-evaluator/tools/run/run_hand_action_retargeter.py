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
SOURCE_BACKEND = dict(name="cpu_pipeline", pipeline="cpu", sim_device="cuda:0",
                      tensor_device="cuda:0", physx_use_gpu=True)
SOURCE_ACTOR_EXECUTION = dict(layout="environment_rows_direct", copies=1)


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
    if packet.get("source_backend") != SOURCE_BACKEND:
        raise ValueError("source backend does not match the fixed CPU-pipeline contract")
    actor_execution = packet.get("replay_identity", {}).get("actor_execution", {})
    if (actor_execution.get("layout") != SOURCE_ACTOR_EXECUTION["layout"]
            or actor_execution.get("copies") != SOURCE_ACTOR_EXECUTION["copies"]
            or actor_execution.get("total_rows") != 4):
        raise ValueError("source actor execution does not match direct environment rows")
    if packet["role_outcomes"]["reactive_teacher"]["maximum_held_frames"] < 45:
        raise ValueError("source teacher does not pass the sustained-hold behavior screen")
    return packet


def runtime_source_identity(task, env_count):
    """Read the native runtime identity used by the source-matched contract."""
    params = task.gym.get_sim_params(task.sim)
    backend = dict(SOURCE_BACKEND, physx_use_gpu=bool(params.physx.use_gpu),
                   tensor_device=str(task.device))
    actor_execution = dict(SOURCE_ACTOR_EXECUTION, total_rows=int(env_count))
    return backend, actor_execution


def anchored_future_batch(source_hands, source_hand, live_current, tick,
                          source_per_env, horizon=HORIZON):
    """Anchor a source displacement future to each live environment.

    Broadcast mode carries one env-0 future with shape ``[T,11,3]``; matched
    mode carries one future per env with shape ``[N,T,11,3]``.  Keeping the
    two branches explicit prevents an accidental extra batch axis in the
    broadcast path.
    """
    source_hands = np.asarray(source_hands)
    source_hand = np.asarray(source_hand)
    live_current = np.asarray(live_current)
    indices = np.minimum(int(tick) + np.arange(1, int(horizon) + 1), 542)
    if source_per_env:
        source_future = np.transpose(source_hands[indices], (1, 0, 2, 3))
        source_current = source_hands[int(tick)]
        source_delta = source_future - source_current[:, None]
        future = live_current[:, None] + source_delta
    else:
        source_future = source_hand[indices]
        source_current = source_hand[int(tick)]
        source_delta = source_future - source_current
        future = live_current[:, None] + source_delta[None]
    if future.shape != (live_current.shape[0], int(horizon), 11, 3):
        raise ValueError("anchored future batch shape mismatch")
    return source_future, future


def replace_with_teacher_fingers(command, teacher_action, first_env=1):
    """Broadcast the current live env-0 teacher finger command to test envs.

    This is an attribution-only control: the caller must keep the teacher
    action from the same dispatch tick, and the caller remains responsible for
    replacing any other action coordinates (for example, the fixed wrist).
    """
    value = np.asarray(command, dtype="float32")
    teacher = np.asarray(teacher_action, dtype="float32")
    if value.ndim != 2 or value.shape[1] != 18:
        raise ValueError("command must have shape [N,18]")
    if teacher.shape != (18,):
        raise ValueError("teacher_action must have shape [18]")
    if not 0 <= int(first_env) < value.shape[0]:
        raise ValueError("first_env is outside command batch")
    result = value.copy()
    result[int(first_env):, 6:] = teacher[6:]
    if not np.isfinite(result).all():
        raise ValueError("nonfinite teacher-finger command")
    return result


def replace_with_teacher_action(command, teacher_action, first_env=1):
    """Broadcast the current live env-0 teacher action to test envs."""
    value = np.asarray(command, dtype="float32")
    teacher = np.asarray(teacher_action, dtype="float32")
    if value.ndim != 2 or value.shape[1] != 18:
        raise ValueError("command must have shape [N,18]")
    if teacher.shape != (18,):
        raise ValueError("teacher_action must have shape [18]")
    if not 0 <= int(first_env) < value.shape[0]:
        raise ValueError("first_env is outside command batch")
    result = value.copy()
    result[int(first_env):] = teacher
    if not np.isfinite(result).all():
        raise ValueError("nonfinite teacher action")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=282)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--query-period", type=int, default=HORIZON)
    parser.add_argument("--fixed-wrist-pd-inverse", type=Path,
                        help="freeze train-only analytic wrist decoder and retain learned finger actions")
    parser.add_argument("--teacher-finger-reference", action="store_true",
                        help="attribution audit: replace test-env fingers with live env-0 teacher fingers")
    parser.add_argument("--teacher-action-reference", action="store_true",
                        help="attribution audit: replace test-env actions with the live env-0 teacher action")
    parser.add_argument("--source-per-env", action="store_true",
                        help="diagnostic: use each source env's own teacher future")
    parser.add_argument("--seconds", type=int, default=300)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not is_within(output, ROOT / "outputs/consequence-evaluator"):
        raise ValueError("fresh task-owned execution output required")
    if args.envs != 4 or not 1 <= args.seconds <= 900 or not 1 <= args.query_period <= HORIZON:
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
        CONTEXT_SCHEMA, HandActionRetargeter, ContextHandActionRetargeter,
        Standardizer, chunk_offset, hand_object_context, trajectory_input)
    if args.fixed_wrist_pd_inverse is not None:
        from consequence_evaluator.fixed_wrist_decoder import (
            load_pd_statistics, recover_wrist_sequence,
            replace_wrist_action, root_template)
    if torch.__version__ != "2.4.1+cu121":
        raise ValueError("pinned graspenv runtime required")
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_schema = payload.get("schema")
    if checkpoint_schema not in (SCHEMA, CONTEXT_SCHEMA) or payload.get("manifest", {}).get("target_contract") != (
            "captured full native action A_t:t+24, no actor action at inference"):
        raise ValueError("full-action retarget checkpoint contract mismatch")
    if args.fixed_wrist_pd_inverse is not None and checkpoint_schema != CONTEXT_SCHEMA:
        raise ValueError("fixed-wrist Spike requires the contextual v2 checkpoint")
    if args.teacher_finger_reference and args.fixed_wrist_pd_inverse is None:
        raise ValueError("teacher-finger reference requires the fixed-wrist decoder")
    if args.teacher_finger_reference and args.teacher_action_reference:
        raise ValueError("teacher-finger and full teacher-action references are exclusive")
    if args.teacher_action_reference and args.fixed_wrist_pd_inverse is not None:
        raise ValueError("full teacher-action reference cannot also replace the wrist")
    fixed_wrist_coefficients = fixed_wrist_metadata = None
    if args.fixed_wrist_pd_inverse is not None:
        fixed_wrist_coefficients, fixed_wrist_metadata = load_pd_statistics(
            args.fixed_wrist_pd_inverse)
    if checkpoint_schema == CONTEXT_SCHEMA:
        model = ContextHandActionRetargeter(payload["width"]).to(device).eval()
        required_statistics = {"hand", "state", "context", "action"}
    else:
        model = HandActionRetargeter(payload["width"]).to(device).eval()
        required_statistics = {"hand", "state", "action"}
    model.load_state_dict(payload["state_dict"], strict=True)
    stats = {key: Standardizer(**value) for key, value in payload["statistics"].items()}
    if set(stats) != required_statistics:
        raise ValueError("checkpoint statistics do not match model schema")
    source_hand = np.asarray(source["hand_keypoints"][:, 0], dtype="float32")
    source_hands = np.asarray(source["hand_keypoints"], dtype="float32")
    if args.source_per_env and source_hands.shape[1] != args.envs:
        raise ValueError("per-env source requires one complete teacher trajectory per environment")
    source_sha = sha(args.source.resolve()); checkpoint_sha = sha(args.checkpoint.resolve())
    fixed_wrist_sha = (sha(args.fixed_wrist_pd_inverse.resolve())
                       if args.fixed_wrist_pd_inverse is not None else None)

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
    if args.fixed_wrist_pd_inverse is not None:
        frozen[str(args.fixed_wrist_pd_inverse.resolve())] = fixed_wrist_sha
        frozen[str((TASK / "src/consequence_evaluator/fixed_wrist_decoder.py").resolve())] = sha(
            TASK / "src/consequence_evaluator/fixed_wrist_decoder.py")
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
                    source_backend=source.get("source_backend"),
                    source_actor_execution=source.get("replay_identity", {}).get("actor_execution"),
                    checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_sha,
                    input_sha256=frozen, query_period=args.query_period, model_schema=checkpoint_schema,
                    roles=["reactive_teacher", "retarget_gt_hand", "retarget_gt_hand_repeat",
                           "retarget_gt_hand_repeat2"],
                    privileged_input=("frozen source teacher future hand geometry; current q,dq and live hand"
                                      if checkpoint_schema == SCHEMA else
                                      "frozen source teacher future hand geometry; current q,dq, live hand/object and previous command"),
                    context_contract=(None if checkpoint_schema == SCHEMA else
                                      "query-time hand points in object frame plus previous native command; zero at tick 0"),
                    fixed_wrist_decoder=(None if args.fixed_wrist_pd_inverse is None else dict(
                        schema="ref2dex.fixed-wrist-decoder.v1",
                        coefficients=str(args.fixed_wrist_pd_inverse.resolve()),
                        coefficients_sha256=fixed_wrist_sha,
                        root_contract="reset-calibrated 11-point wrist roots",
                        control_contract="current live q/dq plus future hand geometry only",
                        excluded_inputs=["future q", "future dq", "future force", "future command"],
                        train_statistics_scope=fixed_wrist_metadata.get("statistics_scope"))),
                    teacher_finger_reference=(
                        "current live env-0 teacher action 6:18 broadcast to test envs"
                        if args.teacher_finger_reference else None),
                    teacher_action_reference=(
                        "current live env-0 teacher action 0:18 broadcast to test envs"
                        if args.teacher_action_reference else None),
                    source_hand_contract=("per-env source hand future for matched-layout diagnostic"
                                          if args.source_per_env else "env0 source hand future broadcast to R envs"),
                    contact_capture=dict(
                        hand_field="native_contact_forces: task._contact_forces[:, task._contact_body_ids]",
                        object_field="native_object_contact_forces: task._tar_contact_forces",
                        pair_rule="hand force norm>.1 any AND object force norm>.1",
                        state_alignment="measured at frame t after command t-1, before command t"),
                    claim="GT-hand execution upper-bound Probe; no deployability or Cm claim")
    write(output / "manifest.json", manifest)

    class Player(native.EvalPlayer):
        @torch.no_grad()
        def run(self):
            task = self.env.task; n = task.num_envs
            if n != args.envs or self.is_rnn or len(task.motion_file) != 1:
                raise ValueError("fixed four-env feedforward full-reference contract required")
            runtime_backend, runtime_actor_execution = runtime_source_identity(task, n)
            if runtime_backend != source["source_backend"]:
                raise ValueError("runtime backend differs from source packet")
            expected_actor = source["replay_identity"]["actor_execution"]
            if runtime_actor_execution != expected_actor:
                raise ValueError("runtime actor execution differs from source packet")
            manifest["runtime_backend"] = runtime_backend
            manifest["runtime_actor_execution"] = runtime_actor_execution
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
            actions = []; requested_actions = []; clips = []; clip_elements = []
            queries = []; chunks = []; inputs = []
            logs = {key: [] for key in ("object_pose", "hand_keypoints", "surface_gap", "support_gap",
                                        "table_footprint", "object_velocity", "dof_position", "dof_velocity",
                                        "native_contact_forces", "native_object_contact_forces", "pair", "done")}
            predicted = np.zeros((n, HORIZON, 18), dtype=np.float32)
            fixed_wrist_goals = None
            fixed_wrist_goal_chunks = []
            previous_command = np.zeros((n, 18), dtype=np.float32)
            last_query_tick = None
            capture = None; original_pre = task.pre_physics_step
            def pre(value):
                nonlocal capture
                capture = value.detach().cpu().numpy().copy()
                return original_pre(value)
            task.pre_physics_step = pre

            def measure():
                hand, gap = geometry.measure(task); sup, foot = support.measure(task, geometry)
                contact = task._contact_forces[:, task._contact_body_ids]
                object_contact = task._tar_contact_forces
                if (contact.ndim != 3 or contact.shape[-1] != 3
                        or object_contact.shape != (n, 3)):
                    raise ValueError("unexpected native contact-force tensor shape")
                pair = ((contact.norm(dim=-1) > .1).any(-1)
                        & (object_contact.norm(dim=-1) > .1))
                value = dict(object_pose=poses(task._target_states), hand_keypoints=hand, surface_gap=gap,
                             support_gap=sup, table_footprint=foot,
                             object_velocity=task._target_states[:, 7:13],
                             dof_position=task._dof_pos, dof_velocity=task._dof_vel,
                             native_contact_forces=contact, native_object_contact_forces=object_contact,
                             pair=pair)
                for key, item in value.items(): logs[key].append(item.detach().cpu().numpy().copy())
                return value

            measured = measure()
            for tick in range(542):
                if time.monotonic() - started > args.seconds:
                    raise TimeoutError("retarget execution deadline")
                base = self.get_action(obs, True).clamp(-1, 1).to(self.device)
                if last_query_tick is None or tick - last_query_tick >= args.query_period:
                    if args.source_per_env:
                        source_future, future = anchored_future_batch(
                            source_hands, source_hand, measured["hand_keypoints"].detach().cpu().numpy(),
                            tick, True)
                    else:
                        source_future, future = anchored_future_batch(
                            source_hands, source_hand, measured["hand_keypoints"].detach().cpu().numpy(),
                            tick, False)
                    # The model was trained on future hand displacement from
                    # the query hand.  The helper preserves that displacement
                    # while anchoring it to each live environment's current
                    # hand; feeding source absolute coordinates would inject
                    # an unregistered initial-frame offset into R.
                    current = measured["hand_keypoints"].detach().cpu().numpy()
                    q = task._dof_pos.detach().cpu().numpy()
                    dq = task._dof_vel.detach().cpu().numpy()
                    future_batch = future if args.source_per_env else np.broadcast_to(
                        future, (n, HORIZON, 11, 3))
                    hand = trajectory_input(current, future_batch)
                    state = np.concatenate((q, dq), axis=-1).astype("float32")
                    with torch.no_grad():
                        hand_tensor = torch.as_tensor(stats["hand"].encode(hand), device=self.device)
                        state_tensor = torch.as_tensor(stats["state"].encode(state), device=self.device)
                        if checkpoint_schema == CONTEXT_SCHEMA:
                            context = hand_object_context(
                                current, measured["object_pose"].detach().cpu().numpy(), previous_command)
                            context_tensor = torch.as_tensor(stats["context"].encode(context), device=self.device)
                            model_output = model(hand_tensor, state_tensor, context_tensor)
                        else:
                            model_output = model(hand_tensor, state_tensor)
                        predicted = stats["action"].decode(model_output.cpu().numpy())
                    if args.fixed_wrist_pd_inverse is not None:
                        fixed_wrist_goals = recover_wrist_sequence(
                            future_batch,
                            root_template(source_hand[0], source["dof_position"][0, 0]),
                            q)
                        fixed_wrist_goal_chunks.append(fixed_wrist_goals.copy())
                    queries.append(tick); chunks.append(predicted.copy())
                    query_input = dict(current_hand=current.copy(), future_hand=future.copy(),
                                       source_future=source_future.copy(), state=state.copy())
                    if checkpoint_schema == CONTEXT_SCHEMA:
                        query_input.update(
                            object_pose=measured["object_pose"].detach().cpu().numpy().copy(),
                            previous_command=previous_command.copy(), context=context.copy())
                    inputs.append(query_input)
                    last_query_tick = tick
                offset = chunk_offset(tick, last_query_tick)
                command = base.detach().cpu().numpy().copy()
                command[1:] = predicted[1:, offset]
                if args.teacher_action_reference:
                    command = replace_with_teacher_action(command, command[0])
                elif args.teacher_finger_reference:
                    command = replace_with_teacher_fingers(command, command[0])
                if fixed_wrist_goals is not None:
                    # The model query state is frozen for the chunk, but the
                    # analytic one-step inverse must use the live state at
                    # every dispatch tick (not q/dq from the query boundary).
                    live_q = measured["dof_position"].detach().cpu().numpy()
                    live_dq = measured["dof_velocity"].detach().cpu().numpy()
                    hybrid = replace_wrist_action(
                        predicted[:, offset], live_q, live_dq, fixed_wrist_goals[:, offset],
                        fixed_wrist_coefficients)
                    command[1:, :6] = hybrid[1:, :6]
                intended = command.copy(); command = np.clip(command, -1, 1); command[~active] = 0
                obs, _, done, info = self.env_step(self.env, torch.as_tensor(command, device=self.device))
                if not isinstance(obs, dict): obs = {"obs": obs}
                if capture is None or not np.allclose(capture, command, atol=1e-7, rtol=0):
                    raise ValueError("native executed action differs from requested action")
                actions.append(capture.copy()); requested_actions.append(command.copy())
                clip_mask = np.abs(intended - command) > 1e-7
                clips.append(clip_mask.any(-1)); clip_elements.append(clip_mask)
                previous_command = capture.copy()
                logs["done"].append(done.detach().cpu().numpy().reshape(-1).astype(bool))
                measured = measure(); self._post_step(info)
                lengths[active] += 1; active &= ~done.detach().cpu().numpy().reshape(-1).astype(bool)
                if not active.any(): break
            if active.any():
                raise ValueError("retarget episode ended before 542 steps")
            arrays = {key: np.stack(value) for key, value in logs.items()}
            arrays.update(actions=np.asarray(actions), requested_actions=np.asarray(requested_actions),
                          clipped=np.asarray(clips), clipped_elements=np.asarray(clip_elements), lengths=lengths)
            outcomes = {}
            for env, name in enumerate(manifest["roles"]):
                packet = {key: arrays[key][:, env] for key in ("object_pose", "surface_gap", "support_gap",
                                                                  "table_footprint", "object_velocity")}
                outcomes[name] = episode_outcome(packet)
            source_target = source_hands if args.source_per_env else source_hand[:, None]
            hand_error = np.sqrt(np.mean((arrays["hand_keypoints"] - source_target) ** 2,
                                         axis=(0, 2, 3))) * 1000
            teacher = outcomes["reactive_teacher"]["maximum_held_frames"]
            reference_held = max(teacher, source["role_outcomes"]["reactive_teacher"]["maximum_held_frames"])
            gate = {name: bool(teacher >= 45 and outcomes[name]["maximum_held_frames"] >= .9 * reference_held
                               and hand_error[index] < 40.) for index, name in enumerate(manifest["roles"])}
            manifest.update(status="COMPLETED", elapsed_s=time.monotonic() - started,
                            action_max_abs_error=float(np.max(np.abs(arrays["actions"] - arrays["requested_actions"]))),
                            clipping_counts=np.sum(arrays["clipped"], axis=0).tolist(),
                            clipping_wrist_counts=np.sum(arrays["clipped_elements"][..., :6], axis=(0, 2)).tolist(),
                            clipping_finger_counts=np.sum(arrays["clipped_elements"][..., 6:], axis=(0, 2)).tolist(),
                            reference_held=reference_held,
                            outcomes=outcomes, hand_coordinate_rmse_mm=hand_error.tolist(), gate=gate,
                            query_ticks=queries, teacher_source_outcome=source["role_outcomes"]["reactive_teacher"])
            with (output / "execution.pkl").open("wb") as stream:
                pickle.dump(dict(schema=manifest["schema"], manifest=manifest, arrays=arrays,
                                 source_hand=(source_hands if args.source_per_env else source_hand),
                                 query_ticks=np.asarray(queries),
                                 predicted_chunks=np.asarray(chunks), retarget_inputs=inputs,
                                 fixed_wrist_goal_chunks=(np.asarray(fixed_wrist_goal_chunks)
                                                          if fixed_wrist_goal_chunks else None)), stream, protocol=4)
            write(output / "manifest.json", manifest)
            write(output / "result.json", dict(status="PROMISING" if all(gate.values()) else "UNCLEAR",
                                                outcomes=outcomes, hand_coordinate_rmse_mm=hand_error.tolist(),
                                                gate=gate, clipping_counts=np.sum(arrays["clipped"], axis=0).tolist(),
                                                clipping_wrist_counts=np.sum(arrays["clipped_elements"][..., :6], axis=(0, 2)).tolist(),
                                                clipping_finger_counts=np.sum(arrays["clipped_elements"][..., 6:], axis=(0, 2)).tolist(),
                                                reference_held=reference_held,
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

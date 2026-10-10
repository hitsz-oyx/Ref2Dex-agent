"""Bounded ref7_3 oracle-reference residual PPO Probe using owned native inputs."""
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def gpu_state(gpu):
    text = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu),
        "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], text=True)
    utilization, used, total = [int(v.strip()) for v in text.strip().split(",")]
    processes = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits"], text=True).strip()
    return dict(utilization=utilization, used_mib=used, total_mib=total, processes=processes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--seed", type=int, default=271)
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--updates", type=int, default=128)
    p.add_argument("--seconds", type=int, default=1200)
    p.add_argument("--mode", choices=("train", "evaluate", "smoke"), required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--wrist-feedforward", action="store_true")
    p.add_argument("--compare-feedforward", action="store_true",
                   help="four randomized roles with the same frozen policy")
    p.add_argument("--tau-only", action="store_true")
    p.add_argument("--geometry-reference", type=Path)
    p.add_argument("--compare-tau", action="store_true")
    p.add_argument("--oracle-checkpoint", type=Path)
    args = p.parse_args()
    if args.tau_only:
        if args.geometry_reference is None:
            raise ValueError("tau-only execution requires verified geometric reference")
        args.geometry_reference = args.geometry_reference.resolve()
        args.wrist_feedforward = True
    elif args.geometry_reference or args.compare_tau:
        raise ValueError("tau comparison/geometry requires tau-only mode")
    if args.compare_tau:
        if args.mode != "evaluate" or args.oracle_checkpoint is None or args.compare_feedforward:
            raise ValueError("tau comparison requires independent owned oracle checkpoint")
        args.oracle_checkpoint = args.oracle_checkpoint.resolve()
    if args.compare_feedforward and (args.mode != "evaluate" or args.wrist_feedforward):
        raise ValueError("comparison requires evaluation and selects feedforward per role")
    if args.checkpoint:
        args.checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    try:
        output.relative_to(ROOT / "outputs/consequence-evaluator")
    except ValueError:
        raise ValueError("output must belong to consequence-evaluator")
    if (output.exists()
            or not 4 <= args.envs <= 96 or args.envs % 4
            or not 1 <= args.updates <= 128 or not 60 <= args.seconds <= 1800):
        raise ValueError("fresh bounded task-owned reference tracking run required")
    if args.mode == "evaluate" and args.checkpoint is None:
        raise ValueError("evaluation requires a frozen tracker checkpoint")
    before = gpu_state(args.gpu)
    # Some unrelated simulators create tiny contexts on every GPU. Preserve them;
    # use only a device without active compute and with at least 20 GiB available.
    if before["utilization"] > 10 or before["used_mib"] > 512 or before["total_mib"] - before["used_mib"] < 20000:
        raise RuntimeError("GPU is not idle with sufficient memory: " + repr(before))
    scratch = ROOT / "tmp/reference-tracking"
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), TMPDIR=str(scratch),
                      TORCH_EXTENSIONS_DIR=str(scratch / "torch-extensions"),
                      TRITON_CACHE_DIR=str(scratch / "triton"), OMP_NUM_THREADS="2",
                      PYTHONDONTWRITEBYTECODE="1")
    from isaacgym import gymtorch  # noqa: Isaac Gym must precede Torch
    import torch
    from consequence_evaluator.reference_tracking import (
        ACTIVE, SCHEMA, ReferenceTracker, advantages, features, native_action, tracking_reward,
        reference_velocity, wrist_feedforward)
    from consequence_evaluator.tau_tracking import (
        SCHEMA as TAU_SCHEMA, TauTracker, tau_features, tau_reward)
    from consequence_evaluator.native_reset import install_reset_patch
    from consequence_evaluator.physical_geometry import PhysicalGeometry, poses
    from consequence_evaluator.value_geometry import TableSupport
    from consequence_evaluator.gate1 import episode_outcome
    from consequence_evaluator.contracts import HAND_LINKS
    from oracle_y_utility import align_native_reference_tables
    import evaluate as native
    from env.tasks.base_dexplore_task import DexploreTask

    if torch.__version__ != "2.4.1+cu121":
        raise ValueError("pinned graspenv Torch runtime required")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    cfg = json.loads(args.inputs.read_text())
    essential = [cfg["actor"], cfg["cfg_env"], cfg["cfg_train"],
                 str(ROOT / "third_party/DExplore/dexplore/evaluate.py"),
                 str(ROOT / "third_party/DExplore/dexplore/env/tasks/base_dexplore_task.py"),
                 str(ROOT / "third_party/DExplore/dexplore/env/tasks/dexplore_inspire.py")]
    essential += [path for path in cfg["input_sha256"] if path.endswith("interaction_hand_inspire.pt")]
    for path in essential:
        if sha(path) != cfg["input_sha256"][path]:
            raise ValueError("native input drift: " + path)
    # Only measured reference fields are exposed; stored teacher commands are
    # deliberately omitted. This run does not inherit ref7_2 held-out identity.
    with args.reference.open("rb") as stream:
        packet = pickle.load(stream)
    if (packet.get("seed") != 282 or packet.get("source_backend", {}).get("pipeline") != "cpu"
            or packet["dof_position"].shape != (543, 4, 18)):
        raise ValueError("frozen complete CPU-source reference required")
    reference = {}
    fields = ("hand_keypoints",) if args.tau_only and args.mode == "train" else (
        "dof_position", "hand_keypoints", "object_pose")
    for name in fields:
        value = np.asarray(packet[name][:, 0], dtype=np.float32)
        if not np.isfinite(value).all():
            raise ValueError("nonfinite reference: " + name)
        reference[name] = torch.as_tensor(value, device=device)
    initial_height = float(packet["object_pose"][0, 0, 2, 3])
    reference_held = int(packet["role_outcomes"]["reactive_teacher"]["maximum_held_frames"])
    initial_palm_height = float(packet["hand_keypoints"][0, 0, 0, 2])
    del packet
    geometry_data = None
    if args.tau_only:
        geometry_manifest = json.loads((args.geometry_reference / "manifest.json").read_text())
        if (geometry_manifest.get("status") != "COMPLETED"
                or geometry_manifest.get("schema") != "ref2dex.tau-geometry-reference.v1"
                or geometry_manifest["reference_sha256"] != sha(args.reference)
                or geometry_manifest["geometry_sha256"] != sha(args.geometry_reference / "geometry.npz")):
            raise ValueError("verified tau-only geometry provenance required")
        with np.load(args.geometry_reference / "geometry.npz", allow_pickle=False) as stream:
            geometry_data = {key: stream[key].copy() for key in stream.files}
        if (geometry_data["q"].shape != (543, 18)
                or not np.array_equal(geometry_data["target_points"], reference["hand_keypoints"].cpu().numpy())):
            raise ValueError("geometry/tau alignment mismatch")
        reference["geometric_q"] = torch.as_tensor(geometry_data["q"], device=device)
    install_reset_patch()
    reset = DexploreTask._reset_ref_state_init
    def aligned(task, ids):
        align_native_reference_tables(task)
        return reset(task, ids)
    DexploreTask._reset_ref_state_init = aligned
    loader = native.torch_ext.load_checkpoint
    def load(path):
        payload = loader(path); keys = list(payload["model"])
        compiled = [key.startswith("_orig_mod.") for key in keys]
        if any(compiled) and not all(compiled):
            raise ValueError("mixed compiled teacher keys")
        if all(compiled):
            payload = dict(payload, model={key[10:]: val for key, val in payload["model"].items()})
        return payload
    native.torch_ext.load_checkpoint = load
    frozen = {str(args.inputs.resolve()): sha(args.inputs), str(args.reference.resolve()): sha(args.reference)}
    for path in essential + [__file__, str(TASK / "src/consequence_evaluator/reference_tracking.py"),
                             str(TASK / "src/consequence_evaluator/native_reset.py")]:
        frozen[str(Path(path).resolve())] = sha(path)
    for path in Path(cfg["motions"]).rglob("*.pt"):
        frozen[str(path.resolve())] = sha(path)
    if args.checkpoint:
        frozen[str(args.checkpoint.resolve())] = sha(args.checkpoint)
    if args.tau_only:
        for path in (args.geometry_reference / "manifest.json", args.geometry_reference / "geometry.npz",
                     TASK / "src/consequence_evaluator/tau_tracking.py"):
            frozen[str(path.resolve())] = sha(path)
    if args.oracle_checkpoint:
        frozen[str(args.oracle_checkpoint)] = sha(args.oracle_checkpoint)
    output.mkdir(parents=True)
    started = time.monotonic()
    manifest = dict(schema=TAU_SCHEMA if args.tau_only else SCHEMA, status="RUNNING", run_id=output.name, task="consequence-evaluator",
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    mode=args.mode, seed=args.seed, num_envs=args.envs, updates=args.updates,
                    wrist_feedforward=args.wrist_feedforward, compare_feedforward=args.compare_feedforward,
                    tau_only=args.tau_only, compare_tau=args.compare_tau,
                    geometry_reference=str(args.geometry_reference) if args.geometry_reference else None,
                    physical_gpu=args.gpu, gpu_before=before, budget_s=args.seconds,
                    input_sha256=frozen, reference_held=reference_held,
                    reference_contract="privileged measured teacher robot-q/hand/object reference; no teacher commanded actions",
                    inference_contract="live q/dq/hand/object/velocity, fixed future hand and next robot/object reference, previous residual",
                    action_contract="bounded absolute residual PD targets around next reference q, encoded through native Inspire adapter",
                    initialization=("warm-start declared owned tracker checkpoint; no official actor or future command labels"
                                if args.checkpoint else "zero residual actor, random small critic; no official actor or future command labels"),
                    training_contract="on-policy PPO only; no imitation/action-label supervision; fixed final checkpoint",
                    source_authorization="2026-10-10 user approved ref7_3 control upper-bound Probe; old diagnostic training_allowed=false is not inherited as a data split",
                    claim="single-motion oracle robot-reference execution Probe; no tau-only deployment or Cm utility claim")
    if args.tau_only:
        manifest.update(
            inference_contract="live q/dq/hand/object/velocity, future11-point tau, tau-derived joint error, previous residual; no true future q/object or tactile input",
            action_contract="tau-derived coupled geometric target + tau-derived wrist velocity feedforward + learned residual; full native command",
            reference_contract="GT future hand upper bound; geometry from tau/reset/static URDF only; q/object labels only for independent comparison/evaluation",
            training_contract="on-policy PPO; tau-only hand/lift/hold reward, no future object reward or command supervision",
            claim="single-motion tau-conditioned geometry/closed-loop execution Probe; no learned high-level tau or Cm benefit")
    write(output / "manifest.json", manifest)

    class Player(native.EvalPlayer):
        def run(self):
            task = self.env.task; n = task.num_envs
            if n != args.envs or self.is_rnn or len(task.motion_file) != 1:
                raise ValueError("single-motion feedforward native task required")
            if str(task.device) != "cpu" or not task.gym.get_sim_params(task.sim).physx.use_gpu:
                raise ValueError("CPU tensor exchange/GPU PhysX contract required")
            task._enable_early_termination = False; task._adaptive_kappa_enabled = False
            task._state_init = DexploreTask.StateInit.Start; task._hybrid_init_prob = 1.
            self.model.eval()
            ids = torch.arange(n, device=task.device)
            obs = self.env_reset(ids); self.get_batch_size(obs["obs"], 1)
            if (task.start_times != 0).any():
                raise ValueError("frame-0 reference initialization required")
            names = task.gym.get_actor_rigid_body_names(task.envs[0], task.humanoid_handles[0])
            key_ids = [names.index(name) for name in HAND_LINKS]
            offset = task._pd_action_offset.to(device); scale = task._pd_action_scale.to(device)
            manifest["native_controller"] = dict(
                dof_names=task.gym.get_actor_dof_names(task.envs[0], task.humanoid_handles[0]),
                offset=offset.cpu().tolist(), scale=scale.cpu().tolist())
            props = task.gym.get_actor_dof_properties(task.envs[0], task.humanoid_handles[0])
            kp = torch.as_tensor(props["stiffness"][:6].copy(), device=device)
            kd = torch.as_tensor(props["damping"][:6].copy(), device=device)
            if (kp <= 0).any() or not torch.allclose(kd / kp, torch.full_like(kp, .1)):
                raise ValueError("unexpected native wrist PD gains")
            wrist_gain_ratio = kd / kp
            control_dt = task.gym.get_sim_params(task.sim).dt * task.control_freq_inv
            base_reference = reference["geometric_q"] if args.tau_only else reference["dof_position"]
            ref_velocity = reference_velocity(base_reference, control_dt)
            if args.tau_only and not np.array_equal(geometry_data["reset_q"], task._dof_pos[0].cpu().numpy()):
                raise ValueError("tau geometry calibration is not actual initial live q")
            manifest["wrist_feedforward_contract"] = dict(
                stiffness=kp.cpu().tolist(), damping=kd.cpu().tolist(), control_dt=control_dt,
                derivative="central position differences, wrapped native rotations, one-sided endpoints",
                scope=("first six native wrist DOFs only; next-frame aligned; tau-derived velocity" if args.tau_only
                       else "first six native wrist DOFs only; next-frame aligned; no teacher command labels"))
            policy = (TauTracker() if args.tau_only else ReferenceTracker()).to(device)
            if args.checkpoint:
                checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
                allowed = (SCHEMA, TAU_SCHEMA) if args.tau_only else (SCHEMA,)
                if checkpoint.get("schema") not in allowed:
                    raise ValueError("tracker checkpoint schema mismatch")
                trained_ff = checkpoint.get("manifest", {}).get("wrist_feedforward", False)
                if args.mode != "train" and trained_ff and not (args.wrist_feedforward or args.compare_feedforward):
                    raise ValueError("feedforward-trained checkpoint requires explicit feedforward controller")
                manifest["checkpoint_control"] = dict(trained_wrist_feedforward=trained_ff,
                                                       evaluation_override=args.compare_feedforward)
                if args.tau_only and checkpoint["schema"] == SCHEMA:
                    policy.warmstart(checkpoint["state_dict"])
                    manifest["checkpoint_control"]["migration"] = "drop future-object-error columns only"
                else:
                    policy.load_state_dict(checkpoint["state_dict"], strict=True)
            oracle_policy = None
            if args.compare_tau:
                oracle_checkpoint = torch.load(args.oracle_checkpoint, map_location=device, weights_only=False)
                if oracle_checkpoint.get("schema") != SCHEMA or not oracle_checkpoint["manifest"].get("wrist_feedforward"):
                    raise ValueError("independent verified feedforward oracle required")
                oracle_policy = ReferenceTracker().to(device)
                oracle_policy.load_state_dict(oracle_checkpoint["state_dict"], strict=True)
                oracle_policy.eval()
                oracle_velocity = reference_velocity(reference["dof_position"], control_dt)
            initial_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
            previous = torch.zeros(n, 12, device=device)
            capture = None
            original_pre = task.pre_physics_step
            def pre(action):
                nonlocal capture
                capture = action.detach().clone().to(device)
                return original_pre(action)
            task.pre_physics_step = pre

            def measure():
                bodies = task._rigid_body_state.view(n, -1, 13)
                return dict(q=task._dof_pos.to(device), dq=task._dof_vel.to(device),
                            hand=bodies[:, key_ids, :3].to(device), obj=poses(task._target_states).to(device),
                            velocity=task._target_states[:, 7:13].to(device),
                            pair=((task._contact_forces[:, task._contact_body_ids].norm(dim=-1) > .1).any(-1)
                                  & (task._tar_contact_forces.norm(dim=-1) > .1)).to(device))

            def observation():
                measured = measure()
                tick = task.progress_buf.to(device).long().clamp(0, 542)
                index = (tick[:, None] + torch.arange(1, 25, device=device)[None]).clamp_max(542)
                next_index = (tick + 1).clamp_max(542)
                next_q = base_reference[next_index]
                if args.tau_only:
                    next_obj = None
                    value = tau_features(measured["q"], measured["dq"], measured["hand"], measured["obj"],
                                         measured["velocity"], reference["hand_keypoints"][index], next_q, previous)
                else:
                    next_obj = reference["object_pose"][next_index]
                    value = features(measured["q"], measured["dq"], measured["hand"], measured["obj"],
                                     measured["velocity"], reference["hand_keypoints"][index], next_q, next_obj, previous)
                return value, measured, next_q, next_obj, reference["hand_keypoints"][next_index]

            def step(latent, next_q, measured):
                nonlocal obs, previous
                base = next_q
                if args.wrist_feedforward:
                    index = (task.progress_buf.to(device).long() + 1).clamp_max(542)
                    base = wrist_feedforward(next_q, ref_velocity[index], wrist_gain_ratio)
                target = policy.target(base, latent)
                intended = native_action(target, measured["q"], offset, scale)
                command = intended.clamp(-1, 1)
                obs, reward_unused, done, info = self.env_step(self.env, command.clone())
                if capture is None or not torch.equal(capture, command):
                    raise ValueError("requested/applied native tracking action mismatch")
                previous = torch.tanh(latent).detach()
                return done.to(device).bool().reshape(-1), (intended - command).abs().amax(-1) > 1e-6

            work_started = time.monotonic()

            def monitor(update, frames):
                elapsed = time.monotonic() - started
                state = gpu_state(args.gpu)
                total = args.updates if args.mode == "train" else (8 if args.mode == "smoke" else 542)
                eta = (time.monotonic() - work_started) / max(update, 1) * max(total - update, 0)
                print(json.dumps(dict(update=update, frames=frames, elapsed_s=round(elapsed, 1),
                                      eta_s=round(eta, 1), gpu=state,
                                      torch_peak_mib=round(torch.cuda.max_memory_allocated() / 2**20, 1))), flush=True)
                with (output / "monitor.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(update=update, elapsed_s=elapsed, gpu=state)) + "\n")
                # Preserve passive contexts seen at launch, reject additional heavy users.
                baseline_pids = {row.split(",")[0].strip() for row in before["processes"].splitlines()}
                for row in state["processes"].splitlines():
                    parts = row.split(",")
                    if len(parts) == 2 and parts[0].strip() not in baseline_pids | {str(os.getpid())}:
                        if int(parts[1].strip()) > 512:
                            raise RuntimeError("new active foreign GPU process; stop own run")

            def save(name, updates):
                path = output / name
                if path.exists():
                    raise ValueError("checkpoint path already exists")
                torch.save(dict(schema=TAU_SCHEMA if args.tau_only else SCHEMA, state_dict=policy.state_dict(), updates=updates,
                                manifest=manifest), path)
                return sha(path)

            if args.mode == "train":
                manifest["initial_sha256"] = save("initial.pt", 0)
                optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
                history = []; frames = 0; total_clipping = 0
                for update in range(1, args.updates + 1):
                    if time.monotonic() - started > args.seconds - 30:
                        raise TimeoutError("training budget exhausted before frozen update count")
                    records = {k: [] for k in ("obs", "latent", "logp", "value", "reward", "done")}
                    for _ in range(32):
                        with torch.no_grad():
                            x, measured, next_q, next_obj, next_hand = observation()
                            dist = policy.distribution(x); latent = dist.sample()
                            value = policy.value(x); logp = dist.log_prob(latent).sum(-1)
                            done, clipped = step(latent, next_q, measured)
                            after = measure()
                            if args.tau_only:
                                reward = tau_reward(after["hand"], after["obj"], next_hand, after["pair"],
                                                    initial_height, initial_palm_height, latent)
                            else:
                                reward = tracking_reward(after["hand"], after["obj"], next_hand, next_obj,
                                                         after["pair"], initial_height, latent)
                            for key, val in (("obs", x), ("latent", latent), ("logp", logp),
                                             ("value", value), ("reward", reward), ("done", done)):
                                records[key].append(val.detach())
                            frames += n; total_clipping += int(clipped.sum())
                            if done.any():
                                reset_ids = torch.nonzero(done).flatten().to(task.device)
                                obs = self.env_reset(reset_ids)
                                previous[done] = 0
                    batch = {key: torch.stack(val) for key, val in records.items()}
                    with torch.no_grad():
                        bootstrap = policy.value(observation()[0])
                        adv, returns = advantages(batch["reward"], batch["value"], batch["done"], bootstrap)
                        adv = (adv - adv.mean()) / adv.std().clamp_min(1e-6)
                    flat = {key: val.flatten(0, 1) for key, val in batch.items()}
                    adv = adv.flatten(); returns = returns.flatten()
                    metrics = []
                    with torch.enable_grad():
                        for _ in range(4):
                            permutation = torch.randperm(len(adv), device=device)
                            for begin in range(0, len(adv), 512):
                                select = permutation[begin:begin + 512]
                                dist = policy.distribution(flat["obs"][select])
                                logp = dist.log_prob(flat["latent"][select]).sum(-1)
                                ratio = (logp - flat["logp"][select]).exp()
                                actor_loss = -torch.minimum(ratio * adv[select], ratio.clamp(.8, 1.2) * adv[select]).mean()
                                critic_loss = (policy.value(flat["obs"][select]) - returns[select]).square().mean()
                                loss = actor_loss + .5 * critic_loss - .001 * dist.entropy().sum(-1).mean()
                                if not torch.isfinite(loss):
                                    raise FloatingPointError("nonfinite PPO loss")
                                optimizer.zero_grad(); loss.backward()
                                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
                                optimizer.step()
                                metrics.append((float(actor_loss.detach()), float(critic_loss.detach())))
                    row = dict(update=update, frames=frames, mean_reward=float(batch["reward"].mean()),
                               actor_loss=float(np.mean([m[0] for m in metrics])),
                               critic_loss=float(np.mean([m[1] for m in metrics])), clipped_steps=total_clipping)
                    history.append(row)
                    with (output / "training.jsonl").open("a") as stream:
                        stream.write(json.dumps(row) + "\n")
                    if update == 1 or update % 4 == 0:
                        monitor(update, frames)
                checksum = save("final.pt", args.updates)
                changed = sum(not torch.equal(val, initial_state[key]) for key, val in policy.state_dict().items())
                if changed == 0:
                    raise ValueError("PPO did not change tracker parameters")
                result = dict(status="COMPLETED", frames=frames, updates=args.updates,
                              checkpoint_sha256=checksum, changed_tensors=changed, history=history,
                              clipping_rate=total_clipping / frames, claim="trained policy only; native behavior evaluation pending")
            else:
                policy.eval()
                geometry = PhysicalGeometry(task, ROOT / "third_party/DExplore/dexplore/data/assets", distance_device=device)
                support = TableSupport(ROOT / "third_party/DExplore/dexplore/data/assets", task.device)
                quarter = n // 4
                if args.compare_tau:
                    roles = np.repeat(["oracle", "tau_nominal", "tracker", "tau_shifted"], quarter)
                    roles = np.random.default_rng(args.seed).permutation(roles).tolist()
                elif args.compare_feedforward:
                    roles = np.repeat(["teacher", "nominal", "tracker", "tracker_feedforward"], quarter)
                    roles = np.random.default_rng(args.seed).permutation(roles).tolist()
                else:
                    roles = ["teacher"] * quarter + ["nominal"] * quarter + ["tracker"] * (2 * quarter)
                teacher_mask = torch.tensor([r == "teacher" for r in roles], device=device)
                nominal_mask = torch.tensor([r in ("nominal", "tau_nominal") for r in roles], device=device)
                ff_mask = torch.tensor([r == "tracker_feedforward" or
                                        (args.wrist_feedforward and r != "teacher") for r in roles], device=device)
                oracle_mask = torch.tensor([r == "oracle" for r in roles], device=device)
                shifted_mask = torch.tensor([r == "tau_shifted" for r in roles], device=device)
                manifest["roles"] = roles
                if args.compare_tau:
                    manifest["tau_shift_ticks"] = 271
                arrays = {key: [] for key in ("object_pose", "hand_keypoints", "surface_gap", "support_gap",
                                              "table_footprint", "object_velocity", "pair", "action", "clipped",
                                              "dof_position", "dof_velocity", "latent", "pd_targets")}
                applied_target = None
                original_encode = task._action_to_pd_targets
                def encode(action):
                    nonlocal applied_target
                    target = original_encode(action)
                    applied_target = target.detach().clone().to(device)
                    return target
                task._action_to_pd_targets = encode
                def record():
                    hand, gap = geometry.measure(task); sup, foot = support.measure(task, geometry)
                    current = measure()
                    for key, val in (("object_pose", current["obj"]), ("hand_keypoints", hand),
                                     ("surface_gap", gap), ("support_gap", sup), ("table_footprint", foot),
                                     ("object_velocity", current["velocity"]), ("pair", current["pair"]),
                                     ("dof_position", current["q"]), ("dof_velocity", current["dq"])):
                        arrays[key].append(val.detach().cpu().numpy().copy())
                record()
                controls = 8 if args.mode == "smoke" else 542
                work_started = time.monotonic()
                for tick in range(controls):
                    with torch.no_grad():
                        x, measured, next_q, _, _ = observation()
                        ref_indices = torch.full((n,), tick + 1, device=device, dtype=torch.long)
                        if shifted_mask.any():
                            ref_indices[shifted_mask] = (tick + 1 + 271) % 543
                            shifted_future = (tick + torch.arange(1, 25, device=device) + 271) % 543
                            next_q = next_q.clone()
                            next_q[shifted_mask] = base_reference[ref_indices[shifted_mask]]
                            x[shifted_mask] = tau_features(
                                measured["q"][shifted_mask], measured["dq"][shifted_mask],
                                measured["hand"][shifted_mask], measured["obj"][shifted_mask],
                                measured["velocity"][shifted_mask],
                                reference["hand_keypoints"][shifted_future].expand(int(shifted_mask.sum()), -1, -1, -1),
                                next_q[shifted_mask], previous[shifted_mask])
                        latent = policy.actor(x)
                        latent[teacher_mask | nominal_mask] = 0
                        base = next_q.clone()
                        base[ff_mask] = wrist_feedforward(next_q[ff_mask],
                            ref_velocity[ref_indices[ff_mask]], wrist_gain_ratio)
                        intended = native_action(policy.target(base, latent), measured["q"], offset, scale)
                        if oracle_mask.any():
                            index = (tick + torch.arange(1, 25, device=device)).clamp_max(542)
                            oracle_q = reference["dof_position"][tick + 1].expand(n, -1)
                            oracle_x = features(measured["q"], measured["dq"], measured["hand"], measured["obj"],
                                measured["velocity"], reference["hand_keypoints"][index].expand(n, -1, -1, -1),
                                oracle_q, reference["object_pose"][tick + 1].expand(n, -1, -1), previous)
                            oracle_latent = oracle_policy.actor(oracle_x)
                            oracle_base = wrist_feedforward(oracle_q, oracle_velocity[tick + 1].expand(n, -1), wrist_gain_ratio)
                            oracle_action = native_action(oracle_policy.target(oracle_base, oracle_latent), measured["q"], offset, scale)
                            intended[oracle_mask] = oracle_action[oracle_mask]
                            latent[oracle_mask] = oracle_latent[oracle_mask]
                        # Teacher action is used only for independent native control rows.
                        actor_obs = obs if isinstance(obs, dict) else {"obs": obs}
                        if teacher_mask.any():
                            teacher = self.get_action(actor_obs, True).to(device).clone().clamp(-1, 1)
                            intended[teacher_mask] = teacher[teacher_mask]
                        command = intended.clamp(-1, 1)
                        obs, _, done, _ = self.env_step(self.env, command.clone())
                        if capture is None or not torch.equal(capture, command):
                            raise ValueError("evaluation requested/applied command mismatch")
                        arrays["action"].append(command.cpu().numpy().copy())
                        arrays["latent"].append(latent.cpu().numpy().copy())
                        arrays["pd_targets"].append(applied_target.cpu().numpy().copy())
                        arrays["clipped"].append(((intended - command).abs().amax(-1) > 1e-6).cpu().numpy())
                        previous = torch.tanh(latent)
                        record()
                        if tick < controls - 1 and done.any():
                            raise ValueError("unexpected early terminal state in complete evaluation")
                    if tick % 128 == 0:
                        monitor(tick + 1, (tick + 1) * n)
                packed = {key: np.stack(value) for key, value in arrays.items()}
                np.savez_compressed(output / "trajectory.npz", **packed)
                if args.mode == "smoke":
                    result = dict(status="COMPLETED", controls=controls, roles=roles, claim="engineering wiring only")
                else:
                    outcomes = []
                    for env in range(n):
                        outcome = episode_outcome({key: packed[key][:, env] for key in (
                            "object_pose", "surface_gap", "support_gap", "table_footprint", "object_velocity")})
                        outcome.update(role=roles[env], env=env, clipping_count=int(packed["clipped"][:, env].sum()))
                        outcome["hand_rmse_m"] = float(np.sqrt(np.mean((
                            packed["hand_keypoints"][:, env] - reference["hand_keypoints"].cpu().numpy())**2)))
                        outcome["object_position_rmse_m"] = float(np.sqrt(np.mean((
                            packed["object_pose"][:, env, :3, 3] - reference["object_pose"].cpu().numpy()[:, :3, 3])**2)))
                        outcomes.append(outcome)
                    summary = {}
                    for role in sorted(set(roles)):
                        rows = [row for row in outcomes if row["role"] == role]
                        summary[role] = dict(episodes=len(rows), mean_held=float(np.mean([r["maximum_held_frames"] for r in rows])),
                                             median_held=float(np.median([r["maximum_held_frames"] for r in rows])),
                                             near_teacher_count=sum(r["maximum_held_frames"] >= .9 * reference_held for r in rows),
                                             hold45_count=sum(r["maximum_held_frames"] >= 45 for r in rows),
                                             mean_hand_rmse_m=float(np.mean([r["hand_rmse_m"] for r in rows])),
                                             mean_object_position_rmse_m=float(np.mean([r["object_position_rmse_m"] for r in rows])),
                                             loss_events=sum(r["intermediate_loss_events"] for r in rows),
                                             clipping_count=sum(r["clipping_count"] for r in rows))
                    result = dict(status="UNCLEAR", outcomes=outcomes, summary=summary,
                                  reference_held=reference_held, claim=manifest["claim"])
            manifest.update(status="COMPLETED", elapsed_s=time.monotonic() - started,
                            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated())
            write(output / "result.json", result); write(output / "manifest.json", manifest)

    native.EvalPlayer = Player
    sys.argv = [sys.argv[0], "--task", "Dexplore_Inspire", "--cfg_env", cfg["cfg_env"], "--cfg_train", cfg["cfg_train"],
                "--checkpoint", cfg["actor"], "--motion_file", cfg["motions"], "--headless", "--num_envs", str(args.envs),
                "--seed", str(args.seed), "--sim_device", "cuda:0", "--rl_device", "cuda:0", "--pipeline", "cpu",
                "--graphics_device_id", "0", "--num_threads", "1", "--disable-early-termination",
                "--output", str(output / "native-unused.json"), "--output_path", str(output / "native")]
    cwd = Path.cwd()
    def deadline(signum, frame):
        raise TimeoutError("owned reference tracking run deadline")
    previous_handler = signal.signal(signal.SIGALRM, deadline); signal.alarm(args.seconds)
    try:
        os.chdir(ROOT / "third_party/DExplore"); native.main()
    except BaseException as error:
        manifest.update(status="FAILED", error=repr(error), elapsed_s=time.monotonic() - started)
        write(output / "manifest.json", manifest)
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous_handler); os.chdir(cwd)


if __name__ == "__main__":
    main()

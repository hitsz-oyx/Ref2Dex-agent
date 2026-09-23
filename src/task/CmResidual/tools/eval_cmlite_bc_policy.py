"""Evaluate a no-checkpoint BC policy with an optional CmLite action selector.

The evaluator runs the same locally trained policy in two matched arms.  The
off arm executes the policy action.  The on arm scores five deterministic local
proposals with the frozen CmLite model and executes the selected proposal.  No
DExplore actor checkpoint is loaded by either arm.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback


def _pin_requested_gpu() -> None:
    """Pin CUDA visibility before importing Isaac Gym or torch."""
    if "--physical-gpu" not in sys.argv:
        return
    index = sys.argv.index("--physical-gpu") + 1
    if index >= len(sys.argv) or not sys.argv[index].isdigit():
        raise ValueError("--physical-gpu must be a non-negative integer")
    requested = sys.argv[index]
    existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing not in (None, "", requested):
        raise ValueError("CUDA_VISIBLE_DEVICES conflicts with --physical-gpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = requested


_pin_requested_gpu()

import numpy as np

if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int

# Allow direct execution from the repository root or from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

# Isaac Gym must be imported before torch in its pinned runtime.
import isaacgym  # noqa: F401
import torch
from rl_games.algos_torch import torch_ext

from src.task.CmResidual.cmlite import FrozenCmLite
from src.task.CmResidual.cmlite_policy_select import (
    ProposalConfig, select_cmlite_action, select_cmlite_candidates,
)
from src.task.CmResidual.dexplore_bc_policy import DExploreBcPolicy, normalized_action


ROOT = Path(__file__).resolve().parents[4]
DEXPLORE = ROOT / "third_party/DExplore"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_bc(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("official_policy_checkpoint", "invalid") is not None:
        raise ValueError("BC checkpoint provenance must explicitly exclude official weights")
    model = DExploreBcPolicy(
        int(payload["observation_dim"]), int(payload["action_dim"]),
        tuple(payload["hidden_dims"])).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, payload["observation_mean"].to(device), payload["observation_std"].to(device)


@torch.inference_mode()
def _run_batched_eval(player, output: Path, checkpoint_path: Path, cm_path: Path,
                      mode: str, cm_sha: str) -> None:
    """Evaluate the first complete episode from every parallel environment."""
    task = player.env.task
    task._adaptive_kappa_enabled = False
    task._enable_early_termination = False
    player.has_batch_dimension = True
    player.env_reset()
    device = task.obs_buf.device
    batch = int(task.num_envs)
    initial_z = task._target_states[:, 2].clone()
    start_frames = task.start_times.clone()
    motion_ids = task.data_id.clone()
    lengths = task.max_episode_length[task.data_id].long()
    teacher_mode = mode == "teacher"
    router_mode = mode == "router"
    ensemble_mode = mode == "ensemble"
    model = mean = std = None
    if not teacher_mode:
        model, mean, std = _load_bc(checkpoint_path, device)
    cm = FrozenCmLite(str(cm_path), device, cm_sha) if mode in ("on", "ensemble") else None
    router_models = {}
    route_names = None
    expert_names = sorted(json.loads(os.environ.get(
        "REF2DEX_POLICY_ROUTE_CHECKPOINTS", "{}")).keys())
    if router_mode:
        route_map = {int(key): str(value) for key, value in json.loads(
            os.environ["REF2DEX_POLICY_ROUTE_MAP"]).items()}
        if not route_map or any(value == "teacher" for value in route_map.values()):
            raise ValueError("router map must contain BC/PPO expert names")
        route_frames = torch.tensor(sorted(route_map), device=device)
        nearest = (start_frames[:, None] - route_frames[None]).abs().argmin(1)
        route_names = [route_map[int(route_frames[index].item())] for index in nearest]
        checkpoint_map = json.loads(os.environ["REF2DEX_POLICY_ROUTE_CHECKPOINTS"])
        required = set(route_map.values()) - {"bc"}
        if required != set(checkpoint_map):
            raise ValueError("router checkpoint names must exactly match routed PPO experts")
    elif ensemble_mode:
        checkpoint_map = json.loads(os.environ["REF2DEX_POLICY_ROUTE_CHECKPOINTS"])
        if len(checkpoint_map) < 1:
            raise ValueError("ensemble requires at least one PPO expert")
    else:
        checkpoint_map = {}
    if router_mode or ensemble_mode:
        for name, path in checkpoint_map.items():
            payload = torch_ext.load_checkpoint(path)
            if "running_mean_std" not in payload or "model" not in payload:
                raise ValueError(f"router expert {name} lacks model/RMS state")
            expert = copy.deepcopy(player.model).to(device)
            expert.load_state_dict(payload["model"], strict=True)
            expert.eval()
            rms = copy.deepcopy(player.running_mean_std).to(device)
            rms.load_state_dict(payload["running_mean_std"])
            rms.eval()
            router_models[name] = (expert, rms)
    proposal_config = ProposalConfig(
        wrist_z_delta=float(os.environ.get("REF2DEX_CMLITE_WRIST_Z_DELTA", "0.01")),
        finger_delta=float(os.environ.get("REF2DEX_CMLITE_FINGER_DELTA", "0.04")),
        contact_threshold=float(os.environ.get("REF2DEX_CMLITE_CONTACT_THRESHOLD", "0.5")),
    )
    max_steps = int((lengths - start_frames).clamp_min(1).max().item()) + 2
    recorded = torch.zeros(batch, dtype=torch.bool, device=device)
    rewards = torch.zeros(batch, device=device)
    steps = torch.zeros(batch, device=device)
    max_lift = torch.zeros(batch, device=device)
    max_contact_lift = torch.zeros(batch, device=device)
    contact_steps = torch.zeros(batch, device=device)
    airborne_steps = torch.zeros(batch, device=device)
    lift_run = torch.zeros(batch, dtype=torch.long, device=device)
    max_lift_run = torch.zeros(batch, dtype=torch.long, device=device)
    success = torch.zeros(batch, dtype=torch.bool, device=device)
    stable_contact_steps = torch.zeros(batch, dtype=torch.long, device=device)
    teacher_grip_latched = torch.zeros(batch, dtype=torch.bool, device=device)
    proposal_count = torch.zeros(batch, dtype=torch.long, device=device)
    proposal_histogram = torch.zeros(5, dtype=torch.long, device=device)
    results = []
    done_indices = torch.empty(0, dtype=torch.long, device=device)
    for _ in range(max_steps):
        player.env_reset(done_indices)
        if teacher_mode:
            from utils.reference_action import inspire_reference_action
            base_action = inspire_reference_action(task, lead=1)
            occupancy = (task._contact_forces[:, task._contact_body_ids].norm(dim=-1) > .1).float().mean(-1)
            teacher_grip_latched |= occupancy >= float(os.environ.get(
                "REF2DEX_REFERENCE_CONTACT_THRESHOLD", "0.2"))
            grip_extra = float(os.environ.get("REF2DEX_REFERENCE_GRIP_EXTRA_RAD", "0.1"))
            indices = torch.as_tensor((6, 8, 10, 12, 14, 15), device=device)
            grip_action_delta = 2.0 * grip_extra / task._pd_action_scale[indices]
            base_action[:, indices] += teacher_grip_latched[:, None] * grip_action_delta[None]
            base_action.clamp_(-1, 1)
        elif router_mode:
            base_action = torch.zeros((batch, 18), device=device)
            raw_obs = task.obs_buf
            for name in set(route_names):
                indices = torch.tensor(
                    [i for i, value in enumerate(route_names) if value == name],
                    dtype=torch.long, device=device)
                if name == "bc":
                    base_action[indices] = normalized_action(
                        model, raw_obs[indices], mean, std)
                else:
                    expert, rms = router_models[name]
                    result = expert({
                        "is_train": False, "prev_actions": None,
                        "obs": rms(raw_obs[indices]), "rnn_states": None,
                    })
                    base_action[indices] = result["mus"].clamp(-1, 1)
        elif ensemble_mode:
            raw_obs = task.obs_buf
            candidate_actions = [normalized_action(model, raw_obs, mean, std)]
            for name in expert_names:
                expert, rms = router_models[name]
                result = expert({
                    "is_train": False, "prev_actions": None,
                    "obs": rms(raw_obs), "rnn_states": None,
                })
                candidate_actions.append(result["mus"].clamp(-1, 1))
            candidates = torch.stack(candidate_actions, dim=1)
            progress = task.progress_buf.long()
            goal_index = torch.minimum(progress + 1, lengths - 1)
            goal = task.hoi_refs[task.data_id, task.ref_index, goal_index, 106:109]
            action, _, selected_id = select_cmlite_candidates(
                cm, task._dof_pos, candidates, task._target_states, goal,
                actual_contact=stable_contact_steps >= 5, config=proposal_config)
            base_action = candidates[:, 0]
        else:
            base_action = normalized_action(model, task.obs_buf, mean, std)
        if ensemble_mode:
            pass
        elif cm is None:
            action = base_action
            selected_id = torch.zeros(batch, dtype=torch.long, device=device)
        else:
            progress = task.progress_buf.long()
            goal_index = torch.minimum(progress + 1, lengths - 1)
            goal = task.hoi_refs[task.data_id, task.ref_index, goal_index, 106:109]
            action, _, selected_id = select_cmlite_action(
                cm, task._dof_pos, base_action, task._target_states, goal,
                actual_contact=stable_contact_steps >= 5, config=proposal_config)
        _, reward, done, _ = player.env_step(player.env, action)
        active = ~recorded
        rewards += reward.reshape(-1) * active
        steps += active
        lift = task._target_states[:, 2] - initial_z
        hand_contact = (task._contact_forces[:, task._contact_body_ids].norm(dim=-1) > .1).any(dim=-1)
        object_contact = task._tar_contact_forces.norm(dim=-1) > .1
        held = hand_contact & object_contact
        stable_contact_steps = torch.where(
            held, stable_contact_steps + 1, torch.zeros_like(stable_contact_steps))
        active_lift = torch.where(active, lift, torch.zeros_like(lift))
        max_lift = torch.maximum(max_lift, active_lift)
        max_contact_lift = torch.maximum(
            max_contact_lift, torch.where(active & held, lift, torch.zeros_like(lift)))
        contact_steps += (active & held).float()
        airborne_steps += (active & (lift >= .03)).float()
        lift_run = torch.where(active & held & (lift >= .03), lift_run + 1,
                               torch.zeros_like(lift_run))
        max_lift_run = torch.maximum(max_lift_run, lift_run)
        success |= max_lift_run >= 5
        proposal_count += (active & (selected_id != 0)).long()
        proposal_histogram += torch.bincount(selected_id[active], minlength=5)
        done_indices = done.bool().nonzero(as_tuple=False).reshape(-1)
        new_done = done_indices[~recorded[done_indices]]
        for index in new_done.tolist():
            episode_steps = max(int(steps[index].item()), 1)
            results.append({
                "env_id": index, "motion_id": int(motion_ids[index].item()),
                "start_frame": int(start_frames[index].item()),
                "steps": episode_steps, "reward": float(rewards[index].item()),
                "max_lift_m": float(max_lift[index].item()),
                "max_contact_lift_m": float(max_contact_lift[index].item()),
                "hand_object_contact_fraction": float(contact_steps[index].item() / episode_steps),
                "airborne_fraction": float(airborne_steps[index].item() / episode_steps),
                "max_lift_contact_run_steps": int(max_lift_run[index].item()),
                "lift_success": bool(success[index].item()),
                "proposal_selection_rate": float(proposal_count[index].item() / episode_steps),
            })
        recorded[new_done] = True
        if bool(recorded.all()):
            break
    if len(results) != batch:
        raise RuntimeError(f"only completed {len(results)}/{batch} first episodes")
    lift_success_rate = float(np.mean([item["lift_success"] for item in results]))
    summary = {
        "mode": {"on": "cmlite_on", "off": "cm_off", "teacher": "reference_teacher",
                 "router": "fixed_start_frame_router",
                 "ensemble": "cmlite_expert_ensemble"}[mode],
        "num_episodes": batch, "lift_success_rate": lift_success_rate,
        "mean_max_lift_m": float(np.mean([item["max_lift_m"] for item in results])),
        "mean_max_contact_lift_m": float(np.mean(
            [item["max_contact_lift_m"] for item in results])),
        "mean_contact_fraction": float(np.mean(
            [item["hand_object_contact_fraction"] for item in results])),
        "proposal_selection_rate": float(proposal_count.sum().item() / steps.sum().item()),
        "proposal_histogram": proposal_histogram.cpu().tolist(),
        "lift_success_definition": "object dz >= 0.03 m with hand+object contact for >=5 consecutive steps",
        "episode_sampling": "first completed episode from every parallel environment",
    }
    if router_mode:
        summary["route_histogram"] = {
            name: route_names.count(name) for name in sorted(set(route_names))}
    if ensemble_mode:
        names = ["bc"] + expert_names
        summary["expert_selection_histogram"] = {
            name: int(proposal_histogram[index].item())
            for index, name in enumerate(names)}
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "results.json", {"summary": summary, "per_episode": results})
    _write(output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


@torch.inference_mode()
def _run_episode(player) -> None:
    output = Path(os.environ["REF2DEX_CMLITE_BC_OUTPUT"]).resolve()
    checkpoint_path = Path(os.environ["REF2DEX_CMLITE_BC_CHECKPOINT"]).resolve()
    cm_path = Path(os.environ["REF2DEX_CMLITE_CHECKPOINT"]).resolve()
    mode = os.environ.get("REF2DEX_CMLITE_BC_MODE", "off")
    cm_enabled = mode == "on"
    cm_sha = os.environ["REF2DEX_CMLITE_SHA256"]
    task = player.env.task
    if int(task.control_freq_inv) != 2:
        raise ValueError("CmLite BC evaluation requires a 30 Hz environment")
    if int(task.num_envs) > 1 or mode == "teacher":
        return _run_batched_eval(
            player, output, checkpoint_path, cm_path, mode, cm_sha)
    task._adaptive_kappa_enabled = False
    task._enable_early_termination = False
    player.has_batch_dimension = True
    player.env_reset()
    device = task.obs_buf.device
    model, mean, std = _load_bc(checkpoint_path, device)
    cm = FrozenCmLite(str(cm_path), device, cm_sha) if cm_enabled else None
    proposal_config = ProposalConfig(
        wrist_z_delta=float(os.environ.get("REF2DEX_CMLITE_WRIST_Z_DELTA", "0.01")),
        finger_delta=float(os.environ.get("REF2DEX_CMLITE_FINGER_DELTA", "0.04")),
        contact_threshold=float(os.environ.get("REF2DEX_CMLITE_CONTACT_THRESHOLD", "0.5")),
    )
    initial_z = float(task._target_states[0, 2])
    start = int(task.start_times[0])
    length = int(task.max_episode_length[task.data_id[0]])
    max_steps = max(1, length - start)
    poses, contacts, actions, base_actions, proposal_ids, rewards = [], [], [], [], [], []
    transition_q, transition_states, transition_next_states = [], [], []
    transition_hand_contact, transition_object_contact = [], []
    stable_contact_steps = 0
    cm_scores = []
    for _ in range(max_steps):
        progress = int(task.progress_buf[0])
        base_action = normalized_action(model, task.obs_buf, mean, std)
        if cm is None:
            action = base_action
            selected_id = torch.zeros(1, dtype=torch.long, device=device)
            scores = torch.zeros(1, 5, device=device)
        else:
            hand_contact = (task._contact_forces[:, task._contact_body_ids].norm(dim=-1) > 0.1).any(dim=-1)
            object_contact = task._tar_contact_forces.norm(dim=-1) > 0.1
            goal_index = min(progress + 1, length - 1)
            goal = task.hoi_refs[task.data_id, task.ref_index, goal_index, 106:109]
            action, scores, selected_id = select_cmlite_action(
                cm, task._dof_pos, base_action, task._target_states, goal,
                actual_contact=torch.full((1,), stable_contact_steps >= 5,
                                          dtype=torch.bool, device=device), config=proposal_config)
        q_before = task._dof_pos.clone()
        state_before = task._target_states.clone()
        _, reward, done, _ = player.env_step(player.env, action)
        hand_contact_after = (task._contact_forces[:, task._contact_body_ids].norm(dim=-1) > .1).any(dim=-1)
        object_contact_after = task._tar_contact_forces.norm(dim=-1) > .1
        transition_q.append(q_before[0].cpu().clone())
        transition_states.append(state_before[0].cpu().clone())
        transition_next_states.append(task._target_states[0].cpu().clone())
        transition_hand_contact.append(hand_contact_after[0].cpu().clone())
        transition_object_contact.append(object_contact_after[0].cpu().clone())
        stable_contact_steps = stable_contact_steps + 1 if bool((hand_contact_after & object_contact_after)[0]) else 0
        poses.append(task._target_states[0, :7].cpu().numpy().copy())
        contacts.append(float((task._contact_forces[0, task._contact_body_ids].norm(dim=-1) > .1).float().mean()))
        actions.append(action[0].cpu().numpy().copy())
        base_actions.append(base_action[0].cpu().numpy().copy())
        proposal_ids.append(int(selected_id[0]))
        rewards.append(float(reward.reshape(-1)[0]))
        cm_scores.append(scores[0].cpu().numpy().copy())
        if bool(done.reshape(-1)[0]):
            break
    object_pose = np.asarray(poses, dtype=np.float32)
    lift = object_pose[:, 2] - initial_z
    summary = {
        "mode": "cmlite_on" if cm_enabled else "cm_off",
        "steps": len(poses), "initial_object_z_m": initial_z,
        "max_lift_m": float(lift.max()), "final_lift_m": float(lift[-1]),
        "frames_lift_gt_2cm": int((lift > .02).sum()),
        "frames_lift_gt_5cm": int((lift > .05).sum()),
        "mean_contact_occupancy": float(np.mean(contacts)),
        "max_contact_occupancy": float(np.max(contacts)),
        "mean_reward": float(np.mean(rewards)),
        "proposal_selection_rate": float(np.mean(np.asarray(proposal_ids) != 0)),
        "proposal_histogram": np.bincount(proposal_ids, minlength=5).tolist(),
    }
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "rollout.npz", object_pose=object_pose,
        contact_occupancy=np.asarray(contacts), actions=np.asarray(actions),
        base_actions=np.asarray(base_actions), proposal_ids=np.asarray(proposal_ids),
        cm_scores=np.asarray(cm_scores), rewards=np.asarray(rewards))
    if cm is None:
        torch.save({
            "schema": "ref2dex.cmlite_transition.v1",
            "q": torch.stack(transition_q), "action": torch.from_numpy(np.asarray(actions)),
            "object_state": torch.stack(transition_states),
            "next_object_state": torch.stack(transition_next_states),
            "hand_contact": torch.stack(transition_hand_contact).reshape(-1, 1),
            "object_contact": torch.stack(transition_object_contact).reshape(-1, 1),
        }, output / "transitions.pt")
    _write(output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-checkpoint", type=Path, required=True)
    parser.add_argument("--cmlite-checkpoint", type=Path, required=True)
    parser.add_argument("--cmlite-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("off", "on", "teacher", "router", "ensemble"), required=True)
    parser.add_argument("--router-checkpoint", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--router-map", help="JSON object mapping start frames to expert names")
    parser.add_argument("--physical-gpu", type=int, required=True)
    args, passthrough = parser.parse_known_args(argv)
    for path in (args.bc_checkpoint, args.cmlite_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if len(args.cmlite_sha256) != 64 or _sha256(args.cmlite_checkpoint) != args.cmlite_sha256:
        raise ValueError("CmLite checkpoint SHA256 mismatch")
    if args.output.exists():
        raise FileExistsError(args.output)
    router_checkpoints = {}
    for item in args.router_checkpoint:
        if "=" not in item:
            raise ValueError("--router-checkpoint must be NAME=PATH")
        name, raw_path = item.split("=", 1)
        path = Path(raw_path).resolve()
        if not name or name in router_checkpoints or not path.is_file():
            raise ValueError("router checkpoint names must be unique and files must exist")
        router_checkpoints[name] = path
    if args.mode == "router":
        if not args.router_map or not router_checkpoints:
            raise ValueError("router mode requires --router-map and PPO checkpoints")
        parsed_route_map = json.loads(args.router_map)
        if not isinstance(parsed_route_map, dict):
            raise ValueError("router map must be a JSON object")
    elif args.mode == "ensemble":
        if args.router_map or not router_checkpoints:
            raise ValueError("ensemble mode requires PPO checkpoints and no router map")
    elif args.router_map or router_checkpoints:
        raise ValueError("router arguments require --mode router")
    manifest = {
        "run_status": "RUNNING", "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "physical_gpu": args.physical_gpu, "mode": args.mode,
        "bc_checkpoint": None if args.mode == "teacher" else {
            "path": str(args.bc_checkpoint.resolve()), "sha256": _sha256(args.bc_checkpoint),
            "official_policy_checkpoint": None},
        "cmlite_checkpoint": {"path": str(args.cmlite_checkpoint.resolve()), "sha256": args.cmlite_sha256},
        "router_map": json.loads(args.router_map) if args.router_map else None,
        "router_checkpoints": {
            name: {"path": str(path), "sha256": _sha256(path),
                   "official_policy_checkpoint": None}
            for name, path in router_checkpoints.items()},
        "official_policy_checkpoint": None,
    }
    _write(args.output / "run_manifest.json", manifest)
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(args.physical_gpu),
        "REF2DEX_CMLITE_BC_OUTPUT": str((args.output / "rollout").resolve()),
        "REF2DEX_CMLITE_BC_CHECKPOINT": str(args.bc_checkpoint.resolve()),
        "REF2DEX_CMLITE_CHECKPOINT": str(args.cmlite_checkpoint.resolve()),
        "REF2DEX_CMLITE_SHA256": args.cmlite_sha256,
        "REF2DEX_CMLITE_BC_MODE": args.mode,
        "PYTHONPATH": os.pathsep.join((str(ROOT), str(DEXPLORE), env.get("PYTHONPATH", ""))),
    })
    if args.mode in ("router", "ensemble"):
        if args.mode == "router":
            env["REF2DEX_POLICY_ROUTE_MAP"] = args.router_map
        env["REF2DEX_POLICY_ROUTE_CHECKPOINTS"] = json.dumps(
            {name: str(path) for name, path in router_checkpoints.items()}, sort_keys=True)
    env["REF2DEX_CMLITE_BC_OUTPUT"] = str((args.output / "rollout").resolve())
    sys.path.insert(0, str(DEXPLORE / "dexplore"))
    try:
        import run as dexplore_run
        from learning import dexplore_players

        original_player = dexplore_players.DexplorePlayerContinuous

        class CmLiteBcPlayer(original_player):
            def __init__(self, params=None, **kwargs):
                # The pinned rl_games release passes the runner config by
                # keyword, while the legacy player accepts it positionally.
                super().__init__(params if params is not None else kwargs)

            def run(self):
                return _run_episode(self)

        # run.py's player factory resolves this module attribute at creation
        # time, so replacing it keeps the standard config and environment path.
        dexplore_players.DexplorePlayerContinuous = CmLiteBcPlayer
        os.environ.update(env)
        sys.argv = [str(DEXPLORE / "dexplore/run.py")] + passthrough
        dexplore_run.main()
        summary = json.loads((args.output / "rollout/summary.json").read_text())
        manifest.update(run_status="COMPLETED", completed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), summary=summary)
        _write(args.output / "run_manifest.json", manifest)
        return 0
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        _write(args.output / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

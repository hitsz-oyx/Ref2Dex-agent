"""Evaluate a no-checkpoint BC policy with an optional CmLite action selector.

The evaluator runs the same locally trained policy in two matched arms.  The
off arm executes the policy action.  The on arm scores five deterministic local
proposals with the frozen CmLite model and executes the selected proposal.  No
DExplore actor checkpoint is loaded by either arm.
"""
from __future__ import annotations

import argparse
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

from src.task.CmResidual.cmlite import FrozenCmLite
from src.task.CmResidual.cmlite_policy_select import ProposalConfig, select_cmlite_action
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
def _run_episode(player) -> None:
    output = Path(os.environ["REF2DEX_CMLITE_BC_OUTPUT"]).resolve()
    checkpoint_path = Path(os.environ["REF2DEX_CMLITE_BC_CHECKPOINT"]).resolve()
    cm_path = Path(os.environ["REF2DEX_CMLITE_CHECKPOINT"]).resolve()
    cm_enabled = os.environ.get("REF2DEX_CMLITE_BC_MODE", "off") == "on"
    cm_sha = os.environ["REF2DEX_CMLITE_SHA256"]
    task = player.env.task
    if int(task.num_envs) != 1 or int(task.control_freq_inv) != 2:
        raise ValueError("CmLite BC evaluation requires one 30 Hz environment")
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
    parser.add_argument("--mode", choices=("off", "on"), required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args, passthrough = parser.parse_known_args(argv)
    for path in (args.bc_checkpoint, args.cmlite_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if len(args.cmlite_sha256) != 64 or _sha256(args.cmlite_checkpoint) != args.cmlite_sha256:
        raise ValueError("CmLite checkpoint SHA256 mismatch")
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = {
        "run_status": "RUNNING", "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "physical_gpu": args.physical_gpu, "mode": args.mode,
        "bc_checkpoint": {"path": str(args.bc_checkpoint.resolve()), "sha256": _sha256(args.bc_checkpoint),
                          "official_policy_checkpoint": None},
        "cmlite_checkpoint": {"path": str(args.cmlite_checkpoint.resolve()), "sha256": args.cmlite_sha256},
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

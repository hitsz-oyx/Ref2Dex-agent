"""Exact HF02 six-expert temporal-option evaluator.

The module is import-safe on a CPU-only machine.  Isaac Gym is imported only
after the contract has passed ``--dry-run`` and a real collection was
explicitly requested.  The collector is intentionally Cm-off: it records a
known-propensity option dataset for the offline Probe.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable, List, Mapping, Optional


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.task.CmResidual.six_expert_support_adapter import (  # noqa: E402
    validate_payload as validate_support_payload,
)

COLLECTOR_CONFIG = ROOT / "src/task/CmResidual/configs/airplane_temporal_expert_probe.json"
ROUTE_CONFIG = ROOT / "src/task/CmResidual/configs/hf02_temporal_canonical_route.json"
EVALUATOR_PATH = Path(__file__).resolve()
PINNED_ROOT = Path(os.environ.get("REF2DEX_MAIN_ROOT", str(ROOT))).resolve()


def _artifact_path(relative: str | Path) -> Path:
    value = Path(relative)
    if value.is_absolute():
        if value.is_file():
            return value.resolve()
        try:
            relative = value.relative_to(ROOT)
        except ValueError:
            return value
        return (PINNED_ROOT / relative).resolve()
    local = (ROOT / value).resolve()
    if local.is_file():
        return local
    return (PINNED_ROOT / value).resolve()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _observation_hash(observation) -> str:
    array = observation.detach().cpu().contiguous().numpy()
    return _sha256_bytes(array.tobytes())

# The contract module uses PyTorch for tensor/schema checks.  Isaac Gym must
# be imported before PyTorch in a real simulator process, while ``--dry-run``
# must remain import-safe on CPU-only hosts.  Load the contract lazily so the
# real path can establish Isaac Gym's required import order first.
_CONTRACT_LOADED = False


def _load_contract() -> None:
    global _CONTRACT_LOADED
    if _CONTRACT_LOADED:
        return
    from src.task.CmResidual.temporal_option_contract import (  # noqa: E402
        RECORD_SCHEMA,
        RUN_SCHEMA,
        balanced_environment_assignment,
        eligible_trigger_mask,
        git_blob_sha1,
        option_active_mask,
        record_future_step,
        sha256_path,
        validate_frozen_contract,
        validate_record_payload,
    )
    globals().update({
        "RECORD_SCHEMA": RECORD_SCHEMA,
        "RUN_SCHEMA": RUN_SCHEMA,
        "balanced_environment_assignment": balanced_environment_assignment,
        "eligible_trigger_mask": eligible_trigger_mask,
        "git_blob_sha1": git_blob_sha1,
        "option_active_mask": option_active_mask,
        "record_future_step": record_future_step,
        "sha256_path": sha256_path,
        "validate_frozen_contract": validate_frozen_contract,
        "validate_record_payload": validate_record_payload,
    })
    _CONTRACT_LOADED = True


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _flag_value(argv: Iterable[str], flag: str) -> str:
    values = list(argv)
    for index, value in enumerate(values):
        if value == flag:
            if index + 1 >= len(values):
                raise ValueError(f"missing value for {flag}")
            return values[index + 1]
        if value.startswith(flag + "="):
            return value.split("=", 1)[1]
    raise ValueError(f"missing required evaluator argument {flag}")


def _has_flag(argv: Iterable[str], flag: str) -> bool:
    return any(value == flag or value.startswith(flag + "=")
               for value in argv)


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _preflight(collector_path: Path, route_path: Path, verify_artifacts: bool) -> dict:
    _load_contract()
    return validate_frozen_contract(
        ROOT, collector_path, route_path, verify_artifacts=verify_artifacts,
        artifact_root=PINNED_ROOT)


def _dry_run_report(provenance: Mapping[str, object], split: Optional[str]) -> dict:
    collector = provenance["collector"]
    route = provenance["route"]
    splits = collector["collections"]
    chosen = [split] if split else ["fit", "holdout"]
    assignments = {}
    for name in chosen:
        spec = splits[name]
        values = balanced_environment_assignment(
            spec["num_envs"], collector["assignment"]["arms"],
            spec["assignment_seed"])
        assignments[name] = {
            "num_envs": int(spec["num_envs"]),
            "assignment_seed": int(spec["assignment_seed"]),
            "arm_counts": {
                str(arm): int((values == arm).sum())
                for arm in range(collector["assignment"]["arms"])
            },
            "propensity": "1/6",
        }
    return {
        "schema": RUN_SCHEMA,
        "dry_run": True,
        "isaacgym_imported": any(
            name == "isaacgym" or name.startswith("isaacgym.")
            for name in sys.modules),
        "collector_config_sha256": provenance["collector_config_sha256"],
        "route_config_sha256": provenance["route_config_sha256"],
        "route_mode": route["route_mode"],
        "objects": route["objects"],
        "candidate_experts": collector["candidate_experts"],
        "motions": collector["motions"],
        "history_steps": collector["temporal_contract"]["history_steps"],
        "option_steps": collector["temporal_contract"]["option_steps"],
        "future_steps": collector["temporal_contract"]["future_steps"],
        "assignments": assignments,
        "artifact_hashes_verified": bool(provenance["artifact_sizes"]),
        "evaluator_sha256": sha256_path(EVALUATOR_PATH),
        "evaluator_git_blob_sha1": git_blob_sha1(EVALUATOR_PATH),
    }


def _runtime_contract(args: argparse.Namespace, remaining: List[str],
                      provenance: Mapping[str, object]) -> dict:
    collector = provenance["collector"]
    route = provenance["route"]
    if args.split not in ("fit", "holdout"):
        raise ValueError("a fit or holdout split is required for collection")
    split = collector["collections"][args.split]
    simulator_seed = int(_flag_value(remaining, "--seed"))
    assignment_seed = int(args.assignment_seed)
    if simulator_seed != split["simulator_seed"]:
        raise ValueError("simulator seed does not own the selected split")
    if assignment_seed != split["assignment_seed"]:
        raise ValueError("assignment seed does not own the selected split")
    num_envs = int(_flag_value(remaining, "--num_envs"))
    if num_envs != split["num_envs"]:
        raise ValueError("num_envs differs from the frozen split contract")
    if not _has_flag(remaining, "--disable-early-termination"):
        raise ValueError("first-episode boundary requires --disable-early-termination")
    for forbidden in ("--cm-checkpoint", "--reference-action-lead"):
        if _has_flag(remaining, forbidden):
            raise ValueError(f"forbidden non-Cm-off option: {forbidden}")
    if _has_flag(remaining, "--cm-mode") and _flag_value(remaining, "--cm-mode") != "off":
        raise ValueError("temporal option collection must remain Cm-off")

    checkpoint = _artifact_path(_flag_value(remaining, "--checkpoint"))
    expected_checkpoint = _artifact_path(
        route["experts"][route["base_expert"]]["checkpoint"])
    if checkpoint != expected_checkpoint:
        raise ValueError("base checkpoint must be canonical source_e260")
    motion_root = _artifact_path(_flag_value(remaining, "--motion_file"))
    expected_motion_root = _artifact_path(collector["motion_root"])
    if motion_root != expected_motion_root:
        raise ValueError("motion root differs from canonical three-airplane root")
    output = Path(_flag_value(remaining, "--output")).resolve()
    record_output = Path(args.record_output).resolve()
    if output.parent != record_output.parent:
        raise ValueError("record output must be beside the evaluator result")
    if output.exists() or output.parent.exists() or record_output.exists():
        raise FileExistsError("all collection outputs must be new")

    router_spec = collector.get("frozen_c1_router", {})
    router_model = (Path(args.observation_router_model).resolve()
                    if args.observation_router_model is not None
                    else _artifact_path(router_spec.get("path", "")))
    router_sha = args.observation_router_sha256 or router_spec.get("sha256")
    if not router_model.is_file():
        raise FileNotFoundError(f"frozen C1 router missing: {router_model}")
    if not router_sha or _sha256_bytes(router_model.read_bytes()) != router_sha:
        raise ValueError("frozen C1 router hash drift")

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit():
        raise RuntimeError("one explicit physical CUDA_VISIBLE_DEVICES index required")
    used = subprocess.check_output([
        "nvidia-smi", f"--id={visible}", "--query-gpu=memory.used",
        "--format=csv,noheader,nounits"], text=True).strip()
    if int(used) > 1024:
        raise RuntimeError(f"GPU{visible} occupied: {used} MiB")

    assignments = balanced_environment_assignment(
        num_envs, collector["assignment"]["arms"], assignment_seed)
    return {
        "split": args.split,
        "simulator_seed": simulator_seed,
        "assignment_seed": assignment_seed,
        "num_envs": num_envs,
        "minimum_rows_per_arm": split["minimum_valid_rows_per_arm"],
        "checkpoint": checkpoint,
        "motion_root": motion_root,
        "output": output,
        "record_output": record_output,
        "router_model": router_model,
        "router_model_sha256": router_sha,
        "physical_gpu": int(visible),
        "assignments": assignments,
        "contract_provenance": provenance,
    }


def _build_player(original, routed, torch, collector: Mapping[str, object],
                  runtime: Mapping[str, object]):
    """Create the Isaac Gym player only after CPU contract validation."""
    # ``main`` replaces ``original.EvalPlayer`` with the temporal subclass
    # below.  Capture the untouched class now so candidate action generation
    # cannot recursively call the replacement class.
    base_eval_player = original.EvalPlayer
    history_steps = collector["temporal_contract"]["history_steps"]
    option_steps = collector["temporal_contract"]["option_steps"]
    future_steps = collector["temporal_contract"]["future_steps"]
    experts = list(collector["candidate_experts"])
    motion_names = list(collector["motions"])
    expert_provenance = [
        {"name": name, "checkpoint_sha256": runtime["contract_provenance"]
         ["expert_checkpoint_sha256"][name]}
        for name in experts
    ]
    router_model_sha256 = runtime["router_model_sha256"]
    route_config_sha256 = runtime["contract_provenance"]["route_config_sha256"]
    collector_config_sha256 = runtime["contract_provenance"]["collector_config_sha256"]

    class TemporalOptionPlayer(routed.RoutedPlayer):
        """Run the frozen route and intervene with one assigned expert option."""

        def __init__(self, config):
            super().__init__(config)
            self.step_index = 0
            self.assignment = None
            self.assignment_propensity = None
            self.triggered = None
            self.episode_ended = None
            self.option_elapsed = None
            self.future_valid = None
            self.future_elapsed = None
            self.future_contact_mask = None
            self.future_contact_supported_lift_m = None
            self.trigger_step = None
            self.trigger_motion = None
            self.trigger_motion_name = None
            self.trigger_start_frame = None
            self.trigger_object_id = None
            self.trigger_history_state = None
            self.trigger_history_action = None
            self.trigger_history_contact = None
            self.trigger_state = None
            self.trigger_base_action = None
            self.trigger_candidate_actions = None
            self.trigger_pre_action_observation = None
            self.trigger_episode_id = None
            self.trigger_split = None
            self.trigger_object_pose_t = None
            self.trigger_object_pose_t_plus_1 = None
            self.trigger_target_delta = None
            self.trigger_object_state_t = None
            self.trigger_pose_pending = None
            self.trigger_executed_action = None
            self.trigger_router_teacher_candidate_id = None
            self.trigger_router_teacher_action = None
            self.trigger_router_input_state_sha256 = None
            self.trigger_start_z = None
            self.history_state = None
            self.history_action = None
            self.history_contact = None
            self.history_count = None
            self.last_option_active = None
            self.option_candidate_action = None
            self.option_executed_action = None

        @staticmethod
        def _contact(task):
            hand = (task._contact_forces[:, task._contact_body_ids]
                    .norm(dim=-1) > .1).any(-1)
            obj = task._tar_contact_forces.norm(dim=-1) > .1
            return hand & obj

        @staticmethod
        def _state(task):
            q = task._dof_pos.clone()
            q[:, :3] -= task._target_states[:, :3]
            return torch.cat((q, task._dof_vel.clone(),
                              task._target_states.clone()), dim=1)

        def _router_teacher(self, obs_dict, rows):
            if self.observation_router is None:
                raise RuntimeError("C1 router teacher is unavailable")
            observation = obs_dict.get("obs")
            if observation is None or observation.ndim != 2 or observation.shape[1] != 1442:
                raise ValueError("canonical pre-action observation must be [N,1442]")
            values = observation.index_select(0, rows)
            if not torch.isfinite(values).all():
                raise FloatingPointError("nonfinite canonical pre-action observation")
            prediction = self.observation_router.predict(
                values.detach().cpu().numpy())
            unknown = set(prediction) - set(experts)
            if unknown:
                raise ValueError(f"C1 router predicted unknown experts: {sorted(unknown)}")
            choices = torch.tensor(
                [experts.index(name) for name in prediction],
                dtype=torch.long, device=self.device)
            hashes = [_observation_hash(values[index])
                      for index in range(values.shape[0])]
            return values, choices, hashes

        @staticmethod
        def _object_local_delta(state_t, state_next):
            if (state_t.ndim != 2 or state_next.ndim != 2 or
                    state_t.shape != state_next.shape or state_t.shape[1] != 13):
                raise ValueError("object state must be [N,13] for local transform")
            from src.task.CmResidual.cmlite import local_translation_target
            delta = local_translation_target(state_t, state_next)
            if not torch.isfinite(delta).all():
                raise FloatingPointError("nonfinite object-local one-step delta")
            return delta

        def restore(self, filename):
            super().restore(filename)
            task = self.env.task
            if list(self.expert_names) != experts:
                raise ValueError("expert order differs from canonical six-arm route")
            if set(task.object_name) != {"airplane"}:
                raise ValueError("temporal option collector requires airplane only")
            if task.num_motions != len(motion_names):
                raise ValueError("simulator motion count differs from canonical route")
            if task.num_envs != runtime["num_envs"]:
                raise ValueError("environment count differs from selected split")
            source_index = self.expert_names.index(collector["base_expert"])
            if self.route_by_motion is None or not torch.all(
                    self.route_by_motion == source_index):
                # Observation routing is the teacher label; the post-option
                # fallback remains the canonical source_e260 route.
                self.route_by_motion = torch.full(
                    (task.num_envs,), source_index, dtype=torch.long,
                    device=self.device)
            if self.observation_router is None:
                raise ValueError("frozen C1 observation router is required")

            n, device = task.num_envs, self.device
            self.assignment = runtime["assignments"].to(device=device)
            self.assignment_propensity = torch.full(
                (n,), 1.0 / len(experts), dtype=torch.float32, device=device)
            self.triggered = torch.zeros(n, dtype=torch.bool, device=device)
            self.episode_ended = torch.zeros(n, dtype=torch.bool, device=device)
            self.option_elapsed = torch.zeros(n, dtype=torch.long, device=device)
            self.future_valid = torch.zeros(n, dtype=torch.bool, device=device)
            self.future_elapsed = torch.zeros(n, dtype=torch.long, device=device)
            self.future_contact_mask = torch.zeros(
                n, future_steps, dtype=torch.bool, device=device)
            self.future_contact_supported_lift_m = torch.zeros(
                n, future_steps, dtype=torch.float32, device=device)
            self.trigger_step = torch.full((n,), -1, dtype=torch.long, device=device)
            self.trigger_motion = torch.full((n,), -1, dtype=torch.long, device=device)
            self.trigger_motion_name = [None] * n
            self.trigger_start_frame = torch.full((n,), -1, dtype=torch.long, device=device)
            self.trigger_object_id = torch.full((n,), -1, dtype=torch.long, device=device)
            self.trigger_history_state = torch.zeros(
                n, history_steps, 49, device=device)
            self.trigger_history_action = torch.zeros(
                n, history_steps, 18, device=device)
            self.trigger_history_contact = torch.zeros(
                n, history_steps, dtype=torch.bool, device=device)
            self.trigger_state = torch.zeros(n, 49, device=device)
            self.trigger_base_action = torch.zeros(n, 18, device=device)
            self.trigger_candidate_actions = torch.zeros(
                n, len(experts), 18, device=device)
            self.trigger_pre_action_observation = torch.zeros(
                n, 1442, device=device)
            self.trigger_object_pose_t = torch.zeros(n, 3, device=device)
            self.trigger_object_pose_t_plus_1 = torch.zeros(n, 3, device=device)
            self.trigger_target_delta = torch.zeros(n, 3, device=device)
            self.trigger_object_state_t = torch.zeros(n, 13, device=device)
            self.trigger_pose_pending = torch.zeros(n, dtype=torch.bool, device=device)
            self.trigger_executed_action = torch.zeros(n, 18, device=device)
            self.trigger_router_teacher_candidate_id = torch.full(
                (n,), -1, dtype=torch.long, device=device)
            self.trigger_router_teacher_action = torch.zeros(n, 18, device=device)
            self.trigger_router_input_state_sha256 = [None] * n
            self.trigger_episode_id = [None] * n
            self.trigger_split = [None] * n
            self.trigger_start_z = torch.full((n,), float("nan"), device=device)
            self.history_state = torch.zeros(n, history_steps, 49, device=device)
            self.history_action = torch.zeros(n, history_steps, 18, device=device)
            self.history_contact = torch.zeros(
                n, history_steps, dtype=torch.bool, device=device)
            self.history_count = torch.zeros(n, dtype=torch.long, device=device)
            self.last_option_active = torch.zeros(n, dtype=torch.bool, device=device)
            self.option_candidate_action = torch.zeros(
                n, option_steps, 18, device=device)
            self.option_executed_action = torch.zeros(
                n, option_steps, 18, device=device)

        def env_reset(self, env_ids=None):
            result = super().env_reset(env_ids)
            if self.history_state is None:
                return result
            if env_ids is None:
                ids = torch.arange(self.history_count.numel(), device=self.device)
            else:
                # rl_games may pass reset IDs as a Python list rather than a
                # tensor.  Normalize both forms before indexing the temporal
                # buffers; the first-episode boundary semantics are unchanged.
                ids = torch.as_tensor(env_ids, device=self.device).reshape(-1).long()
            if ids.numel():
                self.history_state[ids] = 0
                self.history_action[ids] = 0
                self.history_contact[ids] = False
                self.history_count[ids] = 0
            return result

        def _candidate_actions(self, obs_dict, deterministic):
            source_model, source_rms = self.model, self.running_mean_std
            actions = []
            try:
                for name in experts:
                    self.model, self.running_mean_std = self.expert_models[name]
                    actions.append(base_eval_player.get_action(
                        self, obs_dict, deterministic).detach().clone())
            finally:
                self.model, self.running_mean_std = source_model, source_rms
            candidates = torch.stack(actions, dim=1)
            if (not torch.isfinite(candidates).all() or
                    (candidates.abs() > 1 + 1e-5).any()):
                raise FloatingPointError("invalid expert candidate action")
            return candidates

        @torch.no_grad()
        def get_action(self, obs_dict, is_determenistic=False):
            task = self.env.task
            candidates = self._candidate_actions(obs_dict, is_determenistic)
            # The canonical post-option route remains source_e260.  The C1
            # observation router is recorded as a teacher label at trigger
            # time and is never silently replaced by this fallback.
            choice = self.route_by_motion[task.data_id.long()]
            rows = torch.arange(choice.shape[0], device=self.device)
            base = candidates[rows, choice]
            state = self._state(task)
            contact = self._contact(task)
            eligible = eligible_trigger_mask(
                contact, self.triggered, self.episode_ended,
                task.reset_buf.reshape(-1), task.progress_buf,
                self.history_count, history_steps)
            if eligible.any():
                trigger_rows = eligible.nonzero(as_tuple=False).reshape(-1)
                observation, teacher_id, input_hashes = self._router_teacher(
                    obs_dict, trigger_rows)
                self.triggered[trigger_rows] = True
                self.future_valid[trigger_rows] = True
                self.trigger_step[trigger_rows] = task.progress_buf[trigger_rows].long()
                self.trigger_motion[trigger_rows] = task.data_id[trigger_rows].long()
                self.trigger_start_frame[trigger_rows] = task.start_times[trigger_rows].long()
                self.trigger_object_id[trigger_rows] = task.object_id[
                    task.data_id[trigger_rows].long()].long()
                self.trigger_start_z[trigger_rows] = task._target_states[
                    trigger_rows, 2]
                self.trigger_history_state[trigger_rows] = self.history_state[trigger_rows]
                self.trigger_history_action[trigger_rows] = self.history_action[trigger_rows]
                self.trigger_history_contact[trigger_rows] = self.history_contact[trigger_rows]
                self.trigger_state[trigger_rows] = state[trigger_rows]
                self.trigger_base_action[trigger_rows] = base[trigger_rows]
                self.trigger_candidate_actions[trigger_rows] = candidates[trigger_rows]
                self.trigger_pre_action_observation[trigger_rows] = observation
                self.trigger_object_state_t[trigger_rows] = task._target_states[
                    trigger_rows]
                self.trigger_pose_pending[trigger_rows] = True
                self.trigger_router_teacher_candidate_id[trigger_rows] = teacher_id
                self.trigger_router_teacher_action[trigger_rows] = (
                    candidates[trigger_rows, teacher_id])
                for index, row in enumerate(trigger_rows.tolist()):
                    self.trigger_router_input_state_sha256[row] = input_hashes[index]
                    self.trigger_episode_id[row] = (
                        f"{runtime['split']}-seed{runtime['simulator_seed']}-"
                        f"env{row}-episode0")
                    self.trigger_split[row] = runtime["split"]
                for row in trigger_rows.tolist():
                    motion_id = int(self.trigger_motion[row].item())
                    if motion_id < 0 or motion_id >= len(motion_names):
                        raise ValueError("simulator motion id outside canonical route")
                    self.trigger_motion_name[row] = motion_names[motion_id]

            active = option_active_mask(
                self.triggered, self.episode_ended, self.option_elapsed,
                option_steps)
            executed = base.clone()
            if active.any():
                active_rows = active.nonzero(as_tuple=False).reshape(-1)
                selected = self.assignment[active_rows]
                executed[active_rows] = candidates[active_rows, selected]
                slots = self.option_elapsed[active_rows].long()
                self.option_candidate_action[active_rows, slots] = (
                    executed[active_rows])
                self.option_executed_action[active_rows, slots] = (
                    executed[active_rows])
            if eligible.any():
                self.trigger_executed_action[trigger_rows] = executed[trigger_rows]

            if not torch.isfinite(executed).all():
                raise FloatingPointError("nonfinite executed action")
            live = ~self.episode_ended & ~task.reset_buf.reshape(-1).bool()
            live_rows = live.nonzero(as_tuple=False).reshape(-1)
            if live_rows.numel():
                self.history_state[live_rows, :-1] = self.history_state[
                    live_rows, 1:].clone()
                self.history_action[live_rows, :-1] = self.history_action[
                    live_rows, 1:].clone()
                self.history_contact[live_rows, :-1] = self.history_contact[
                    live_rows, 1:].clone()
                self.history_state[live_rows, -1] = state[live_rows]
                self.history_action[live_rows, -1] = executed[live_rows]
                self.history_contact[live_rows, -1] = contact[live_rows]
                self.history_count[live_rows] = torch.clamp(
                    self.history_count[live_rows] + 1, max=history_steps)
            self.last_option_active = active.detach().clone()
            return executed

        def env_step(self, env, action):
            result = super().env_step(env, action)
            task = env.task
            done = result[2].reshape(-1).bool()
            pending = self.trigger_pose_pending
            if pending.any():
                pose_rows = pending.nonzero(as_tuple=False).reshape(-1)
                state_next = task._target_states[pose_rows].clone()
                delta = self._object_local_delta(
                    self.trigger_object_state_t[pose_rows], state_next)
                # The pair is expressed in the t object-local frame: pose_t
                # is its origin and pose_t+1 is the lossless transformed
                # translation.  No world pose is substituted.
                self.trigger_object_pose_t[pose_rows] = 0
                self.trigger_object_pose_t_plus_1[pose_rows] = delta
                self.trigger_target_delta[pose_rows] = delta
                self.trigger_pose_pending[pose_rows] = False
            active_option = self.last_option_active
            self.option_elapsed[active_option] += 1
            post_contact = self._contact(task)
            lift = task._target_states[:, 2] - self.trigger_start_z
            record_future_step(
                triggered=self.triggered,
                episode_ended=self.episode_ended,
                future_valid=self.future_valid,
                future_elapsed=self.future_elapsed,
                future_contact_mask=self.future_contact_mask,
                future_contact_supported_lift_m=self.future_contact_supported_lift_m,
                post_contact=post_contact,
                lift_m=lift,
                done=done,
                future_steps=future_steps)
            self.episode_ended |= done
            self.step_index += 1
            return result

        def run(self):
            super().run()
            by_env = {int(row["env_id"]): row for row in self.episode_results}
            complete = (self.triggered & self.future_valid &
                        self.future_elapsed.eq(future_steps) &
                        self.option_elapsed.ge(option_steps))
            ids = complete.nonzero(as_tuple=False).reshape(-1).tolist()
            if not ids:
                raise ValueError("no complete first-episode temporal option rows")
            index = torch.tensor(ids, dtype=torch.long, device=self.device)
            if self.trigger_pose_pending[index].any():
                raise ValueError("missing t+1 object-local pose for a complete row")
            if any(self.trigger_episode_id[i] is None or
                   self.trigger_router_input_state_sha256[i] is None
                   for i in ids):
                raise ValueError("missing canonical episode/router provenance")
            if len({self.trigger_episode_id[i] for i in ids}) != len(ids):
                raise ValueError("first-episode episode_id is not globally unique")
            for env_id in ids:
                if env_id not in by_env:
                    raise ValueError(f"missing first-episode result for env {env_id}")
                row = by_env[env_id]
                if (int(row["motion_id"]) != int(self.trigger_motion[env_id]) or
                        int(row["start_frame"]) != int(self.trigger_start_frame[env_id])):
                    raise ValueError("episode reset changed motion/start_frame provenance")

            mask = self.future_contact_mask[index].detach().cpu()
            supported = self.future_contact_supported_lift_m[index].detach().cpu()
            records = {
                "env_id": index.cpu(),
                "motion_id": self.trigger_motion[index].detach().cpu(),
                "motion_name": [self.trigger_motion_name[i] for i in ids],
                "object_name": ["airplane"] * len(ids),
                "simulator_object_id": self.trigger_object_id[index].detach().cpu(),
                "route_expert": [collector["base_expert"]] * len(ids),
                "assignment": self.assignment[index].detach().cpu().to(torch.int8),
                "assignment_propensity": self.assignment_propensity[index].detach().cpu(),
                "trigger_step": self.trigger_step[index].detach().cpu(),
                "start_frame": self.trigger_start_frame[index].detach().cpu(),
                "state": self.trigger_state[index].detach().cpu(),
                "base_action": self.trigger_base_action[index].detach().cpu(),
                "candidate_actions": self.trigger_candidate_actions[index].detach().cpu(),
                "history_state": self.trigger_history_state[index].detach().cpu(),
                "history_action": self.trigger_history_action[index].detach().cpu(),
                "history_contact": self.trigger_history_contact[index].detach().cpu(),
                "option_candidate_action": self.option_candidate_action[index].detach().cpu(),
                "option_executed_action": self.option_executed_action[index].detach().cpu(),
                "future_contact_mask": mask,
                "episode_id": [self.trigger_episode_id[i] for i in ids],
                "split": [self.trigger_split[i] for i in ids],
                "pre_action_observation": self.trigger_pre_action_observation[
                    index].detach().cpu(),
                "object_pose_t_object_local_frame": self.trigger_object_pose_t[
                    index].detach().cpu(),
                "object_pose_t_plus_1_object_local_frame": self.trigger_object_pose_t_plus_1[
                    index].detach().cpu(),
                "target_delta_object_local_1": self.trigger_target_delta[
                    index].detach().cpu(),
                "contact_mask_t_plus_1_to_t_plus_5": mask[:, :5],
                "candidate_expert_names_and_checkpoint_sha256": [
                    expert_provenance for _ in ids],
                "executed_action": self.trigger_executed_action[index].detach().cpu(),
                "router_teacher_candidate_id": self.trigger_router_teacher_candidate_id[
                    index].detach().cpu().to(torch.int8),
                "router_teacher_action": self.trigger_router_teacher_action[index].detach().cpu(),
                "router_model_sha256": [router_model_sha256] * len(ids),
                "router_input_state_sha256": [
                    self.trigger_router_input_state_sha256[i] for i in ids],
                "router_teacher_source": ["c1_observation_router"] * len(ids),
                "route_config_sha256": [route_config_sha256] * len(ids),
                "collector_config_sha256": [collector_config_sha256] * len(ids),
                "future_contact_supported_lift_m": supported,
                "followup_contact_fraction": mask.float().mean(dim=1),
                "followup_max_contact_lift_m": supported.max(dim=1).values.clamp_min(0),
                "final_lift_success": torch.tensor(
                    [bool(by_env[i]["lift_success"]) for i in ids], dtype=torch.bool),
                "final_max_contact_lift_m": torch.tensor(
                    [float(by_env[i]["max_contact_lift_m"]) for i in ids], dtype=torch.float32),
                "final_contact_fraction": torch.tensor(
                    [float(by_env[i]["hand_object_contact_fraction"]) for i in ids], dtype=torch.float32),
                "final_episode_steps": torch.tensor(
                    [int(by_env[i]["steps"]) for i in ids], dtype=torch.int64),
            }
            payload = {
                "schema": RECORD_SCHEMA,
                "run_status": "COMPLETED",
                "candidate_experts": experts,
                "base_expert": collector["base_expert"],
                "post_option_policy": collector["temporal_contract"][
                    "post_option_policy"],
                "history_steps": history_steps,
                "option_steps": option_steps,
                "future_steps": future_steps,
                "provenance": runtime["record_provenance"],
                "records": records,
            }
            summary = validate_record_payload(
                payload, collector, runtime["contract_provenance"])
            if any(count < runtime["minimum_rows_per_arm"]
                   for count in summary["arm_counts"].values()):
                raise ValueError(
                    "valid rows per arm below the frozen split minimum: "
                    f"{summary['arm_counts']}")
            # Admission is fail-closed and happens before torch.save.  The
            # adapter sends every canonical row through main's validator and
            # separately enforces the one-split collection support minimum.
            support_summary = validate_support_payload(
                payload,
                expected_provenance={
                    "expert_checkpoint_sha256": runtime["contract_provenance"][
                        "expert_checkpoint_sha256"],
                    "router_model_sha256": router_model_sha256,
                    "route_config_sha256": route_config_sha256,
                    "collector_config_sha256": collector_config_sha256,
                },
                min_rows_per_arm=runtime["minimum_rows_per_arm"],
            )
            summary["support_contract"] = support_summary
            torch.save(payload, runtime["record_output"])
            runtime["record_summary"] = summary
            print("REF2DEX_TEMPORAL_OPTION " + json.dumps({
                "rows": summary["rows"],
                "arm_counts": summary["arm_counts"],
                "output": str(runtime["record_output"]),
            }, sort_keys=True), flush=True)

    return TemporalOptionPlayer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collector-config", type=Path,
                        default=COLLECTOR_CONFIG)
    parser.add_argument("--route-config", type=Path, default=ROUTE_CONFIG)
    parser.add_argument("--split", choices=("fit", "holdout"))
    parser.add_argument("--assignment-seed", type=int)
    parser.add_argument("--record-output", type=Path)
    parser.add_argument("--observation-router-model", type=Path)
    parser.add_argument("--observation-router-sha256")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-artifact-hashes", action="store_true",
                        help="only for CPU contract smoke; never valid for collection")
    args, remaining = parser.parse_known_args()
    collector_path = args.collector_config.resolve()
    route_path = args.route_config.resolve()
    if not collector_path.is_file() or not route_path.is_file():
        raise FileNotFoundError("collector or canonical route config missing")
    if not args.dry_run:
        # Isaac Gym's dependency guard rejects a process that imported torch
        # first.  The contract is loaded by _preflight immediately after this
        # import, before any simulator/player construction.
        from isaacgym import gymapi  # noqa: F401
    provenance = _preflight(
        collector_path, route_path,
        verify_artifacts=not args.skip_artifact_hashes)
    if args.dry_run:
        report = _dry_run_report(provenance, args.split)
        print("REF2DEX_TEMPORAL_OPTION_DRY_RUN " +
              json.dumps(report, sort_keys=True), flush=True)
        return
    if args.skip_artifact_hashes:
        raise ValueError("artifact hash skipping is allowed only for --dry-run")
    if args.assignment_seed is None or args.record_output is None:
        raise ValueError("collection requires --assignment-seed and --record-output")
    runtime = _runtime_contract(args, remaining, provenance)
    collector = provenance["collector"]
    output = runtime["output"]
    runtime["record_provenance"] = {
        "collector_config": str(collector_path),
        "collector_config_sha256": provenance["collector_config_sha256"],
        "route_config": str(route_path),
        "route_config_sha256": provenance["route_config_sha256"],
        "canonical_route_sha256": collector["canonical_route"]["sha256"],
        "expert_checkpoint_sha256": provenance["expert_checkpoint_sha256"],
        "motion_sha256": provenance["motion_sha256"],
        "router_model": str(runtime["router_model"]),
        "router_model_sha256": runtime["router_model_sha256"],
        "evaluator_path": str(EVALUATOR_PATH),
        "evaluator_sha256": sha256_path(EVALUATOR_PATH),
        "evaluator_git_blob_sha1": git_blob_sha1(EVALUATOR_PATH),
    }

    # Isaac Gym and the simulator modules are deliberately imported only here.
    import torch
    import evaluate as original
    import evaluate_object_router as routed

    output.parent.mkdir(parents=True)
    manifest_path = output.parent / "run_manifest.json"
    manifest = {
        "schema": RUN_SCHEMA,
        "run_status": "STARTED",
        "run_id": output.parent.name,
        "created_at": _now(),
        "git_commit": _git_commit(),
        "split": runtime["split"],
        "simulator_seed": runtime["simulator_seed"],
        "assignment_seed": runtime["assignment_seed"],
        "num_envs": runtime["num_envs"],
        "route_mode": collector["route_mode"],
        "object_name": collector["object_name"],
        "base_expert": collector["base_expert"],
        "candidate_experts": collector["candidate_experts"],
        "history_steps": collector["temporal_contract"]["history_steps"],
        "option_steps": collector["temporal_contract"]["option_steps"],
        "future_steps": collector["temporal_contract"]["future_steps"],
        "assignment_propensity": "1/6",
        "first_episode_only": True,
        "cm_mode": "off",
        "observation_router_model": str(runtime["router_model"]),
        "observation_router_model_sha256": runtime["router_model_sha256"],
        "checkpoint": str(runtime["checkpoint"]),
        "motion_root": str(runtime["motion_root"]),
        "record_output": str(runtime["record_output"]),
        "physical_gpu": runtime["physical_gpu"],
        "max_gpu_count": 1,
        "provenance": runtime["record_provenance"],
        "command": [sys.executable, *sys.argv],
        "stop_rule": "route/checkpoint/motion drift, incomplete first episode/future, nonfinite action, or row minimum failure",
    }
    _write_json(manifest_path, manifest)

    routed.CONFIG = provenance["route"]
    routed.PINNED_ROOT = PINNED_ROOT
    routed.MODEL_PATH = runtime["router_model"]
    routed.OUTPUT_PATH = output
    routed.CM_MODEL = None
    routed.CM_SHA = None
    routed.CM_MODE = "off"
    routed.CM_CANDIDATE_MODE = "experts"
    routed.CM_CONTACT_GATE = "instant"
    sys.argv = [sys.argv[0], *remaining]
    original.EvalPlayer = _build_player(
        original, routed, torch, collector, runtime)
    try:
        original.main()
        result = json.loads(output.read_text())
        if (result["summary"]["num_episodes"] != runtime["num_envs"] or
                not result["summary"]["early_termination_disabled"]):
            raise ValueError("incomplete first-episode evaluation")
        if not runtime.get("record_output", Path()).is_file():
            raise FileNotFoundError(runtime["record_output"])
        manifest.update(
            run_status="COMPLETED",
            completed_at=_now(),
            summary=result["summary"],
            record_summary=runtime.get("record_summary"),
            result_sha256=sha256_path(output),
            record_sha256=sha256_path(runtime["record_output"]),
        )
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["completed_at"] = _now()
        _write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()

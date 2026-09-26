"""Evaluate a frozen, simulator-object-ID route over self-trained PPO actors."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from isaacgym import gymapi  # noqa: F401 - import before torch
import joblib
import torch

import evaluate as original


ROOT = Path(__file__).resolve().parents[3]
CONFIG = None
MODEL_PATH = None
OUTPUT_PATH = None
CM_MODEL = None
CM_SHA = None
CM_MODE = "off"
CM_CANDIDATE_MODE = "experts"
CM_CONTACT_GATE = "instant"
CM_STATS = {}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


class RoutedPlayer(original.EvalPlayer):
    def __init__(self, config):
        super().__init__(config)
        self.expert_models = {}
        self.route_by_motion = None
        self.observation_router = None
        self.cm = None
        self.cm_override_steps = 0
        self.cm_scored_steps = 0
        self.cm_histogram = None
        self.cm_stable_contact_steps = None

    def restore(self, filename):
        super().restore(filename)
        target_keys = set(self.model.state_dict())
        for name, spec in CONFIG["experts"].items():
            path = ROOT / spec["checkpoint"]
            payload = torch.load(path, map_location=self.device, weights_only=False)
            model = copy.deepcopy(self.model)
            # rl_games may compile the training actor while the evaluation
            # actor remains eager (or vice versa). Normalize only that wrapper.
            state = {}
            for key, value in payload["model"].items():
                bare_key = key[len("_orig_mod."):] if key.startswith("_orig_mod.") else key
                target_key = bare_key if bare_key in target_keys else "_orig_mod." + bare_key
                state[target_key] = value
            if set(state) != target_keys:
                raise ValueError(f"expert architecture differs: {name}")
            model.load_state_dict(state, strict=True)
            model.eval().requires_grad_(False)
            rms = copy.deepcopy(self.running_mean_std)
            rms.load_state_dict(payload["running_mean_std"], strict=True)
            rms.eval()
            self.expert_models[name] = (model, rms)
        task = self.env.task
        names = list(self.expert_models)
        self.expert_names = names
        self.cm_stable_contact_steps = torch.zeros(
            task.num_envs, dtype=torch.long, device=self.device)
        if CM_MODEL is not None:
            from src.task.CmResidual.cmlite import FrozenCmLite
            self.cm = FrozenCmLite(str(CM_MODEL), self.device, CM_SHA)
            # Bin zero is the fixed object-route action; the following bins
            # correspond to the frozen experts in ``actions`` order.
            candidate_count = 5 if CM_CANDIDATE_MODE == "local" else len(names) + 1
            self.cm_histogram = torch.zeros(candidate_count, dtype=torch.long,
                                            device=self.device)
            print("REF2DEX_ROUTER_CM " + json.dumps({
                "mode": CM_MODE, "candidate_mode": CM_CANDIDATE_MODE,
                "contact_gate": CM_CONTACT_GATE,
                "checkpoint_sha256": sha256(CM_MODEL),
                "experts": names,
            }, sort_keys=True), flush=True)
        if MODEL_PATH is not None:
            self.observation_router = joblib.load(MODEL_PATH)
            if set(self.observation_router.classes_) - set(names):
                raise ValueError("observation router predicts an unknown expert")
            if self.observation_router.n_features_in_ != 1442:
                raise ValueError("observation router input dimension differs")
            self._needs_route = torch.ones(task.num_envs, device=self.device, dtype=torch.bool)
            self._choice_by_env = torch.full((task.num_envs,), -1, device=self.device,
                                             dtype=torch.long)
            self.initial_expert_names = None
            print("REF2DEX_OBSERVATION_ROUTE_MODEL " + json.dumps({
                "model_sha256": sha256(MODEL_PATH), "experts": names,
            }, sort_keys=True), flush=True)
            return
        route = CONFIG["object_route"]
        motion_objects = [task.object_name[int(object_id)] for object_id in task.object_id]
        unknown = set(motion_objects) - set(route)
        if unknown:
            raise ValueError(f"unmapped objects: {sorted(unknown)}")
        self.route_by_motion = torch.tensor(
            [names.index(route[obj]) for obj in motion_objects],
            device=self.device, dtype=torch.long)
        print("REF2DEX_OBJECT_ROUTE " + json.dumps({
            "motion_objects": motion_objects,
            "motion_experts": [route[obj] for obj in motion_objects],
        }, sort_keys=True), flush=True)

    def env_reset(self, env_ids=None):
        result = super().env_reset(env_ids)
        if self.cm_stable_contact_steps is not None:
            if env_ids is None:
                self.cm_stable_contact_steps.zero_()
            elif len(env_ids):
                self.cm_stable_contact_steps[env_ids.reshape(-1).long()] = 0
        if self.observation_router is not None:
            if env_ids is None:
                self._needs_route[:] = True
            elif len(env_ids):
                self._needs_route[env_ids.reshape(-1).long()] = True
        return result

    @torch.no_grad()
    def get_action(self, obs_dict, is_determenistic=False):
        if self.observation_router is not None:
            pending = self._needs_route.nonzero(as_tuple=False).reshape(-1)
            if pending.numel():
                observation = obs_dict["obs"][pending].detach().cpu().numpy()
                prediction = self.observation_router.predict(observation)
                expert_indices = [self.expert_names.index(name) for name in prediction]
                self._choice_by_env[pending] = torch.tensor(
                    expert_indices, device=self.device, dtype=torch.long)
                self._needs_route[pending] = False
            choice = self._choice_by_env
            if self.initial_expert_names is None and not self._needs_route.any():
                self.initial_expert_names = [self.expert_names[index]
                                             for index in choice.tolist()]
        else:
            if self.route_by_motion is None:
                raise RuntimeError("object route not initialized")
            choice = self.route_by_motion[self.env.task.data_id.long()]
        if (choice < 0).any():
            raise RuntimeError("unassigned observation route")
        source_model, source_rms = self.model, self.running_mean_std
        actions = []
        try:
            for index in range(len(self.expert_names)):
                self.model, self.running_mean_std = self.expert_models[
                    self.expert_names[index]]
                candidate = super().get_action(obs_dict, is_determenistic)
                actions.append(candidate)
        finally:
            self.model, self.running_mean_std = source_model, source_rms
        selected = torch.stack(actions, dim=1)[
            torch.arange(choice.shape[0], device=self.device), choice]
        if self.cm is not None:
            from src.task.CmResidual.cmlite_policy_select import (
                proposal_actions, select_cmlite_candidates)
            task = self.env.task
            base_selected = selected
            if CM_CANDIDATE_MODE == "local":
                candidates = proposal_actions(base_selected)
            else:
                candidates = torch.stack([base_selected] + actions, dim=1)
                # Candidate zero is the fixed object route. The remaining
                # actions are the same frozen experts for every arm; Cm ranks them.
            goal_index = (task.progress_buf + 1).clamp_max(task.hoi_data.shape[1] - 1)
            goal_position = task.hoi_data[task.data_id, goal_index, 106:109]
            hand_contact = (task._contact_forces[:, task._contact_body_ids].norm(dim=-1) > .1).any(dim=-1)
            object_contact = task._tar_contact_forces.norm(dim=-1) > .1
            actual_contact = hand_contact & object_contact
            self.cm_stable_contact_steps = torch.where(
                actual_contact, self.cm_stable_contact_steps + 1,
                torch.zeros_like(self.cm_stable_contact_steps))
            if CM_CONTACT_GATE == "stable":
                actual_contact = self.cm_stable_contact_steps >= 5
            selected, _, selected_id = select_cmlite_candidates(
                self.cm, task._dof_pos, candidates, task._target_states,
                goal_position, actual_contact=actual_contact)
            active = torch.ones(choice.shape[0], dtype=torch.bool, device=self.device)
            self.cm_histogram += torch.bincount(selected_id[active],
                                                minlength=candidates.shape[1])
            self.cm_scored_steps += int(active.sum())
            self.cm_override_steps += int((selected_id.ne(0) & active).sum())
        if selected is None or not torch.isfinite(selected).all():
            raise FloatingPointError("invalid routed action")
        return selected

    def run(self):
        global CM_STATS
        super().run()
        if self.cm is not None:
            CM_STATS = {
                "cm_scored_steps": int(self.cm_scored_steps),
                "cm_override_steps": int(self.cm_override_steps),
                "cm_selection_histogram": self.cm_histogram.detach().cpu().tolist(),
                "cm_candidate_mode": CM_CANDIDATE_MODE,
                "cm_contact_gate": CM_CONTACT_GATE,
            }
        if self.observation_router is not None:
            if self.initial_expert_names is None:
                raise RuntimeError("initial observation route was not recorded")
            write(OUTPUT_PATH.parent / "initial_routes.json", {
                "expert_by_env": self.initial_expert_names,
                "model_sha256": sha256(MODEL_PATH),
            })


def main() -> None:
    global CONFIG, MODEL_PATH, OUTPUT_PATH, CM_MODEL, CM_SHA, CM_MODE, CM_STATS
    global CM_CANDIDATE_MODE, CM_CONTACT_GATE
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--route-config", type=Path, required=True)
    parser.add_argument("--observation-router-model", type=Path)
    parser.add_argument("--observation-router-sha256")
    parser.add_argument("--cm-checkpoint", type=Path)
    parser.add_argument("--cm-sha256")
    parser.add_argument("--cm-mode", choices=("off", "cm"), default="off")
    parser.add_argument("--cm-candidate-mode", choices=("experts", "local"),
                        default="experts")
    parser.add_argument("--cm-contact-gate", choices=("instant", "stable"),
                        default="instant")
    args, remaining = parser.parse_known_args()
    if bool(args.observation_router_model) != bool(args.observation_router_sha256):
        parser.error("observation router model and SHA256 must be specified together")
    if args.observation_router_model is not None:
        MODEL_PATH = args.observation_router_model.resolve()
        if not MODEL_PATH.is_file() or sha256(MODEL_PATH) != args.observation_router_sha256:
            raise ValueError("observation router model drift")
    if bool(args.cm_checkpoint) != bool(args.cm_sha256):
        parser.error("Cm checkpoint and SHA256 must be specified together")
    if args.cm_mode == "cm" and args.cm_checkpoint is None:
        parser.error("cm mode requires --cm-checkpoint")
    CM_MODE = args.cm_mode
    CM_CANDIDATE_MODE = args.cm_candidate_mode
    CM_CONTACT_GATE = args.cm_contact_gate
    if args.cm_checkpoint is not None:
        CM_MODEL = args.cm_checkpoint.resolve()
        CM_SHA = args.cm_sha256
        if not CM_MODEL.is_file() or sha256(CM_MODEL) != CM_SHA:
            raise ValueError("Cm checkpoint drift")
        if args.cm_mode != "cm":
            parser.error("--cm-checkpoint requires --cm-mode cm")
    elif args.cm_mode == "off":
        CM_MODEL = None
        CM_SHA = None
    config_path = args.route_config.resolve()
    CONFIG = json.loads(config_path.read_text())
    for name, spec in CONFIG["experts"].items():
        path = ROOT / spec["checkpoint"]
        if not path.is_file() or sha256(path) != spec["sha256"]:
            raise ValueError(f"expert checkpoint drift: {name}")
    if set(CONFIG["object_route"].values()) - set(CONFIG["experts"]):
        raise ValueError("route refers to missing expert")
    output = Path(remaining[remaining.index("--output") + 1]).resolve()
    OUTPUT_PATH = output
    if output.exists() or output.parent.exists():
        raise FileExistsError("new output directory required")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit():
        raise RuntimeError("one explicit physical GPU required")
    used = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True)
    memory = {int(line.split(",")[0]): int(line.split(",")[1])
              for line in used.splitlines()}[int(visible)]
    if memory > 1024:
        raise RuntimeError(f"GPU{visible} occupied: {memory} MiB")
    output.parent.mkdir(parents=True)
    manifest_path = output.parent / "run_manifest.json"
    manifest = {"run_status": "STARTED", "created_at": now(),
                "run_id": output.parent.name, "work_version": "multitrajectory-object-router-probe",
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                    cwd=ROOT, text=True).strip(),
                "route_config": str(config_path), "route_config_sha256": sha256(config_path),
                "physical_gpu": int(visible), "command": [sys.executable, *sys.argv],
                "checkpoint_roles": "self_trained_only", "cm_enabled": CM_MODEL is not None,
                "route_mode": "observation" if MODEL_PATH is not None else "simulator_object_id",
                "cm_mode": CM_MODE, "cm_checkpoint_sha256": CM_SHA,
                "cm_candidate_mode": CM_CANDIDATE_MODE,
                "cm_contact_gate": CM_CONTACT_GATE,
                "observation_router_model_sha256": sha256(MODEL_PATH) if MODEL_PATH else None,
                "budget": {"gpu_count": 1, "wall_minutes": 10, "output_mb": 50},
                "stop_rule": "input drift, GPU conflict, invalid route or incomplete evaluation"}
    write(manifest_path, manifest)
    sys.argv = [sys.argv[0], *remaining]
    original.EvalPlayer = RoutedPlayer
    try:
        original.main()
        result = json.loads(output.read_text())
        summary = result["summary"]
        if CM_MODEL is not None:
            summary.update(CM_STATS)
        if summary["num_episodes"] != 64 or not summary["early_termination_disabled"]:
            raise ValueError("incomplete first-episode evaluation")
        manifest.update(run_status="COMPLETED", summary=summary)
        print("REF2DEX_ROUTER_RESULT " + json.dumps({
            "run_id": output.parent.name,
            "lift_success_rate": summary["lift_success_rate"],
        }, sort_keys=True), flush=True)
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["completed_at"] = now()
        write(manifest_path, manifest)


if __name__ == "__main__":
    main()

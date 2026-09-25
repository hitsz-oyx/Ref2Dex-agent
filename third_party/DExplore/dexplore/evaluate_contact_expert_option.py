"""Randomized self-trained ten-step expert option at first object contact."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from isaacgym import gymapi  # noqa: F401 - import before torch
import numpy as np
import torch

import evaluate as original
import evaluate_object_router as routed

BASE_PLAYER = original.EvalPlayer


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


class ContactOptionPlayer(routed.RoutedPlayer):
    def restore(self, filename):
        super().restore(filename)
        task = self.env.task
        if set(task.object_name) != {"airplane"} or task.num_motions != 3 or task.num_envs != 64:
            raise ValueError("expected three airplane motions and 64 environments")
        if self.route_by_motion.unique().tolist() != [self.expert_names.index("source_e260")]:
            raise ValueError("source route differs")
        rng = np.random.default_rng(20260925234)
        assignment = np.zeros(task.num_envs, dtype=np.int8)
        initial_motion = np.arange(task.num_envs) % task.num_motions
        for motion in range(task.num_motions):
            ids = np.flatnonzero(initial_motion == motion)
            rng.shuffle(ids)
            assignment[ids[:len(ids) // 2]] = 1
        self.assignment = torch.as_tensor(assignment, device=self.device)
        self.triggered = torch.zeros(64, device=self.device, dtype=torch.bool)
        self.valid = torch.zeros_like(self.triggered)
        self.elapsed = torch.zeros(64, device=self.device, dtype=torch.long)
        self.contact_count = torch.zeros(64, device=self.device, dtype=torch.long)
        self.start_z = torch.full((64,), float("nan"), device=self.device)
        self.final_z = torch.full((64,), float("nan"), device=self.device)
        self.initial_gap = torch.full((64,), float("nan"), device=self.device)
        self.trigger_motion = torch.full((64,), -1, device=self.device, dtype=torch.long)
        self.trigger_step = torch.full((64,), -1, device=self.device, dtype=torch.long)
        self.trigger_obs = torch.full((64, 1442), float("nan"), device=self.device)
        self.trigger_q = torch.full((64, 18), float("nan"), device=self.device)
        self.trigger_object = torch.full((64, 13), float("nan"), device=self.device)
        self.trigger_base_action = torch.full((64, 18), float("nan"), device=self.device)
        self.trigger_candidate_action = torch.full((64, 18), float("nan"), device=self.device)
        self.step_index = 0

    @torch.no_grad()
    def get_action(self, obs_dict, is_determenistic=False):
        base = super().get_action(obs_dict, is_determenistic)
        old_model, old_rms = self.model, self.running_mean_std
        try:
            self.model, self.running_mean_std = self.expert_models["balanced_e360"]
            candidate = BASE_PLAYER.get_action(self, obs_dict, is_determenistic)
        finally:
            self.model, self.running_mean_std = old_model, old_rms
        if (base.shape != (64, 18) or candidate.shape != base.shape or
                not torch.isfinite(base).all() or not torch.isfinite(candidate).all()):
            raise ValueError("invalid source or candidate action")
        self.candidate_action = candidate.detach().clone()
        self.current_obs = obs_dict["obs"].detach().clone()
        return base

    @staticmethod
    def contact(task):
        hand = (task._contact_forces[:, task._contact_body_ids].norm(dim=-1) > .1).any(-1)
        obj = task._tar_contact_forces.norm(dim=-1) > .1
        return hand & obj

    def env_step(self, env, action):
        task = env.task
        self.step_index += 1
        pre_contact = self.contact(task)
        new = (pre_contact & ~self.triggered & (task.reset_buf.reshape(-1) == 0) &
               (task.progress_buf > 0))
        if new.any():
            self.triggered[new] = True
            self.valid[new] = True
            self.start_z[new] = task._target_states[new, 2]
            self.initial_gap[new] = (self.candidate_action[new] - action[new]).norm(dim=-1)
            self.trigger_motion[new] = task.data_id[new]
            self.trigger_step[new] = self.step_index
            self.trigger_obs[new] = self.current_obs[new]
            self.trigger_q[new] = task._dof_pos[new]
            self.trigger_object[new] = task._target_states[new]
            self.trigger_base_action[new] = action[new]
            self.trigger_candidate_action[new] = self.candidate_action[new]
        active = self.triggered & (self.elapsed < 20)
        candidate_mask = active & (self.elapsed < 10) & (self.assignment == 1)
        executed = torch.where(candidate_mask[:, None], self.candidate_action, action)
        result = super().env_step(env, executed)
        if active.any():
            done = result[2].reshape(-1).bool()
            self.valid[active] &= ~done[active]
            post_contact = self.contact(task)
            self.contact_count[active & self.valid] += post_contact[active & self.valid].long()
            self.elapsed[active] += 1
            complete = active & (self.elapsed == 20) & self.valid
            self.final_z[complete] = task._target_states[complete, 2]
        return result

    def run(self):
        super().run()
        payload = {
            "schema": "ref2dex.contact_expert_option.v1",
            "assignment": self.assignment.detach().cpu(),
            "triggered": self.triggered.detach().cpu(),
            "followup_valid": (self.valid & (self.elapsed >= 20)).detach().cpu(),
            "elapsed": self.elapsed.detach().cpu(),
            "contact_count": self.contact_count.detach().cpu(),
            "start_z": self.start_z.detach().cpu(),
            "final_z": self.final_z.detach().cpu(),
            "initial_gap": self.initial_gap.detach().cpu(),
            "trigger_motion": self.trigger_motion.detach().cpu(),
            "trigger_step": self.trigger_step.detach().cpu(),
            "observation": self.trigger_obs.detach().cpu(),
            "q": self.trigger_q.detach().cpu(),
            "object_state": self.trigger_object.detach().cpu(),
            "source_action": self.trigger_base_action.detach().cpu(),
            "candidate_action": self.trigger_candidate_action.detach().cpu(),
        }
        torch.save(payload, routed.OUTPUT_PATH.parent / "option_records.pt")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--route-config", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    config_path = args.route_config.resolve()
    routed.CONFIG = json.loads(config_path.read_text())
    routed.MODEL_PATH = None
    for name, spec in routed.CONFIG["experts"].items():
        path = routed.ROOT / spec["checkpoint"]
        if not path.is_file() or routed.sha256(path) != spec["sha256"]:
            raise ValueError(f"checkpoint drift: {name}")
    output = Path(remaining[remaining.index("--output") + 1]).resolve()
    if output.exists() or output.parent.exists():
        raise FileExistsError("new output directory required")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit():
        raise RuntimeError("one explicit physical GPU required")
    used = subprocess.check_output([
        "nvidia-smi", f"--id={visible}", "--query-gpu=memory.used",
        "--format=csv,noheader,nounits"], text=True).strip()
    if int(used) > 1024:
        raise RuntimeError(f"GPU{visible} occupied: {used} MiB")
    output.parent.mkdir(parents=True)
    routed.OUTPUT_PATH = output
    manifest_path = output.parent / "run_manifest.json"
    manifest = {"run_status": "STARTED", "created_at": now(),
                "run_id": output.parent.name, "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=routed.ROOT, text=True).strip(),
                "route_config_sha256": routed.sha256(config_path),
                "checkpoint_roles": "self_trained_only", "cm_enabled": False,
                "assignment_seed": 20260925234, "option_steps": 10,
                "followup_steps": 20, "physical_gpu": int(visible),
                "budget": {"gpu_count": 1, "wall_minutes": 15, "output_mb": 100},
                "stop_rule": "input drift, GPU conflict, incomplete episode or nonfinite action",
                "command": [sys.executable, *sys.argv]}
    write(manifest_path, manifest)
    sys.argv = [sys.argv[0], *remaining]
    original.EvalPlayer = ContactOptionPlayer
    try:
        original.main()
        result = json.loads(output.read_text())
        if result["summary"]["num_episodes"] != 64 or not result["summary"]["early_termination_disabled"]:
            raise ValueError("incomplete first-episode evaluation")
        record_path = output.parent / "option_records.pt"
        if not record_path.is_file():
            raise FileNotFoundError(record_path)
        manifest.update(run_status="COMPLETED", summary=result["summary"],
                        record_sha256=routed.sha256(record_path))
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["completed_at"] = now()
        write(manifest_path, manifest)


if __name__ == "__main__":
    main()

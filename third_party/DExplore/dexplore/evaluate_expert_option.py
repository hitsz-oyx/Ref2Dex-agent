"""RCT: expert-origin 10-step option versus self-trained option at same states.

The expert is a read-only candidate-action generator for simulator diagnosis,
not a final policy initialization or Cm-on treatment.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from isaacgym import gymapi  # noqa: F401 - must precede torch
import torch

import evaluate as original
from src.task.CmResidual.randomized_action import balanced_assignment
from src.task.CmResidual.randomized_source import validate as validate_source
from src.task.CmResidual.tools.probe_intervention_handflow import sha256


ROOT = Path(__file__).resolve().parents[3]
SELF_TRAINED_SHA = "3212bcc195d374a4d1cb31b21051e4a0a93f63fc345a0fe987279a562092dccb"
EXPERT_SHA = "8f6823db752288f1bddd6d042981d33514e29dac5a68e58726e76215fea6d553"
SPLIT_SHA = "35fffcb500f1f3db76fb8a59c940f5da0a113627bb0be9db0d68b0112fe7f516"
EXPERT = Path("/home2/wyy/oyx_ws/_external/dexplore_official_v120/checkpoint/inspire.pth")
CONFIG = None


class ProbeDone(Exception):
    """Normal stop after final 20-step followup."""


class DualPlayer(original.EvalPlayer):
    def __init__(self, config):
        super().__init__(config)
        self.step_index = 0
        self.generator = torch.Generator(device="cpu").manual_seed(CONFIG["assignment_seed"])
        self.macro_pending = None
        self.followups = []
        self.records = []
        self.failure = None
        self.expert_model = None
        self.expert_rms = None
        self.expert_action = None

    def restore(self, filename):
        super().restore(filename)
        if sha256(EXPERT) != EXPERT_SHA:
            raise ValueError("read-only expert checkpoint SHA drift")
        payload = torch.load(EXPERT, map_location=self.device, weights_only=False)
        model = copy.deepcopy(self.model)
        target_keys = set(model.state_dict())
        candidate = {key if key in target_keys else "_orig_mod." + key: value
                     for key, value in payload["model"].items()}
        if set(candidate) != target_keys:
            raise ValueError("expert and self-trained policy architecture differ")
        model.load_state_dict(candidate, strict=True)
        model.eval().requires_grad_(False)
        rms = copy.deepcopy(self.running_mean_std)
        rms.load_state_dict(payload["running_mean_std"], strict=True)
        rms.eval()
        self.expert_model = model
        self.expert_rms = rms

    @torch.no_grad()
    def get_action(self, obs_dict, is_determenistic=False):
        base = super().get_action(obs_dict, is_determenistic)
        if self.expert_model is None or self.expert_rms is None:
            raise RuntimeError("expert candidate model not restored")
        source_model, source_rms = self.model, self.running_mean_std
        try:
            self.model, self.running_mean_std = self.expert_model, self.expert_rms
            expert = super().get_action(obs_dict, is_determenistic)
        finally:
            self.model, self.running_mean_std = source_model, source_rms
        if (base.shape != expert.shape or base.ndim != 2 or base.shape[1] != 18 or
                not torch.isfinite(base).all() or not torch.isfinite(expert).all() or
                (base.abs() > 1 + 1e-5).any() or (expert.abs() > 1 + 1e-5).any()):
            raise ValueError("candidate action shape/range/finite mismatch")
        self.expert_action = expert.detach().clone()
        return base

    @staticmethod
    def _contact(task):
        return ((task._contact_forces[:, task._contact_body_ids].norm(dim=-1) > .1).any(-1) &
                (task._tar_contact_forces.norm(dim=-1) > .1))

    def env_step(self, env, action):
        self.step_index += 1
        task = env.task
        started = self.step_index in CONFIG["steps"]
        if started:
            if self.macro_pending is not None:
                raise RuntimeError("overlapping dual-policy option")
            contact = self._contact(task)
            valid = contact & (task.reset_buf.reshape(-1) == 0) & (task.progress_buf > 0)
            assignment = balanced_assignment(valid, self.generator)
            record = {
                "q": task._dof_pos.clone(), "dof_vel": task._dof_vel.clone(),
                "object_state": task._target_states.clone(),
                "base_action": action.detach().clone(),
                "expert_action": self.expert_action.clone(),
                "assignment": assignment.clone(),
                "pre_contact": contact.clone(),
                "progress": task.progress_buf.clone(),
                "motion_id": task.data_id.clone(),
                "start_frame": task.start_times.clone(),
                "global_step": torch.full_like(task.progress_buf, self.step_index),
                "followup_contact_count": torch.zeros_like(task.progress_buf),
                "followup_alive": torch.ones_like(contact),
                "option_action_gap_sum": torch.zeros_like(task.progress_buf,
                                                          dtype=action.dtype),
                "option_steps": torch.zeros_like(task.progress_buf),
            }
            self.macro_pending = (self.step_index, assignment, record)
            self.followups.append((self.step_index, record))
        if self.macro_pending is not None:
            option_start, assignment, record = self.macro_pending
            if self.step_index - option_start >= 10:
                raise RuntimeError("dual-policy option lifetime exceeded")
            gap = (self.expert_action - action).norm(dim=-1)
            record["option_action_gap_sum"] += gap * (assignment != 0)
            record["option_steps"] += (assignment != 0).long()
            executed = torch.where((assignment == 1)[:, None],
                                   self.expert_action, action)
            result = super().env_step(env, executed)
            if started:
                record["executed_action"] = executed.clone()
                record["next_q"] = task._dof_pos.clone()
                record["next_object_state"] = task._target_states.clone()
            if self.step_index - option_start == 9:
                self.macro_pending = None
        else:
            result = super().env_step(env, action)
        contact = self._contact(task)
        remaining = []
        for option_start, record in self.followups:
            elapsed = self.step_index - option_start + 1
            record["followup_alive"] &= ((task.reset_buf.reshape(-1) == 0) &
                                         (task.progress_buf == record["progress"] + elapsed))
            record["followup_contact_count"] += (contact & record["followup_alive"]).long()
            if elapsed == 20:
                record.update(followup_object_state=task._target_states.clone(),
                              followup_contact=contact.clone(),
                              followup_progress=task.progress_buf.clone(),
                              followup_reset=task.reset_buf.reshape(-1).clone())
                if not all(torch.isfinite(value.float()).all() for value in record.values()):
                    raise FloatingPointError("non-finite dual-policy record")
                self.records.append({key: value.detach().cpu()
                                     for key, value in record.items()})
            else:
                remaining.append((option_start, record))
        self.followups = remaining
        if started:
            print("REF2DEX_EXPERT_OPTION_STEP " + json.dumps({
                "step": self.step_index, "selected": int(valid.sum()),
                "expert": int((assignment == 1).sum()),
                "self_trained": int((assignment == -1).sum()),
                "candidate_gap_mean": float(gap[assignment != 0].mean())
                if (assignment != 0).any() else None,
            }, sort_keys=True), flush=True)
        if self.step_index >= CONFIG["stop_step"]:
            raise ProbeDone
        return result

    def run(self):
        status = "COMPLETED"
        try:
            return super().run()
        except ProbeDone:
            return None
        except BaseException as error:
            status = "FAILED"
            self.failure = f"{type(error).__name__}: {error}"
            raise
        finally:
            output = CONFIG["output"]
            records = ({key: torch.cat([part[key] for part in self.records])
                        for key in self.records[0]}
                       if self.records else {})
            torch.save({"schema": "ref2dex.expert_option_h20.v1",
                        "run_status": status, "failure": self.failure,
                        "source_actor_role": "self_trained",
                        "candidate_actor_role": "official_data_collector",
                        "followup_horizon": 20, "option_horizon": 10,
                        "assignment_seed": CONFIG["assignment_seed"],
                        "records": records}, output)
            assignment = records.get("assignment", torch.empty(0, dtype=torch.int8))
            summary = {"run_status": status, "failure": self.failure,
                       "rows": len(assignment), "steps_collected": len(self.records),
                       "expert": int((assignment == 1).sum()),
                       "self_trained": int((assignment == -1).sum())}
            output.with_suffix(".json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n")
            print("REF2DEX_EXPERT_OPTION_SUMMARY " + json.dumps(summary,
                                                              sort_keys=True), flush=True)


def main():
    global CONFIG
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--intervention-output", type=Path, required=True)
    parser.add_argument("--intervention-first", type=int, default=50)
    parser.add_argument("--intervention-last", type=int, default=190)
    parser.add_argument("--assignment-seed", type=int, default=20260924191)
    parser.add_argument("--source-motion-manifest", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if (args.intervention_output.exists() or
            args.intervention_output.with_suffix(".json").exists() or
            not 1 <= args.intervention_first <= args.intervention_last <= 480 or
            (args.intervention_last - args.intervention_first) % 20):
        raise ValueError("new output and 20-step intervention schedule required")
    if sha256(EXPERT) != EXPERT_SHA or sha256(args.source_motion_manifest) != SPLIT_SHA:
        raise ValueError("expert or object-split input SHA drift")
    checkpoint = Path(remaining[remaining.index("--checkpoint") + 1]).resolve()
    motion_root = Path(remaining[remaining.index("--motion_file") + 1]).resolve()
    validate_source(checkpoint=checkpoint, checkpoint_sha256=SELF_TRAINED_SHA,
                    motion_root=motion_root,
                    manifest_path=args.source_motion_manifest,
                    manifest_sha256=SPLIT_SHA, partition="train")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit():
        raise RuntimeError("one explicit physical CUDA_VISIBLE_DEVICES index required")
    memory = int(subprocess.check_output([
        "nvidia-smi", f"--id={visible}", "--query-gpu=memory.used",
        "--format=csv,noheader,nounits"], text=True).strip())
    if memory > 512:
        raise RuntimeError(f"GPU{visible} occupied: {memory} MiB")
    output = args.intervention_output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.parent / "run_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    steps = list(range(args.intervention_first, args.intervention_last + 1, 20))
    manifest = {"run_status": "STARTED", "run_id": output.parent.name,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       cwd=ROOT, text=True).strip(),
                "source_actor_role": "self_trained",
                "candidate_actor_role": "official_data_collector",
                "source_checkpoint_sha256": SELF_TRAINED_SHA,
                "candidate_checkpoint_sha256": EXPERT_SHA,
                "motion_manifest_sha256": SPLIT_SHA,
                "intervention_steps": steps, "assignment_seed": args.assignment_seed,
                "option_horizon": 10, "followup_horizon": 20,
                "physical_gpu": int(visible), "max_gpu_count": 1,
                "wall_budget_minutes": 30, "output_budget_mb": 100,
                "stop_rule": "source drift, non-finite action, incomplete followup or wall budget",
                "command": [sys.executable, *sys.argv]}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    CONFIG = {"output": output, "steps": set(steps),
              "stop_step": steps[-1] + 19,
              "assignment_seed": args.assignment_seed}
    sys.argv = [sys.argv[0], *remaining]
    original.EvalPlayer = DualPlayer
    try:
        original.main()
        manifest.update(run_status="COMPLETED",
                        transitions_sha256=sha256(output),
                        summary_sha256=sha256(output.with_suffix(".json")))
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

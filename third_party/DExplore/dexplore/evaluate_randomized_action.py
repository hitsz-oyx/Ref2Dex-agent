"""Randomized, actually executed one-step wrist-z interventions in DExplore.

Treatment is randomized among pre-contact environments at prespecified
global steps. This identifies a population treatment effect, not an
individual same-state counterfactual. The actor checkpoint is self-trained.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from isaacgym import gymapi  # noqa: F401 - must precede torch
import torch

import evaluate_paired as pinned
from src.task.CmResidual.randomized_action import balanced_assignment


original = pinned.original
BASE_PLAYER = original.EvalPlayer
CONFIG = None


class ProbeDone(Exception):
    """Normal early stop after the last randomized intervention."""


class RandomizedPlayer(BASE_PLAYER):
    def __init__(self, config):
        super().__init__(config)
        if CONFIG is None:
            raise RuntimeError("randomized action configuration missing")
        self.probe_step = 0
        self.probe_records = []
        self.probe_error = None
        self.probe_generator = torch.Generator(device="cpu").manual_seed(CONFIG["assignment_seed"])

    def env_step(self, env, action):
        self.probe_step += 1
        task = env.task
        selected_step = self.probe_step in CONFIG["steps"]
        if selected_step:
            pre_contact = ((task._contact_forces[:, task._contact_body_ids].norm(dim=-1) > .1).any(-1) &
                           (task._tar_contact_forces.norm(dim=-1) > .1))
            valid = pre_contact & (task.reset_buf.reshape(-1) == 0) & (task.progress_buf > 0)
            assignment = balanced_assignment(valid, self.probe_generator)
            executed = action.detach().clone()
            executed[:, 2] = (executed[:, 2] + CONFIG["delta_z"] * assignment).clamp(-1, 1)
            before = {
                "q": task._dof_pos.clone(),
                "dof_vel": task._dof_vel.clone(),
                "object_state": task._target_states.clone(),
                "base_action": action.detach().clone(),
                "executed_action": executed.clone(),
                "assignment": assignment.clone(),
                "pre_contact": pre_contact.clone(),
                "progress": task.progress_buf.clone(),
                "motion_id": task.data_id.clone(),
                "start_frame": task.start_times.clone(),
            }
            result = BASE_PLAYER.env_step(self, env, executed)
            before.update(next_q=task._dof_pos.clone(),
                          next_object_state=task._target_states.clone(),
                          global_step=torch.full_like(task.progress_buf, self.probe_step))
            if not all(torch.isfinite(value).all() for value in before.values()):
                raise FloatingPointError("non-finite randomized physical transition")
            self.probe_records.append({key: value.detach().cpu() for key, value in before.items()})
            print("REF2DEX_RANDOMIZED_STEP " + json.dumps({
                "step": self.probe_step, "selected": int(valid.sum()),
                "plus": int((assignment == 1).sum()),
                "minus": int((assignment == -1).sum()),
            }, sort_keys=True), flush=True)
        else:
            result = BASE_PLAYER.env_step(self, env, action)
        if self.probe_step >= CONFIG["stop_step"]:
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
            self.probe_error = f"{type(error).__name__}: {error}"
            raise
        finally:
            output = CONFIG["output"]
            if self.probe_records:
                keys = self.probe_records[0]
                records = {key: torch.cat([item[key] for item in self.probe_records])
                           for key in keys}
            else:
                records = {}
            torch.save({"schema": "ref2dex.randomized_action_transitions.v1",
                        "run_status": status, "failure": self.probe_error,
                        "assignment_seed": CONFIG["assignment_seed"],
                        "delta_z_action": CONFIG["delta_z"],
                        "records": records}, output)
            selected = records.get("assignment", torch.empty(0, dtype=torch.int8))
            summary = {
                "run_status": status, "steps_collected": len(self.probe_records),
                "rows": len(selected), "plus": int((selected == 1).sum()),
                "minus": int((selected == -1).sum()),
                "failure": self.probe_error,
            }
            output.with_suffix(".json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n")
            print("REF2DEX_RANDOMIZED_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


def main():
    global CONFIG
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--intervention-output", type=Path, required=True)
    parser.add_argument("--intervention-first", type=int, default=50)
    parser.add_argument("--intervention-last", type=int, default=150)
    parser.add_argument("--intervention-stride", type=int, default=10)
    parser.add_argument("--intervention-delta-z", type=float, default=.3)
    parser.add_argument("--assignment-seed", type=int, default=20260924)
    args, remaining = parser.parse_known_args()
    if (args.intervention_output.exists() or
            args.intervention_output.with_suffix(".json").exists() or
            not 1 <= args.intervention_first <= args.intervention_last <= 500 or
            not 1 <= args.intervention_stride <= 100 or
            not 0 < args.intervention_delta_z <= .5):
        raise ValueError("invalid randomized intervention design or existing output")
    checkpoint = Path(pinned.argument_value(remaining, "--checkpoint")).resolve()
    motion_root = Path(pinned.argument_value(remaining, "--motion_file")).resolve()
    if (checkpoint != pinned.CHECKPOINT.resolve() or
            pinned.sha256(checkpoint) != pinned.CHECKPOINT_SHA256 or
            motion_root != pinned.MOTION_ROOT.resolve() or
            pinned.sha256(pinned.MOTION_MANIFEST) != pinned.MOTION_MANIFEST_SHA256):
        raise ValueError("pinned self-trained actor or motion source drift")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit():
        raise RuntimeError("one explicit physical CUDA_VISIBLE_DEVICES index required")
    memory = int(subprocess.check_output([
        "nvidia-smi", f"--id={visible}", "--query-gpu=memory.used",
        "--format=csv,noheader,nounits"], text=True).strip())
    if memory > 512:
        raise RuntimeError(f"GPU{visible} occupied: {memory} MiB")
    args.intervention_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.intervention_output.parent / "run_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    steps = list(range(args.intervention_first, args.intervention_last + 1,
                       args.intervention_stride))
    manifest = {
        "run_status": "STARTED", "schema": "ref2dex.randomized_action_transitions.v1",
        "run_id": args.intervention_output.parent.name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                               cwd=pinned.ROOT, text=True).strip(),
        "physical_gpu": int(visible), "max_gpu_count": 1,
        "intervention_steps": steps, "stop_step": steps[-1],
        "delta_z_action": args.intervention_delta_z,
        "assignment_seed": args.assignment_seed,
        "checkpoint_sha256": pinned.CHECKPOINT_SHA256,
        "motion_manifest_sha256": pinned.MOTION_MANIFEST_SHA256,
        "wall_budget_minutes": 30, "output_budget_mb": 100,
        "stop_rule": "input drift, GPU conflict, non-finite state or wall budget",
        "command": [sys.executable, *sys.argv],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    CONFIG = {"output": args.intervention_output.resolve(), "steps": set(steps),
              "stop_step": steps[-1], "delta_z": args.intervention_delta_z,
              "assignment_seed": args.assignment_seed}
    sys.argv = [sys.argv[0], *remaining]
    original.EvalPlayer = RandomizedPlayer
    try:
        original.main()
        manifest.update(run_status="COMPLETED",
                        transitions_sha256=pinned.sha256(args.intervention_output),
                        summary_sha256=pinned.sha256(args.intervention_output.with_suffix(".json")))
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

"""Matched base, always-down and frozen contact-aware Cm action choice."""
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
from src.task.CmResidual.cm_online_choice import (
    apply_wrist_z_down, select_contact_preserving_down,
)
from src.task.CmResidual.contact_aware_cm import RawContactAwareCm
from src.task.CmResidual.tools.probe_randomized_geometric_cm import raw_inputs


original = pinned.original
BASE_PLAYER = original.EvalPlayer
CONFIG = None
FIRST_STEP, LAST_STEP, STRIDE = 50, 150, 2
DELTA_Z = .1
MODEL_PATH = (pinned.ROOT /
              "outputs/CmResidual/agent_contact_aware_cm_s151152_train_s153154_test/raw_action.pt")
MODEL_SHA256 = "ce711dca743fb47ff84adf45f3f580e4afd0237f7f0defd55a4ec411a5c34d5d"


class ContactChoicePlayer(BASE_PLAYER):
    def __init__(self, config):
        super().__init__(config)
        if CONFIG is None:
            raise RuntimeError("contact Cm choice configuration missing")
        task = self.env.task
        self.choice_step = 0
        self.choice_scored_steps = 0
        self.choice_eligible = torch.zeros((), dtype=torch.long, device=task._dof_pos.device)
        self.choice_selected = torch.zeros((), dtype=torch.long, device=task._dof_pos.device)
        self.choice_error = None
        self.mode = CONFIG["mode"]
        if self.mode == "cm_down":
            device = task._dof_pos.device
            self.cm = RawContactAwareCm(torch.zeros(67), torch.ones(67)).to(device).eval()
            payload = torch.load(MODEL_PATH, map_location=device, weights_only=False)
            if payload.get("schema") != "ref2dex.contact_aware_cm.v1" or (
                    payload.get("name") != "raw_action"):
                raise ValueError("contact Cm checkpoint schema/name mismatch")
            self.cm.load_state_dict(payload["model"], strict=True)

    @torch.no_grad()
    def _cm_selected(self, task, action, contact):
        ids = contact.nonzero(as_tuple=False).reshape(-1)
        selected = torch.zeros(len(action), dtype=torch.bool, device=action.device)
        if not len(ids):
            return selected
        count = len(ids)
        baseline = action[ids].detach()
        down = baseline.clone()
        down[:, 2] = (down[:, 2] - DELTA_Z).clamp(-1, 1)
        q = task._dof_pos[ids]
        velocity = task._dof_vel[ids]
        obj = task._target_states[ids]
        rows = {"q": torch.cat((q, q)),
                "dof_vel": torch.cat((velocity, velocity)),
                "object_state": torch.cat((obj, obj)),
                "action": torch.cat((baseline, down))}
        output = self.cm(raw_inputs(rows))
        chosen = select_contact_preserving_down(
            output["contact_fraction"][:count], output["contact_fraction"][count:],
            output["followup_delta_local"][:count],
            output["followup_delta_local"][count:], obj)
        selected[ids] = chosen
        self.choice_scored_steps += 1
        return selected

    def env_step(self, env, action):
        self.choice_step += 1
        if (self.mode == "base" or not FIRST_STEP <= self.choice_step <= LAST_STEP or
                self.choice_step % STRIDE):
            return BASE_PLAYER.env_step(self, env, action)
        task = env.task
        contact = ((task._contact_forces[:, task._contact_body_ids].norm(dim=-1) > .1).any(-1) &
                   (task._tar_contact_forces.norm(dim=-1) > .1) &
                   (task.reset_buf.reshape(-1) == 0))
        self.choice_eligible += contact.sum()
        selected = (contact if self.mode == "always_down"
                    else self._cm_selected(task, action, contact))
        self.choice_selected += selected.sum()
        changed = apply_wrist_z_down(action, selected, DELTA_Z)
        return BASE_PLAYER.env_step(self, env, changed)

    def run(self):
        status = "COMPLETED"
        try:
            return super().run()
        except BaseException as error:
            status = "FAILED"
            self.choice_error = f"{type(error).__name__}: {error}"
            raise
        finally:
            summary = {"run_status": status, "mode": self.mode,
                       "global_steps": self.choice_step,
                       "contact_eligible": int(self.choice_eligible),
                       "down_selected": int(self.choice_selected),
                       "cm_scored_steps": self.choice_scored_steps,
                       "failure": self.choice_error}
            CONFIG["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            print("REF2DEX_CONTACT_CM_DOWN " + json.dumps(summary, sort_keys=True), flush=True)


def main():
    global CONFIG
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--choice-mode", choices=("base", "always_down", "cm_down"), required=True)
    parser.add_argument("--choice-summary", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if args.choice_summary.exists():
        raise FileExistsError(args.choice_summary)
    checkpoint = Path(pinned.argument_value(remaining, "--checkpoint")).resolve()
    motion_root = Path(pinned.argument_value(remaining, "--motion_file")).resolve()
    if (checkpoint != pinned.CHECKPOINT.resolve() or
            pinned.sha256(checkpoint) != pinned.CHECKPOINT_SHA256 or
            motion_root != pinned.MOTION_ROOT.resolve() or
            pinned.sha256(pinned.MOTION_MANIFEST) != pinned.MOTION_MANIFEST_SHA256):
        raise ValueError("pinned actor or motion source drift")
    if pinned.sha256(MODEL_PATH) != MODEL_SHA256:
        raise ValueError("contact-aware Cm model drift")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit():
        raise RuntimeError("one explicit physical CUDA_VISIBLE_DEVICES index required")
    memory = int(subprocess.check_output([
        "nvidia-smi", f"--id={visible}", "--query-gpu=memory.used",
        "--format=csv,noheader,nounits"], text=True).strip())
    if memory > 512:
        raise RuntimeError(f"physical GPU{visible} occupied: {memory}MiB")
    num_envs = int(pinned.argument_value(remaining, "--num_envs"))
    evaluation_output = Path(pinned.argument_value(remaining, "--output")).resolve()
    if not 1 <= num_envs <= 64:
        raise ValueError("online Probe limited to 1-64 env")
    args.choice_summary.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.choice_summary.parent / "run_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    manifest = {"run_status": "STARTED", "run_id": args.choice_summary.parent.name,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       cwd=pinned.ROOT, text=True).strip(),
                "mode": args.choice_mode, "physical_gpu": int(visible), "max_gpu_count": 1,
                "evaluation_output": str(evaluation_output), "num_envs": num_envs,
                "checkpoint_sha256": pinned.CHECKPOINT_SHA256,
                "motion_manifest_sha256": pinned.MOTION_MANIFEST_SHA256,
                "contact_cm_sha256": MODEL_SHA256,
                "window": [FIRST_STEP, LAST_STEP], "stride": STRIDE,
                "delta_z_action": -DELTA_Z, "min_predicted_contact_gain": .05,
                "max_predicted_lift_loss_m": .01,
                "wall_budget_minutes": 20, "output_budget_mb": 100,
                "stop_rule": "input drift, occupied GPU, non-finite Cm, evaluation failure or wall budget",
                "command": [sys.executable, *sys.argv]}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    CONFIG = {"mode": args.choice_mode, "summary": args.choice_summary.resolve()}
    sys.argv = [sys.argv[0], *remaining]
    original.EvalPlayer = ContactChoicePlayer
    try:
        original.main()
        manifest.update(run_status="COMPLETED",
                        summary_sha256=pinned.sha256(args.choice_summary),
                        evaluation_sha256=pinned.sha256(evaluation_output))
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

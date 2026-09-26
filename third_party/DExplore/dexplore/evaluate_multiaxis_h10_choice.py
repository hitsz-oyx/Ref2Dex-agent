"""Matched base/always-x/always-z/frozen-Cm complete-episode evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from isaacgym import gymapi  # noqa: F401 - Isaac Gym import order
import torch

import evaluate_paired as pinned
from src.task.CmResidual.cm_multiaxis_choice import apply_choice, select_x_or_z
from src.task.CmResidual.cmlite import local_to_world_translation
from src.task.CmResidual.contact_aware_cm import RawContactAwareCm
from src.task.CmResidual.tools.probe_randomized_geometric_cm import raw_inputs


original = pinned.original
BASE_PLAYER = original.EvalPlayer
CONFIG = None
FIRST_STEP, LAST_STEP, STRIDE = 50, 200, 10
DELTA = .1
MODEL_PATH = (pinned.ROOT /
              "outputs/CmResidual/agent_multiaxis_h10_cm_s161162_train_s163_test/raw_action.pt")
MODEL_SHA256 = "3023a6f9715d1b2dbbb0c82b38184e6c84affd57987f6ba4d3d786477b188174"


class MultiAxisChoicePlayer(BASE_PLAYER):
    def __init__(self, config):
        super().__init__(config)
        if CONFIG is None:
            raise RuntimeError("multiaxis choice configuration missing")
        self.mode = CONFIG["mode"]
        self.choice_step = 0
        self.scored_steps = 0
        self.eligible_count = 0
        self.selected_x = 0
        self.selected_z = 0
        self.choice_error = None
        if self.mode == "cm":
            device = self.env.task._dof_pos.device
            self.cm = RawContactAwareCm(torch.zeros(67), torch.ones(67)).to(device).eval()
            payload = torch.load(MODEL_PATH, map_location=device, weights_only=False)
            if (payload.get("schema") != "ref2dex.multiaxis_h10_cm.v1" or
                    payload.get("name") != "raw_action"):
                raise ValueError("multiaxis Cm checkpoint schema/name mismatch")
            self.cm.load_state_dict(payload["model"], strict=True)
            self.cm.requires_grad_(False)

    @torch.no_grad()
    def _cm_choice(self, task, action, eligible):
        ids = eligible.nonzero(as_tuple=False).reshape(-1)
        result = torch.zeros(len(action), dtype=torch.int8, device=action.device)
        if not len(ids):
            return result
        baseline = action[ids].detach()
        x_plus, z_plus = baseline.clone(), baseline.clone()
        x_plus[:, 0] += DELTA
        z_plus[:, 2] += DELTA
        count = len(ids)
        rows = {"q": task._dof_pos[ids].repeat(3, 1),
                "dof_vel": task._dof_vel[ids].repeat(3, 1),
                "object_state": task._target_states[ids].repeat(3, 1),
                "action": torch.cat((baseline, x_plus, z_plus))}
        output = self.cm(raw_inputs(rows))
        future_world = local_to_world_translation(
            rows["object_state"], output["followup_delta_local"])
        choice = select_x_or_z(future_world.reshape(3, count, 3),
                               output["contact_fraction"].reshape(3, count))
        result[ids] = choice
        self.scored_steps += 1
        return result

    def env_step(self, env, action):
        self.choice_step += 1
        if (not FIRST_STEP <= self.choice_step <= LAST_STEP or
                (self.choice_step - FIRST_STEP) % STRIDE):
            return BASE_PLAYER.env_step(self, env, action)
        task = env.task
        contact = ((task._contact_forces[:, task._contact_body_ids].norm(dim=-1) > .1).any(-1) &
                   (task._tar_contact_forces.norm(dim=-1) > .1) &
                   (task.reset_buf.reshape(-1) == 0))
        safe = (action[:, [0, 2]].abs() <= 1 - DELTA).all(-1)
        eligible = contact & safe
        self.eligible_count += int(eligible.sum())
        if self.mode == "base":
            return BASE_PLAYER.env_step(self, env, action)
        if self.mode == "cm":
            choice = self._cm_choice(task, action, eligible)
        else:
            choice = torch.zeros(len(action), dtype=torch.int8, device=action.device)
            choice[eligible] = 1 if self.mode == "always_x" else 2
        self.selected_x += int((choice == 1).sum())
        self.selected_z += int((choice == 2).sum())
        return BASE_PLAYER.env_step(self, env, apply_choice(action, choice, DELTA))

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
                       "contact_eligible": self.eligible_count,
                       "selected_x": self.selected_x, "selected_z": self.selected_z,
                       "cm_scored_steps": self.scored_steps,
                       "failure": self.choice_error}
            CONFIG["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            print("REF2DEX_MULTIAXIS_H10_CHOICE " + json.dumps(summary, sort_keys=True),
                  flush=True)


def main():
    global CONFIG
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--choice-mode", choices=("base", "always_x", "always_z", "cm"),
                        required=True)
    parser.add_argument("--choice-summary", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if args.choice_summary.exists():
        raise FileExistsError(args.choice_summary)
    checkpoint = Path(pinned.argument_value(remaining, "--checkpoint")).resolve()
    motion_root = Path(pinned.argument_value(remaining, "--motion_file")).resolve()
    if (checkpoint != pinned.CHECKPOINT.resolve() or
            pinned.sha256(checkpoint) != pinned.CHECKPOINT_SHA256 or
            motion_root != pinned.MOTION_ROOT.resolve() or
            pinned.sha256(pinned.MOTION_MANIFEST) != pinned.MOTION_MANIFEST_SHA256 or
            pinned.sha256(MODEL_PATH) != MODEL_SHA256):
        raise ValueError("pinned actor, motion or multiaxis Cm drift")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit():
        raise RuntimeError("one explicit physical GPU required")
    memory = int(subprocess.check_output([
        "nvidia-smi", f"--id={visible}", "--query-gpu=memory.used",
        "--format=csv,noheader,nounits"], text=True).strip())
    if memory > 512:
        raise RuntimeError(f"physical GPU{visible} occupied: {memory} MiB")
    num_envs = int(pinned.argument_value(remaining, "--num_envs"))
    output = Path(pinned.argument_value(remaining, "--output")).resolve()
    if not 1 <= num_envs <= 64 or output.exists():
        raise ValueError("evaluation output must be new and limited to 64 env")
    args.choice_summary.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.choice_summary.parent / "run_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    manifest = {"run_status": "STARTED", "run_id": args.choice_summary.parent.name,
                "experiment_id": "P-20260924-multiaxis-h10-online-choice",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       cwd=pinned.ROOT, text=True).strip(),
                "mode": args.choice_mode, "physical_gpu": int(visible),
                "num_envs": num_envs, "checkpoint_sha256": pinned.CHECKPOINT_SHA256,
                "motion_manifest_sha256": pinned.MOTION_MANIFEST_SHA256,
                "multiaxis_cm_sha256": MODEL_SHA256,
                "window": [FIRST_STEP, LAST_STEP], "stride": STRIDE,
                "delta_action": DELTA,
                "thresholds": {"z_gain_m": .005, "z_contact_loss": .02,
                               "x_contact_gain": .03, "x_z_loss_m": .005},
                "wall_budget_minutes": 20, "output_budget_mb": 100,
                "stop_rule": "source drift, GPU conflict, non-finite score or invalid episode",
                "command": [sys.executable, *sys.argv]}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    CONFIG = {"mode": args.choice_mode, "summary": args.choice_summary.resolve()}
    sys.argv = [sys.argv[0], *remaining]
    original.EvalPlayer = MultiAxisChoicePlayer
    try:
        original.main()
        manifest.update(run_status="COMPLETED",
                        summary_sha256=pinned.sha256(args.choice_summary),
                        evaluation_sha256=pinned.sha256(output))
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

"""Collect validated same-state, different-action DExplore physics pairs.

This wraps the existing self-trained actor evaluator.  It never substitutes
the official actor checkpoint and refuses hidden-state replay mismatch.
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

from isaacgym import gymapi, gymtorch  # noqa: F401 - Isaac Gym must precede torch
import torch

import evaluate as original
from src.task.CmResidual.paired_sim_step import paired_sim_step


CONFIG = None
ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = (ROOT / "outputs/Dexplore/agent_v139_s3_backtrack_s70_e260/train/"
              "inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000260.pth")
CHECKPOINT_SHA256 = "16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f"
MOTION_ROOT = ROOT / "outputs/CmResidual/agent_v129_s3_coordfix/corrected_converted_r2"
MOTION_MANIFEST = ROOT / "outputs/CmResidual/agent_v129_s3_coordfix/corrected_manifest_r2.json"
MOTION_MANIFEST_SHA256 = "2878bd20d1dd849f6844883c832a3777602d2d2bc73f7581b3bc23d30725f038"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def argument_value(arguments, name):
    if name not in arguments:
        raise ValueError(f"missing {name}")
    position = arguments.index(name)
    if position + 1 >= len(arguments):
        raise ValueError(f"missing value for {name}")
    return arguments[position + 1]


class PairedPlayer(original.EvalPlayer):
    def __init__(self, config):
        super().__init__(config)
        if CONFIG is None:
            raise RuntimeError("paired evaluator config missing")
        self.paired_output = CONFIG["output"]
        self.paired_schedule = CONFIG["schedule"]
        self.paired_delta_z = CONFIG["delta_z"]
        self.paired_step_index = 0
        self.paired_records = []
        self.paired_error = None

    def env_step(self, env, action):
        self.paired_step_index += 1
        if self.paired_step_index in self.paired_schedule:
            task = env.task
            action = action.detach()
            alternate = action.clone()
            alternate[:, 2] = (alternate[:, 2] + self.paired_delta_z).clamp(-1, 1)
            pre_contact = ((task._contact_forces[:, task._contact_body_ids].norm(dim=-1) > .1).any(-1) &
                           (task._tar_contact_forces.norm(dim=-1) > .1))
            pre_progress = task.progress_buf.clone()
            pre_motion = task.data_id.clone()
            pre_start = task.start_times.clone()
            pairs = paired_sim_step(task, action, alternate, gymtorch.unwrap_tensor)
            valid = ((pairs["action_gap_l2"] > .001) &
                     (task.reset_buf.reshape(-1) == 0) &
                     (pre_progress > 0))
            if valid.any():
                record = {key: value[valid].detach().cpu() for key, value in pairs.items()}
                record.update(
                    env_id=torch.arange(len(action), device=action.device)[valid].cpu(),
                    global_step=torch.full((int(valid.sum()),), self.paired_step_index, dtype=torch.int64),
                    progress=pre_progress[valid].cpu(),
                    motion_id=pre_motion[valid].cpu(),
                    start_frame=pre_start[valid].cpu(),
                    pre_contact=pre_contact[valid].cpu(),
                )
                self.paired_records.append(record)
            print("REF2DEX_PAIRED_STEP " + json.dumps({
                "global_step": self.paired_step_index,
                "kept": int(valid.sum()),
                "pre_contact_kept": int((valid & pre_contact).sum()),
                "repeat_object_max_mm": float(pairs["same_action_repeat_object_mm"].max()),
                "effect_mean_mm": float(pairs["actual_effect_mm"][valid].mean())
                if valid.any() else None,
            }, sort_keys=True), flush=True)
        return super().env_step(env, action)

    def _save_pairs(self, status):
        path = self.paired_output
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.paired_records:
            keys = self.paired_records[0]
            merged = {key: torch.cat([item[key] for item in self.paired_records]) for key in keys}
        else:
            merged = {}
        data = {"schema": "ref2dex.paired_sim_actions.v1",
                "run_status": status, "schedule": self.paired_schedule,
                "delta_z_action": self.paired_delta_z,
                "failure": self.paired_error,
                "records": merged}
        torch.save(data, path)
        if merged:
            effect = merged["actual_effect_mm"]
            repeated = merged["same_action_repeat_object_mm"]
            contact = merged["pre_contact"].bool()
            summary = {"run_status": status, "samples": len(effect),
                       "pre_contact_samples": int(contact.sum()),
                       "repeat_p95_mm": float(torch.quantile(repeated, .95)),
                       "repeat_max_mm": float(repeated.max()),
                       "effect_mean_mm": float(effect.mean()),
                       "effect_p95_mm": float(torch.quantile(effect, .95)),
                       "contact_effect_gt_0_2mm": int(((effect > .2) & contact).sum()),
                       "failure": self.paired_error}
        else:
            summary = {"run_status": status, "samples": 0,
                       "failure": self.paired_error}
        path.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print("REF2DEX_PAIRED_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)

    def run(self):
        status = "COMPLETED"
        try:
            return super().run()
        except BaseException as error:
            status = "FAILED"
            self.paired_error = f"{type(error).__name__}: {error}"
            raise
        finally:
            self._save_pairs(status)


def main():
    global CONFIG
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--paired-output", type=Path, required=True)
    parser.add_argument("--paired-schedule", type=int, action="append", required=True)
    parser.add_argument("--paired-delta-z", type=float, default=.1)
    parser.add_argument("--paired-cpu-smoke", action="store_true")
    args, remaining = parser.parse_known_args()
    if (args.paired_output.exists() or args.paired_output.with_suffix(".json").exists() or
            not args.paired_schedule or len(set(args.paired_schedule)) != len(args.paired_schedule) or
            any(step < 1 for step in args.paired_schedule) or
            not 0 < args.paired_delta_z <= .5):
        raise ValueError("paired output must be new; schedule and action delta must be valid")
    checkpoint = Path(argument_value(remaining, "--checkpoint")).resolve()
    motion_root = Path(argument_value(remaining, "--motion_file")).resolve()
    if (checkpoint != CHECKPOINT.resolve() or sha256(checkpoint) != CHECKPOINT_SHA256 or
            motion_root != MOTION_ROOT.resolve() or
            sha256(MOTION_MANIFEST) != MOTION_MANIFEST_SHA256):
        raise ValueError("pinned self-trained actor or motion source drift")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if args.paired_cpu_smoke:
        if (argument_value(remaining, "--sim_device") != "cpu" or
                argument_value(remaining, "--rl_device") != "cpu" or
                "--use_gpu" in remaining or "--use_gpu_pipeline" in remaining):
            raise RuntimeError("CPU smoke requires CPU simulation, CPU policy, no GPU flags")
        physical_gpu = None
    else:
        if not visible.isdigit():
            raise RuntimeError("paired evaluation requires one explicit physical CUDA_VISIBLE_DEVICES index")
        memory = int(subprocess.check_output([
            "nvidia-smi", f"--id={visible}", "--query-gpu=memory.used",
            "--format=csv,noheader,nounits"], text=True).strip())
        if memory > 1024:
            raise RuntimeError(f"physical GPU{visible} already occupied: {memory} MiB")
        physical_gpu = int(visible)
    args.paired_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.paired_output.parent / "run_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    manifest = {"run_status": "STARTED", "run_id": args.paired_output.parent.name,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       cwd=ROOT, text=True).strip(),
                "schema": "ref2dex.paired_sim_actions.v1",
                "physical_gpu": physical_gpu, "max_gpu_count": 0 if args.paired_cpu_smoke else 1,
                "cpu_smoke_only": args.paired_cpu_smoke,
                "wall_budget_minutes": 60, "output_budget_mb": 100,
                "stop_rule": "input drift, restore/replay error, GPU conflict or wall budget",
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "motion_manifest_sha256": MOTION_MANIFEST_SHA256,
                "schedule": args.paired_schedule, "delta_z_action": args.paired_delta_z,
                "command": [sys.executable, *sys.argv]}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    CONFIG = {"output": args.paired_output.resolve(),
              "schedule": tuple(sorted(args.paired_schedule)),
              "delta_z": args.paired_delta_z}
    sys.argv = [sys.argv[0]] + remaining
    original.EvalPlayer = PairedPlayer
    try:
        original.main()
        manifest.update(run_status="COMPLETED",
                        paired_sha256=sha256(args.paired_output),
                        paired_summary_sha256=sha256(args.paired_output.with_suffix(".json")),
                        completed_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        manifest.update(run_status="FAILED",
                        failure=f"{type(error).__name__}: {error}",
                        completed_at=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

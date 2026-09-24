"""Collect one same-run physical action pair using matched parallel envs.

The policy is the pinned self-trained DExplore actor. All envs are forced to
start at reference frame zero, evolve under identical commands, and fork for
one physical step only after visible prestate and same-action replay checks.
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

import evaluate_paired as sequential
from src.task.CmResidual.parallel_sim_pair import parallel_sim_pair_step


original = sequential.original
BASE_PLAYER = original.EvalPlayer
BASE_LOAD_CFG = original.load_cfg
CONFIG = None


class PairCollected(Exception):
    """Normal early exit after the requested physical pair has been saved."""


def load_paired_cfg(args):
    cfg, cfg_train, logdir = BASE_LOAD_CFG(args)
    cfg["env"]["stateInit"] = "Start"
    cfg["env"]["hybridInitProb"] = 1.0
    return cfg, cfg_train, logdir


class ParallelPlayer(BASE_PLAYER):
    def __init__(self, config):
        super().__init__(config)
        if CONFIG is None:
            raise RuntimeError("parallel pair configuration missing")
        self.pair_step_index = 0
        self.pair_record = None
        self.pair_failure = None

    def env_step(self, env, action):
        self.pair_step_index += 1
        if self.pair_step_index in (1, 2, 5, 10, 20, 40, CONFIG["step"]):
            task = env.task
            root = task._root_states.view(task.num_envs, -1, 13)
            rigid = task._rigid_body_state.view(task.num_envs, -1, 13)
            print("REF2DEX_PARALLEL_DRIFT " + json.dumps({
                "step": self.pair_step_index,
                "root_actor_max": (root[1:] - root[0]).abs().amax(dim=(0, 2)).detach().cpu().tolist(),
                "dof_max": float((task._dof_state.view(task.num_envs, -1, 2)[1:] -
                                  task._dof_state.view(task.num_envs, -1, 2)[0]).abs().amax()),
                "rigid_max": float((rigid[1:] - rigid[0]).abs().amax()),
                "action_max": float((action[1:] - action[0]).abs().amax()),
                "progress": task.progress_buf.detach().cpu().tolist(),
                "start_times": task.start_times.detach().cpu().tolist(),
            }, sort_keys=True), flush=True)
        if self.pair_step_index != CONFIG["step"]:
            return BASE_PLAYER.env_step(self, env, action)
        _, record = parallel_sim_pair_step(
            env.task, action.detach(),
            lambda executed: BASE_PLAYER.env_step(self, env, executed),
            delta_z=CONFIG["delta_z"])
        self.pair_record = {key: value.detach().cpu() for key, value in record.items()}
        raise PairCollected

    def run(self):
        status = "COMPLETED"
        try:
            return super().run()
        except PairCollected:
            return None
        except BaseException as error:
            status = "FAILED"
            self.pair_failure = f"{type(error).__name__}: {error}"
            raise
        finally:
            output = CONFIG["output"]
            records = self.pair_record if self.pair_record is not None else {}
            torch.save({"schema": "ref2dex.parallel_physical_pairs.v1",
                        "run_status": status, "failure": self.pair_failure,
                        "records": records}, output)
            if records:
                effect = records["actual_effect_mm"]
                repeat = records["same_action_repeat_object_mm"]
                contact = records["pre_contact"].bool()
                summary = {
                    "run_status": status, "samples": len(effect),
                    "pre_contact_samples": int(contact.sum()),
                    "repeat_max_mm": float(repeat.max()),
                    "effect_mean_mm": float(effect.mean()),
                    "effect_max_mm": float(effect.max()),
                    "contact_effect_gt_0_2mm": int(((effect > .2) & contact).sum()),
                    "pre_root_gap_max": float(records["pre_root_gap"].max()),
                    "pre_dof_gap_max": float(records["pre_dof_gap"].max()),
                    "pre_rigid_gap_max": float(records["pre_rigid_gap"].max()),
                }
            else:
                summary = {"run_status": status, "samples": 0,
                           "failure": self.pair_failure}
            output.with_suffix(".json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n")
            print("REF2DEX_PARALLEL_PAIR " + json.dumps(summary, sort_keys=True), flush=True)


def main():
    global CONFIG
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--paired-output", type=Path, required=True)
    parser.add_argument("--paired-step", type=int, required=True)
    parser.add_argument("--paired-delta-z", type=float, default=.1)
    args, remaining = parser.parse_known_args()
    if (args.paired_output.exists() or args.paired_output.with_suffix(".json").exists() or
            args.paired_step < 1 or not 0 < args.paired_delta_z <= .5):
        raise ValueError("new output path, positive step and 0<delta<=0.5 required")
    num_envs = int(sequential.argument_value(remaining, "--num_envs"))
    if num_envs < 3 or num_envs % 3:
        raise ValueError("num_envs must contain complete base/repeat/alternate triplets")
    checkpoint = Path(sequential.argument_value(remaining, "--checkpoint")).resolve()
    motion_root = Path(sequential.argument_value(remaining, "--motion_file")).resolve()
    if (checkpoint != sequential.CHECKPOINT.resolve() or
            sequential.sha256(checkpoint) != sequential.CHECKPOINT_SHA256 or
            motion_root != sequential.MOTION_ROOT.resolve() or
            sequential.sha256(sequential.MOTION_MANIFEST) !=
            sequential.MOTION_MANIFEST_SHA256):
        raise ValueError("pinned self-trained actor or motion source drift")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit():
        raise RuntimeError("one explicit physical CUDA_VISIBLE_DEVICES index required")
    memory = int(subprocess.check_output([
        "nvidia-smi", f"--id={visible}", "--query-gpu=memory.used",
        "--format=csv,noheader,nounits"], text=True).strip())
    if memory > 512:
        raise RuntimeError(f"GPU{visible} occupied: {memory} MiB")
    args.paired_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.paired_output.parent / "run_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    manifest = {
        "run_status": "STARTED", "schema": "ref2dex.parallel_physical_pairs.v1",
        "run_id": args.paired_output.parent.name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                               cwd=sequential.ROOT, text=True).strip(),
        "physical_gpu": int(visible), "max_gpu_count": 1,
        "num_envs": num_envs, "paired_step": args.paired_step,
        "delta_z_action": args.paired_delta_z,
        "state_init_override": {"stateInit": "Start", "hybridInitProb": 1.0},
        "checkpoint_sha256": sequential.CHECKPOINT_SHA256,
        "motion_manifest_sha256": sequential.MOTION_MANIFEST_SHA256,
        "wall_budget_minutes": 30, "output_budget_mb": 100,
        "stop_rule": "input drift, GPU conflict, prestate/replay mismatch or wall budget",
        "command": [sys.executable, *sys.argv],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    CONFIG = {"output": args.paired_output.resolve(), "step": args.paired_step,
              "delta_z": args.paired_delta_z}
    sys.argv = [sys.argv[0], *remaining]
    original.EvalPlayer = ParallelPlayer
    original.load_cfg = load_paired_cfg
    try:
        original.main()
        manifest.update(run_status="COMPLETED",
                        pairs_sha256=sequential.sha256(args.paired_output),
                        summary_sha256=sequential.sha256(args.paired_output.with_suffix(".json")))
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

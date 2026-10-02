#!/usr/bin/env python3
"""Matched small policy Probe for Cm task-value auxiliary supervision."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent-baseline")
PYTHON = "/home2/wyy/miniconda3/envs/graspenv/bin/python"
SOURCE = BASE / "outputs/Dexplore/agent_v139_s3_backtrack_s70_e260/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000260.pth"
MOTIONS = BASE / "outputs/CmResidual/agent_contact_option_airplane_motions"
PHYSICAL = ROOT / "src/task/CmResidual/research/physical_value/output/P-20260930-cm-physical-value/r7/models/tier_1000000.pt"
REPRESENTATION = ROOT / "src/task/CmResidual/research/physical_value/output/P-20261003-cm-task-representation/cm_residual_mlp.pt"
ENV_CFG = ROOT / "src/task/CmResidual/research/physical_value/output/P-20260930-cm-physical-value/r7/environment.yaml"
TRAIN_CFG = ROOT / "src/task/CmResidual/research/physical_value/output/P-20260930-cm-physical-value/r7/training.yaml"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command, output, gpu, representation=False, timeout=3600):
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1", LOCAL_RANK="0",
        RANK="0", WORLD_SIZE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
    )
    env["LD_LIBRARY_PATH"] = "/home2/wyy/miniconda3/envs/graspenv/lib:" + env.get("LD_LIBRARY_PATH", "")
    if representation:
        env["REF2DEX_CM_REPRESENTATION_CHECKPOINT"] = str(REPRESENTATION.resolve())
    started = time.monotonic()
    with output.open("w") as stream:
        subprocess.run(command, cwd=ROOT / "third_party/DExplore", env=env,
                       stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=timeout)
    return time.monotonic() - started


def train_command(arm, output, seed, epochs, envs):
    return [
        PYTHON, str(ROOT / "src/task/CmResidual/tools/dexplore_physical_value_bootstrap.py"),
        "--physical-value-arm", arm,
        "--physical-value-checkpoint", str(PHYSICAL), "--physical-value-sha256", sha(PHYSICAL),
        "--cm-distill-coef", "0", "--approach-reward-coef", "2",
        "--held-lift-reward-coef", "10", "--lift-progress-reward-coef", "5",
        "--actual-epochs", str(epochs), "--scratch-resume-checkpoint", str(SOURCE),
        "--scratch-resume-sha256", sha(SOURCE), "--save-frequency", "20",
        "--learning-rate", "1e-5", "--task", "Dexplore_Inspire",
        "--cfg_env", str(ENV_CFG), "--cfg_train", str(TRAIN_CFG), "--checkpoint", str(SOURCE),
        "--motion_file", str(MOTIONS), "--headless", "--num_envs", str(envs),
        "--seed", str(seed), "--sim_device", "cuda:0", "--rl_device", "cuda:0",
        "--graphics_device_id", "0", "--output_path", str(output),
    ]


def evaluate_command(checkpoint, output, seed, envs):
    return [
        PYTHON, str(ROOT / "scripts/run_cm_physical_value_environment.py"),
        "--mode", "evaluate", "--run-dir", str(output), "--checkpoint-sha256", sha(checkpoint),
        "--rows", "0", "--wall-seconds", "1200", "--assignment-seed", str(20261030000 + seed),
        "--seed-namespace", str(seed), "--task", "Dexplore_Inspire", "--cfg_env", str(ENV_CFG),
        "--cfg_train", str(TRAIN_CFG), "--checkpoint", str(checkpoint), "--motion_file", str(MOTIONS),
        "--headless", "--num_envs", str(envs), "--seed", str(seed), "--sim_device", "cuda:0",
        "--rl_device", "cuda:0", "--graphics_device_id", "0", "--disable-early-termination",
        "--output", str(output / "unused.json"), "--output_path", str(output / "player"),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--training-seed", type=int, default=286)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--train-envs", type=int, default=64)
    parser.add_argument("--eval-envs", type=int, default=96)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[288, 289])
    args = parser.parse_args()
    # DExplore is launched with its own working directory.  Resolve this once
    # so all training/evaluation artifacts stay under the requested repo path.
    args.output = args.output.resolve()
    for path in (SOURCE, MOTIONS, PHYSICAL, REPRESENTATION, ENV_CFG, TRAIN_CFG):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "run_manifest.json"
    manifest = {
        "schema": "ref2dex.cm_task_value_aux_policy_probe.v1",
        "experiment_id": "P-20261003-cm-task-value-aux-policy",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": sha(SOURCE), "physical_checkpoint_sha256": sha(PHYSICAL),
        "representation_sha256": sha(REPRESENTATION), "training_seed": args.training_seed,
        "epochs": args.epochs, "train_envs": args.train_envs, "eval_envs": args.eval_envs,
        "eval_seeds": args.eval_seeds, "arms": ["cm_task_aux_off", "cm_task_aux"],
        "runs": {}, "run_status": "STARTED",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for arm in manifest["arms"]:
        root = args.output / ("train_" + arm)
        elapsed = run(train_command(arm, root, args.training_seed, args.epochs, args.train_envs),
                      args.output / ("train_" + arm + ".log"), args.gpu,
                      representation=arm in ("cm_task_aux", "cm_task_aux_off"))
        checkpoint = root / "inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn" / ("GRAB_%08d.pth" % args.epochs)
        if not checkpoint.exists():
            checkpoint = root / "inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB.pth"
        if not checkpoint.exists():
            raise FileNotFoundError("training checkpoint", checkpoint)
        manifest["runs"][arm] = {"train_seconds": elapsed, "checkpoint": str(checkpoint),
                                  "checkpoint_sha256": sha(checkpoint)}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for arm in manifest["arms"]:
        checkpoint = Path(manifest["runs"][arm]["checkpoint"])
        manifest["runs"][arm]["evaluations"] = {}
        for seed in args.eval_seeds:
            root = args.output / ("eval_%s_s%d" % (arm, seed))
            elapsed = run(evaluate_command(checkpoint, root, seed, args.eval_envs),
                          args.output / ("eval_%s_s%d.log" % (arm, seed)), args.gpu,
                          timeout=1800)
            result = json.loads((root / "results.json").read_text())
            manifest["runs"][arm]["evaluations"][str(seed)] = {
                "seconds": elapsed, "results": str((root / "results.json").resolve()),
                "stable_success_count": result["stable_success_count"],
                "drop_after_success_count": result["drop_after_success_count"],
                "complete_episodes": result["complete_episodes"],
            }
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    direct = manifest["runs"]["cm_task_aux_off"]["evaluations"]
    cm = manifest["runs"]["cm_task_aux"]["evaluations"]
    success_delta = sum(cm[str(seed)]["stable_success_count"] - direct[str(seed)]["stable_success_count"] for seed in args.eval_seeds)
    drop_delta = sum(cm[str(seed)]["drop_after_success_count"] - direct[str(seed)]["drop_after_success_count"] for seed in args.eval_seeds)
    manifest["result"] = {"stable_success_delta": success_delta, "drop_after_success_delta": drop_delta,
                           "decision_boundary": "Probe only; policy utility requires stable-hold and drop gates."}
    manifest["run_status"] = "COMPLETED"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

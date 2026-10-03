#!/usr/bin/env python3
"""Matched policy Probe for a Cm model-based critic-training checkpoint."""
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
PHYSICAL_OFF = ROOT / "src/task/CmResidual/research/physical_value/output/P-20260930-cm-physical-value/r7/models/tier_1000000.pt"
PHYSICAL_ON = ROOT / "src/task/CmResidual/research/physical_value/output/P-20261003-cm-model-based-critic/r1/cm_model_based_physical_value.pt"
ENV_CFG = ROOT / "src/task/CmResidual/research/physical_value/output/P-20260930-cm-physical-value/r7/environment.yaml"
TRAIN_CFG = ROOT / "src/task/CmResidual/research/physical_value/output/P-20260930-cm-physical-value/r7/training.yaml"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command, output, gpu, timeout=3600):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1", LOCAL_RANK="0",
               RANK="0", WORLD_SIZE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    env["LD_LIBRARY_PATH"] = "/home2/wyy/miniconda3/envs/graspenv/lib:" + env.get("LD_LIBRARY_PATH", "")
    started = time.monotonic()
    with output.open("w") as stream:
        subprocess.run(command, cwd=ROOT / "third_party/DExplore", env=env,
                       stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=timeout)
    return time.monotonic() - started


def train_command(physical, output, seed, epochs, envs):
    return [
        PYTHON, str(ROOT / "src/task/CmResidual/tools/dexplore_physical_value_bootstrap.py"),
        "--physical-value-arm", "direct_q",
        "--physical-value-checkpoint", str(physical), "--physical-value-sha256", sha(physical),
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
    parser.add_argument("--training-seed", type=int, default=292)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--train-envs", type=int, default=64)
    parser.add_argument("--eval-envs", type=int, default=96)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[293, 294])
    args = parser.parse_args()
    args.output = args.output.resolve()
    for path in (SOURCE, MOTIONS, PHYSICAL_OFF, PHYSICAL_ON, ENV_CFG, TRAIN_CFG):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "run_manifest.json"
    manifest = {
        "schema": "ref2dex.cm_model_based_critic_policy_probe.v1",
        "experiment_id": "P-20261003-cm-model-based-critic-policy",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": sha(SOURCE), "physical_off_sha256": sha(PHYSICAL_OFF),
        "physical_on_sha256": sha(PHYSICAL_ON), "training_seed": args.training_seed,
        "epochs": args.epochs, "train_envs": args.train_envs, "eval_envs": args.eval_envs,
        "eval_seeds": args.eval_seeds, "arms": ["direct_q", "cm_model_based_q"],
        "runs": {}, "run_status": "STARTED",
        "contract": "Cm is used only to pretrain direct-Q through conservative synthetic transitions; actor observation/action wiring is unchanged.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for name, physical in (("direct_q", PHYSICAL_OFF), ("cm_model_based_q", PHYSICAL_ON)):
        root = args.output / ("train_" + name)
        elapsed = run(train_command(physical, root, args.training_seed, args.epochs, args.train_envs),
                      args.output / ("train_" + name + ".log"), args.gpu)
        checkpoint = root / "inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn" / ("GRAB_%08d.pth" % args.epochs)
        if not checkpoint.exists():
            checkpoint = root / "inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB.pth"
        if not checkpoint.exists():
            raise FileNotFoundError("training checkpoint", checkpoint)
        manifest["runs"][name] = {"train_seconds": elapsed, "checkpoint": str(checkpoint),
                                  "checkpoint_sha256": sha(checkpoint), "evaluations": {}}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for name in manifest["arms"]:
        checkpoint = Path(manifest["runs"][name]["checkpoint"])
        for seed in args.eval_seeds:
            root = args.output / ("eval_%s_s%d" % (name, seed))
            elapsed = run(evaluate_command(checkpoint, root, seed, args.eval_envs),
                          args.output / ("eval_%s_s%d.log" % (name, seed)), args.gpu, timeout=1800)
            result = json.loads((root / "results.json").read_text())
            manifest["runs"][name]["evaluations"][str(seed)] = {
                "seconds": elapsed, "results": str((root / "results.json").resolve()),
                "stable_success_count": result["stable_success_count"],
                "drop_after_success_count": result["drop_after_success_count"],
                "complete_episodes": result["complete_episodes"],
            }
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    off = manifest["runs"]["direct_q"]["evaluations"]
    on = manifest["runs"]["cm_model_based_q"]["evaluations"]
    per_seed = {str(seed): {
        "stable_delta": on[str(seed)]["stable_success_count"] - off[str(seed)]["stable_success_count"],
        "drop_delta": on[str(seed)]["drop_after_success_count"] - off[str(seed)]["drop_after_success_count"],
    } for seed in args.eval_seeds}
    stable_delta = sum(v["stable_delta"] for v in per_seed.values())
    manifest["result"] = {
        "per_seed": per_seed, "stable_success_delta": stable_delta,
        "drop_after_success_delta": sum(v["drop_delta"] for v in per_seed.values()),
        "upgrade_gate": "Cm model-based Q nonnegative stable success each seed and aggregate +8pp; otherwise close",
        "decision_boundary": "Probe only; no Validation claim",
    }
    manifest["run_status"] = "COMPLETED"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

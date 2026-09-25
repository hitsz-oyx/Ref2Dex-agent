"""Collect paired self-trained expert outcomes and initial proposal actions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
DEXPLORE = ROOT / "third_party/DExplore"
EVALUATE = DEXPLORE / "dexplore/evaluate_object_router.py"
CONFIGS = ROOT / "outputs/CmResidual/agent_expert_choice_headroom_s219/configs"
MOTIONS = ROOT / "outputs/Dexplore/agent_multitrajectory12_s70_e300/motions"
CHECKPOINT = ROOT / ("outputs/Dexplore/agent_v139_s3_backtrack_s70_e260/train/"
                     "inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/"
                     "nn/GRAB_00000260.pth")
EXPERTS = ("source_e260", "mixed12_e300", "train5_e320", "balanced_e360", "duck_e340")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    if args.gpu < 0:
        raise ValueError("physical GPU required")
    root = ROOT / "outputs/CmResidual/agent_cm_option_value_dataset_20260925"
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    manifest = {"experiment_id": "P-20260925-cm-option-value",
                "run_id": root.name, "run_status": "STARTED", "created_at": now(),
                "physical_gpu": args.gpu,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                      cwd=ROOT, text=True).strip(),
                "steps": []}
    write(root / "run_manifest.json", manifest)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    plan = [(seed, expert) for seed in (223, 224) for expert in EXPERTS]
    plan.append((224, "fixed_a"))
    try:
        for seed, expert in plan:
            directory = root / f"s{seed}" / expert
            output = directory / "results.json"
            features = directory / "initial_features.pt"
            config = CONFIGS / f"{expert}.json"
            command = [sys.executable, str(EVALUATE), "--route-config", str(config),
                       "--task", "Dexplore_Inspire", "--cfg_env",
                       "dexplore/data/cfg/inspire_object_balanced.yaml", "--cfg_train",
                       "dexplore/data/cfg/train/rlg/inspire.yaml", "--motion_file",
                       str(MOTIONS), "--checkpoint", str(CHECKPOINT),
                       "--disable-early-termination", "--headless", "--sim_device",
                       "cuda:0", "--rl_device", "cuda:0", "--graphics_device_id", "0",
                       "--num_envs", "64", "--seed", str(seed),
                       "--output", str(output)]
            if expert != "fixed_a":
                command += ["--initial-feature-output", str(features)]
            step = {"seed": seed, "expert": expert, "status": "STARTED",
                    "started_at": now(), "command": command}
            manifest["steps"].append(step)
            write(root / "run_manifest.json", manifest)
            with (root / f"s{seed}_{expert}.log").open("w") as log:
                code = subprocess.run(command, cwd=DEXPLORE, env=environment,
                                      stdout=log, stderr=subprocess.STDOUT).returncode
            if code:
                raise RuntimeError(f"seed{seed} {expert} exited {code}")
            child = json.loads((directory / "run_manifest.json").read_text())
            if child["run_status"] != "COMPLETED" or (
                    expert != "fixed_a" and not features.is_file()):
                raise RuntimeError(f"seed{seed} {expert} incomplete")
            step.update(status="COMPLETED", completed_at=now(),
                        held_lift_rate=child["summary"]["lift_success_rate"])
            write(root / "run_manifest.json", manifest)
            print(f"seed{seed} {expert}: {step['held_lift_rate']:.4f}", flush=True)
        manifest.update(run_status="COMPLETED", completed_at=now())
    except BaseException as error:
        manifest.update(run_status="FAILED", completed_at=now(),
                        failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(root / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()

"""Run a frozen five-expert option headroom Probe on one evaluation seed."""
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
ROUTE = ROOT / "src/task/CmResidual/configs/multitrajectory_object_router_probe.json"
MOTIONS = ROOT / "outputs/Dexplore/agent_multitrajectory12_s70_e300/motions"
CHECKPOINT = ROOT / ("outputs/Dexplore/agent_v139_s3_backtrack_s70_e260/train/"
                     "inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/"
                     "nn/GRAB_00000260.pth")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=219)
    args = parser.parse_args()
    if args.seed != 219 or args.gpu < 0:
        raise ValueError("this predeclared Probe uses seed219 and a physical GPU")
    config = json.loads(ROUTE.read_text())
    root = ROOT / f"outputs/CmResidual/agent_expert_choice_headroom_s{args.seed}"
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    config_dir = root / "configs"
    config_dir.mkdir()
    experiment = {"experiment_id": "P-20260925-expert-choice-headroom",
                  "run_id": root.name, "run_status": "STARTED", "created_at": now(),
                  "physical_gpu": args.gpu, "seed": args.seed,
                  "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                        cwd=ROOT, text=True).strip(),
                  "steps": []}
    write(root / "run_manifest.json", experiment)
    names = ["fixed_a", "fixed_b", *config["experts"]]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    try:
        for name in names:
            step_config = config.copy()
            if name not in ("fixed_a", "fixed_b"):
                step_config["object_route"] = {obj: name for obj in config["object_route"]}
            config_path = config_dir / f"{name}.json"
            write(config_path, step_config)
            output = root / name / "results.json"
            command = [sys.executable, str(EVALUATE), "--route-config", str(config_path),
                       "--task", "Dexplore_Inspire", "--cfg_env",
                       "dexplore/data/cfg/inspire_object_balanced.yaml", "--cfg_train",
                       "dexplore/data/cfg/train/rlg/inspire.yaml", "--motion_file",
                       str(MOTIONS), "--checkpoint", str(CHECKPOINT),
                       "--disable-early-termination", "--headless", "--sim_device",
                       "cuda:0", "--rl_device", "cuda:0", "--graphics_device_id", "0",
                       "--num_envs", "64", "--seed", str(args.seed),
                       "--output", str(output)]
            step = {"name": name, "status": "STARTED", "started_at": now(),
                    "command": command}
            experiment["steps"].append(step)
            write(root / "run_manifest.json", experiment)
            with (root / f"{name}.log").open("w") as log:
                code = subprocess.run(command, cwd=DEXPLORE, env=env,
                                      stdout=log, stderr=subprocess.STDOUT).returncode
            if code:
                raise RuntimeError(f"{name} exited {code}; see {root / (name + '.log')}")
            child = json.loads((output.parent / "run_manifest.json").read_text())
            if child["run_status"] != "COMPLETED":
                raise RuntimeError(f"{name} incomplete")
            step.update(status="COMPLETED", completed_at=now(),
                        lift_success_rate=child["summary"]["lift_success_rate"])
            write(root / "run_manifest.json", experiment)
            print(f"{name}: {step['lift_success_rate']:.4f}", flush=True)
        experiment.update(run_status="COMPLETED", completed_at=now())
    except BaseException as error:
        experiment.update(run_status="FAILED", completed_at=now(),
                          failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(root / "run_manifest.json", experiment)


if __name__ == "__main__":
    main()

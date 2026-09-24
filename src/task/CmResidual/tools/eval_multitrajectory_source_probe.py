"""Evaluate the frozen pre-continuation actor on the same motion mixture."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[4]
DEXPLORE = ROOT / "third_party/DExplore"
SOURCE = ROOT / ("outputs/Dexplore/agent_v139_s3_backtrack_s70_e260/train/"
                 "inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/"
                 "GRAB_00000260.pth")
SOURCE_SHA = "16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    baseline = args.baseline_run.resolve()
    output = baseline / f"eval_source_e260_s{args.seed}_full"
    if output.exists():
        raise FileExistsError(output)
    used = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True)
    if {int(line.split(",")[0]): int(line.split(",")[1]) for line in used.splitlines()}[args.gpu] > 1024:
        raise RuntimeError("requested physical GPU is occupied")
    if sha256(SOURCE) != SOURCE_SHA:
        raise ValueError("source checkpoint drift")
    training = json.loads((baseline / "run_manifest.json").read_text())
    if training["run_status"] != "COMPLETED":
        raise ValueError("baseline training incomplete")
    motion_root = baseline / "motions"
    input_manifest = baseline / "input_manifest.json"
    if sha256(input_manifest) != training["input_manifest_sha256"]:
        raise ValueError("motion manifest drift")
    command = [sys.executable, str(DEXPLORE / "dexplore/evaluate.py"),
               "--task", "Dexplore_Inspire", "--cfg_env",
               "dexplore/data/cfg/inspire_object_balanced.yaml", "--cfg_train",
               "dexplore/data/cfg/train/rlg/inspire.yaml", "--motion_file", str(motion_root),
               "--checkpoint", str(SOURCE), "--disable-early-termination", "--headless",
               "--sim_device", "cuda:0", "--rl_device", "cuda:0",
               "--graphics_device_id", "0", "--num_envs", "64", "--seed", str(args.seed),
               "--output", str(output / "results.json")]
    output.mkdir()
    manifest_path = output / "run_manifest.json"
    manifest = {"run_status": "STARTED", "created_at": now(),
                "run_id": output.name, "work_version": "multitrajectory-baseline-source-control",
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                    cwd=ROOT, text=True).strip(),
                "command": command, "physical_gpu": args.gpu, "seed": args.seed,
                "checkpoint": str(SOURCE), "checkpoint_sha256": SOURCE_SHA,
                "motion_root": str(motion_root), "input_manifest": str(input_manifest),
                "input_manifest_sha256": sha256(input_manifest),
                "budget": {"gpu_count": 1, "wall_minutes": 20, "output_gb": 0.1},
                "stop_rule": "checkpoint/input drift, GPU conflict, failed or incomplete evaluation"}
    write(manifest_path, manifest)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    started = time.monotonic()
    try:
        with (output / "eval.log").open("w") as log:
            result = subprocess.run(command, cwd=DEXPLORE, env=env,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=1200)
        if result.returncode:
            raise RuntimeError(f"evaluation exit code {result.returncode}")
        data = json.loads((output / "results.json").read_text())
        summary = data["summary"]
        if summary["num_episodes"] != 64 or not summary["early_termination_disabled"]:
            raise ValueError("incomplete first-episode evaluation")
        manifest.update(run_status="COMPLETED", completed_at=now(),
                        elapsed_seconds=time.monotonic() - started,
                        summary=summary)
        print(json.dumps({"run_id": output.name, "run_status": "COMPLETED",
                          "lift_success_rate": summary["lift_success_rate"]}), flush=True)
    except BaseException as error:
        manifest.update(run_status="FAILED", completed_at=now(),
                        elapsed_seconds=time.monotonic() - started,
                        failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(manifest_path, manifest)


if __name__ == "__main__":
    main()

"""Bounded, resource-safe launcher for one pinned paired-simulation smoke.

This is operational scheduling only. The evaluator owns the physical-pair
checks and scientific summary. This launcher never runs on an occupied GPU.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[4]
EVALUATOR = ROOT / "third_party/DExplore/dexplore/evaluate_paired.py"
PAIR_STEP = ROOT / "src/task/CmResidual/paired_sim_step.py"
CHECKPOINT = (ROOT / "outputs/Dexplore/agent_v139_s3_backtrack_s70_e260/train/"
              "inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000260.pth")
MOTION = ROOT / "outputs/CmResidual/agent_v129_s3_coordfix/corrected_converted_r2"
MOTION_MANIFEST = ROOT / "outputs/CmResidual/agent_v129_s3_coordfix/corrected_manifest_r2.json"
CHECKPOINT_SHA256 = "16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f"
MOTION_MANIFEST_SHA256 = "2878bd20d1dd849f6844883c832a3777602d2d2bc73f7581b3bc23d30725f038"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def idle_gpus() -> list[int]:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True, timeout=15)
    available = []
    for line in output.splitlines():
        index, memory, utilization = [int(value.strip()) for value in line.split(",")]
        if memory <= 512 and utilization <= 5:
            available.append(index)
    return available


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--evaluator-sha256", required=True)
    parser.add_argument("--pair-step-sha256", required=True)
    parser.add_argument("--wait-minutes", type=int, default=600)
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args()
    if not 1 <= args.wait_minutes <= 600 or not 60 <= args.poll_seconds <= 600:
        parser.error("invalid bounded wait/poll interval")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        parser.error(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    manifest_path = output_dir / "wait_manifest.json"
    started = datetime.now(timezone.utc)
    deadline = started + timedelta(minutes=args.wait_minutes)
    manifest = {
        "schema": "ref2dex.idle_gpu_pair_wait.v1",
        "run_id": output_dir.name,
        "run_status": "STARTED",
        "started_at": started.isoformat(),
        "deadline": deadline.isoformat(),
        "source_commit": args.source_commit,
        "source_sha256": {"evaluator": args.evaluator_sha256,
                          "pair_step": args.pair_step_sha256},
        "input_sha256": {"checkpoint": CHECKPOINT_SHA256,
                         "motion_manifest": MOTION_MANIFEST_SHA256},
        "max_gpu_count": 1,
        "candidate_gpu": None,
        "poll_seconds": args.poll_seconds,
        "idle_checks_required": 2,
        "poll_count": 0,
        "stop_rule": "deadline, STOP file, code/input drift, GPU conflict, evaluator failure",
    }
    write_manifest(manifest_path, manifest)
    stable_gpu = None
    stable_count = 0
    try:
        while datetime.now(timezone.utc) < deadline:
            if (output_dir / "STOP").exists():
                manifest.update(run_status="STOPPED", reason="STOP file requested")
                break
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                             cwd=ROOT, text=True).strip()
            if (commit != args.source_commit or
                    sha256(EVALUATOR) != args.evaluator_sha256 or
                    sha256(PAIR_STEP) != args.pair_step_sha256 or
                    sha256(CHECKPOINT) != CHECKPOINT_SHA256 or
                    sha256(MOTION_MANIFEST) != MOTION_MANIFEST_SHA256):
                manifest.update(run_status="STOPPED", reason="code or input drift")
                break
            idle = idle_gpus()
            manifest["poll_count"] += 1
            if stable_gpu not in idle:
                stable_gpu = idle[0] if idle else None
                stable_count = 1 if idle else 0
            else:
                stable_count += 1
            manifest.update(run_status="RUNNING", candidate_gpu=stable_gpu,
                            stable_idle_checks=stable_count,
                            last_poll_at=datetime.now(timezone.utc).isoformat())
            write_manifest(manifest_path, manifest)
            if stable_count >= 2:
                command = [
                    sys.executable, "dexplore/evaluate_paired.py",
                    "--paired-output", str(output_dir / "pairs.pt"),
                    "--paired-schedule", "80", "--paired-delta-z", "0.1",
                    "--task", "Dexplore_Inspire",
                    "--cfg_env", "dexplore/data/cfg/inspire.yaml",
                    "--cfg_train", "dexplore/data/cfg/train/rlg/inspire.yaml",
                    "--motion_file", str(MOTION), "--checkpoint", str(CHECKPOINT),
                    "--disable-early-termination", "--headless",
                    "--sim_device", "cuda:0", "--rl_device", "cuda:0",
                    "--graphics_device_id", "0", "--num_envs", "16",
                    "--seed", "145", "--output", str(output_dir / "results.json"),
                ]
                manifest.update(physical_gpu=stable_gpu, command=command,
                                launched_at=datetime.now(timezone.utc).isoformat())
                write_manifest(manifest_path, manifest)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(stable_gpu)
                env["OMP_NUM_THREADS"] = "8"
                with (output_dir / "evaluate.log").open("w") as log:
                    result = subprocess.run(command, cwd=ROOT / "third_party/DExplore",
                                            env=env, stdout=log, stderr=subprocess.STDOUT,
                                            timeout=900, check=False)
                manifest.update(run_status="COMPLETED" if result.returncode == 0 else "FAILED",
                                returncode=result.returncode)
                break
            time.sleep(min(args.poll_seconds,
                           max(0, (deadline - datetime.now(timezone.utc)).total_seconds())))
        else:
            manifest.update(run_status="STOPPED", reason="wait deadline, no stable idle GPU")
    except Exception as error:
        manifest.update(run_status="FAILED", reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        write_manifest(manifest_path, manifest)
        print(json.dumps({"run_status": manifest["run_status"],
                          "reason": manifest.get("reason"),
                          "physical_gpu": manifest.get("physical_gpu"),
                          "output_dir": str(output_dir)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

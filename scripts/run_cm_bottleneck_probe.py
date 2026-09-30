#!/usr/bin/env python3
"""Bounded, logged matched physical probe with pinned read-only inputs."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent-baseline")
PYTHON = "/home2/wyy/miniconda3/envs/graspenv/bin/python"
SOURCE = BASE / "outputs/Dexplore/agent_v139_s3_backtrack_s70_e260/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000260.pth"
MOTIONS = BASE / "outputs/CmResidual/agent_contact_option_airplane_motions"
PHYSICAL = ROOT / "outputs/P-20260930-scratch-mlp-feasibility/cpu_s278_r1/model.pt"
ARMS = ("on", "off", "random", "action")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_memory(index):
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
    return {int(row.split(",")[0]): int(row.split(",")[1]) for row in output.splitlines()}[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if gpu_memory(args.gpu) > 512:
        raise RuntimeError("selected GPU is occupied")
    route = json.loads((ROOT / "src/task/CmResidual/configs/hf02_temporal_canonical_route.json").read_text())
    source_hash = sha(SOURCE)
    if source_hash != route["experts"]["source_e260"]["sha256"]:
        raise ValueError("source checkpoint hash drift")
    motion_hashes = {}
    for motion in route["motions"]:
        path = MOTIONS / motion["name"] / "interaction_hand_inspire.pt"
        if sha(path) != motion["interaction_hand_sha256"]:
            raise ValueError("motion hash drift")
        motion_hashes[str(path)] = sha(path)
    output.mkdir(parents=True)
    cfg_env = ROOT / "third_party/DExplore/dexplore/data/cfg/inspire_object_balanced.yaml"
    cfg_train = ROOT / "third_party/DExplore/dexplore/data/cfg/train/rlg/inspire.yaml"
    manifest = dict(run_status="STARTED", phase="preflight", experiment_id="P-20260930-cm-inference-bottleneck",
                    physical_gpu=args.gpu, gpu_count=1, wall_limit_seconds=3600,
                    storage_limit_bytes=5 * 1024 ** 3,
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    inputs=dict(source_sha256=source_hash, motion_hashes=motion_hashes,
                                physical_sha256=sha(PHYSICAL), cfg_env_sha256=sha(cfg_env), cfg_train_sha256=sha(cfg_train)),
                    commands=[], evaluation_seeds=[281, 282], training_seed=278,
                    statement="single-training-seed physical Probe only; no formal causal claim")
    started = time.monotonic()
    def save():
        manifest["elapsed_seconds"] = time.monotonic() - started
        (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    save()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"
    # Conda's C++ runtime is needed by the Isaac Gym extension.
    env["LD_LIBRARY_PATH"] = "/home2/wyy/miniconda3/envs/graspenv/lib:" + env.get("LD_LIBRARY_PATH", "")
    def run(phase, command):
        if gpu_memory(args.gpu) > 512:
            raise RuntimeError("GPU collision before next phase")
        if sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) > manifest["storage_limit_bytes"]:
            raise RuntimeError("storage limit exceeded")
        remaining = 3600 - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("whole-probe wall budget")
        manifest["phase"] = phase
        manifest["commands"].append(dict(phase=phase, argv=command))
        save()
        print(json.dumps(dict(phase=phase, status="STARTED")), flush=True)
        with (output / f"{phase}.log").open("x") as log:
            subprocess.run(command, cwd=ROOT / "third_party/DExplore", env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=remaining)
        print(json.dumps(dict(phase=phase, status="COMPLETED")), flush=True)

    def evaluate(name, seed, student=None, export=False):
        run_dir = output / name
        command = [PYTHON, str(ROOT / "scripts/evaluate_cm_bottleneck_student.py"),
                   "--run-dir", str(run_dir), "--task", "Dexplore_Inspire", "--cfg_env", str(cfg_env),
                   "--cfg_train", str(cfg_train), "--checkpoint", str(SOURCE), "--motion_file", str(MOTIONS),
                   "--headless", "--num_envs", "64", "--seed", str(seed), "--sim_device", "cuda:0",
                   "--rl_device", "cuda:0", "--graphics_device_id", "0", "--output", str(run_dir / "results.json"),
                   "--disable-early-termination"]
        if export:
            command += ["--transition-output", str(run_dir / "trajectory.pt")]
        if student is not None:
            command += ["--student", student["checkpoint"], "--student-sha256", student["sha256"]]
        run(name, command)

    try:
        evaluate("fit_s279", 279, export=True)
        evaluate("holdout_s280", 280, export=True)
        run("training", [PYTHON, str(ROOT / "scripts/train_cm_bottleneck_students.py"),
                         "--fit", str(output / "fit_s279/trajectory.pt"),
                         "--holdout", str(output / "holdout_s280/trajectory.pt"),
                         "--physical", str(PHYSICAL), "--output", str(output / "students")])
        training = json.loads((output / "students/results.json").read_text())
        counts = {arm: [] for arm in ARMS}
        paired_outcomes = {}
        for seed in (281, 282):
            results = {}
            for arm in ARMS:
                name = f"{arm}_s{seed}"
                evaluate(name, seed, student=training["arms"][arm])
                native = json.loads((output / name / "results.json").read_text())
                results[arm] = sorted(native["per_episode"], key=lambda row: row["env_id"])
                if len({row["env_id"] for row in results[arm]}) != 64:
                    raise ValueError("nonunique evaluation environments")
                counts[arm].append(sum(bool(row["lift_success"]) for row in results[arm]))
            for arm in ARMS[1:]:
                for base, other in zip(results["on"], results[arm]):
                    if any(base[key] != other[key] for key in ("env_id", "motion_id", "start_frame", "steps")):
                        raise ValueError(f"physical episode pairing failed: {seed}/{arm}")
            paired_outcomes[str(seed)] = {arm: dict(
                gains=sum(bool(a["lift_success"]) and not bool(b["lift_success"]) for a, b in zip(results["on"], results[arm])),
                losses=sum(not bool(a["lift_success"]) and bool(b["lift_success"]) for a, b in zip(results["on"], results[arm]))) for arm in ARMS[1:]}
        differences = {arm: (sum(counts["on"]) - sum(counts[arm])) / 128 for arm in ARMS[1:]}
        positive = all(value >= .05 for value in differences.values()) and all(a >= b for a, b in zip(counts["on"], counts["off"]))
        result = dict(run_status="COMPLETED", conclusion="PROMISING" if positive else "UNPROMISING",
                      held_lift_counts=counts, on_minus_control=differences, paired_outcomes=paired_outcomes,
                      native_pairing_valid=True, training=training,
                      scope="fixed three-airplane task; one BC training seed; self-trained expert required at inference")
        (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        manifest.update(run_status="COMPLETED", phase="complete", conclusion=result["conclusion"])
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()

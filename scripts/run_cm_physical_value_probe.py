#!/usr/bin/env python3
"""Pinned and budgeted phase launcher for HF08; no process interference."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent-baseline")
PYTHON = "/home2/wyy/miniconda3/envs/graspenv/bin/python"
SOURCE = BASE / "outputs/Dexplore/agent_v139_s3_backtrack_s70_e260/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000260.pth"
MOTIONS = BASE / "outputs/CmResidual/agent_contact_option_airplane_motions"
SOURCE_SHA = "16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def idle(gpu):
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
    usage = {int(line.split(",")[0]): int(line.split(",")[1]) for line in rows.splitlines()}
    if usage[gpu] > 512:
        raise RuntimeError("GPU is occupied; refusing to start another phase")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--stage", choices=("smoke", "collect", "fit", "train", "evaluate", "all"), required=True)
    a = p.parse_args()
    out = a.output.resolve()
    if ROOT not in out.parents:
        raise ValueError("new research artifacts must stay in this worktree; outputs is a shared symlink")
    idle(a.gpu)
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError("source drift")
    route = json.loads((ROOT / "src/task/CmResidual/configs/hf02_temporal_canonical_route.json").read_text())
    motion_hashes = {}
    for row in route["motions"]:
        path = MOTIONS / row["name"] / "interaction_hand_inspire.pt"
        if sha(path) != row["interaction_hand_sha256"]:
            raise ValueError("motion drift")
        motion_hashes[str(path)] = sha(path)
    out.mkdir(parents=True, exist_ok=True)
    env_cfg = out / "environment.yaml"
    train_cfg = out / "training.yaml"
    if not env_cfg.exists():
        cfg = yaml.safe_load((ROOT / "third_party/DExplore/dexplore/data/cfg/inspire_object_balanced.yaml").read_text())
        cfg["env"]["hybridInitProb"] = .5
        cfg["env"]["enableEarlyTermination"] = False
        env_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))
        cfg = yaml.safe_load((ROOT / "third_party/DExplore/dexplore/data/cfg/train/rlg/inspire.yaml").read_text())
        cfg["params"]["config"].update(horizon_length=32, minibatch_size=256, save_frequency=40)
        train_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))
    inputs = dict(source_sha256=SOURCE_SHA, motion_hashes=motion_hashes,
                  environment_sha256=sha(env_cfg), training_sha256=sha(train_cfg))
    manifest_path = out / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["inputs"] != inputs:
            raise ValueError("phase inputs changed")
    else:
        manifest = dict(experiment_id="P-20260930-cm-physical-value", family="HF08", inputs=inputs,
                        phases=[], scientific_elapsed_seconds=0, storage_limit_bytes=20 * 1024**3,
                        wall_limit_seconds=21600, run_status="STARTED", physical_gpu=a.gpu)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(a.gpu), PYTHONUNBUFFERED="1", LOCAL_RANK="0", RANK="0", WORLD_SIZE="1",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    env["LD_LIBRARY_PATH"] = "/home2/wyy/miniconda3/envs/graspenv/lib:" + env.get("LD_LIBRARY_PATH", "")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    def run(name, cmd, wall, scientific=True):
        idle(a.gpu)
        if any(x["name"] == name for x in manifest["phases"]):
            raise ValueError("phase already attempted; retain evidence and explicitly choose a new run")
        remaining = min(wall, manifest["wall_limit_seconds"] - manifest["scientific_elapsed_seconds"])
        if remaining <= 0:
            raise TimeoutError("whole Probe budget exhausted")
        if sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) > manifest["storage_limit_bytes"]:
            raise RuntimeError("storage limit")
        phase = dict(name=name, command=cmd, git_commit=commit, run_status="STARTED", scientific=scientific)
        manifest["phases"].append(phase)
        manifest.update(run_status="RUNNING", current_phase=name)
        save()
        start = time.monotonic()
        print(json.dumps(phase), flush=True)
        try:
            with (out / (name + ".log")).open("x") as log:
                subprocess.run(cmd, cwd=ROOT / "third_party/DExplore", env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True, timeout=remaining)
            phase["run_status"] = "COMPLETED"
            manifest["run_status"] = "COMPLETED"
        except BaseException as error:
            phase.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}")
            manifest["run_status"] = "FAILED"
            raise
        finally:
            phase["elapsed_seconds"] = time.monotonic() - start
            if scientific:
                manifest["scientific_elapsed_seconds"] += phase["elapsed_seconds"]
            save()
        print(json.dumps(dict(name=name, run_status=phase["run_status"], elapsed_seconds=phase["elapsed_seconds"])), flush=True)

    def environment(name, mode, seed, rows, envs, checkpoint=SOURCE, wall=2700):
        directory = out / name
        cmd = [PYTHON, str(ROOT / "scripts/run_cm_physical_value_environment.py"), "--mode", mode,
               "--run-dir", str(directory), "--checkpoint-sha256", sha(checkpoint),
               "--rows", str(rows), "--wall-seconds", str(wall),
               "--assignment-seed", str(20260930000 + seed), "--seed-namespace", str(seed),
               "--task", "Dexplore_Inspire", "--cfg_env", str(env_cfg), "--cfg_train", str(train_cfg),
               "--checkpoint", str(checkpoint), "--motion_file", str(MOTIONS), "--headless",
               "--num_envs", str(envs), "--seed", str(seed), "--sim_device", "cuda:0",
               "--rl_device", "cuda:0", "--graphics_device_id", "0", "--disable-early-termination",
               "--output", str(directory / "unused.json")]
        run(name, cmd, wall, scientific=not name.startswith("smoke"))

    if a.stage == "smoke":
        environment("smoke_collect_s83", "collect", 83, 512, 6, wall=600)
        environment("smoke_evaluate_s84", "evaluate", 84, 0, 6, wall=600)
    if a.stage in ("collect", "all"):
        environment("collect_s283", "collect", 283, 500000, 64)
        environment("collect_s284", "collect", 284, 500000, 64)
    if a.stage in ("fit", "all"):
        run("fit", [PYTHON, str(ROOT / "scripts/train_cm_physical_value.py"), "--collections",
                    str(out / "collect_s283"), str(out / "collect_s284"), "--output", str(out / "models"),
                    "--device", "cuda:0"], 7200)
    if a.stage in ("train", "evaluate", "all"):
        raise NotImplementedError("PPO and evaluation phase wiring must pass preflight before launch")


if __name__ == "__main__":
    main()

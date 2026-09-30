#!/usr/bin/env python3
"""Complete fixed HF08 matrix on two GPUs; all three paired arms share a GPU."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import time
from cm_physical_value_evaluation import ARMS, TRAINING_SEEDS, EPOCHS, EVALUATION_SEEDS, checkpoint, summarize
from run_cm_physical_value_probe import ROOT, SOURCE, SOURCE_SHA, PYTHON, MOTIONS, sha, idle


async def execute(out, gpus):
    path = out / "run_manifest.json"
    manifest = json.loads(path.read_text())
    if sha(SOURCE) != SOURCE_SHA or manifest["inputs"]["source_sha256"] != SOURCE_SHA:
        raise ValueError("source drift")
    for key, name in (("environment_sha256", "environment.yaml"), ("training_sha256", "training.yaml")):
        if sha(out/name) != manifest["inputs"][key]:
            raise ValueError("configuration drift")
    for motion, expected in manifest["inputs"]["motion_hashes"].items():
        if sha(motion) != expected:
            raise ValueError("motion drift")
    for t in TRAINING_SEEDS:
        for arm in ARMS:
            audit = json.loads((out/f"train_{arm}_s{t}"/"checkpoint_audit.json").read_text())
            if audit["epoch"] != 420 or audit["physical_value"]["initial_model_sha256"] != manifest["initial_actor_sha256"]:
                raise ValueError("training completion/initialization missing")
    if any(p["name"].startswith("eval_") or p["name"] == "evaluate_parallel" for p in manifest["phases"]):
        raise ValueError("evaluation already attempted; do not overwrite evidence")
    for gpu in gpus:
        idle(gpu)
    budget = min(3600, manifest["wall_limit_seconds"] - manifest["scientific_elapsed_seconds"])
    if budget <= 0:
        raise TimeoutError("whole Probe budget exhausted")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    group = dict(name="evaluate_parallel", git_commit=commit, run_status="STARTED", scientific=True,
                 physical_gpus=gpus, wall_limit_seconds=budget, accounting="group wall; native GPU seconds separate")
    manifest["phases"].append(group)
    manifest.update(run_status="RUNNING", current_phase="evaluate_parallel")
    def save():
        path.write_text(json.dumps(manifest, indent=2)+"\n")
    save()
    started = time.monotonic()
    deadline = started + budget
    queues = [[], []]
    for t in TRAINING_SEEDS:
        for epoch in EPOCHS:
            for seed in EVALUATION_SEEDS:
                queues[(t+seed) % 2].append((t, epoch, seed))

    async def worker(index):
        gpu = gpus[index]
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), LOCAL_RANK="0", RANK="0", WORLD_SIZE="1",
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1")
        env["LD_LIBRARY_PATH"] = "/home2/wyy/miniconda3/envs/graspenv/lib:"+env.get("LD_LIBRARY_PATH", "")
        for t, epoch, seed in queues[index]:
            for arm in ARMS:
                idle(gpu)
                roots = [out] + [Path(p["path"]) for p in manifest.get("prior_attempts", [])]
                if sum(f.stat().st_size for root in roots for f in root.rglob("*") if f.is_file()) > manifest["storage_limit_bytes"]:
                    raise RuntimeError("storage limit")
                remaining = min(300, deadline-time.monotonic())
                if remaining <= 0:
                    raise TimeoutError("evaluation wall budget exhausted")
                ckpt = checkpoint(out, arm, t, epoch, SOURCE)
                name = f"eval_{arm}_t{t}_e{epoch}_s{seed}"
                directory = out/name
                cmd = [PYTHON, str(ROOT/"scripts/run_cm_physical_value_environment.py"), "--mode", "evaluate",
                    "--run-dir", str(directory), "--checkpoint-sha256", sha(ckpt), "--rows", "0", "--wall-seconds", "300",
                    "--assignment-seed", str(20260930000+seed), "--seed-namespace", str(seed),
                    "--task", "Dexplore_Inspire", "--cfg_env", str(out/"environment.yaml"), "--cfg_train", str(out/"training.yaml"),
                    "--checkpoint", str(ckpt), "--motion_file", str(MOTIONS), "--headless", "--num_envs", "96", "--seed", str(seed),
                    "--sim_device", "cuda:0", "--rl_device", "cuda:0", "--graphics_device_id", "0", "--disable-early-termination",
                    "--output", str(directory/"unused.json"), "--output_path", str(directory/"player")]
                phase = dict(name=name, command=cmd, git_commit=commit, physical_gpu=gpu,
                    run_status="STARTED", scientific=True, accounting="included in evaluate_parallel wall")
                manifest["phases"].append(phase); save()
                print(json.dumps(dict(name=name, physical_gpu=gpu, run_status="STARTED")), flush=True)
                begin = time.monotonic()
                process = None
                try:
                    with (out/(name+".log")).open("x") as log:
                        process = await asyncio.create_subprocess_exec(*cmd, cwd=ROOT/"third_party/DExplore", env=env, stdout=log, stderr=subprocess.STDOUT)
                        phase["pid"] = process.pid; save()
                        code = await asyncio.wait_for(process.wait(), timeout=remaining)
                        if code:
                            raise RuntimeError(f"native evaluation exit {code}: {name}")
                    phase["run_status"] = "COMPLETED"
                except BaseException as error:
                    if process is not None and process.returncode is None:
                        process.terminate()
                        await process.wait()
                    phase.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}")
                    raise
                finally:
                    phase["elapsed_seconds"] = time.monotonic()-begin
                    save()
                print(json.dumps(dict(name=name, run_status="COMPLETED", elapsed_seconds=phase["elapsed_seconds"])), flush=True)
    tasks = [asyncio.create_task(worker(i)) for i in range(2)]
    try:
        await asyncio.gather(*tasks)
        result = summarize(out, SOURCE)
        (out/"results.json").write_text(json.dumps(result, indent=2)+"\n")
        group["run_status"] = "COMPLETED"
        manifest.update(run_status="COMPLETED", conclusion=result["conclusion"])
    except BaseException as error:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        group.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}")
        manifest["run_status"] = "FAILED"
        raise
    finally:
        group["elapsed_seconds"] = time.monotonic()-started
        group["gpu_seconds"] = sum(p.get("elapsed_seconds", 0) for p in manifest["phases"] if p["name"].startswith("eval_"))
        manifest["scientific_elapsed_seconds"] += group["elapsed_seconds"]
        save()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpus", nargs=2, type=int, default=[1, 2])
    a = p.parse_args()
    out = a.output.resolve()
    if ROOT not in out.parents or len(set(a.gpus)) != 2:
        raise ValueError("owned output and two distinct GPUs required")
    asyncio.run(execute(out, a.gpus))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fixed six-arm HF08 training; each seed's paired arms use one GPU."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import time
from cm_physical_value_evaluation import ARMS, TRAINING_SEEDS, NN
from run_cm_physical_value_probe import ROOT, SOURCE, SOURCE_SHA, PYTHON, MOTIONS, sha, idle


async def execute(out, gpus):
    manifest_path = out/"run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    report = json.loads((out/"models/results.json").read_text())
    if report["run_status"] != "COMPLETED" or report["smoke"]:
        raise ValueError("full pretraining required")
    model = Path(report["selected_checkpoint"])
    if sha(model) != report["tiers"][str(report["selected_tier"])]["sha256"] or sha(SOURCE) != SOURCE_SHA:
        raise ValueError("checkpoint drift")
    for key, name in (("environment_sha256", "environment.yaml"), ("training_sha256", "training.yaml")):
        if sha(out/name) != manifest["inputs"][key]:
            raise ValueError("configuration drift")
    for motion, expected in manifest["inputs"]["motion_hashes"].items():
        if sha(motion) != expected:
            raise ValueError("motion drift")
    if any(p["name"].startswith("train_") or p["name"] == "train_parallel" for p in manifest["phases"]):
        raise ValueError("training already attempted; preserve evidence")
    for gpu in gpus:
        idle(gpu)
    budget = min(5400, manifest["wall_limit_seconds"]-manifest["scientific_elapsed_seconds"])
    if budget <= 0:
        raise TimeoutError("whole Probe budget exhausted")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    group = dict(name="train_parallel", git_commit=commit, run_status="STARTED", scientific=True,
                 physical_gpus=gpus, wall_limit_seconds=budget, accounting="group wall; native GPU seconds separate")
    manifest["phases"].append(group)
    manifest.update(run_status="RUNNING", current_phase="train_parallel", physical_checkpoint_sha256=sha(model))
    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    save()
    started = time.monotonic()
    deadline = started+budget

    async def worker(index):
        seed, gpu = TRAINING_SEEDS[index], gpus[index]
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), LOCAL_RANK="0", RANK="0", WORLD_SIZE="1",
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1")
        env["LD_LIBRARY_PATH"] = "/home2/wyy/miniconda3/envs/graspenv/lib:"+env.get("LD_LIBRARY_PATH", "")
        for arm in ARMS:
            idle(gpu)
            roots = [out]+[Path(p["path"]) for p in manifest.get("prior_attempts", [])]
            if sum(f.stat().st_size for root in roots for f in root.rglob("*") if f.is_file()) > manifest["storage_limit_bytes"]:
                raise RuntimeError("storage limit")
            remaining = deadline-time.monotonic()-30
            if remaining <= 0:
                raise TimeoutError("training wall budget exhausted")
            name = f"train_{arm}_s{seed}"
            directory = out/name
            cmd = [PYTHON, str(ROOT/"src/task/CmResidual/tools/dexplore_physical_value_bootstrap.py"),
                "--physical-value-arm", arm, "--physical-value-checkpoint", str(model), "--physical-value-sha256", sha(model),
                "--cm-distill-coef", "0", "--approach-reward-coef", "2", "--held-lift-reward-coef", "10", "--lift-progress-reward-coef", "5",
                "--actual-epochs", "420", "--scratch-resume-checkpoint", str(SOURCE), "--scratch-resume-sha256", SOURCE_SHA,
                "--save-frequency", "20", "--learning-rate", "1e-5", "--task", "Dexplore_Inspire", "--cfg_env", str(out/"environment.yaml"),
                "--cfg_train", str(out/"training.yaml"), "--checkpoint", str(SOURCE), "--motion_file", str(MOTIONS), "--headless",
                "--num_envs", "64", "--seed", str(seed), "--sim_device", "cuda:0", "--rl_device", "cuda:0", "--graphics_device_id", "0",
                "--output_path", str(directory)]
            phase = dict(name=name, command=cmd, git_commit=commit, physical_gpu=gpu,
                         run_status="STARTED", scientific=True, accounting="included in train_parallel wall")
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
                        raise RuntimeError(f"native training exit {code}: {name}")
                nn_dir = directory/NN
                code = "import json,sys,torch; p=torch.load(sys.argv[1],map_location='cpu',weights_only=False); print(json.dumps({'epoch':p['epoch'],'frame':p['frame'],'physical_value':{k:v for k,v in p['physical_value'].items() if k in ('arm','initial_model_sha256','behavior_integrity_verified','actor_supervision_gradient_verified')}}))"
                audit_process = await asyncio.create_subprocess_exec(PYTHON, "-c", code, str(nn_dir/"GRAB.pth"), env=env, stdout=asyncio.subprocess.PIPE)
                audit_stdout, _ = await asyncio.wait_for(audit_process.communicate(), timeout=30)
                if audit_process.returncode:
                    raise RuntimeError("saved checkpoint audit failed")
                audit = json.loads(audit_stdout)
                pv = audit["physical_value"]
                if audit["epoch"] != 420 or audit["frame"] != 327680 or pv["arm"] != arm:
                    raise ValueError("epoch/frame/arm mismatch")
                if arm != "plain_off" and not (pv["behavior_integrity_verified"] and pv["actor_supervision_gradient_verified"]):
                    raise ValueError("teacher behavior/gradient integrity missing")
                initial = manifest.setdefault("initial_actor_sha256", pv["initial_model_sha256"])
                if initial != pv["initial_model_sha256"]:
                    raise ValueError("initial actors differed")
                audit["checkpoint_sha256"] = {str(e): sha(nn_dir/("GRAB_%08d.pth" % e)) for e in (300, 340, 420)}
                (directory/"checkpoint_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
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
        group["run_status"] = "COMPLETED"
        manifest["run_status"] = "COMPLETED"
    except BaseException as error:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        group.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}")
        manifest["run_status"] = "FAILED"
        raise
    finally:
        group["elapsed_seconds"] = time.monotonic()-started
        group["gpu_seconds"] = sum(p.get("elapsed_seconds", 0) for p in manifest["phases"] if p["name"].startswith("train_"))
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

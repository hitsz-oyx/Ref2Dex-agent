#!/usr/bin/env python3
"""Run the frozen five-seed observation-router C1 Validation matrix."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
DEXPLORE = ROOT / "third_party/DExplore"
OUTPUT = ROOT / "outputs/CmResidual"
PROBE = OUTPUT / "agent_obs_router_reliability_s260/preflight.json"
PARENT = OUTPUT / "val_observation_router_c1"
SEEDS = (400, 401, 402, 403, 404)
GPU = 1
WALL_SECONDS = 3600
OUTPUT_BYTES = 100 * 1024 * 1024


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def option(command, flag):
    return command[command.index(flag) + 1]


def check_inputs(probe):
    expected = probe["input_hashes"]
    command = probe["commands"]["observation"]
    files = {
        "route_config": Path(option(command, "--route-config")),
        "evaluator": DEXPLORE / command[1],
        "router_model": Path(option(command, "--observation-router-model")),
        "cfg_env": DEXPLORE / option(command, "--cfg_env"),
        "cfg_train": DEXPLORE / option(command, "--cfg_train"),
    }
    for name, path in files.items():
        if sha(path) != expected[name]:
            raise RuntimeError("input drift: " + name)
    for name, info in expected["checkpoints"].items():
        if sha(info["path"]) != info["sha256"]:
            raise RuntimeError("checkpoint drift: " + name)
    motions = Path(option(command, "--motion_file"))
    # Motion subdirectories are symlinks, which Path.rglob does not traverse.
    actual = sorted(str(path.relative_to(motions)) for directory in motions.iterdir()
                    for path in [directory / "interaction_hand_inspire.pt"] if path.is_file())
    planned = sorted(relative for relative, _ in probe["motion_files"])
    if actual != planned:
        raise RuntimeError("motion file set drift")
    for relative, digest in probe["motion_files"]:
        if sha(motions / relative) != digest:
            raise RuntimeError("motion file drift: " + relative)


def check_gpu():
    lines = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    memory = {int(parts[0].strip()): int(parts[1].strip()) for line in lines
              for parts in [line.split(",")]}
    if memory.get(GPU, 999999) > 1024:
        raise RuntimeError("GPU 1 occupied: {} MiB".format(memory.get(GPU)))


def command_for(probe, seed, arm):
    command = list(probe["commands"]["observation" if arm == "obs" else "fixed"])
    command[command.index("--seed") + 1] = str(seed)
    command[command.index("--output") + 1] = str(
        OUTPUT / "val_observation_router_c1_s{}_{}".format(seed, arm) / "results.json"
    )
    return command


def output_size():
    return sum(path.stat().st_size for path in OUTPUT.glob("val_observation_router_c1*/**/*")
               if path.is_file())


def preflight():
    if PARENT.exists():
        raise RuntimeError("Validation output parent already exists")
    probe = json.loads(PROBE.read_text())
    check_inputs(probe)
    check_gpu()
    for seed in SEEDS:
        for arm in ("fixed", "obs"):
            if (OUTPUT / "val_observation_router_c1_s{}_{}".format(seed, arm)).exists():
                raise RuntimeError("Validation child output already exists")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    plan = {
        "schema": "ref2dex.validation_preflight.v1",
        "validation_id": "VAL-20260926-observation-six-expert-c1",
        "created_at": now(), "branch": "agent/observation-router-reliability",
        "execution_commit": head, "frozen_evaluator_sha256": probe["input_hashes"]["evaluator"],
        "physical_gpu": GPU, "wall_seconds": WALL_SECONDS, "output_bytes": OUTPUT_BYTES,
        "seeds": list(SEEDS), "input_hashes": probe["input_hashes"],
        "motion_files": probe["motion_files"],
        "commands": {str(seed): {arm: command_for(probe, seed, arm)
                                 for arm in ("fixed", "obs")} for seed in SEEDS},
    }
    write(PARENT / "preflight.json", plan)
    return plan


def run():
    plan_path = PARENT / "preflight.json"
    plan = json.loads(plan_path.read_text())
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != plan["execution_commit"]:
        raise RuntimeError("execution commit drift")
    if (PARENT / "run_manifest.json").exists():
        raise RuntimeError("Validation parent manifest already exists")
    probe = json.loads(PROBE.read_text())
    if plan["input_hashes"] != probe["input_hashes"]:
        raise RuntimeError("preflight input contract drift")
    check_inputs(probe)
    check_gpu()
    manifest = {"schema": "ref2dex.validation_run.v1", "validation_id": plan["validation_id"],
                "run_status": "STARTED", "started_at": now(), "execution_commit": head,
                "physical_gpu": GPU, "completed": [], "current": None,
                "preflight_sha256": sha(plan_path)}
    manifest_path = PARENT / "run_manifest.json"
    write(manifest_path, manifest)
    started = datetime.now(timezone.utc)
    try:
        for seed in SEEDS:
            check_inputs(probe)
            for arm in ("fixed", "obs"):
                elapsed = (datetime.now(timezone.utc) - started).total_seconds()
                if elapsed >= WALL_SECONDS:
                    raise TimeoutError("Validation wall budget exceeded")
                if output_size() > OUTPUT_BYTES:
                    raise RuntimeError("Validation output budget exceeded")
                check_gpu()
                command = plan["commands"][str(seed)][arm]
                target = Path(option(command, "--output")).parent
                if target.exists():
                    raise RuntimeError("child output already exists: " + str(target))
                manifest["current"] = {"seed": seed, "arm": arm, "started_at": now(),
                                        "command": command}
                write(manifest_path, manifest)
                log = PARENT / "s{}_{}.log".format(seed, arm)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(GPU)
                env["PYTHONUNBUFFERED"] = "1"
                with log.open("w") as handle:
                    child = subprocess.Popen(command, cwd=DEXPLORE, env=env,
                                             stdout=handle, stderr=subprocess.STDOUT,
                                             start_new_session=True)
                    print("START", seed, arm, child.pid, flush=True)
                    try:
                        code = child.wait(timeout=max(1, WALL_SECONDS - elapsed))
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                        raise TimeoutError("child exceeded Validation wall budget")
                if code:
                    raise RuntimeError("child failed: seed {} {} exit {}".format(seed, arm, code))
                native = json.loads((target / "run_manifest.json").read_text())
                if native.get("run_status") != "COMPLETED" or native.get("git_commit") != head:
                    raise RuntimeError("child native manifest invalid: seed {} {}".format(seed, arm))
                manifest["completed"].append({"seed": seed, "arm": arm,
                                              "native_manifest": str(target / "run_manifest.json"),
                                              "native_manifest_sha256": sha(target / "run_manifest.json"),
                                              "finished_at": now()})
                manifest["current"] = None
                write(manifest_path, manifest)
                print("DONE", seed, arm, native["summary"]["lift_success_rate"], flush=True)
        if output_size() > OUTPUT_BYTES:
            raise RuntimeError("Validation output budget exceeded")
        manifest["run_status"] = "COMPLETED"
    except BaseException as error:
        manifest["run_status"] = "FAILED"
        manifest["failure"] = "{}: {}".format(type(error).__name__, error)
        raise
    finally:
        manifest["finished_at"] = now()
        manifest["output_bytes"] = output_size()
        write(manifest_path, manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "run"))
    args = parser.parse_args()
    try:
        preflight() if args.mode == "preflight" else run()
    except BaseException as error:
        print("VALIDATION ERROR: {}: {}".format(type(error).__name__, error), file=sys.stderr)
        raise

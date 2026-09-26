"""Run two already-fixed Probes once an idle GPU is available."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[4]
DEXPLORE = ROOT / "third_party/DExplore"
PYTHON = Path("/home2/wyy/miniconda3/envs/graspenv/bin/python")
TRAINER = ROOT / "src/task/CmResidual/tools/run_multitrajectory_baseline_probe.py"
EVALUATOR = ROOT / "src/task/CmResidual/tools/eval_dexplore_full_episode_grid.py"
ROUTER = DEXPLORE / "dexplore/evaluate_object_router.py"
SPEC = ROOT / "src/task/CmResidual/configs/single_waterbottle_motion_probe.json"
ROUTE_CONFIG = ROOT / "src/task/CmResidual/configs/multitrajectory_object_router_probe.json"
SOURCE = ROOT / ("outputs/Dexplore/agent_waterbottle_specialist_s70_e340/train/"
                 "inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/"
                 "GRAB_00000340.pth")
SOURCE_SHA = "e2a1611bb07e721c84029edea1dcb893c5bf5953317595f0693162e9d06d62e0"
BASE = ROOT / "outputs/Dexplore/agent_multitrajectory12_s70_e300"
BASE_CHECKPOINT = ROOT / ("outputs/Dexplore/agent_v139_s3_backtrack_s70_e260/train/"
                          "inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/"
                          "GRAB_00000260.pth")
TRAIN_RUN = ROOT / "outputs/Dexplore/agent_waterbottle_anneal_s70_e400"
QUEUE = ROOT / "outputs/CmResidual/agent_deferred_multitrajectory_probes_20260924"
WAIT_UNTIL = datetime(2026, 9, 25, 1, 30, tzinfo=timezone.utc)  # 09:30 CST


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                   text=True).strip()


def idle_gpu() -> int | None:
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True)
    values = [tuple(int(field.strip()) for field in row.split(","))
              for row in rows.splitlines()]
    free = [index for index, memory, utilization in values
            if memory <= 512 and utilization <= 5]
    return free[-1] if free else None


def wait_gpu(expected_commit: str, manifest: dict, path: Path) -> int:
    previous = None
    while datetime.now(timezone.utc) < WAIT_UNTIL:
        if revision() != expected_commit:
            raise RuntimeError("repository commit changed while waiting")
        current = idle_gpu()
        if current is not None and current == previous:
            return current
        previous = current
        manifest.update(run_status="RUNNING", last_wait_check=now(),
                        wait_reason="all candidate GPUs occupied")
        write(path, manifest)
        time.sleep(60)
    raise TimeoutError("no idle GPU before 2026-09-25 09:30 CST")


def run_step(name: str, command: list[str], cwd: Path, gpu: int,
             expected_commit: str, manifest: dict, path: Path) -> None:
    if revision() != expected_commit or idle_gpu() != gpu:
        raise RuntimeError(f"code or GPU drift before {name}")
    manifest["current_step"] = name
    manifest["current_gpu"] = gpu
    manifest["step_started_at"] = now()
    manifest.setdefault("steps", []).append({"name": name, "gpu": gpu,
                                               "command": command,
                                               "started_at": now()})
    write(path, manifest)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "src/task/CmResidual/tools"),
         environment.get("PYTHONPATH", "")])
    log = QUEUE / f"{name}.log"
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=cwd, env=environment,
                                stdout=stream, stderr=subprocess.STDOUT,
                                timeout=3900 if name == "waterbottle_train" else 1200)
    if result.returncode:
        raise RuntimeError(f"{name} exit code {result.returncode}; see {log}")
    manifest["steps"][-1].update(status="COMPLETED", completed_at=now(),
                                 log=str(log))
    manifest.pop("current_step", None)
    manifest.pop("current_gpu", None)
    write(path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    if QUEUE.exists() or TRAIN_RUN.exists():
        raise FileExistsError("deferred queue or waterbottle run already exists")
    if revision() != args.expected_commit or sha256(SOURCE) != SOURCE_SHA:
        raise ValueError("code or source checkpoint drift")
    if not BASE.is_dir() or not SPEC.is_file() or not ROUTE_CONFIG.is_file():
        raise FileNotFoundError("frozen input or route config missing")
    QUEUE.mkdir(parents=True)
    path = QUEUE / "run_manifest.json"
    manifest = {"run_status": "STARTED", "created_at": now(),
                "run_id": QUEUE.name, "work_version": "deferred-multitrajectory-probes",
                "git_commit": args.expected_commit, "max_concurrent_gpus": 1,
                "source_checkpoint_sha256": SOURCE_SHA,
                "spec_sha256": sha256(SPEC), "route_config_sha256": sha256(ROUTE_CONFIG),
                "wait_deadline": WAIT_UNTIL.isoformat(),
                "stop_rule": "no idle GPU by deadline, code/input drift, child failure or incomplete evaluation",
                "steps": []}
    write(path, manifest)
    try:
        gpu = wait_gpu(args.expected_commit, manifest, path)
        train = [str(PYTHON), str(TRAINER), "--output", str(TRAIN_RUN),
                 "--gpu", str(gpu), "--target-epoch", "400", "--spec", str(SPEC),
                 "--work-version", "waterbottle-start-anneal-probe",
                 "--source-checkpoint", str(SOURCE), "--source-sha256", SOURCE_SHA,
                 "--source-epoch", "340", "--anneal-start", "340",
                 "--anneal-end", "380"]
        run_step("waterbottle_train", train, ROOT, gpu, args.expected_commit,
                 manifest, path)
        gpu = wait_gpu(args.expected_commit, manifest, path)
        evaluation = [str(PYTHON), str(EVALUATOR), "--run-dir", str(TRAIN_RUN),
                      "--gpu", str(gpu), "--seed", "209", "--epochs", "400",
                      "--cfg-env", "dexplore/data/cfg/inspire_object_balanced.yaml",
                      "--work-version", "waterbottle-start-anneal-probe"]
        run_step("waterbottle_eval_s209", evaluation, ROOT, gpu,
                 args.expected_commit, manifest, path)
        for seed in (211, 212, 213):
            gpu = wait_gpu(args.expected_commit, manifest, path)
            output = ROOT / f"outputs/CmResidual/agent_multitrajectory_object_router_s{seed}/results.json"
            command = [str(PYTHON), str(ROUTER), "--route-config", str(ROUTE_CONFIG),
                       "--task", "Dexplore_Inspire", "--cfg_env",
                       "dexplore/data/cfg/inspire_object_balanced.yaml", "--cfg_train",
                       "dexplore/data/cfg/train/rlg/inspire.yaml", "--motion_file",
                       str(BASE / "motions"), "--checkpoint", str(BASE_CHECKPOINT),
                       "--disable-early-termination", "--headless", "--sim_device",
                       "cuda:0", "--rl_device", "cuda:0", "--graphics_device_id", "0",
                       "--num_envs", "64", "--seed", str(seed), "--output", str(output)]
            run_step(f"router_eval_s{seed}", command, DEXPLORE, gpu,
                     args.expected_commit, manifest, path)
        manifest.update(run_status="COMPLETED", completed_at=now())
    except BaseException as error:
        manifest.update(run_status="STOPPED" if isinstance(error, TimeoutError) else "FAILED",
                        completed_at=now(), failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(path, manifest)


if __name__ == "__main__":
    main()

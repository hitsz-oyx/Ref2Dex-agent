"""Run the frozen V1.45 route/baseline replication matrix on GPUs 5 and 6."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
DEXPLORE = ROOT / "third_party/DExplore"
PYTHON = Path("/home2/wyy/miniconda3/envs/graspenv/bin/python")
EVALUATOR = ROOT / "src/task/CmResidual/tools/eval_cmlite_bc_policy.py"
OUTPUT = ROOT / "outputs/CmResidual/agent_v145_replication"
BC = Path("/home2/wyy/oyx_ws/ai_ws/Ref2Dex-review-cm-world/outputs/CmResidual/agent_reference_dagger2_bc_seed42_20260922/policy.pt")
CM = ROOT / "outputs/CmLite/V1.37/balanced_s1x24_s3_e160_e180/best.pt"
MAPS = {
    "route": ROOT / "outputs/CmResidual/agent_v140_s3_router_fit/route.json",
    "baseline": ROOT / "src/task/CmResidual/config/v145_back260_only_route.json",
}
NETWORK = "train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn"
EXPERTS = {
    "source": ROOT / f"outputs/Dexplore/agent_v135_s3_cmoff_s70_e180/{NETWORK}/GRAB_00000180.pth",
    "back240": ROOT / f"outputs/Dexplore/agent_v139_s3_backtrack_s70_e260/{NETWORK}/GRAB_00000240.pth",
    "back260": ROOT / f"outputs/Dexplore/agent_v139_s3_backtrack_s70_e260/{NETWORK}/GRAB_00000260.pth",
}
MOTION = ROOT / "outputs/CmResidual/agent_v129_s3_coordfix/corrected_converted_r2"
EXPECTED = {
    "bc": "a0b0c1da197abb169b501fc252345cc0f29f8a46b60f57d5b304c058da2a944c",
    "cm": "396f5e0c92ecf67b2a096568c1a538b7c86d40779a594729f786df6c64ed204a",
    "route_map": "7db4686f8eb273e0ac94bfbbd3625ddd35209a1f4922e8451bcbf519f6395a79",
    "baseline_map": "1c396812c46d2e9d433e168925e73c2d3c55a5ac9bce6ff932a7c1845716e0bd",
    "source": "863443513a746155f2b04662fe4dddd8c7b5a672d3e73533c1ff555d90c0237c",
    "back240": "a88924a590966c964b029e067985fea41465f230d459454b8048899cbe8670c2",
    "back260": "16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def gpu_free(index: int) -> bool:
    response = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True)
    used = {int(row.split(",")[0]): int(row.split(",")[1]) for row in response.splitlines()}
    return used[index] <= 1024


def command(policy: str, seed: int, repeat: int, gpu: int) -> tuple[list[str], Path]:
    if policy not in MAPS or seed not in range(99, 104) or repeat not in (0, 1):
        raise ValueError("V1.45 command outside frozen matrix")
    output = ROOT / f"outputs/CmResidual/agent_v145_{policy}_s{seed}_r{repeat}_n64"
    expert_names = ("source", "back240", "back260") if policy == "route" else ("back260",)
    args = [str(PYTHON), str(EVALUATOR),
            "--bc-checkpoint", str(BC), "--cmlite-checkpoint", str(CM),
            "--cmlite-sha256", EXPECTED["cm"], "--mode", "router",
            "--router-map-file", str(MAPS[policy]),
            "--grip-reflex-mode", "off", "--physical-gpu", str(gpu),
            "--output", str(output)]
    for name in expert_names:
        args += ["--router-checkpoint", f"{name}={EXPERTS[name]}"]
    args += ["--task", "Dexplore_Inspire", "--cfg_env", "dexplore/data/cfg/inspire.yaml",
             "--cfg_train", "dexplore/data/cfg/train/rlg/inspire.yaml",
             "--motion_file", str(MOTION), "--checkpoint", str(EXPERTS["source"]),
             "--test", "--headless", "--sim_device", "cuda:0", "--rl_device", "cuda:0",
             "--graphics_device_id", "0", "--num_envs", "64", "--seed", str(seed)]
    return args, output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    schedule = []
    for seed in range(99, 104):
        for repeat in (0, 1):
            route_gpu = 5 if repeat == 0 else 6
            baseline_gpu = 6 if repeat == 0 else 5
            schedule.append([
                ("route", seed, repeat, route_gpu),
                ("baseline", seed, repeat, baseline_gpu),
            ])
    if args.dry_run:
        print(json.dumps([[{"policy": p, "seed": s, "repeat": r, "gpu": g,
                           "command": command(p, s, r, g)[0]}
                          for p, s, r, g in round_jobs] for round_jobs in schedule], indent=2))
        return
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    existing = [str(command(p, s, r, g)[1]) for round_jobs in schedule
                for p, s, r, g in round_jobs if command(p, s, r, g)[1].exists()]
    if existing:
        raise FileExistsError(f"V1.45 output already exists: {existing}")
    verified = {"bc": sha256(BC), "cm": sha256(CM),
                "route_map": sha256(MAPS["route"]),
                "baseline_map": sha256(MAPS["baseline"]),
                **{name: sha256(path) for name, path in EXPERTS.items()}}
    if verified != EXPECTED or not MOTION.is_dir() or not PYTHON.is_file():
        raise ValueError("frozen V1.45 input or runtime is missing/mismatched")
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         text=True).strip()
    manifest = {
        "run_status": "STARTED", "created_at": now(), "work_version": "V1.45",
        "git_commit": git_commit, "run_id": OUTPUT.name,
        "seeds": list(range(99, 104)), "repeats_per_seed": 2,
        "max_concurrent_gpus": 2, "gpus": [5, 6],
        "checkpoint_and_map_sha256": verified,
        "motion_root": str(MOTION), "official_policy_checkpoint": None,
        "stop_rule": "complete fixed 5x2x2 matrix unless technical failure or external GPU occupation",
        "output_budget_gb": 1, "completed_runs": [],
    }
    write(OUTPUT / "run_manifest.json", manifest)
    for round_jobs in schedule:
        if not gpu_free(5) or not gpu_free(6):
            manifest.update(run_status="FAILED", completed_at=now(),
                            failure="GPU 5 or 6 occupied before next round")
            write(OUTPUT / "run_manifest.json", manifest)
            raise RuntimeError(manifest["failure"])
        processes = []
        for policy, seed, repeat, gpu in round_jobs:
            cmd, output = command(policy, seed, repeat, gpu)
            log_path = OUTPUT / "logs" / f"{output.name}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("w", encoding="utf-8")
            env = os.environ.copy()
            env.pop("CUDA_VISIBLE_DEVICES", None)
            process = subprocess.Popen(cmd, cwd=DEXPLORE, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            processes.append((policy, seed, repeat, gpu, process, log, output, log_path))
        failures = []
        for policy, seed, repeat, gpu, process, log, output, log_path in processes:
            code = process.wait()
            log.close()
            run_manifest = output / "run_manifest.json"
            if code or not run_manifest.is_file():
                failures.append(f"{output.name}: exit={code}; log={log_path}")
                continue
            result = json.loads(run_manifest.read_text())
            if result.get("run_status") != "COMPLETED" or result.get("cm_used_by_policy"):
                failures.append(f"{output.name}: wrong status or Cm used")
                continue
            rate = result["summary"]["lift_success_rate"]
            manifest["completed_runs"].append({
                "policy": policy, "seed": seed, "repeat": repeat, "gpu": gpu,
                "run_manifest": str(run_manifest.resolve()),
                "successes": round(float(rate) * 64),
                "created_at": result["created_at"],
                "completed_at": result["completed_at"],
            })
            print(json.dumps(manifest["completed_runs"][-1], sort_keys=True), flush=True)
        if failures:
            manifest.update(run_status="FAILED", completed_at=now(), failure=failures)
            write(OUTPUT / "run_manifest.json", manifest)
            raise RuntimeError(failures)
        write(OUTPUT / "run_manifest.json", manifest)
    manifest.update(run_status="COMPLETED", completed_at=now())
    write(OUTPUT / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()

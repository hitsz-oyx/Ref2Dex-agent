"""Run the fixed V1.46 Cm-on/off e300 strict full-episode evaluation matrix."""
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
PYTHON = Path("/home2/wyy/miniconda3/envs/graspenv/bin/python")
EVAL = ROOT / "src/task/CmResidual/tools/eval_dexplore_full_episode_grid.py"
OUTPUT = ROOT / "outputs/CmResidual/agent_v146_eval_matrix"
RUNS = {
    "off": ROOT / "outputs/Dexplore/agent_v146_cmoff_s70_e300",
    "on": ROOT / "outputs/Dexplore/agent_v146_cmon_s70_e300",
}
EXPECTED = {
    "off": "36ff2ac7ffd4433b5f60b32dce7603cff5db24a0bab9866b9ab469d1ce645929",
    "on": "278fc7a1b65837de7495d0e5da12a78ab177937a14210ae733134940deac15b2",
}
INPUT_SHA = "2878bd20d1dd849f6844883c832a3777602d2d2bc73f7581b3bc23d30725f038"
SOURCE_SHA = "16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f"
SEEDS = tuple(range(104, 109))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def gpu_free(index: int) -> bool:
    response = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True)
    used = {int(row.split(",")[0]): int(row.split(",")[1]) for row in response.splitlines()}
    return used[index] <= 1024


def command(arm: str, seed: int, repeat: int, gpu: int) -> tuple[list[str], Path]:
    if arm not in RUNS or seed not in SEEDS or repeat not in (0, 1) or gpu not in (5, 6):
        raise ValueError("V1.46 evaluation outside fixed matrix")
    run = RUNS[arm]
    output = run / f"eval_s{seed}_e300_full_v146_r{repeat}"
    cmd = [str(PYTHON), str(EVAL), "--run-dir", str(run), "--gpu", str(gpu),
           "--seed", str(seed), "--epochs", "300", "--tag", f"v146_r{repeat}",
           "--work-version", "V1.46"]
    return cmd, output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    schedule = []
    for seed in SEEDS:
        for repeat in (0, 1):
            schedule.append([
                ("on", seed, repeat, 5 if repeat == 0 else 6),
                ("off", seed, repeat, 6 if repeat == 0 else 5),
            ])
    if args.dry_run:
        print(json.dumps([[{"arm": arm, "seed": seed, "repeat": repeat,
                           "gpu": gpu, "command": command(arm, seed, repeat, gpu)[0]}
                          for arm, seed, repeat, gpu in round_jobs]
                          for round_jobs in schedule], indent=2))
        return
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    existing = [str(command(*job)[1]) for round_jobs in schedule for job in round_jobs
                if command(*job)[1].exists()]
    if existing:
        raise FileExistsError(f"V1.46 evaluation output already exists: {existing}")
    for arm, run in RUNS.items():
        training = json.loads((run / "run_manifest.json").read_text())
        config = json.loads((run / "config.json").read_text())
        checkpoint = next((run / "train").rglob("GRAB_00000300.pth"))
        if (training.get("run_status") != "COMPLETED" or
                config.get("scratch_resume_sha256") != SOURCE_SHA or
                sha256(checkpoint) != EXPECTED[arm] or
                sha256(Path(config["input_manifest"])) != INPUT_SHA or
                (arm == "on") != (config.get("cmlite_reward_coef") == 0.1)):
            raise ValueError(f"V1.46 {arm} training provenance mismatch")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    manifest = {
        "run_status": "STARTED", "created_at": now(), "run_id": OUTPUT.name,
        "work_version": "V1.46", "git_commit": revision,
        "seeds": list(SEEDS), "repeats_per_seed": 2, "gpus": [5, 6],
        "max_concurrent_gpus": 2, "checkpoint_sha256": EXPECTED,
        "input_manifest_sha256": INPUT_SHA,
        "source_checkpoint_sha256": SOURCE_SHA,
        "stop_rule": "complete fixed 5x2x2 matrix unless technical failure or external GPU occupation",
        "output_budget_gb": 1, "official_policy_checkpoint": None,
        "completed_runs": [],
    }
    write(OUTPUT / "run_manifest.json", manifest)
    for round_jobs in schedule:
        if not gpu_free(5) or not gpu_free(6):
            manifest.update(run_status="FAILED", completed_at=now(),
                            failure="GPU 5 or 6 occupied before next round")
            write(OUTPUT / "run_manifest.json", manifest)
            raise RuntimeError(manifest["failure"])
        processes = []
        for arm, seed, repeat, gpu in round_jobs:
            cmd, output = command(arm, seed, repeat, gpu)
            log_path = OUTPUT / "logs" / f"{output.name}_{arm}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("w", encoding="utf-8")
            env = os.environ.copy()
            env.pop("CUDA_VISIBLE_DEVICES", None)
            process = subprocess.Popen(cmd, cwd=ROOT, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            processes.append((arm, seed, repeat, gpu, process, log, output, log_path))
        failures = []
        for arm, seed, repeat, gpu, process, log, output, log_path in processes:
            code = process.wait()
            log.close()
            child_path = output / "run_manifest.json"
            if code or not child_path.is_file():
                failures.append(f"{output.name}: exit={code}, log={log_path}")
                continue
            child = json.loads(child_path.read_text())
            if (child.get("run_status") != "COMPLETED" or
                    child.get("checkpoint_sha256") != EXPECTED[arm] or
                    child.get("input_manifest_sha256") != INPUT_SHA or
                    child.get("early_termination_disabled") is not True or
                    child["summary"].get("selector_enabled") is not False):
                failures.append(f"{output.name}: provenance/status mismatch")
                continue
            results = json.loads((output / "results.json").read_text())
            episodes = results["per_episode"]
            if len(episodes) != 64 or {row["env_id"] for row in episodes} != set(range(64)):
                failures.append(f"{output.name}: expected 64 first episodes")
                continue
            successes = sum(bool(row["lift_success"]) for row in episodes)
            manifest["completed_runs"].append({
                "arm": arm, "seed": seed, "repeat": repeat, "gpu": gpu,
                "successes": successes, "run_manifest": str(child_path.resolve()),
                "created_at": child["created_at"], "completed_at": child["completed_at"],
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

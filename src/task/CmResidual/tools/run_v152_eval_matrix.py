"""Fixed new-seed three-arm test of Cm PPO weighting and permuted placebo."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from src.task.CmResidual.tools.run_v146_eval_matrix import (
    PYTHON, EVAL, ROOT, sha256, gpu_free, write,
)


OUTPUT = ROOT / "outputs/CmResidual/agent_v152_eval_matrix"
RUNS = {
    "on": ROOT / "outputs/Dexplore/agent_v151_cmppo_s70_e300",
    "off": ROOT / "outputs/Dexplore/agent_v151_cmoff_s70_e300",
    "placebo": ROOT / "outputs/Dexplore/agent_v152_cmplacebo_s70_e300",
}
FIXED_SHA = {
    "on": "21d4972eb1956258b6d36b12fc31bbcf2ba3dd4dbe622949ee20ffc51547aad8",
    "off": "d61a8fdec5948dd3a7df7d3021c0e8a98d0fa6dce9dd00d86d1e805af89c4521",
}
SOURCE_SHA = "16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f"
INPUT_SHA = "2878bd20d1dd849f6844883c832a3777602d2d2bc73f7581b3bc23d30725f038"
SEEDS = tuple(range(124, 129))
SOURCE_FILES = (
    "third_party/DExplore/dexplore/evaluate.py",
    "third_party/DExplore/dexplore/env/tasks/base_dexplore_task.py",
    "third_party/DExplore/dexplore/env/tasks/dexplore_inspire.py",
    "third_party/DExplore/dexplore/learning/dexplore_agent.py",
    "third_party/DExplore/dexplore/learning/dexplore_network_builder.py",
    "src/task/CmResidual/tools/eval_dexplore_full_episode_grid.py",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def source_hashes() -> dict[str, str]:
    return {name: sha256(ROOT / name) for name in SOURCE_FILES}


def command(arm: str, seed: int, repeat: int, gpu: int) -> tuple[list[str], Path]:
    if arm not in RUNS or seed not in SEEDS or repeat not in (0, 1) or gpu not in (5, 6):
        raise ValueError("V1.52 evaluation outside fixed matrix")
    run = RUNS[arm]
    output = run / f"eval_s{seed}_e300_full_v152_r{repeat}"
    return ([str(PYTHON), str(EVAL), "--run-dir", str(run), "--gpu", str(gpu),
             "--seed", str(seed), "--epochs", "300", "--tag", f"v152_r{repeat}",
             "--work-version", "V1.52"], output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placebo-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(args.placebo_sha256) != 64:
        raise ValueError("expected full placebo checkpoint SHA256")
    expected = {**FIXED_SHA, "placebo": args.placebo_sha256}
    jobs = [(arm, seed, repeat) for seed in SEEDS for repeat in (0, 1)
            for arm in ("on", "off", "placebo")]
    schedule = [[(arm, seed, repeat, 5 if pair_index % 2 == job_index else 6)
                 for job_index, (arm, seed, repeat) in enumerate(jobs[2*pair_index:2*pair_index+2])]
                for pair_index in range(len(jobs)//2)]
    if args.dry_run:
        print(json.dumps([[{"arm": arm, "seed": seed, "repeat": repeat,
                           "gpu": gpu, "command": command(arm, seed, repeat, gpu)[0]}
                          for arm, seed, repeat, gpu in pair] for pair in schedule], indent=2))
        return
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if any(command(arm, seed, repeat, gpu)[1].exists() for pair in schedule
           for arm, seed, repeat, gpu in pair):
        raise FileExistsError("V1.52 child evaluation output already exists")
    for arm, run in RUNS.items():
        training = json.loads((run / "run_manifest.json").read_text())
        config = json.loads((run / "config.json").read_text())
        checkpoint = next((run / "train").rglob("GRAB_00000300.pth"))
        if (training.get("run_status") != "COMPLETED" or
                config.get("scratch_resume_sha256") != SOURCE_SHA or
                sha256(Path(config["input_manifest"])) != INPUT_SHA or
                sha256(checkpoint) != expected[arm] or
                config.get("seed") != 70 or config.get("actual_epochs") != 300 or
                (arm != "off") != (config.get("cm_actor_weight_coef") == 1.0) or
                (arm == "placebo") != bool(config.get("permute_cm_actor_weights", False)) or
                (arm == "off" and config.get("cm_distill_coef") != 0.0)):
            raise ValueError(f"V1.52 {arm} training provenance mismatch")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    hashes = source_hashes()
    manifest = {"run_status": "STARTED", "created_at": now(), "run_id": OUTPUT.name,
                "work_version": "V1.52", "git_commit": revision, "seeds": list(SEEDS),
                "repeats_per_seed": 2, "arms": ["on", "off", "placebo"],
                "gpus": [5, 6], "max_concurrent_gpus": 2,
                "checkpoint_sha256": expected, "source_checkpoint_sha256": SOURCE_SHA,
                "input_manifest_sha256": INPUT_SHA, "evaluation_source_sha256": hashes,
                "output_budget_gb": 8, "official_policy_checkpoint": None,
                "stop_rule": "complete fixed 5x2x3 matrix unless technical failure, GPU conflict, or source drift",
                "completed_runs": []}
    write(OUTPUT / "run_manifest.json", manifest)
    for pair in schedule:
        if source_hashes() != hashes:
            manifest.update(run_status="INVALIDATED", completed_at=now(),
                            failure="evaluation source changed before pair")
            write(OUTPUT / "run_manifest.json", manifest)
            raise RuntimeError(manifest["failure"])
        if not gpu_free(5) or not gpu_free(6):
            manifest.update(run_status="FAILED", completed_at=now(),
                            failure="GPU 5 or 6 occupied before next pair")
            write(OUTPUT / "run_manifest.json", manifest)
            raise RuntimeError(manifest["failure"])
        processes = []
        for arm, seed, repeat, gpu in pair:
            cmd, output = command(arm, seed, repeat, gpu)
            log_path = OUTPUT / "logs" / f"{output.name}_{arm}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("w", encoding="utf-8")
            env = os.environ.copy()
            env.pop("CUDA_VISIBLE_DEVICES", None)
            process = subprocess.Popen(cmd, cwd=ROOT, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            processes.append((arm, seed, repeat, process, log, output, log_path))
        failures = []
        for arm, seed, repeat, process, log, output, log_path in processes:
            code = process.wait()
            log.close()
            child_path = output / "run_manifest.json"
            if code or not child_path.is_file():
                failures.append(f"{output.name}: exit={code}, log={log_path}")
                continue
            child = json.loads(child_path.read_text())
            if (child.get("run_status") != "COMPLETED" or
                    child.get("checkpoint_sha256") != expected[arm] or
                    child.get("input_manifest_sha256") != INPUT_SHA or
                    child.get("early_termination_disabled") is not True or
                    child["summary"].get("selector_enabled") is not False):
                failures.append(f"{output.name}: child provenance mismatch")
                continue
            episodes = json.loads((output / "results.json").read_text())["per_episode"]
            if len(episodes) != 64 or {row["env_id"] for row in episodes} != set(range(64)):
                failures.append(f"{output.name}: expected 64 first episodes")
                continue
            record = {"arm": arm, "seed": seed, "repeat": repeat, "gpu": gpu,
                      "successes": sum(bool(row["lift_success"]) for row in episodes),
                      "run_manifest": str(child_path.resolve()),
                      "created_at": child["created_at"],
                      "completed_at": child["completed_at"]}
            manifest["completed_runs"].append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
        if source_hashes() != hashes:
            manifest.update(run_status="INVALIDATED", completed_at=now(),
                            failure="evaluation source changed during pair")
            write(OUTPUT / "run_manifest.json", manifest)
            raise RuntimeError(manifest["failure"])
        if failures:
            manifest.update(run_status="FAILED", completed_at=now(), failure=failures)
            write(OUTPUT / "run_manifest.json", manifest)
            raise RuntimeError(failures)
        write(OUTPUT / "run_manifest.json", manifest)
    manifest.update(run_status="COMPLETED", completed_at=now())
    write(OUTPUT / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()

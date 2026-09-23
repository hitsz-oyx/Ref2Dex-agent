"""Fixed heldout full-episode matrix: Cm PPO weighting vs matched Cm-off."""
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


OUTPUT = ROOT / "outputs/CmResidual/agent_v151_eval_matrix"
RUNS = {
    "on": ROOT / "outputs/Dexplore/agent_v151_cmppo_s70_e300",
    "off": ROOT / "outputs/Dexplore/agent_v151_cmoff_s70_e300",
}
SOURCE_SHA = "16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f"
INPUT_SHA = "2878bd20d1dd849f6844883c832a3777602d2d2bc73f7581b3bc23d30725f038"
SEEDS = tuple(range(119, 124))


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def command(arm: str, seed: int, repeat: int, gpu: int) -> tuple[list[str], Path]:
    if arm not in RUNS or seed not in SEEDS or repeat not in (0, 1) or gpu not in (5, 6):
        raise ValueError("V1.51 evaluation outside fixed matrix")
    run = RUNS[arm]
    output = run / f"eval_s{seed}_e300_full_v151_r{repeat}"
    return ([str(PYTHON), str(EVAL), "--run-dir", str(run), "--gpu", str(gpu),
             "--seed", str(seed), "--epochs", "300", "--tag", f"v151_r{repeat}",
             "--work-version", "V1.51"], output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--on-sha256", required=True)
    parser.add_argument("--off-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(args.on_sha256) != 64 or len(args.off_sha256) != 64:
        raise ValueError("expected full checkpoint SHA256 for both arms")
    expected = {"on": args.on_sha256, "off": args.off_sha256}
    schedule = [[("on", seed, repeat, 5 if repeat == 0 else 6),
                 ("off", seed, repeat, 6 if repeat == 0 else 5)]
                for seed in SEEDS for repeat in (0, 1)]
    if args.dry_run:
        print(json.dumps([[{"arm": arm, "seed": seed, "repeat": repeat,
                           "gpu": gpu, "command": command(arm, seed, repeat, gpu)[0]}
                          for arm, seed, repeat, gpu in pair] for pair in schedule], indent=2))
        return
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if any(command(*job)[1].exists() for pair in schedule for job in pair):
        raise FileExistsError("V1.51 child evaluation output already exists")
    for arm, run in RUNS.items():
        training = json.loads((run / "run_manifest.json").read_text())
        config = json.loads((run / "config.json").read_text())
        checkpoint = next((run / "train").rglob("GRAB_00000300.pth"))
        if (training.get("run_status") != "COMPLETED" or
                config.get("scratch_resume_sha256") != SOURCE_SHA or
                sha256(Path(config["input_manifest"])) != INPUT_SHA or
                sha256(checkpoint) != expected[arm] or
                (arm == "on") != (config.get("cm_actor_weight_coef") == 1.0) or
                (arm == "off" and config.get("cm_distill_coef") != 0.0) or
                config.get("seed") != 70 or config.get("actual_epochs") != 300):
            raise ValueError(f"V1.51 {arm} training provenance mismatch")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    manifest = {"run_status": "STARTED", "created_at": now(), "run_id": OUTPUT.name,
                "work_version": "V1.51", "git_commit": revision, "seeds": list(SEEDS),
                "repeats_per_seed": 2, "gpus": [5, 6], "max_concurrent_gpus": 2,
                "checkpoint_sha256": expected, "source_checkpoint_sha256": SOURCE_SHA,
                "input_manifest_sha256": INPUT_SHA, "output_budget_gb": 5,
                "official_policy_checkpoint": None,
                "stop_rule": "complete fixed 5x2x2 matrix unless technical failure or GPU conflict",
                "completed_runs": []}
    write(OUTPUT / "run_manifest.json", manifest)
    for pair in schedule:
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
            record = {"arm": arm, "seed": seed, "repeat": repeat,
                      "successes": sum(bool(row["lift_success"]) for row in episodes),
                      "run_manifest": str(child_path.resolve()),
                      "created_at": child["created_at"],
                      "completed_at": child["completed_at"]}
            manifest["completed_runs"].append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
        if failures:
            manifest.update(run_status="FAILED", completed_at=now(), failure=failures)
            write(OUTPUT / "run_manifest.json", manifest)
            raise RuntimeError(failures)
        write(OUTPUT / "run_manifest.json", manifest)
    manifest.update(run_status="COMPLETED", completed_at=now())
    write(OUTPUT / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()

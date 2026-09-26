#!/usr/bin/env python3
"""Fixed two-seed, four-arm strict first-episode Cm weight-head probe."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from src.task.CmResidual.tools.run_v146_eval_matrix import (
    EVAL, PYTHON, ROOT, gpu_free, now, sha256, write,
)
from src.task.CmResidual.tools.run_v152_eval_matrix import SOURCE_FILES


OUTPUT = ROOT / "outputs/CmResidual/agent_cm_weight_heads_probe"
RUNS = {
    "joint": ROOT / "outputs/Dexplore/agent_v151_cmppo_s70_e300",
    "off": ROOT / "outputs/Dexplore/agent_v151_cmoff_s70_e300",
    "contact_rank": ROOT / "outputs/Dexplore/agent_cm_weight_contact_s70_e300",
    "effect_rank": ROOT / "outputs/Dexplore/agent_cm_weight_effect_s70_e300",
}
CHECKPOINT_SHA = {
    "joint": "21d4972eb1956258b6d36b12fc31bbcf2ba3dd4dbe622949ee20ffc51547aad8",
    "off": "d61a8fdec5948dd3a7df7d3021c0e8a98d0fa6dce9dd00d86d1e805af89c4521",
    "contact_rank": "1128223798bb5905ae53d5c7eae7ddb0bc78d83242cfffcc972b03581959a42c",
    "effect_rank": "12bb9f921cc95fe240f69185118b34c9028b944b90ec9727e88eb3af261f1607",
}
SOURCE_SHA = "16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f"
INPUT_SHA = "2878bd20d1dd849f6844883c832a3777602d2d2bc73f7581b3bc23d30725f038"
SEEDS = (129, 130)


def source_hashes() -> dict[str, str]:
    return {name: sha256(ROOT / name) for name in SOURCE_FILES}


def command(arm: str, seed: int, gpu: int) -> tuple[list[str], Path]:
    if arm not in RUNS or seed not in SEEDS or gpu not in (5, 6):
        raise ValueError("evaluation outside fixed probe")
    output = RUNS[arm] / f"eval_s{seed}_e300_full_cmheads"
    return ([str(PYTHON), str(EVAL), "--run-dir", str(RUNS[arm]),
             "--gpu", str(gpu), "--seed", str(seed), "--epochs", "300",
             "--tag", "cmheads", "--work-version", "cm_weight_heads_probe"], output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    pairs = [[("joint", seed, 5), ("contact_rank", seed, 6)] for seed in SEEDS]
    pairs += [[("effect_rank", seed, 5), ("off", seed, 6)] for seed in SEEDS]
    if args.dry_run:
        print(json.dumps([[{"arm": arm, "seed": seed, "gpu": gpu,
                           "command": command(arm, seed, gpu)[0]}
                          for arm, seed, gpu in pair] for pair in pairs], indent=2))
        return
    if OUTPUT.exists() or any(command(*job)[1].exists() for pair in pairs for job in pair):
        raise FileExistsError("probe or child output already exists")
    for arm, run in RUNS.items():
        training = json.loads((run / "run_manifest.json").read_text())
        config = json.loads((run / "config.json").read_text())
        checkpoint = next((run / "train").rglob("GRAB_00000300.pth"))
        expected_mode = arm if arm in ("contact_rank", "effect_rank") else "joint"
        actual_mode = config.get("cm_actor_weight_component", "joint")
        if (training.get("run_status") != "COMPLETED" or config.get("seed") != 70 or
                config.get("actual_epochs") != 300 or
                config.get("scratch_resume_sha256") != SOURCE_SHA or
                sha256(Path(config["input_manifest"])) != INPUT_SHA or
                sha256(checkpoint) != CHECKPOINT_SHA[arm] or
                actual_mode != expected_mode or
                (arm != "off") != (config.get("cm_actor_weight_coef") == 1.0)):
            raise ValueError(f"{arm} training provenance mismatch")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    hashes = source_hashes()
    manifest = {"run_status": "STARTED", "created_at": now(),
                "run_id": OUTPUT.name, "git_commit": revision,
                "seeds": list(SEEDS), "arms": list(RUNS), "gpus": [5, 6],
                "max_concurrent_gpus": 2, "checkpoint_sha256": CHECKPOINT_SHA,
                "input_manifest_sha256": INPUT_SHA,
                "source_checkpoint_sha256": SOURCE_SHA,
                "evaluation_source_sha256": hashes, "output_budget_gb": 2,
                "official_policy_checkpoint": None,
                "stop_rule": "fixed eight runs unless technical failure, GPU conflict, or source drift",
                "completed_runs": []}
    write(OUTPUT / "run_manifest.json", manifest)
    for pair in pairs:
        if source_hashes() != hashes or not gpu_free(5) or not gpu_free(6):
            manifest.update(run_status="FAILED", completed_at=now(),
                            failure="source drift or GPU conflict before pair")
            write(OUTPUT / "run_manifest.json", manifest)
            raise RuntimeError(manifest["failure"])
        processes = []
        for arm, seed, gpu in pair:
            cmd, output = command(arm, seed, gpu)
            log_path = OUTPUT / "logs" / f"{output.name}_{arm}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("w", encoding="utf-8")
            env = os.environ.copy()
            env.pop("CUDA_VISIBLE_DEVICES", None)
            process = subprocess.Popen(cmd, cwd=ROOT, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            processes.append((arm, seed, gpu, process, log, output, log_path))
        failures = []
        for arm, seed, gpu, process, log, output, log_path in processes:
            code = process.wait()
            log.close()
            child_path = output / "run_manifest.json"
            if code or not child_path.is_file():
                failures.append(f"{arm} seed{seed}: exit={code}, log={log_path}")
                continue
            child = json.loads(child_path.read_text())
            if (child.get("run_status") != "COMPLETED" or
                    child.get("checkpoint_sha256") != CHECKPOINT_SHA[arm] or
                    child.get("input_manifest_sha256") != INPUT_SHA or
                    child.get("early_termination_disabled") is not True or
                    child["summary"].get("selector_enabled") is not False):
                failures.append(f"{arm} seed{seed}: child provenance mismatch")
                continue
            episodes = json.loads((output / "results.json").read_text())["per_episode"]
            if len(episodes) != 64 or {row["env_id"] for row in episodes} != set(range(64)):
                failures.append(f"{arm} seed{seed}: expected 64 first episodes")
                continue
            record = {"arm": arm, "seed": seed, "gpu": gpu,
                      "successes": sum(bool(row["lift_success"]) for row in episodes),
                      "run_manifest": str(child_path.resolve())}
            manifest["completed_runs"].append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
        if source_hashes() != hashes:
            failures.append("evaluation source changed during pair")
        if failures:
            manifest.update(run_status="FAILED", completed_at=now(), failure=failures)
            write(OUTPUT / "run_manifest.json", manifest)
            raise RuntimeError(failures)
        write(OUTPUT / "run_manifest.json", manifest)
    manifest.update(run_status="COMPLETED", completed_at=now())
    write(OUTPUT / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()

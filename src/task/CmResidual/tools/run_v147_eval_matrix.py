"""Run the preregistered V1.47 s3 heldout matrix with GPU-crossed repeats."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[4]
PYTHON = Path("/home2/wyy/miniconda3/envs/graspenv/bin/python")
EVAL = ROOT / "src/task/CmResidual/tools/eval_dexplore_full_episode_grid.py"
OUTPUT = ROOT / "outputs/CmResidual/agent_v147_eval_matrix"
RUNS = {
    "mixed": ROOT / "outputs/Dexplore/agent_v147_s1s3_s70_e320",
    "control": ROOT / "outputs/Dexplore/agent_v147_s3only_s70_e320",
}
S3_ROOT = ROOT / "outputs/CmResidual/agent_v129_s3_coordfix/corrected_converted_r2"
S3_MANIFEST = ROOT / "outputs/CmResidual/agent_v129_s3_coordfix/corrected_manifest_r2.json"
MIXED_MANIFEST = ROOT / "src/task/CmResidual/configs/v147_s1_s3_motion_manifest.json"
SOURCE_SHA = "16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f"
TRAIN_COMMIT = "70b082fad3976f363470b01cb48f3840c2f9f8a0"
SEEDS = tuple(range(109, 114))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def gpu_free(index: int) -> bool:
    response = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True)
    memory = {int(row.split(",")[0]): int(row.split(",")[1])
              for row in response.splitlines()}
    return memory[index] <= 1024


def command(arm: str, seed: int, repeat: int, gpu: int) -> tuple[list[str], Path]:
    run = RUNS[arm]
    tag = f"v147_s3_r{repeat}"
    output = run / f"eval_s{seed}_e320_full_{tag}"
    cmd = [str(PYTHON), str(EVAL), "--run-dir", str(run), "--gpu", str(gpu),
           "--seed", str(seed), "--epochs", "320", "--tag", tag,
           "--motion-root-override", str(S3_ROOT),
           "--input-manifest-override", str(S3_MANIFEST),
           "--work-version", "V1.47"]
    return cmd, output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    schedule = [[("mixed", seed, repeat, 5 if repeat == 0 else 6),
                 ("control", seed, repeat, 6 if repeat == 0 else 5)]
                for seed in SEEDS for repeat in (0, 1)]
    if args.dry_run:
        print(json.dumps([[command(*job)[0] for job in round_jobs]
                          for round_jobs in schedule], indent=2))
        return
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    existing = [str(command(*job)[1]) for pair in schedule for job in pair
                if command(*job)[1].exists()]
    if existing:
        raise FileExistsError(existing)
    checkpoints = {}
    for arm, run in RUNS.items():
        training = json.loads((run / "run_manifest.json").read_text())
        config = json.loads((run / "config.json").read_text())
        matches = list((run / "train").rglob("GRAB_00000320.pth"))
        expected_input = MIXED_MANIFEST if arm == "mixed" else S3_MANIFEST
        if (training.get("run_status") != "COMPLETED" or
                training.get("git_commit") != TRAIN_COMMIT or
                config.get("scratch_resume_sha256") != SOURCE_SHA or
                config.get("cm_distill_coef") != 0 or
                config.get("actual_epochs") != 320 or
                config.get("seed") != 70 or
                config.get("num_envs_per_rank") != 64 or
                Path(config.get("input_manifest", "")).resolve() != expected_input.resolve() or
                len(matches) != 1):
            raise ValueError(f"V1.47 {arm} training provenance mismatch")
        checkpoints[arm] = sha256(matches[0])
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    evaluation_input_sha = sha256(S3_MANIFEST)
    parent = {
        "run_status": "STARTED", "created_at": now(), "run_id": OUTPUT.name,
        "work_version": "V1.47", "git_commit": revision,
        "training_commit": TRAIN_COMMIT, "seeds": list(SEEDS),
        "repeats_per_seed": 2, "gpus": [5, 6], "max_concurrent_gpus": 2,
        "checkpoint_sha256": checkpoints,
        "training_input_manifest_sha256": {
            "mixed": sha256(MIXED_MANIFEST), "control": evaluation_input_sha},
        "evaluation_input_manifest_sha256": evaluation_input_sha,
        "source_checkpoint_sha256": SOURCE_SHA,
        "stop_rule": "complete fixed 5x2x2 matrix unless technical failure or GPU occupation",
        "output_budget_gb": 1, "official_policy_checkpoint": None,
        "completed_runs": [],
    }
    write(OUTPUT / "run_manifest.json", parent)
    for pair in schedule:
        if not gpu_free(5) or not gpu_free(6):
            parent.update(run_status="FAILED", completed_at=now(),
                          failure="GPU 5 or 6 occupied before next round")
            write(OUTPUT / "run_manifest.json", parent)
            raise RuntimeError(parent["failure"])
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
                    child.get("checkpoint_sha256") != checkpoints[arm] or
                    child.get("input_manifest_sha256") != evaluation_input_sha or
                    child.get("early_termination_disabled") is not True or
                    child["summary"].get("selector_enabled") is not False):
                failures.append(f"{output.name}: provenance/status mismatch")
                continue
            episodes = json.loads((output / "results.json").read_text())["per_episode"]
            if len(episodes) != 64 or {row["env_id"] for row in episodes} != set(range(64)):
                failures.append(f"{output.name}: expected 64 first episodes")
                continue
            parent["completed_runs"].append({
                "arm": arm, "seed": seed, "repeat": repeat, "gpu": gpu,
                "successes": sum(bool(row["lift_success"]) for row in episodes),
                "run_manifest": str(child_path.resolve()),
                "created_at": child["created_at"], "completed_at": child["completed_at"],
            })
            print(json.dumps(parent["completed_runs"][-1], sort_keys=True), flush=True)
        if failures:
            parent.update(run_status="FAILED", completed_at=now(), failure=failures)
            write(OUTPUT / "run_manifest.json", parent)
            raise RuntimeError(failures)
        write(OUTPUT / "run_manifest.json", parent)
    parent.update(run_status="COMPLETED", completed_at=now())
    write(OUTPUT / "run_manifest.json", parent)


if __name__ == "__main__":
    main()

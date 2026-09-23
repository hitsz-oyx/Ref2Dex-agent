#!/usr/bin/env python3
"""Frozen matched training/evaluation matrix for Cm effect-ranked PPO."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from src.task.CmResidual.tools.run_v146_eval_matrix import (
    EVAL, PYTHON, ROOT, gpu_free, now, sha256, write,
)


EXPERIMENT_ID = "VAL-20260923-CM-EFFECT-PPO"
WORK_VERSION = "cm_effect_validation"
OUTPUT = ROOT / "outputs/CmResidual/cm_effect_validation"
TRAIN_ENTRY = ROOT / "src/task/CmResidual/tools/run_dexplore_v120_ddp.py"
WEIGHT_BOOTSTRAP = ROOT / "src/task/CmResidual/tools/dexplore_cm_ppo_weight_rank_bootstrap.py"
OFF_BOOTSTRAP = ROOT / "src/task/CmResidual/tools/dexplore_cm_off_rank_bootstrap.py"
SOURCE_CHECKPOINT = ROOT / ("outputs/Dexplore/agent_v139_s3_backtrack_s70_e260/train/"
                            "inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/"
                            "nn/GRAB_00000260.pth")
CMLITE = ROOT / "outputs/CmLite/V1.37/balanced_s1x24_s3_e160_e180/best.pt"
MOTION_ROOT = ROOT / "outputs/CmResidual/agent_v129_s3_coordfix/corrected_converted_r2"
INPUT_MANIFEST = ROOT / "outputs/CmResidual/agent_v129_s3_coordfix/corrected_manifest_r2.json"
SOURCE_SHA = "16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f"
CMLITE_SHA = "396f5e0c92ecf67b2a096568c1a538b7c86d40779a594729f786df6c64ed204a"
INPUT_SHA = "2878bd20d1dd849f6844883c832a3777602d2d2bc73f7581b3bc23d30725f038"
TRAIN_SEEDS = (71, 72, 73, 74)
EVAL_SEEDS = (133, 134, 135, 136, 137, 138)
ARMS = ("effect_rank", "action_shuffled", "off")
SOURCE_FILES = (
    "src/task/CmResidual/tools/run_cm_effect_validation.py",
    "src/task/CmResidual/tools/run_dexplore_v120_ddp.py",
    "src/task/CmResidual/tools/dexplore_ddp_rank_bootstrap.py",
    "src/task/CmResidual/tools/dexplore_cm_ppo_weight_rank_bootstrap.py",
    "src/task/CmResidual/tools/dexplore_cm_off_rank_bootstrap.py",
    "src/task/CmResidual/tools/eval_dexplore_full_episode_grid.py",
    "src/task/CmResidual/cm_ppo_weight.py",
    "src/task/CmResidual/cmlite.py",
    "src/task/CmResidual/dexplore_cm_ppo_weight_agent.py",
    "src/task/CmResidual/dexplore_approach_agent.py",
    "third_party/DExplore/dexplore/run.py",
    "third_party/DExplore/dexplore/evaluate.py",
    "third_party/DExplore/dexplore/env/tasks/base_dexplore_task.py",
    "third_party/DExplore/dexplore/env/tasks/dexplore_inspire.py",
    "third_party/DExplore/dexplore/learning/dexplore_agent.py",
    "third_party/DExplore/dexplore/learning/dexplore_network_builder.py",
    "third_party/DExplore/dexplore/data/cfg/inspire.yaml",
    "third_party/DExplore/dexplore/data/cfg/train/rlg/inspire.yaml",
)


def revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                   text=True).strip()


def source_hashes() -> dict[str, str]:
    return {name: sha256(ROOT / name) for name in SOURCE_FILES}


def frozen_inputs() -> dict[str, object]:
    motion_files = sorted(path for path in MOTION_ROOT.rglob("*") if path.is_file())
    if len(motion_files) != 1:
        raise ValueError(f"expected one converted motion tensor, found {motion_files}")
    actual = {"source_checkpoint": sha256(SOURCE_CHECKPOINT),
              "cmlite": sha256(CMLITE), "input_manifest": sha256(INPUT_MANIFEST)}
    expected = {"source_checkpoint": SOURCE_SHA, "cmlite": CMLITE_SHA,
                "input_manifest": INPUT_SHA}
    if actual != expected:
        raise ValueError(f"frozen input fingerprint mismatch: {actual}")
    return {**actual, "motion_tensor": {"path": str(motion_files[0]),
                                        "sha256": sha256(motion_files[0])}}


def run_dir(train_seed: int, arm: str) -> Path:
    if train_seed not in TRAIN_SEEDS or arm not in ARMS:
        raise ValueError("job outside frozen training matrix")
    return ROOT / f"outputs/Dexplore/cm_effect_val_t{train_seed}_{arm}"


def train_command(train_seed: int, arm: str, gpu: int) -> list[str]:
    if gpu not in (5, 6):
        raise ValueError("validation may use only physical GPU 5 or 6")
    base = [str(PYTHON), str(TRAIN_ENTRY), "--gpus", str(gpu),
            "--run-id", run_dir(train_seed, arm).name]
    if arm == "off":
        base += ["--rank-bootstrap", str(OFF_BOOTSTRAP), "--cm-distill-coef", "0"]
    else:
        mode = "effect_rank" if arm == "effect_rank" else "effect_action_shuffled_rank"
        base += ["--rank-bootstrap", str(WEIGHT_BOOTSTRAP),
                 "--cm-actor-weight-coef", "1.0",
                 "--cm-actor-weight-component", mode,
                 "--cmlite-checkpoint", str(CMLITE), "--cmlite-sha256", CMLITE_SHA]
    return base + [
        "--actual-epochs", "300", "--approach-reward-coef", "2.0",
        "--held-lift-reward-coef", "10.0", "--lift-progress-reward-coef", "5.0",
        "--grasp-link-reward-coef", "0.0", "--min-grasp-links", "0",
        "--scratch-resume-checkpoint", str(SOURCE_CHECKPOINT),
        "--scratch-resume-sha256", SOURCE_SHA, "--learning-rate", "1e-5",
        "--contact-before", "3", "--contact-after", "3",
        "--contact-fraction", "0.5", "--lift-fraction", "0.25",
        "--curriculum-backtrack-start", "180", "--curriculum-backtrack-end", "220",
        "--save-frequency", "20", "--motion-root", str(MOTION_ROOT),
        "--input-manifest", str(INPUT_MANIFEST), "--num-envs", "64",
        "--horizon-length", "32", "--minibatch-size", "256",
        "--max-iterations", "300", "--seed", str(train_seed),
        "--work-version", WORK_VERSION, "--execute"]


def eval_output(train_seed: int, eval_seed: int, arm: str) -> Path:
    if eval_seed not in EVAL_SEEDS:
        raise ValueError("evaluation seed outside frozen matrix")
    return run_dir(train_seed, arm) / f"eval_s{eval_seed}_e300_full_cmval"


def eval_command(train_seed: int, eval_seed: int, arm: str, gpu: int) -> list[str]:
    if gpu not in (5, 6):
        raise ValueError("validation may use only physical GPU 5 or 6")
    return [str(PYTHON), str(EVAL), "--run-dir", str(run_dir(train_seed, arm)),
            "--gpu", str(gpu), "--seed", str(eval_seed), "--epochs", "300",
            "--tag", "cmval", "--work-version", WORK_VERSION]


def training_jobs() -> list[tuple[int, str]]:
    return [(seed, arm) for seed in TRAIN_SEEDS for arm in ARMS]


def evaluation_jobs() -> list[tuple[int, int, str]]:
    return [(train_seed, eval_seed, arm) for train_seed in TRAIN_SEEDS
            for eval_seed in EVAL_SEEDS for arm in ARMS]


def check_identity(manifest: dict) -> None:
    if (revision() != manifest["git_commit"] or
            source_hashes() != manifest["source_sha256"] or
            frozen_inputs() != manifest["input_sha256"]):
        raise RuntimeError("validation code or frozen input drift")


def launch_pair(jobs: list[tuple], command_fn, label: str) -> list[tuple]:
    processes = []
    for gpu, job in zip((5, 6), jobs):
        command = command_fn(*job, gpu)
        log_path = OUTPUT / "logs" / f"{label}_{'_'.join(map(str, job))}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stream = log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        env.pop("CUDA_VISIBLE_DEVICES", None)
        process = subprocess.Popen(command, cwd=ROOT, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT)
        processes.append((job, process, stream, log_path))
    return processes


def wait_pair(processes: list[tuple]) -> None:
    failures = []
    for job, process, stream, log_path in processes:
        code = process.wait()
        stream.close()
        if code:
            failures.append(f"{job}: exit={code}, log={log_path}")
    if failures:
        raise RuntimeError(failures)


def checked_training(train_seed: int, arm: str, manifest: dict) -> dict:
    directory = run_dir(train_seed, arm)
    training = json.loads((directory / "run_manifest.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    checkpoint = next((directory / "train").rglob("GRAB_00000300.pth"))
    expected_mode = ("effect_rank" if arm == "effect_rank" else
                     "effect_action_shuffled_rank" if arm == "action_shuffled" else "joint")
    if (training.get("run_status") != "COMPLETED" or
            training.get("git_commit") != manifest["git_commit"] or
            config.get("seed") != train_seed or config.get("actual_epochs") != 300 or
            config.get("scratch_resume_sha256") != SOURCE_SHA or
            sha256(Path(config["input_manifest"])) != INPUT_SHA or
            config.get("cm_actor_weight_component", "joint") != expected_mode or
            (arm != "off") != (config.get("cm_actor_weight_coef") == 1.0) or
            (arm != "off" and config.get("cmlite_sha256") != CMLITE_SHA)):
        raise ValueError(f"training provenance mismatch: {train_seed} {arm}")
    return {"train_seed": train_seed, "arm": arm, "run_dir": str(directory),
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            "train_log": str(directory / "train.log")}


def checked_evaluation(train_seed: int, eval_seed: int, arm: str,
                       training_record: dict) -> dict:
    directory = eval_output(train_seed, eval_seed, arm)
    child = json.loads((directory / "run_manifest.json").read_text())
    episodes = json.loads((directory / "results.json").read_text())["per_episode"]
    if (child.get("run_status") != "COMPLETED" or
            child.get("checkpoint_sha256") != training_record["checkpoint_sha256"] or
            child.get("input_manifest_sha256") != INPUT_SHA or
            child.get("early_termination_disabled") is not True or
            child.get("seed") != eval_seed or
            child["summary"].get("selector_enabled") is not False or
            len(episodes) != 64 or
            {row["env_id"] for row in episodes} != set(range(64))):
        raise ValueError(f"strict evaluation contract mismatch: {train_seed} {eval_seed} {arm}")
    return {"train_seed": train_seed, "eval_seed": eval_seed, "arm": arm,
            "successes": sum(bool(row["lift_success"]) for row in episodes),
            "run_manifest": str(directory / "run_manifest.json")}


def bootstrap_ci(delta: np.ndarray, rng: np.random.Generator) -> list[float]:
    draws = np.empty(20000, dtype=np.float64)
    for index in range(draws.size):
        rows = rng.integers(0, delta.shape[0], delta.shape[0])
        columns = rng.integers(0, delta.shape[1], delta.shape[1])
        draws[index] = delta[np.ix_(rows, columns)].mean() / 64
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def analyze(manifest: dict) -> dict:
    records = manifest["completed_evaluations"]
    if len(records) != len(evaluation_jobs()):
        raise ValueError("cannot analyze incomplete validation matrix")
    matrix = {arm: np.full((len(TRAIN_SEEDS), len(EVAL_SEEDS)), -1, dtype=np.int64)
              for arm in ARMS}
    for record in records:
        row = TRAIN_SEEDS.index(record["train_seed"])
        column = EVAL_SEEDS.index(record["eval_seed"])
        if matrix[record["arm"]][row, column] != -1:
            raise ValueError("duplicate validation job")
        matrix[record["arm"]][row, column] = record["successes"]
    if any((values < 0).any() for values in matrix.values()):
        raise ValueError("missing validation job")
    rng = np.random.default_rng(20260923)
    comparisons = {}
    for control in ("off", "action_shuffled"):
        delta = matrix["effect_rank"] - matrix[control]
        difference = float(delta.mean() / 64)
        interval = bootstrap_ci(delta, rng)
        by_train = (delta.mean(axis=1) / 64).tolist()
        by_eval = (delta.mean(axis=0) / 64).tolist()
        gate = difference >= 0.08 and all(value > 0 for value in by_train) and interval[0] > 0
        comparisons[control] = {"difference": difference, "ci95": interval,
                                "by_training_seed": dict(zip(TRAIN_SEEDS, by_train)),
                                "by_evaluation_seed": dict(zip(EVAL_SEEDS, by_eval)),
                                "gate_passed": gate}
    if all(value["gate_passed"] for value in comparisons.values()):
        conclusion = "SUPPORTED"
    elif any(value["difference"] <= 0 for value in comparisons.values()):
        conclusion = "REFUTED"
    else:
        conclusion = "INCONCLUSIVE"
    return {"schema": "ref2dex.cm_effect_validation_analysis.v1",
            "validation_id": EXPERIMENT_ID, "run_status": "COMPLETED",
            "conclusion": conclusion, "training_seeds": list(TRAIN_SEEDS),
            "evaluation_seeds": list(EVAL_SEEDS), "num_envs_per_run": 64,
            "matrix": {arm: values.tolist() for arm, values in matrix.items()},
            "totals": {arm: int(values.sum()) for arm, values in matrix.items()},
            "denominator_per_arm": len(TRAIN_SEEDS) * len(EVAL_SEEDS) * 64,
            "comparisons": comparisons,
            "stable_runs_ge58": int((matrix["effect_rank"] >= 58).sum()),
            "total_effect_runs": len(TRAIN_SEEDS) * len(EVAL_SEEDS),
            "bootstrap": {"scheme": "crossed training/evaluation-seed resampling",
                          "draws": 20000, "seed": 20260923}}


def run_train() -> None:
    if OUTPUT.exists() or any(run_dir(*job).exists() for job in training_jobs()):
        raise FileExistsError("validation parent or child training output already exists")
    inputs = frozen_inputs()
    manifest = {"run_status": "STARTED", "phase": "TRAINING",
                "created_at": now(), "validation_id": EXPERIMENT_ID,
                "run_id": OUTPUT.name, "work_version": WORK_VERSION,
                "git_commit": revision(), "source_sha256": source_hashes(),
                "input_sha256": inputs, "training_seeds": list(TRAIN_SEEDS),
                "evaluation_seeds": list(EVAL_SEEDS), "arms": list(ARMS),
                "gpus": [5, 6], "max_concurrent_gpus": 2,
                "output_budget_gb": 10, "official_policy_checkpoint": None,
                "stop_rule": "fixed matrix; stop on technical failure, GPU conflict or source drift",
                "completed_training": [], "completed_evaluations": []}
    write(OUTPUT / "run_manifest.json", manifest)
    jobs = training_jobs()
    try:
        for start in range(0, len(jobs), 2):
            check_identity(manifest)
            if not gpu_free(5) or not gpu_free(6):
                raise RuntimeError("GPU 5 or 6 occupied before training pair")
            pair = jobs[start:start + 2]
            wait_pair(launch_pair(pair, train_command, "train"))
            check_identity(manifest)
            for seed, arm in pair:
                record = checked_training(seed, arm, manifest)
                manifest["completed_training"].append(record)
                print(json.dumps({"phase": "TRAINING", **record}, sort_keys=True), flush=True)
            manifest["run_status"] = "RUNNING"
            write(OUTPUT / "run_manifest.json", manifest)
        manifest.update(run_status="RUNNING", phase="TRAINED", trained_at=now())
        write(OUTPUT / "run_manifest.json", manifest)
    except BaseException as error:
        manifest.update(run_status="FAILED", failed_at=now(), failure=str(error))
        write(OUTPUT / "run_manifest.json", manifest)
        raise


def run_eval() -> None:
    manifest_path = OUTPUT / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("run_status") != "RUNNING" or
            manifest.get("phase") not in ("TRAINED", "EVALUATING") or
            len(manifest["completed_training"]) != len(training_jobs())):
        raise ValueError("validation training must be complete before evaluation")
    check_identity(manifest)
    training = {(row["train_seed"], row["arm"]): row
                for row in manifest["completed_training"]}
    for record in training.values():
        if sha256(Path(record["checkpoint"])) != record["checkpoint_sha256"]:
            raise ValueError("trained checkpoint drift")
    jobs = evaluation_jobs()
    if manifest["phase"] == "TRAINED" and any(
            eval_output(*job).exists() for job in jobs):
        raise FileExistsError("evaluation child output exists before evaluation phase")
    manifest["phase"] = "EVALUATING"
    write(manifest_path, manifest)
    completed = {(row["train_seed"], row["eval_seed"], row["arm"])
                 for row in manifest["completed_evaluations"]}
    try:
        for start in range(0, len(jobs), 2):
            pair = jobs[start:start + 2]
            pending = [job for job in pair if job not in completed]
            if not pending:
                continue
            check_identity(manifest)
            if not gpu_free(5) or not gpu_free(6):
                raise RuntimeError("GPU 5 or 6 occupied before evaluation pair")
            if any(eval_output(*job).exists() for job in pending):
                raise FileExistsError("unrecorded child evaluation output exists")
            wait_pair(launch_pair(pending, eval_command, "eval"))
            check_identity(manifest)
            for seed, eval_seed, arm in pending:
                record = checked_evaluation(seed, eval_seed, arm, training[(seed, arm)])
                manifest["completed_evaluations"].append(record)
                completed.add((seed, eval_seed, arm))
                print(json.dumps({"phase": "EVALUATING", **record}, sort_keys=True), flush=True)
            write(manifest_path, manifest)
        analysis = analyze(manifest)
        write(OUTPUT / "analysis.json", analysis)
        manifest.update(run_status="COMPLETED", phase="ANALYZED",
                        completed_at=now(), conclusion=analysis["conclusion"],
                        analysis=str(OUTPUT / "analysis.json"))
        write(manifest_path, manifest)
        print(json.dumps({"phase": "ANALYZED", "conclusion": analysis["conclusion"],
                          "totals": analysis["totals"],
                          "comparisons": analysis["comparisons"]}, sort_keys=True), flush=True)
    except BaseException as error:
        manifest.update(run_status="FAILED", failed_at=now(), failure=str(error))
        write(manifest_path, manifest)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "eval"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        jobs = training_jobs() if args.phase == "train" else evaluation_jobs()
        fn = train_command if args.phase == "train" else eval_command
        print(json.dumps([fn(*job, 5 if index % 2 == 0 else 6)
                          for index, job in enumerate(jobs)], indent=2))
        return
    if args.phase == "train":
        run_train()
    else:
        run_eval()


if __name__ == "__main__":
    main()

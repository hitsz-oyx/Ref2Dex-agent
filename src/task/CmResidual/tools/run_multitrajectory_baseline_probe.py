"""Continue a self-trained actor on a fixed, identity-balanced motion set."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[4]
DEXPLORE = ROOT / "third_party/DExplore"
DEFAULT_SPEC = ROOT / "src/task/CmResidual/configs/multitrajectory_12_motion_probe.json"
BOOTSTRAP = ROOT / "src/task/CmResidual/tools/dexplore_cm_off_rank_bootstrap.py"
SOURCE = ROOT / ("outputs/Dexplore/agent_v139_s3_backtrack_s70_e260/train/"
                 "inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/"
                 "GRAB_00000260.pth")
SOURCE_SHA = "16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f"
CFG_ENV = DEXPLORE / "dexplore/data/cfg/inspire_object_balanced.yaml"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def gpu_used(index: int) -> int:
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True)
    return {int(row.split(",")[0]): int(row.split(",")[1]) for row in rows.splitlines()}[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--target-epoch", type=int, default=300)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--work-version", default="multitrajectory-baseline-probe")
    parser.add_argument("--source-checkpoint", type=Path, default=SOURCE)
    parser.add_argument("--source-sha256", default=SOURCE_SHA)
    parser.add_argument("--source-epoch", type=int, default=260)
    parser.add_argument("--from-scratch", action="store_true",
                        help="train a self-trained policy with random initialization")
    parser.add_argument("--anneal-start", type=int)
    parser.add_argument("--anneal-end", type=int)
    parser.add_argument("--cfg-env", type=Path, default=CFG_ENV)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=70)
    parser.add_argument("--contact-fraction", type=float, default=.5)
    parser.add_argument("--lift-fraction", type=float, default=.25)
    parser.add_argument("--save-frequency", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning rate must be finite and positive")
    if not args.source_epoch < args.target_epoch <= 500:
        parser.error("target epoch must exceed source and be <=500")
    if (args.anneal_start is None) != (args.anneal_end is None):
        parser.error("anneal start and end must be supplied together")
    if args.anneal_start is not None and not (
            args.source_epoch <= args.anneal_start < args.anneal_end <= args.target_epoch):
        parser.error("anneal window must lie within the continuation")
    output = args.output.resolve()
    if output.exists():
        parser.error("new output directory required")
    source = args.source_checkpoint.resolve()
    cfg_env = args.cfg_env.resolve()
    if args.num_envs < 1 or (args.num_envs * 32) % args.minibatch_size:
        parser.error("rollout batch must divide into complete minibatches")
    if not args.from_scratch and sha256(source) != args.source_sha256:
        raise ValueError("source checkpoint SHA256 drift")
    if gpu_used(args.gpu) > 1024:
        raise RuntimeError(f"physical GPU {args.gpu} is occupied")
    spec_path = args.spec.resolve()
    spec = json.loads(spec_path.read_text())
    input_classification = spec.get("input_classification", "reconstructed_baseline")
    if input_classification not in ("reconstructed_baseline", "filtered_geometric_dexplore"):
        raise ValueError(f"unsupported motion input classification: {input_classification}")
    motions = [ROOT / path for path in spec["motions"]]
    names = [path.name for path in motions]
    if len(names) != len(set(names)):
        raise ValueError("duplicate motion directory name")
    inputs = []
    for motion in motions:
        tensor = motion / "interaction_hand_inspire.pt"
        if not tensor.is_file():
            raise FileNotFoundError(tensor)
        inputs.append({"sequence": motion.name, "path": str(motion),
                       "tensor_sha256": sha256(tensor)})
    motion_root = output / "motions"
    input_manifest = output / "input_manifest.json"
    train_output = output / "train"
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1",
        str(BOOTSTRAP), "--dexplore-run", "dexplore/run.py", "--cm-distill-coef", "0",
        "--actual-epochs", str(args.target_epoch), "--approach-reward-coef", "2.0",
        "--held-lift-reward-coef", "10.0", "--lift-progress-reward-coef", "5.0",
        "--grasp-link-reward-coef", "0.0", "--min-grasp-links", "0",
        "--learning-rate", str(args.learning_rate), "--contact-before", "3", "--contact-after", "3",
        "--contact-fraction", str(args.contact_fraction), "--lift-fraction", str(args.lift_fraction),
        "--save-frequency", str(args.save_frequency), "--task", "Dexplore_Inspire",
        "--cfg_env", str(cfg_env),
        "--cfg_train", "dexplore/data/cfg/train/rlg/inspire.yaml",
        "--motion_file", str(motion_root), "--output_path", str(train_output), "--headless",
        "--sim_device", "cuda:0", "--rl_device", "cuda:0", "--graphics_device_id", "0",
        "--num_envs", str(args.num_envs), "--horizon_length", "32", "--minibatch_size", str(args.minibatch_size),
        "--max_iterations", str(args.target_epoch), "--seed", str(args.seed), "--horovod",
        "--resume", "0" if args.from_scratch else "1",
        "--checkpoint", "Base" if args.from_scratch else str(source),
    ]
    if not args.from_scratch:
        insert_at = command.index("--learning-rate")
        command[insert_at:insert_at] = ["--scratch-resume-checkpoint", str(source),
                                        "--scratch-resume-sha256", args.source_sha256]
    if args.dry_run:
        print(json.dumps({"command": command, "output": str(output),
                          "motion_count": len(inputs), "input_classification": input_classification},
                         indent=2))
        return
    output.mkdir(parents=True)
    motion_root.mkdir()
    for motion in motions:
        (motion_root / motion.name).symlink_to(motion, target_is_directory=True)
    write(input_manifest, {"classification": input_classification,
                           "source_spec": str(spec_path), "source_spec_sha256": sha256(spec_path),
                           "motions": inputs})
    if args.anneal_start is None:
        command[command.index("--save-frequency"):command.index("--save-frequency")] = [
            "--curriculum-backtrack-start", "180", "--curriculum-backtrack-end", "220"]
    else:
        command[command.index("--save-frequency"):command.index("--save-frequency")] = [
            "--curriculum-anneal-start", str(args.anneal_start),
            "--curriculum-anneal-end", str(args.anneal_end)]
    manifest_path = output / "run_manifest.json"
    manifest = {
        "manifest_schema": "ref2dex.run.v1", "run_status": "STARTED",
        "created_at": now(), "run_id": output.name, "work_version": args.work_version,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                              text=True).strip(),
        "command": command, "physical_gpu": args.gpu, "seed": args.seed,
        "source_checkpoint": None if args.from_scratch else str(source),
        "source_checkpoint_sha256": None if args.from_scratch else args.source_sha256,
        "source_epoch": 0 if args.from_scratch else args.source_epoch,
        "curriculum_anneal_window": [args.anneal_start, args.anneal_end]
        if args.anneal_start is not None else None, "cm_enabled": False,
        "motion_root": str(motion_root), "input_manifest": str(input_manifest),
        "input_manifest_sha256": sha256(input_manifest), "motion_count": len(inputs),
        "input_classification": input_classification,
        "env_config": str(cfg_env), "env_config_sha256": sha256(cfg_env),
        "object_sampling": "see pinned env config; all requested inputs retained or fail loudly",
        "initialization": "random_scratch" if args.from_scratch else "pinned_scratch_resume",
        "learning_rate": args.learning_rate,
        "budget": {"gpu_count": 1, "wall_minutes": 60, "output_gb": 5},
        "stop_rule": "input drift, GPU conflict, non-finite training, >60 min, or missing endpoint checkpoint",
    }
    write(output / "config.json", {"num_envs_per_rank": args.num_envs,
                                   "motion_root": str(motion_root),
                                   "input_manifest": str(input_manifest),
                                   "cfg_env": str(cfg_env),
                                   "seed": args.seed})
    write(manifest_path, manifest)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "src/task/CmResidual/tools"), env.get("PYTHONPATH", "")])
    started = time.monotonic()
    try:
        with (output / "train.log").open("w", buffering=1) as log:
            result = subprocess.run(command, cwd=DEXPLORE, env=env,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=3600)
        if result.returncode:
            raise RuntimeError(f"training exit code {result.returncode}")
        checkpoints = list(train_output.rglob(f"GRAB_{args.target_epoch:08d}.pth"))
        if len(checkpoints) != 1:
            raise ValueError(f"expected one endpoint checkpoint, found {checkpoints}")
        manifest.update(run_status="COMPLETED", completed_at=now(),
                        elapsed_seconds=time.monotonic() - started,
                        checkpoint=str(checkpoints[0]), checkpoint_sha256=sha256(checkpoints[0]))
        print(json.dumps({"run_id": output.name, "run_status": "COMPLETED",
                          "checkpoint": str(checkpoints[0])}), flush=True)
    except BaseException as error:
        manifest.update(run_status="FAILED", completed_at=now(),
                        elapsed_seconds=time.monotonic() - started,
                        failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(manifest_path, manifest)


if __name__ == "__main__":
    main()

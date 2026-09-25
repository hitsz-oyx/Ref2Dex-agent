"""Launch a bounded matched temporal-Cm online Probe.

The launcher keeps the two arms identical except for the temporal-Cm reward
bootstrap and records the exact input hashes, command, resource check, and
fixed epoch budget in a run manifest.  It is intentionally small: this is a
Probe launcher, not a general experiment scheduler.
"""
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
DEXPLORE_RUN = ROOT / "third_party/DExplore/dexplore/run.py"
SOURCE_CHECKPOINT = ROOT / (
    "outputs/Dexplore/agent_v139_s3_backtrack_s70_e260/train/"
    "inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/"
    "GRAB_00000260.pth"
)
TEMPORAL_CHECKPOINT = ROOT / (
    "outputs/CmResidual/agent_cm_history_value_probe_20260925_v2/history_action.pt"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def gpu_memory(gpu: int) -> int:
    result = subprocess.run(
        ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"],
        check=True, text=True, stdout=subprocess.PIPE,
    )
    return int(result.stdout.strip())


def motion_inventory(root: Path) -> list[dict[str, object]]:
    if not root.is_dir():
        raise FileNotFoundError(f"missing motion root: {root}")
    records = []
    for path in sorted(root.glob("*/interaction_hand_inspire.pt")):
        records.append({"path": str(path.resolve()), "sha256": sha256(path),
                        "bytes": path.stat().st_size})
    if not records:
        raise ValueError(f"motion root has no Inspire motion files: {root}")
    return records


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("off", "on"), required=True)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--motion-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/Dexplore")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--actual-epochs", type=int, required=True)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--horizon-length", type=int, default=32)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--temporal-coef", type=float, default=1.0)
    parser.add_argument("--temporal-scale-mm", type=float, default=20.0)
    parser.add_argument("--save-frequency", type=int, default=20)
    args = parser.parse_args(argv)
    if args.gpus < 0 or args.actual_epochs < 1 or args.num_envs < 1:
        raise ValueError("GPU, epoch, and environment counts must be positive")
    if args.horizon_length < 1 or args.minibatch_size < 1:
        raise ValueError("horizon and minibatch sizes must be positive")
    if args.mode == "on" and not (args.temporal_coef > 0):
        raise ValueError("temporal coefficient must be positive for Cm-on")
    if args.save_frequency < 1:
        raise ValueError("save frequency must be positive")
    return args


def build_command(args, output: Path, temporal_sha: str, source_sha: str) -> list[str]:
    if args.mode == "on":
        bootstrap = ROOT / "src/task/CmResidual/tools/dexplore_temporal_cm_rank_bootstrap.py"
        bootstrap_args = [
            "--temporal-cm-reward-coef", str(args.temporal_coef),
            "--temporal-cm-reward-scale-mm", str(args.temporal_scale_mm),
            "--temporal-cm-checkpoint", str(TEMPORAL_CHECKPOINT.resolve()),
            "--temporal-cm-sha256", temporal_sha,
            "--actual-epochs", str(args.actual_epochs),
            "--save-frequency", str(args.save_frequency),
        ]
    else:
        bootstrap = ROOT / "src/task/CmResidual/tools/dexplore_cm_off_rank_bootstrap.py"
        bootstrap_args = [
            "--cm-distill-coef", "0",
            "--actual-epochs", str(args.actual_epochs),
            "--save-frequency", str(args.save_frequency),
        ]
    bootstrap_args += [
        "--scratch-resume-checkpoint", str(SOURCE_CHECKPOINT.resolve()),
        "--scratch-resume-sha256", source_sha,
    ]
    dexplore_args = [
        "--task", "Dexplore_Inspire",
        "--cfg_env", "dexplore/data/cfg/inspire.yaml",
        "--cfg_train", "dexplore/data/cfg/train/rlg/inspire.yaml",
        "--motion_file", str(args.motion_root.resolve()),
        "--output_path", str(output / "train"),
        "--headless", "--sim_device", "cuda:0", "--rl_device", "cuda:0",
        "--graphics_device_id", "0", "--num_envs", str(args.num_envs),
        "--horizon_length", str(args.horizon_length),
        "--minibatch_size", str(args.minibatch_size),
        "--max_iterations", str(args.actual_epochs), "--seed", str(args.seed),
        "--horovod", "--resume", "1", "--checkpoint", str(SOURCE_CHECKPOINT.resolve()),
    ]
    return [sys.executable, "-m", "torch.distributed.run", "--standalone",
            "--nproc_per_node=1", str(bootstrap), "--dexplore-run",
            str(DEXPLORE_RUN)] + bootstrap_args + dexplore_args


def main(argv=None):
    args = parse_args(argv)
    output = (args.output_root / args.run_id).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    if not DEXPLORE_RUN.is_file() or not SOURCE_CHECKPOINT.is_file():
        raise FileNotFoundError("required DExplore or source checkpoint is missing")
    if args.mode == "on" and not TEMPORAL_CHECKPOINT.is_file():
        raise FileNotFoundError(f"missing temporal checkpoint: {TEMPORAL_CHECKPOINT}")
    if gpu_memory(args.gpus) > 1024:
        raise RuntimeError(f"GPU{args.gpus} is occupied; refusing to launch")

    motion_root = args.motion_root.resolve()
    motions = motion_inventory(motion_root)
    source_sha = sha256(SOURCE_CHECKPOINT)
    temporal_sha = sha256(TEMPORAL_CHECKPOINT) if args.mode == "on" else None
    command = build_command(args, output, temporal_sha or "", source_sha)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    tools_dir = str((ROOT / "src/task/CmResidual/tools").resolve())
    dexplore_dir = str((ROOT / "third_party/DExplore/dexplore").resolve())
    environment["PYTHONPATH"] = ":".join(
        [tools_dir, str(ROOT.resolve()), dexplore_dir,
         environment.get("PYTHONPATH", "")])

    output.mkdir(parents=True)
    manifest_path = output / "run_manifest.json"
    manifest = {
        "manifest_schema": "ref2dex.run.v1",
        "created_at": now(), "task": "CmResidual",
        "work_version": "V1.21-temporal", "run_id": args.run_id,
        "git_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "run_status": "STARTED", "mode": args.mode, "command": command,
        "runtime": {"python": sys.executable, "cuda_visible_devices": str(args.gpus),
                    "physical_gpus": [args.gpus], "backend": "torch.distributed:nccl",
                    "gpu_memory_before_mib": gpu_memory(args.gpus)},
        "source_checkpoint": str(SOURCE_CHECKPOINT.resolve()),
        "source_checkpoint_sha256": source_sha,
        "temporal_checkpoint": (str(TEMPORAL_CHECKPOINT.resolve())
                                if args.mode == "on" else None),
        "temporal_checkpoint_sha256": temporal_sha,
        "motion_root": str(motion_root), "motion_inventory": motions,
        "seed": args.seed, "actual_epochs": args.actual_epochs,
        "additional_epochs_from_source": args.actual_epochs - 260,
        "num_envs_per_rank": args.num_envs, "horizon_length": args.horizon_length,
        "minibatch_size": args.minibatch_size,
        "budget": {"gpu_count": 1, "wall_minutes": 60, "output_mb": 500},
        "stop_rule": "fixed epoch budget; stop on input drift, nonfinite tensors, GPU conflict, or integration error",
        "train_log": str(output / "train.log"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    try:
        with (output / "train.log").open("w", encoding="utf-8", buffering=1) as stream:
            subprocess.run(command, cwd=DEXPLORE_RUN.parent.parent, env=environment,
                           check=True, stdout=stream, stderr=subprocess.STDOUT)
        checkpoints = sorted((output / "train").rglob("*.pth"))
        if not checkpoints:
            raise RuntimeError("training finished without a checkpoint")
        manifest.update({"run_status": "COMPLETED", "completed_at": now(),
                         "checkpoint": str(checkpoints[-1]),
                         "checkpoint_sha256": sha256(checkpoints[-1])})
    except BaseException as error:
        manifest.update({"run_status": "FAILED", "completed_at": now(),
                         "failure_reason": f"{type(error).__name__}: {error}"})
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

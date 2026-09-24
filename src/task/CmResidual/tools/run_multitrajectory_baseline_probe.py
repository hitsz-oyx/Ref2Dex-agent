"""Continue a self-trained actor on a fixed, identity-balanced motion set."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
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
    args = parser.parse_args()
    if not 261 <= args.target_epoch <= 360:
        parser.error("target epoch must be 261..360")
    output = args.output.resolve()
    if output.exists():
        parser.error("new output directory required")
    if sha256(SOURCE) != SOURCE_SHA:
        raise ValueError("source checkpoint SHA256 drift")
    if gpu_used(args.gpu) > 1024:
        raise RuntimeError(f"physical GPU {args.gpu} is occupied")
    spec_path = args.spec.resolve()
    spec = json.loads(spec_path.read_text())
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
    output.mkdir(parents=True)
    motion_root = output / "motions"
    motion_root.mkdir()
    for motion in motions:
        (motion_root / motion.name).symlink_to(motion, target_is_directory=True)
    input_manifest = output / "input_manifest.json"
    write(input_manifest, {"classification": "reconstructed_baseline",
                           "source_spec": str(spec_path), "source_spec_sha256": sha256(spec_path),
                           "motions": inputs})
    train_output = output / "train"
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1",
        str(BOOTSTRAP), "--dexplore-run", "dexplore/run.py", "--cm-distill-coef", "0",
        "--actual-epochs", str(args.target_epoch), "--approach-reward-coef", "2.0",
        "--held-lift-reward-coef", "10.0", "--lift-progress-reward-coef", "5.0",
        "--grasp-link-reward-coef", "0.0", "--min-grasp-links", "0",
        "--scratch-resume-checkpoint", str(SOURCE), "--scratch-resume-sha256", SOURCE_SHA,
        "--learning-rate", "1e-05", "--contact-before", "3", "--contact-after", "3",
        "--contact-fraction", "0.5", "--lift-fraction", "0.25",
        "--curriculum-backtrack-start", "180", "--curriculum-backtrack-end", "220",
        "--save-frequency", "20", "--task", "Dexplore_Inspire",
        "--cfg_env", "dexplore/data/cfg/inspire_object_balanced.yaml",
        "--cfg_train", "dexplore/data/cfg/train/rlg/inspire.yaml",
        "--motion_file", str(motion_root), "--output_path", str(train_output), "--headless",
        "--sim_device", "cuda:0", "--rl_device", "cuda:0", "--graphics_device_id", "0",
        "--num_envs", "64", "--horizon_length", "32", "--minibatch_size", "256",
        "--max_iterations", str(args.target_epoch), "--seed", "70", "--horovod",
        "--resume", "1", "--checkpoint", str(SOURCE),
    ]
    manifest_path = output / "run_manifest.json"
    manifest = {
        "manifest_schema": "ref2dex.run.v1", "run_status": "STARTED",
        "created_at": now(), "run_id": output.name, "work_version": args.work_version,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                              text=True).strip(),
        "command": command, "physical_gpu": args.gpu, "seed": 70,
        "source_checkpoint": str(SOURCE), "source_checkpoint_sha256": SOURCE_SHA,
        "source_epoch": 260, "target_epoch": args.target_epoch, "cm_enabled": False,
        "motion_root": str(motion_root), "input_manifest": str(input_manifest),
        "input_manifest_sha256": sha256(input_manifest), "motion_count": len(inputs),
        "env_config": str(CFG_ENV), "env_config_sha256": sha256(CFG_ENV),
        "object_sampling": "each listed motion once; hard-object oversampling disabled",
        "budget": {"gpu_count": 1, "wall_minutes": 60, "output_gb": 5},
        "stop_rule": "input drift, GPU conflict, non-finite training, >60 min, or missing endpoint checkpoint",
    }
    write(output / "config.json", {"num_envs_per_rank": 64,
                                   "motion_root": str(motion_root),
                                   "input_manifest": str(input_manifest),
                                   "cfg_env": "dexplore/data/cfg/inspire_object_balanced.yaml",
                                   "seed": 70})
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

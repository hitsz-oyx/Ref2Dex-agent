"""Run the fixed train5 object-balanced Cm-off continuation Probe."""
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
BOOTSTRAP = ROOT / "src/task/CmResidual/tools/dexplore_cm_off_rank_bootstrap.py"
SOURCE = (ROOT / "outputs/Dexplore/agent_crossobject_train5_s179_e320/train/"
          "inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/"
          "GRAB_00000320.pth")
SOURCE_SHA = "6907c12f8ee4ffa9af22ccae8ffe7599d524e401ff2d8f21b8804185fd5d4961"
MOTION_ROOT = ROOT / "outputs/CmResidual/agent_crossobject_pilot_split_v2/train"
SPLIT = ROOT / "outputs/CmResidual/agent_crossobject_pilot_split_v2/manifest.json"
SPLIT_SHA = "694b74889e5d5523be3b96485e5e91bdc31fb420aec2766ec0fb4a2dc40d1c3c"
ENV_CONFIG = "dexplore/data/cfg/inspire_object_balanced.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
    parser.add_argument("--motion-root", type=Path, default=MOTION_ROOT)
    parser.add_argument("--input-manifest", type=Path, default=SPLIT)
    parser.add_argument("--input-manifest-sha256", default=SPLIT_SHA)
    parser.add_argument("--cfg-env", default=ENV_CONFIG)
    parser.add_argument("--sampling-description",
                        default="one motion per object; hard-object oversampling disabled")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("new output directory required")
    motion_root = args.motion_root.resolve()
    input_manifest = args.input_manifest.resolve()
    if (sha256(SOURCE) != SOURCE_SHA or
            sha256(input_manifest) != args.input_manifest_sha256 or
            not motion_root.is_dir()):
        raise ValueError("source checkpoint or split drift")
    if gpu_used(args.gpu) > 1024:
        raise RuntimeError(f"physical GPU {args.gpu} is occupied")
    output.mkdir(parents=True)
    train_output = output / "train"
    config_path = output / "config.json"
    log_path = output / "train.log"
    manifest_path = output / "run_manifest.json"
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1",
        str(BOOTSTRAP), "--dexplore-run", "dexplore/run.py", "--cm-distill-coef", "0",
        "--actual-epochs", "360", "--approach-reward-coef", "2.0",
        "--held-lift-reward-coef", "10.0", "--lift-progress-reward-coef", "5.0",
        "--grasp-link-reward-coef", "0.0", "--min-grasp-links", "0",
        "--scratch-resume-checkpoint", str(SOURCE), "--scratch-resume-sha256", SOURCE_SHA,
        "--learning-rate", "1e-05", "--contact-before", "3", "--contact-after", "3",
        "--contact-fraction", "0.5", "--lift-fraction", "0.25",
        "--curriculum-anneal-start", "40", "--curriculum-anneal-end", "80",
        "--save-frequency", "20", "--task", "Dexplore_Inspire",
        "--cfg_env", args.cfg_env, "--cfg_train", "dexplore/data/cfg/train/rlg/inspire.yaml",
        "--motion_file", str(motion_root), "--output_path", str(train_output), "--headless",
        "--sim_device", "cuda:0", "--rl_device", "cuda:0", "--graphics_device_id", "0",
        "--num_envs", "64", "--horizon_length", "32", "--minibatch_size", "256",
        "--max_iterations", "360", "--seed", "179", "--horovod", "--resume", "1",
        "--checkpoint", str(SOURCE),
    ]
    manifest = {
        "manifest_schema": "ref2dex.run.v1", "run_status": "STARTED",
        "created_at": now(), "run_id": output.name, "work_version": "crossobject-balanced-coverage",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                              text=True).strip(),
        "command": command, "physical_gpu": args.gpu, "seed": 179,
        "source_checkpoint": str(SOURCE), "source_checkpoint_sha256": SOURCE_SHA,
        "source_epoch": 320, "target_epoch": 360, "cm_enabled": False,
        "motion_root": str(motion_root), "input_manifest": str(input_manifest),
        "input_manifest_sha256": args.input_manifest_sha256,
        "env_config": str(DEXPLORE / args.cfg_env),
        "env_config_sha256": sha256(DEXPLORE / args.cfg_env),
        "object_sampling": args.sampling_description,
        "budget": {"gpu_count": 1, "wall_minutes": 20, "output_gb": 5},
        "stop_rule": "input drift, GPU conflict, non-finite training or missing e360 checkpoint",
    }
    write(config_path, {"num_envs_per_rank": 64, "motion_root": str(motion_root),
                        "input_manifest": str(input_manifest), "cfg_env": args.cfg_env,
                        "object_sampling": args.sampling_description, "seed": 179})
    write(manifest_path, manifest)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "src/task/CmResidual/tools"), environment.get("PYTHONPATH", "")])
    started = time.monotonic()
    try:
        with log_path.open("w", buffering=1) as log:
            result = subprocess.run(command, cwd=DEXPLORE, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"training exit code {result.returncode}")
        checkpoints = list(train_output.rglob("GRAB_00000360.pth"))
        if len(checkpoints) != 1:
            raise ValueError(f"expected one e360 checkpoint, found {checkpoints}")
        checkpoint = checkpoints[0]
        manifest.update(run_status="COMPLETED", completed_at=now(),
                        elapsed_seconds=time.monotonic() - started,
                        checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint))
        print(json.dumps({"run_id": output.name, "run_status": "COMPLETED",
                          "checkpoint": str(checkpoint),
                          "checkpoint_sha256": manifest["checkpoint_sha256"]}), flush=True)
    except BaseException as error:
        manifest.update(run_status="FAILED", completed_at=now(),
                        elapsed_seconds=time.monotonic() - started,
                        failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(manifest_path, manifest)


if __name__ == "__main__":
    main()

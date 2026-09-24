"""Export hash-pinned official apple transitions for failure-phase audit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from src.task.CmResidual.tools.probe_intervention_handflow import sha256


ROOT = Path(__file__).resolve().parents[4]
DEXPLORE = ROOT / "third_party/DExplore"
CHECKPOINT = Path("/home2/wyy/oyx_ws/_external/dexplore_official_v120/checkpoint/inspire.pth")
CHECKPOINT_SHA = "8f6823db752288f1bddd6d042981d33514e29dac5a68e58726e76215fea6d553"
SPLIT = ROOT / "outputs/CmResidual/agent_crossobject_split_v1/manifest.json"
SPLIT_SHA = "35fffcb500f1f3db76fb8a59c940f5da0a113627bb0be9db0d68b0112fe7f516"
MOTION = ROOT / "outputs/CmResidual/agent_crossobject_split_v1/heldout"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new output directory required")
    if sha256(CHECKPOINT) != CHECKPOINT_SHA or sha256(SPLIT) != SPLIT_SHA:
        raise ValueError("official checkpoint or split SHA drift")
    split = json.loads(SPLIT.read_text())
    if (split.get("heldout_objects") != ["apple"] or
            Path(split.get("heldout_motion_root", "")).resolve() != MOTION.resolve()):
        raise ValueError("held-out apple motion root drift")
    used = int(subprocess.check_output([
        "nvidia-smi", f"--id={args.gpu}", "--query-gpu=memory.used",
        "--format=csv,noheader,nounits"], text=True).strip())
    if used > 512:
        raise RuntimeError(f"GPU{args.gpu} occupied: {used} MiB")
    output = args.output.resolve()
    output.mkdir(parents=True)
    command = [sys.executable, str(DEXPLORE / "dexplore/evaluate.py"),
               "--task", "Dexplore_Inspire",
               "--cfg_env", "dexplore/data/cfg/inspire.yaml",
               "--cfg_train", "dexplore/data/cfg/train/rlg/inspire.yaml",
               "--motion_file", str(MOTION), "--checkpoint", str(CHECKPOINT),
               "--disable-early-termination", "--headless",
               "--sim_device", "cuda:0", "--rl_device", "cuda:0",
               "--graphics_device_id", "0", "--num_envs", "64",
               "--seed", "174", "--output", str(output / "results.json"),
               "--transition-output", str(output / "transitions.pt")]
    manifest = {"run_status": "STARTED", "started_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       cwd=ROOT, text=True).strip(),
                "source_actor_role": "official_diagnostic",
                "checkpoint_sha256": CHECKPOINT_SHA, "split_sha256": SPLIT_SHA,
                "seed": 174, "num_envs": 64, "physical_gpu": args.gpu,
                "wall_budget_minutes": 20, "output_budget_mb": 100,
                "stop_rule": "input drift, GPU conflict, non-finite export or wall budget",
                "command": command}
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    started = time.monotonic()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    try:
        with (output / "eval.log").open("w") as stream:
            subprocess.run(command, cwd=DEXPLORE, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, timeout=20 * 60, check=True)
        summary = json.loads((output / "results.json").read_text())["summary"]
        transitions = output / "transitions.pt"
        if (summary.get("num_episodes") != 64 or
                summary.get("early_termination_disabled") is not True or
                not transitions.is_file() or not transitions.stat().st_size or
                transitions.stat().st_size > 100 * 1024 * 1024):
            raise ValueError("official diagnostic export contract mismatch")
        manifest.update(run_status="COMPLETED",
                        transition_sha256=sha256(transitions),
                        summary=summary)
        print(json.dumps({"run_status": "COMPLETED",
                          "lift_success_rate": summary["lift_success_rate"],
                          "contact_fraction": summary["mean_hand_object_contact_fraction"]},
                         sort_keys=True), flush=True)
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

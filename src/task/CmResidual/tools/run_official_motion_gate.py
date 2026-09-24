"""Use the read-only official actor only as a physical motion feasibility diagnostic."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-dir", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=174)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new output directory required")
    tensor_path = args.motion_dir / args.sequence / "interaction_hand_inspire.pt"
    if not tensor_path.is_file() or not args.input_manifest.is_file():
        raise FileNotFoundError("motion tensor or input manifest missing")
    if sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError("official checkpoint SHA drift")
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
               "--motion_file", str(args.motion_dir.resolve()),
               "--checkpoint", str(CHECKPOINT),
               "--disable-early-termination", "--headless",
               "--sim_device", "cuda:0", "--rl_device", "cuda:0",
               "--graphics_device_id", "0", "--num_envs", "64",
               "--seed", str(args.seed), "--output", str(output / "results.json")]
    manifest = {"run_status": "STARTED", "started_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       cwd=ROOT, text=True).strip(),
                "source_actor_role": "official_feasibility_diagnostic",
                "checkpoint_sha256": CHECKPOINT_SHA,
                "input_manifest_sha256": sha256(args.input_manifest),
                "motion_tensor_sha256": sha256(tensor_path),
                "sequence": args.sequence, "seed": args.seed, "num_envs": 64,
                "physical_gpu": args.gpu, "wall_budget_minutes": 20,
                "output_budget_mb": 10,
                "stop_rule": "input drift, GPU conflict, non-finite result or wall budget",
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
        if (summary.get("num_episodes") != 64 or
                summary.get("early_termination_disabled") is not True):
            raise ValueError("incomplete physical diagnostic")
        manifest.update(run_status="COMPLETED", summary=summary,
                        results_sha256=sha256(output / "results.json"))
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

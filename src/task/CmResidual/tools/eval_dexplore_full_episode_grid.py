#!/usr/bin/env python3
"""Evaluate frozen scratch PPO checkpoints using the V1.28 full-episode gate."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
DEXPLORE = ROOT / "third_party/DExplore"
EVALUATE = DEXPLORE / "dexplore/evaluate.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _gpu_used_mib(index: int) -> int:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True)
    values = {int(row.split(",")[0].strip()): int(row.split(",")[1].strip())
              for row in output.splitlines()}
    return values[index]


def _checkpoint(run_dir: Path, epoch: int) -> Path:
    matches = list((run_dir / "train").rglob(f"GRAB_{epoch:08d}.pth"))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one epoch {epoch} checkpoint, found {matches}")
    return matches[0].resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, nargs="+", required=True)
    parser.add_argument("--motion-root-override", type=Path)
    parser.add_argument("--input-manifest-override", type=Path)
    parser.add_argument("--tag", help="short output suffix for a transfer probe")
    parser.add_argument("--selector-cmlite-checkpoint", type=Path)
    parser.add_argument("--selector-cmlite-sha256")
    parser.add_argument("--reference-action-lead", type=int,
                        help="diagnostic: execute the reference controller instead of the actor")
    parser.add_argument("--work-version", default="V1.29")
    parser.add_argument("--cfg-env", default="dexplore/data/cfg/inspire.yaml",
                        help="DExplore environment config, relative to the vendor root")
    parser.add_argument("--save-transitions", action="store_true",
                        help="save step-major transition tensors for an offline model audit")
    parser.add_argument("--contact-topology", action="store_true",
                        help="append configured hand-link force magnitudes to transitions")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.gpu < 0 or args.seed < 0 or any(epoch < 1 for epoch in args.epochs):
        raise ValueError("GPU, seed and epochs must be nonnegative/positive")
    if len(set(args.epochs)) != len(args.epochs):
        raise ValueError("duplicate checkpoint epochs")
    override = args.motion_root_override is not None
    if override != (args.input_manifest_override is not None):
        raise ValueError("motion root and input manifest overrides must be paired")
    if override and args.tag is None:
        raise ValueError("a transfer probe requires a unique output tag")
    selector = args.selector_cmlite_checkpoint is not None
    if selector != (args.selector_cmlite_sha256 is not None):
        raise ValueError("CmLite selector checkpoint and SHA256 must be paired")
    if selector and args.tag is None:
        raise ValueError("CmLite selector requires a unique output tag")
    if selector and args.reference_action_lead is not None:
        raise ValueError("reference action and CmLite selector are mutually exclusive")
    if args.contact_topology and not args.save_transitions:
        raise ValueError("contact topology requires transition export")
    if args.reference_action_lead is not None and args.reference_action_lead < 0:
        raise ValueError("reference action lead must be nonnegative")
    if selector and (not args.selector_cmlite_checkpoint.is_file() or
                     _sha256(args.selector_cmlite_checkpoint) != args.selector_cmlite_sha256):
        raise ValueError("CmLite selector checkpoint is missing or SHA256 mismatched")
    if args.tag is not None and (not args.tag or not args.tag.replace("_", "").isalnum()):
        raise ValueError("output tag must contain only letters, digits and underscores")
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    training = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if training.get("run_status") != "COMPLETED" or training.get("seed") is None:
        raise ValueError("source scratch training must have completed with a recorded seed")
    motion_root = (args.motion_root_override or Path(config["motion_root"])).resolve()
    input_manifest = (args.input_manifest_override or Path(config["input_manifest"])).resolve()
    if not motion_root.is_dir() or not input_manifest.is_file():
        raise FileNotFoundError("frozen motion input is missing")
    input_record = json.loads(input_manifest.read_text(encoding="utf-8"))
    if input_record.get("classification") != "reconstructed_baseline":
        raise ValueError("motion input must have reconstructed baseline provenance")
    num_envs = int(config["num_envs_per_rank"])
    if num_envs != 64:
        raise ValueError("the V1.29 strict gate requires 64 environments")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    entries = []
    for epoch in args.epochs:
        checkpoint = _checkpoint(run_dir, epoch)
        suffix = f"_{args.tag}" if args.tag is not None else ""
        output = run_dir / f"eval_s{args.seed}_e{epoch:03d}_full{suffix}"
        if output.exists():
            raise FileExistsError(output)
        command = [sys.executable, str(EVALUATE), "--task", "Dexplore_Inspire",
                   "--cfg_env", args.cfg_env,
                   "--cfg_train", "dexplore/data/cfg/train/rlg/inspire.yaml",
                   "--motion_file", str(motion_root), "--checkpoint", str(checkpoint),
                   "--disable-early-termination", "--headless", "--sim_device", "cuda:0",
                   "--rl_device", "cuda:0", "--graphics_device_id", "0",
                   "--num_envs", str(num_envs), "--seed", str(args.seed),
                   "--output", str(output / "results.json")]
        transition_output = output / "transitions.pt" if args.save_transitions else None
        if transition_output is not None:
            command += ["--transition-output", str(transition_output)]
        if args.contact_topology:
            command += ["--contact-topology"]
        if selector:
            command += ["--cmlite-selector-checkpoint",
                        str(args.selector_cmlite_checkpoint.resolve()),
                        "--cmlite-selector-sha256", args.selector_cmlite_sha256]
        if args.reference_action_lead is not None:
            command += ["--reference-action-lead", str(args.reference_action_lead)]
        entry = {"run_status": "STARTED", "created_at": _now(),
                 "run_id": output.name, "work_version": args.work_version,
                 "evaluation_commit": revision, "training_commit": training["git_commit"],
                 "training_run_id": training["run_id"], "input_manifest": str(input_manifest),
                 "input_manifest_sha256": _sha256(input_manifest),
                 "motion_root": str(motion_root), "input_sequence": input_record.get("sequence"),
                 "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
                 "physical_gpu": args.gpu, "seed": args.seed, "epoch": epoch,
                 "num_envs": num_envs, "early_termination_disabled": True,
                 "selector_cmlite_checkpoint": str(args.selector_cmlite_checkpoint.resolve())
                 if selector else None,
                 "selector_cmlite_sha256": args.selector_cmlite_sha256 if selector else None,
                 "reference_action_lead": args.reference_action_lead,
                 "transition_output": str(transition_output) if transition_output else None,
                 "cfg_env": args.cfg_env,
                 "contact_topology": args.contact_topology,
                 "command": command}
        entries.append((output, command, entry))
    if args.dry_run:
        print(json.dumps([entry for _, _, entry in entries], indent=2))
        return 0
    used = _gpu_used_mib(args.gpu)
    if used > 1024:
        raise RuntimeError(f"physical GPU {args.gpu} is occupied: {used} MiB")
    for output, command, entry in entries:
        output.mkdir(parents=True)
        manifest = output / "run_manifest.json"
        _write(manifest, entry)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        with (output / "eval.log").open("w", encoding="utf-8") as log:
            code = subprocess.run(command, cwd=DEXPLORE, env=env,
                                  stdout=log, stderr=subprocess.STDOUT).returncode
        try:
            if code:
                raise RuntimeError(f"evaluation exit code {code}")
            summary = json.loads((output / "results.json").read_text(encoding="utf-8"))["summary"]
            if (summary.get("num_episodes") != num_envs or
                    summary.get("early_termination_disabled") is not True):
                raise ValueError("strict evaluation summary contract mismatch")
            if summary.get("selector_enabled") is not selector:
                raise ValueError("selector status mismatch in evaluation summary")
            if transition_output is not None:
                if not transition_output.is_file() or not transition_output.stat().st_size:
                    raise ValueError("requested transition tensor is missing or empty")
                entry["transition_sha256"] = _sha256(transition_output)
                entry["transition_bytes"] = transition_output.stat().st_size
            entry.update(run_status="COMPLETED", completed_at=_now(), summary=summary)
            print(json.dumps({"run_id": entry["run_id"], "lift_success_rate":
                              summary["lift_success_rate"], "mean_max_contact_lift_m":
                              summary["mean_max_contact_lift_m"]}), flush=True)
        except BaseException as error:
            entry.update(run_status="FAILED", completed_at=_now(), failure=str(error))
            _write(manifest, entry)
            raise
        _write(manifest, entry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

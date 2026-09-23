"""Build, inspect, and launch the isolated V1.20 torchrun DExplore command."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
BOOTSTRAP = Path(__file__).resolve().with_name("dexplore_ddp_rank_bootstrap.py")
DEFAULT_DEXPLORE_RUN = REPOSITORY_ROOT / "third_party/DExplore/dexplore/run.py"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs/Dexplore"
RUNTIME_ASSETS = (
    (REPOSITORY_ROOT / "data/raw_data/GRAB/objects/airplane/mesh.obj",
     Path("dexplore/data/assets/mjcf/objects/airplane/airplane.obj"),
     "dcbb1cce38e65b3ee608e20f0846cbf93aa3b7bbf863a67582d9944f9a0d64f0"),
    (REPOSITORY_ROOT / "data/raw_data/GRAB/objects/table/mesh.obj",
     Path("dexplore/data/assets/mjcf/objects/table/table.obj"),
     "25c6fb8b774a04f5314a13a538b9c26886716d196d46a68979e817755cf0e383"),
)


def parse_gpus(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item)
    if not 1 <= len(values) <= 2 or len(set(values)) != len(values) or any(item < 0 for item in values):
        raise ValueError("--gpus must name one or two unique non-negative physical GPU indices")
    return values


def torchrun_command(*, gpus: Sequence[int], dexplore_run: Path, dexplore_args: Sequence[str],
                     bootstrap: Path = BOOTSTRAP, bootstrap_args: Sequence[str] = ()) -> list[str]:
    if not bootstrap.is_file():
        raise FileNotFoundError(f"missing DDP bootstrap: {bootstrap}")
    if not dexplore_run.is_file():
        raise FileNotFoundError(f"missing DExplore entrypoint: {dexplore_run}")
    return [sys.executable, "-m", "torch.distributed.run", "--standalone",
            f"--nproc_per_node={len(gpus)}", str(bootstrap),
            "--dexplore-run", str(dexplore_run)] + list(bootstrap_args) + list(dexplore_args)


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_runtime_assets(dexplore_run: Path) -> list[dict[str, str | bool]]:
    """Materialize DExplore's ignored mesh inputs from fixed raw GRAB sources.

    DExplore deliberately ignores this asset directory; copying is byte-for-byte,
    refuses unexpected existing content, and never follows a symlink.
    """
    source_root = dexplore_run.resolve().parents[1]
    records = []
    for raw_source, relative_target, expected_sha256 in RUNTIME_ASSETS:
        if not raw_source.is_file() or raw_source.is_symlink():
            raise FileNotFoundError(f"missing regular raw GRAB asset: {raw_source}")
        source_sha256 = _sha256(raw_source)
        if source_sha256 != expected_sha256:
            raise ValueError(f"raw GRAB asset hash mismatch: {raw_source}")
        target = source_root / relative_target
        copied = False
        if target.exists():
            if target.is_symlink() or not target.is_file() or _sha256(target) != expected_sha256:
                raise ValueError(f"refusing to replace unexpected runtime asset: {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(raw_source, target, follow_symlinks=False)
            copied = True
        records.append({"raw_source": str(raw_source), "target": str(target),
                        "sha256": expected_sha256, "materialized": copied})
    return records


def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True,
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()
    except subprocess.CalledProcessError:
        return None


def smoke_dexplore_args(*, motion_root: Path, output: Path, num_envs: int,
                        horizon_length: int, minibatch_size: int,
                        max_iterations: int, seed: int) -> list[str]:
    if min(num_envs, horizon_length, minibatch_size, max_iterations) < 1:
        raise ValueError("--num-envs, --horizon-length, --minibatch-size, and --max-iterations must be positive")
    return ["--task", "Dexplore_Inspire", "--cfg_env", "dexplore/data/cfg/inspire.yaml",
            "--cfg_train", "dexplore/data/cfg/train/rlg/inspire.yaml",
            "--motion_file", str(motion_root), "--output_path", str(output / "train"),
            "--headless", "--sim_device", "cuda:0", "--rl_device", "cuda:0",
            "--graphics_device_id", "0", "--num_envs", str(num_envs),
            "--horizon_length", str(horizon_length), "--minibatch_size", str(minibatch_size),
            "--max_iterations", str(max_iterations), "--seed", str(seed), "--horovod"]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", required=True, help="comma-separated physical GPU indices")
    parser.add_argument("--dexplore-run", type=Path, default=DEFAULT_DEXPLORE_RUN)
    parser.add_argument("--run-id", help="new output identity; required for --execute")
    parser.add_argument("--motion-root", type=Path, help="converted DExplore tensor root; required for --execute")
    parser.add_argument("--input-manifest", type=Path, help="motion-root provenance manifest; required for --execute")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--rank-bootstrap", type=Path, default=BOOTSTRAP,
                        help="Task-local rank bootstrap; defaults to the V1.20 DDP facade")
    parser.add_argument("--cm-distill-coef", type=float,
                        help="only valid with the V1.21 Cm-off bootstrap and must be exactly zero")
    parser.add_argument("--cm-reward-coef", type=float,
                        help="positive frozen-Cmv2 dense reward coefficient")
    parser.add_argument("--cmlite-reward-coef", type=float,
                        help="positive frozen-CmLite dense reward coefficient")
    parser.add_argument("--approach-reward-coef", type=float,
                        help="matched geometry potential shaping coefficient for both Cm arms")
    parser.add_argument("--held-lift-reward-coef", type=float,
                        help="matched contact-supported lift shaping coefficient for both Cm arms")
    parser.add_argument("--lift-progress-reward-coef", type=float, default=0.0)
    parser.add_argument("--grasp-link-reward-coef", type=float, default=0.0)
    parser.add_argument("--min-grasp-links", type=int, default=0)
    parser.add_argument("--contact-before", type=int,
                        help="optional near-contact curriculum: frames before first reference contact")
    parser.add_argument("--contact-after", type=int, default=0,
                        help="frames after first reference contact; requires --contact-before")
    parser.add_argument("--contact-fraction", type=float, default=1.0,
                        help="fraction of training resets placed near contact")
    parser.add_argument("--lift-fraction", type=float, default=0.0,
                        help="fraction of training resets placed near reference lift onset")
    parser.add_argument("--curriculum-anneal-start", type=int)
    parser.add_argument("--curriculum-anneal-end", type=int)
    parser.add_argument("--curriculum-backtrack-start", type=int)
    parser.add_argument("--curriculum-backtrack-end", type=int)
    parser.add_argument("--cm-reward-positive-only", action="store_true")
    parser.add_argument("--cmv2-checkpoint", type=Path)
    parser.add_argument("--cmv2-sha256")
    parser.add_argument("--cmlite-checkpoint", type=Path)
    parser.add_argument("--cmlite-sha256")
    parser.add_argument("--use-predicted-contact", action="store_true")
    parser.add_argument("--max-cmlite-gap-m", type=float)
    parser.add_argument("--save-frequency", type=int)
    parser.add_argument("--scratch-resume-checkpoint", type=Path)
    parser.add_argument("--scratch-resume-sha256")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--lr-schedule", choices=("constant", "adaptive"))
    parser.add_argument("--schedule-type", choices=("legacy", "standard", "standard_epoch"))
    parser.add_argument("--kl-threshold", type=float)
    parser.add_argument("--mini-epochs", type=int)
    parser.add_argument("--ppo-clip", type=float)
    parser.add_argument("--actual-epochs", type=int,
                        help="exact epoch budget for the V1.21 Cm-off bootstrap")
    parser.add_argument("--work-version", default="V1.21")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--horizon-length", type=int, default=64)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execute", action="store_true", help="create a new run directory and execute the smoke")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("dexplore_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.execute and args.dry_run:
        raise ValueError("--execute and --dry-run are mutually exclusive")
    gpus = parse_gpus(args.gpus)
    bootstrap = args.rank_bootstrap.resolve()
    bootstrap_args: list[str] = []
    modes = sum(value is not None for value in (
        args.cm_distill_coef, args.cm_reward_coef, args.cmlite_reward_coef))
    if modes > 1:
        raise ValueError("Cm-off, Cmv2-reward, and CmLite-reward modes are mutually exclusive")
    if args.cm_reward_positive_only and args.cm_reward_coef is None:
        raise ValueError("--cm-reward-positive-only requires --cm-reward-coef")
    if args.approach_reward_coef is not None and (not math.isfinite(args.approach_reward_coef)
                                                  or args.approach_reward_coef < 0):
        raise ValueError("--approach-reward-coef must be finite and nonnegative")
    if args.held_lift_reward_coef is not None and (not math.isfinite(args.held_lift_reward_coef)
                                                   or args.held_lift_reward_coef < 0):
        raise ValueError("--held-lift-reward-coef must be finite and nonnegative")
    for name, value in (("lift-progress", args.lift_progress_reward_coef),
                        ("grasp-link", args.grasp_link_reward_coef)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name}-reward-coef must be finite and nonnegative")
    if not 0 <= args.min_grasp_links <= 5:
        raise ValueError("--min-grasp-links must be in [0,5]")
    if args.save_frequency is not None and args.save_frequency < 1:
        raise ValueError("--save-frequency must be positive")
    if ((args.scratch_resume_checkpoint is None) != (args.scratch_resume_sha256 is None) or
            (args.scratch_resume_checkpoint is not None and
             (not (args.cmlite_reward_coef is not None or args.cm_distill_coef == 0.0) or
              not args.scratch_resume_checkpoint.is_file() or
              len(args.scratch_resume_sha256) != 64 or
              _sha256(args.scratch_resume_checkpoint) != args.scratch_resume_sha256))):
        raise ValueError("scratch resume requires CmLite or Cm-off mode and a matching checkpoint SHA256")
    if args.learning_rate is not None and (not (args.cmlite_reward_coef is not None or
             args.cm_distill_coef == 0.0) or
            not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        raise ValueError("learning-rate override requires CmLite or Cm-off mode and a positive finite value")
    ppo_overrides = (args.lr_schedule, args.schedule_type, args.kl_threshold,
                     args.mini_epochs, args.ppo_clip)
    if any(value is not None for value in ppo_overrides) and args.cmlite_reward_coef is None:
        raise ValueError("PPO stability overrides require CmLite mode")
    if args.lr_schedule == "adaptive" and args.kl_threshold is None:
        raise ValueError("adaptive learning-rate schedule requires --kl-threshold")
    if args.schedule_type is not None and args.lr_schedule is None:
        raise ValueError("schedule type requires --lr-schedule")
    if args.kl_threshold is not None and not (
            args.lr_schedule == "adaptive" and math.isfinite(args.kl_threshold) and
            args.kl_threshold > 0):
        raise ValueError("KL threshold requires adaptive schedule and must be positive")
    if args.mini_epochs is not None and args.mini_epochs < 1:
        raise ValueError("mini epochs must be positive")
    if args.ppo_clip is not None and not (
            math.isfinite(args.ppo_clip) and 0 < args.ppo_clip < 1):
        raise ValueError("PPO clip must be finite and in (0,1)")
    if args.approach_reward_coef is not None and modes == 0:
        raise ValueError("--approach-reward-coef requires a Cm-off or Cm-reward bootstrap")
    if args.held_lift_reward_coef is not None and modes == 0:
        raise ValueError("--held-lift-reward-coef requires a Cm-off or Cm-reward bootstrap")
    if ((args.contact_before is not None and args.contact_before < 0) or args.contact_after < 0 or
            args.contact_fraction < 0 or args.lift_fraction < 0 or
            not (0.0 < args.contact_fraction + args.lift_fraction <= 1.0) or
            (args.contact_before is None and args.contact_after)):
        raise ValueError("contact curriculum requires --contact-before >= 0 and --contact-after >= 0")
    if args.contact_before is not None and modes == 0:
        raise ValueError("contact curriculum requires a Cm-off or Cm-reward bootstrap")
    if ((args.curriculum_anneal_start is None) != (args.curriculum_anneal_end is None) or
            (args.curriculum_anneal_start is not None and
             (args.contact_before is None or args.curriculum_anneal_start < 0 or
              args.curriculum_anneal_end <= args.curriculum_anneal_start))):
        raise ValueError("curriculum annealing requires a valid epoch window")
    if ((args.curriculum_backtrack_start is None) != (args.curriculum_backtrack_end is None) or
            (args.curriculum_backtrack_start is not None and
             (args.contact_before is None or args.curriculum_backtrack_start < 0 or
              args.curriculum_backtrack_end <= args.curriculum_backtrack_start))):
        raise ValueError("curriculum backtracking requires a valid epoch window")
    if (args.curriculum_anneal_start is not None and
            args.curriculum_backtrack_start is not None):
        raise ValueError("curriculum annealing and backtracking are mutually exclusive")
    shared_shaping = ["--approach-reward-coef", str(args.approach_reward_coef or 0.0),
                      "--held-lift-reward-coef", str(args.held_lift_reward_coef or 0.0),
                      "--lift-progress-reward-coef", str(args.lift_progress_reward_coef),
                      "--grasp-link-reward-coef", str(args.grasp_link_reward_coef),
                      "--min-grasp-links", str(args.min_grasp_links)]
    if args.cm_distill_coef is not None:
        if bootstrap.name != "dexplore_cm_off_rank_bootstrap.py":
            raise ValueError("--cm-distill-coef requires dexplore_cm_off_rank_bootstrap.py")
        if args.cm_distill_coef != 0.0:
            raise ValueError("this launcher only supports Cm-off --cm-distill-coef 0")
        if args.actual_epochs is None or args.actual_epochs < 1:
            raise ValueError("Cm-off launcher requires positive --actual-epochs")
        bootstrap_args = ["--cm-distill-coef", "0", "--actual-epochs", str(args.actual_epochs)]
        bootstrap_args += shared_shaping
        if args.scratch_resume_checkpoint is not None:
            bootstrap_args += ["--scratch-resume-checkpoint",
                               str(args.scratch_resume_checkpoint.resolve()),
                               "--scratch-resume-sha256", args.scratch_resume_sha256]
        if args.learning_rate is not None:
            bootstrap_args += ["--learning-rate", str(args.learning_rate)]
    elif args.cm_reward_coef is not None:
        if bootstrap.name != "dexplore_cm_reward_rank_bootstrap.py":
            raise ValueError("--cm-reward-coef requires dexplore_cm_reward_rank_bootstrap.py")
        if not math.isfinite(args.cm_reward_coef) or args.cm_reward_coef <= 0 or args.actual_epochs is None or args.actual_epochs < 1:
            raise ValueError("Cm-reward launcher requires positive coefficient and actual epochs")
        if args.cmv2_checkpoint is None or not args.cmv2_checkpoint.is_file() or not args.cmv2_sha256:
            raise ValueError("Cm-reward launcher requires checkpoint and SHA256")
        bootstrap_args = ["--cm-reward-coef", str(args.cm_reward_coef),
                          "--cmv2-checkpoint", str(args.cmv2_checkpoint.resolve()),
                          "--cmv2-sha256", args.cmv2_sha256,
                          "--actual-epochs", str(args.actual_epochs)]
        # The older Cmv2 bootstrap predates lift-progress/link shaping.
        bootstrap_args += shared_shaping[:4]
        if args.cm_reward_positive_only:
            bootstrap_args.append("--cm-reward-positive-only")
    elif args.cmlite_reward_coef is not None:
        if bootstrap.name != "dexplore_cmlite_rank_bootstrap.py":
            raise ValueError("--cmlite-reward-coef requires dexplore_cmlite_rank_bootstrap.py")
        if (not math.isfinite(args.cmlite_reward_coef) or args.cmlite_reward_coef <= 0 or
                args.actual_epochs is None or args.actual_epochs < 1):
            raise ValueError("CmLite-reward launcher requires positive coefficient and actual epochs")
        if (args.cmlite_checkpoint is None or not args.cmlite_checkpoint.is_file() or
                not args.cmlite_sha256):
            raise ValueError("CmLite-reward launcher requires checkpoint and SHA256")
        if args.max_cmlite_gap_m is not None and (not args.use_predicted_contact or
                not math.isfinite(args.max_cmlite_gap_m) or args.max_cmlite_gap_m <= 0):
            raise ValueError("CmLite gap requires predicted contact and a positive finite threshold")
        bootstrap_args = ["--cmlite-reward-coef", str(args.cmlite_reward_coef),
                          "--cmlite-checkpoint", str(args.cmlite_checkpoint.resolve()),
                          "--cmlite-sha256", args.cmlite_sha256,
                          "--actual-epochs", str(args.actual_epochs)] + shared_shaping
        if args.use_predicted_contact:
            bootstrap_args.append("--use-predicted-contact")
        if args.max_cmlite_gap_m is not None:
            bootstrap_args += ["--max-cmlite-gap-m", str(args.max_cmlite_gap_m)]
        if args.scratch_resume_checkpoint is not None:
            bootstrap_args += ["--scratch-resume-checkpoint",
                               str(args.scratch_resume_checkpoint.resolve()),
                               "--scratch-resume-sha256", args.scratch_resume_sha256]
        if args.learning_rate is not None:
            bootstrap_args += ["--learning-rate", str(args.learning_rate)]
        if args.lr_schedule is not None:
            bootstrap_args += ["--lr-schedule", args.lr_schedule]
        if args.schedule_type is not None:
            bootstrap_args += ["--schedule-type", args.schedule_type]
        if args.kl_threshold is not None:
            bootstrap_args += ["--kl-threshold", str(args.kl_threshold)]
        if args.mini_epochs is not None:
            bootstrap_args += ["--mini-epochs", str(args.mini_epochs)]
        if args.ppo_clip is not None:
            bootstrap_args += ["--ppo-clip", str(args.ppo_clip)]
    elif args.actual_epochs is not None:
        raise ValueError("--actual-epochs requires a Cm mode")
    if args.contact_before is not None:
        bootstrap_args += ["--contact-before", str(args.contact_before),
                           "--contact-after", str(args.contact_after),
                           "--contact-fraction", str(args.contact_fraction),
                           "--lift-fraction", str(args.lift_fraction)]
    if args.curriculum_anneal_start is not None:
        bootstrap_args += ["--curriculum-anneal-start", str(args.curriculum_anneal_start),
                           "--curriculum-anneal-end", str(args.curriculum_anneal_end)]
    if args.curriculum_backtrack_start is not None:
        bootstrap_args += ["--curriculum-backtrack-start",
                           str(args.curriculum_backtrack_start),
                           "--curriculum-backtrack-end", str(args.curriculum_backtrack_end)]
    if args.save_frequency is not None:
        bootstrap_args += ["--save-frequency", str(args.save_frequency)]
    if args.execute:
        if not args.run_id or args.motion_root is None or args.input_manifest is None:
            raise ValueError("--execute requires --run-id, --motion-root, and --input-manifest")
        if args.dexplore_args:
            raise ValueError("--execute constructs the smoke arguments; do not append free-form DExplore arguments")
        motion_root = args.motion_root.resolve()
        manifest = args.input_manifest.resolve()
        if not motion_root.is_dir() or not manifest.is_file():
            raise FileNotFoundError("motion root or input manifest is missing")
        input_record = json.loads(manifest.read_text(encoding="utf-8"))
        if input_record.get("classification") != "reconstructed_baseline":
            raise ValueError("V1.20 smoke requires a reconstructed_baseline input manifest")
        output = (args.output_root / args.run_id).resolve()
        if output.exists():
            raise FileExistsError(f"refusing to overwrite output: {output}")
        runtime_assets = prepare_runtime_assets(args.dexplore_run)
        dexplore_args = smoke_dexplore_args(motion_root=motion_root, output=output,
                                            num_envs=args.num_envs,
                                            horizon_length=args.horizon_length,
                                            minibatch_size=args.minibatch_size,
                                            max_iterations=args.max_iterations, seed=args.seed)
        if args.scratch_resume_checkpoint is not None:
            dexplore_args += ["--resume", "1", "--checkpoint",
                              str(args.scratch_resume_checkpoint.resolve())]
    else:
        dexplore_args = list(args.dexplore_args)
        if dexplore_args[:1] == ["--"]:
            dexplore_args = dexplore_args[1:]
        output = None
        input_record = None
        manifest = None
        runtime_assets = []
    command = torchrun_command(gpus=gpus, dexplore_run=args.dexplore_run.resolve(), dexplore_args=dexplore_args,
                               bootstrap=bootstrap, bootstrap_args=bootstrap_args)
    source_root = args.dexplore_run.resolve().parents[1]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    environment["PYTHONPATH"] = ":".join((str(Path(__file__).resolve().parent),
                                            str(REPOSITORY_ROOT), environment.get("PYTHONPATH", "")))
    print("CUDA_VISIBLE_DEVICES=" + environment["CUDA_VISIBLE_DEVICES"])
    print(" ".join(command))
    if args.execute:
        output.mkdir(parents=True)
        runtime = {"python": sys.executable, "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"],
                   "physical_gpus": list(gpus), "backend": "torch.distributed:nccl",
                   "horovod_package": False, "legacy_cli_facade": True}
        config = {"command": command, "runtime": runtime, "num_envs_per_rank": args.num_envs,
                  "horizon_length": args.horizon_length, "minibatch_size": args.minibatch_size,
                  "max_iterations": args.max_iterations, "seed": args.seed,
                  "rank_bootstrap": str(bootstrap), "cm_distill_coef": args.cm_distill_coef,
                  "cm_reward_coef": args.cm_reward_coef,
                  "cmlite_reward_coef": args.cmlite_reward_coef,
                  "approach_reward_coef": args.approach_reward_coef or 0.0,
                  "held_lift_reward_coef": args.held_lift_reward_coef or 0.0,
                  "lift_progress_reward_coef": args.lift_progress_reward_coef,
                  "grasp_link_reward_coef": args.grasp_link_reward_coef,
                  "min_grasp_links": args.min_grasp_links,
                  "contact_before": args.contact_before, "contact_after": args.contact_after,
                  "contact_fraction": args.contact_fraction,
                  "lift_fraction": args.lift_fraction,
                  "curriculum_anneal_start": args.curriculum_anneal_start,
                  "curriculum_anneal_end": args.curriculum_anneal_end,
                  "curriculum_backtrack_start": args.curriculum_backtrack_start,
                  "curriculum_backtrack_end": args.curriculum_backtrack_end,
                  "cm_reward_positive_only": args.cm_reward_positive_only,
                  "cmv2_checkpoint": str(args.cmv2_checkpoint.resolve()) if args.cmv2_checkpoint else None,
                  "cmv2_sha256": args.cmv2_sha256,
                  "cmlite_checkpoint": str(args.cmlite_checkpoint.resolve()) if args.cmlite_checkpoint else None,
                  "cmlite_sha256": args.cmlite_sha256,
                  "use_predicted_contact": args.use_predicted_contact,
                  "max_cmlite_gap_m": args.max_cmlite_gap_m,
                  "save_frequency": args.save_frequency,
                  "scratch_resume_checkpoint": str(args.scratch_resume_checkpoint.resolve()) if args.scratch_resume_checkpoint else None,
                  "scratch_resume_sha256": args.scratch_resume_sha256,
                  "learning_rate": args.learning_rate,
                  "lr_schedule": args.lr_schedule,
                  "schedule_type": args.schedule_type,
                  "kl_threshold": args.kl_threshold,
                  "mini_epochs": args.mini_epochs,
                  "ppo_clip": args.ppo_clip,
                  "actual_epochs": args.actual_epochs,
                  "motion_root": str(args.motion_root.resolve()), "input_manifest": str(manifest),
                  "runtime_assets": runtime_assets}
        _write_json(output / "config.json", config)
        manifest_path = output / "run_manifest.json"
        run_manifest = {
            "manifest_schema": "ref2dex.run.v1", "created_at": _timestamp(),
            "task": "CmResidual", "work_version": args.work_version, "run_id": args.run_id,
            "git_commit": _git_commit(REPOSITORY_ROOT), "external_source_commit": _git_commit(source_root),
            "run_status": "STARTED", "command": command, "runtime": runtime,
            "input_manifest": str(manifest), "input_classification": input_record["classification"],
            "runtime_assets": runtime_assets,
            "output_dir": str(output), "config": str(output / "config.json"),
            "train_log": str(output / "train.log"), "seed": args.seed,
        }
        _write_json(manifest_path, run_manifest)
        try:
            with (output / "train.log").open("w", encoding="utf-8", buffering=1) as log:
                subprocess.run(command, cwd=source_root, env=environment, check=True,
                               stdout=log, stderr=subprocess.STDOUT)
            events = sorted((output / "train").rglob("events.out.tfevents*"))
            checkpoints = sorted((output / "train").rglob("*.pth"))
            if not events or not checkpoints:
                raise RuntimeError(f"smoke is missing events={len(events)} checkpoints={len(checkpoints)}")
            run_manifest.update({"run_status": "COMPLETED", "completed_at": _timestamp(),
                                 "tensorboard": str(events[-1]), "checkpoint": str(checkpoints[-1])})
            _write_json(manifest_path, run_manifest)
        except BaseException as error:
            run_manifest.update({"run_status": "FAILED", "completed_at": _timestamp(),
                                 "failure_reason": f"{type(error).__name__}: {error}"})
            _write_json(manifest_path, run_manifest)
            raise
    elif not args.dry_run:
        subprocess.run(command, cwd=source_root, env=environment, check=True)


if __name__ == "__main__":
    main()

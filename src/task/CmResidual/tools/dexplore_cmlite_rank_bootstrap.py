"""Install the frozen CmLite reward agent before launching DExplore."""
from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import runpy
import sys

import numpy as np

import dexplore_ddp_rank_bootstrap as base


if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int


def parse_cmlite_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cmlite-reward-coef", type=float, required=True)
    parser.add_argument("--approach-reward-coef", type=float, default=0.0)
    parser.add_argument("--held-lift-reward-coef", type=float, default=0.0)
    parser.add_argument("--lift-progress-reward-coef", type=float, default=0.0)
    parser.add_argument("--grasp-link-reward-coef", type=float, default=0.0)
    parser.add_argument("--min-grasp-links", type=int, default=0)
    parser.add_argument("--cmlite-checkpoint", type=Path, required=True)
    parser.add_argument("--cmlite-sha256", required=True)
    parser.add_argument("--actual-epochs", type=int, required=True)
    parser.add_argument("--contact-before", type=int)
    parser.add_argument("--contact-after", type=int, default=0)
    parser.add_argument("--contact-fraction", type=float, default=1.0)
    parser.add_argument("--lift-fraction", type=float, default=0.0)
    parser.add_argument("--curriculum-anneal-start", type=int)
    parser.add_argument("--curriculum-anneal-end", type=int)
    parser.add_argument("--curriculum-backtrack-start", type=int)
    parser.add_argument("--curriculum-backtrack-end", type=int)
    parser.add_argument("--allow-negative-cmlite-reward", action="store_true")
    parser.add_argument("--use-predicted-contact", action="store_true")
    parser.add_argument("--max-cmlite-gap-m", type=float)
    parser.add_argument("--save-frequency", type=int)
    parser.add_argument("--scratch-resume-checkpoint", type=Path)
    parser.add_argument("--scratch-resume-sha256")
    parser.add_argument("--learning-rate", type=float)
    args, passthrough = parser.parse_known_args(argv)
    coefficients = (args.cmlite_reward_coef, args.approach_reward_coef,
                    args.held_lift_reward_coef, args.lift_progress_reward_coef,
                    args.grasp_link_reward_coef)
    if (not all(math.isfinite(value) and value >= 0 for value in coefficients) or
            args.cmlite_reward_coef == 0 or args.actual_epochs < 1):
        raise ValueError("invalid CmLite reward coefficients or epoch budget")
    if not args.cmlite_checkpoint.is_file() or len(args.cmlite_sha256) != 64:
        raise ValueError("CmLite requires an existing checkpoint and SHA256")
    if ((args.contact_before is not None and args.contact_before < 0) or
            args.contact_after < 0 or args.contact_fraction < 0 or args.lift_fraction < 0 or
            not (0 < args.contact_fraction + args.lift_fraction <= 1)):
        raise ValueError("invalid reset curriculum")
    if args.contact_before is None and args.contact_after:
        raise ValueError("--contact-after requires --contact-before")
    if ((args.curriculum_anneal_start is None) != (args.curriculum_anneal_end is None) or
            (args.curriculum_anneal_start is not None and
             (args.contact_before is None or args.curriculum_anneal_start < 0 or
              args.curriculum_anneal_end <= args.curriculum_anneal_start))):
        raise ValueError("annealing requires a valid contact curriculum and epoch window")
    if ((args.curriculum_backtrack_start is None) != (args.curriculum_backtrack_end is None) or
            (args.curriculum_backtrack_start is not None and
             (args.contact_before is None or args.curriculum_backtrack_start < 0 or
              args.curriculum_backtrack_end <= args.curriculum_backtrack_start))):
        raise ValueError("backtracking requires a valid contact curriculum and epoch window")
    if (args.curriculum_anneal_start is not None and
            args.curriculum_backtrack_start is not None):
        raise ValueError("curriculum annealing and backtracking are mutually exclusive")
    if ((args.scratch_resume_checkpoint is None) != (args.scratch_resume_sha256 is None) or
            (args.scratch_resume_checkpoint is not None and
             (not args.scratch_resume_checkpoint.is_file() or
              len(args.scratch_resume_sha256) != 64 or
              hashlib.sha256(args.scratch_resume_checkpoint.read_bytes()).hexdigest() !=
              args.scratch_resume_sha256))):
        raise ValueError("scratch resume requires an existing checkpoint and matching SHA256")
    if args.learning_rate is not None and not (
            math.isfinite(args.learning_rate) and args.learning_rate > 0):
        raise ValueError("learning rate override must be finite and positive")
    if args.save_frequency is not None and args.save_frequency < 1:
        raise ValueError("save frequency must be positive")
    if args.max_cmlite_gap_m is not None and not (
            math.isfinite(args.max_cmlite_gap_m) and args.max_cmlite_gap_m > 0):
        raise ValueError("maximum CmLite gap must be finite and positive")
    if args.max_cmlite_gap_m is not None and not args.use_predicted_contact:
        raise ValueError("maximum CmLite gap requires predicted contact")
    if not 0 <= args.min_grasp_links <= 5:
        raise ValueError("minimum grasp links must be in [0,5]")
    return args, passthrough


def main(argv=None):
    args, passthrough = parse_cmlite_args(argv)
    os.environ.update({
        "REF2DEX_CMLITE_REWARD_COEF": str(args.cmlite_reward_coef),
        "REF2DEX_APPROACH_REWARD_COEF": str(args.approach_reward_coef),
        "REF2DEX_HELD_LIFT_REWARD_COEF": str(args.held_lift_reward_coef),
        "REF2DEX_LIFT_PROGRESS_REWARD_COEF": str(args.lift_progress_reward_coef),
        "REF2DEX_GRASP_LINK_REWARD_COEF": str(args.grasp_link_reward_coef),
        "REF2DEX_MIN_GRASP_LINKS": str(args.min_grasp_links),
        "REF2DEX_CMLITE_CHECKPOINT": str(args.cmlite_checkpoint.resolve()),
        "REF2DEX_CMLITE_SHA256": args.cmlite_sha256,
        "REF2DEX_CMLITE_REWARD_POSITIVE_ONLY": "0" if args.allow_negative_cmlite_reward else "1",
        "REF2DEX_CMLITE_USE_PREDICTED_CONTACT": "1" if args.use_predicted_contact else "0",
        "REF2DEX_ACTUAL_EPOCH_BUDGET": str(args.actual_epochs),
        "REF2DEX_SCRATCH_POLICY": "1",
    })
    if args.contact_before is not None:
        os.environ.update({
            "REF2DEX_CONTACT_RESET_BEFORE": str(args.contact_before),
            "REF2DEX_CONTACT_RESET_AFTER": str(args.contact_after),
            "REF2DEX_CONTACT_RESET_FRACTION": str(args.contact_fraction),
            "REF2DEX_LIFT_RESET_FRACTION": str(args.lift_fraction),
        })
    if args.curriculum_anneal_start is not None:
        os.environ["REF2DEX_CURRICULUM_ANNEAL_START"] = str(args.curriculum_anneal_start)
        os.environ["REF2DEX_CURRICULUM_ANNEAL_END"] = str(args.curriculum_anneal_end)
    if args.curriculum_backtrack_start is not None:
        os.environ["REF2DEX_CURRICULUM_BACKTRACK_START"] = str(
            args.curriculum_backtrack_start)
        os.environ["REF2DEX_CURRICULUM_BACKTRACK_END"] = str(
            args.curriculum_backtrack_end)
    if args.scratch_resume_checkpoint is not None:
        os.environ["REF2DEX_SCRATCH_RESUME_CHECKPOINT"] = str(
            args.scratch_resume_checkpoint.resolve())
        os.environ["REF2DEX_SCRATCH_RESUME_SHA256"] = args.scratch_resume_sha256
    if args.learning_rate is not None:
        os.environ["REF2DEX_LEARNING_RATE"] = str(args.learning_rate)
    if args.save_frequency is not None:
        os.environ["REF2DEX_SAVE_FREQUENCY"] = str(args.save_frequency)
    if args.max_cmlite_gap_m is not None:
        os.environ["REF2DEX_CMLITE_MAX_GAP_M"] = str(args.max_cmlite_gap_m)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dexplore-run", default=os.environ.get(
        "REF2DEX_DEXPLORE_RUN", base.DEFAULT_DEXPLORE_RUN))
    launch, dexplore_args = parser.parse_known_args(passthrough)
    dexplore_run = Path(launch.dexplore_run).resolve()
    if not dexplore_run.is_file():
        raise FileNotFoundError(f"missing DExplore entrypoint: {dexplore_run}")

    from isaacgym import torch_utils
    local_rank = int(os.environ["LOCAL_RANK"])
    torch_utils.torch.cuda.set_device(local_rank)
    original_to_torch = torch_utils.to_torch

    def rank_local_to_torch(x, dtype=torch_utils.torch.float, device=None, requires_grad=False):
        return original_to_torch(x, dtype=dtype, device=device or f"cuda:{local_rank}",
                                 requires_grad=requires_grad)

    torch_utils.to_torch = rank_local_to_torch
    from dexplore_ddp_compat import cleanup, install_horovod_facade
    install_horovod_facade()
    sys.path.insert(0, str(dexplore_run.parent))
    base._patch_synchronized_shutdown()
    base._install_contact_curriculum_from_env()
    import learning.dexplore_agent as agent_module
    from src.task.CmResidual.dexplore_cmlite_agent import DExploreCmLiteAgent
    agent_module.DexploreAgent = DExploreCmLiteAgent
    sys.argv = [str(dexplore_run)] + dexplore_args
    try:
        runpy.run_path(str(dexplore_run), run_name="__main__")
    finally:
        cleanup()


if __name__ == "__main__":
    main()

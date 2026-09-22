"""Install the Task-local frozen-Cmv2 reward agent before launching DExplore."""
from __future__ import annotations

import argparse
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


def parse_cm_reward_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cm-reward-coef", type=float, required=True)
    parser.add_argument("--approach-reward-coef", type=float, default=0.0)
    parser.add_argument("--held-lift-reward-coef", type=float, default=0.0)
    parser.add_argument("--cmv2-checkpoint", type=Path, required=True)
    parser.add_argument("--cmv2-sha256", required=True)
    parser.add_argument("--actual-epochs", type=int, required=True)
    parser.add_argument("--contact-before", type=int)
    parser.add_argument("--contact-after", type=int, default=0)
    parser.add_argument("--contact-fraction", type=float, default=1.0)
    parser.add_argument("--lift-fraction", type=float, default=0.0)
    parser.add_argument("--curriculum-anneal-start", type=int)
    parser.add_argument("--curriculum-anneal-end", type=int)
    parser.add_argument("--cm-reward-positive-only", action="store_true")
    args, passthrough = parser.parse_known_args(argv)
    if not math.isfinite(args.cm_reward_coef) or args.cm_reward_coef <= 0 or args.actual_epochs < 1:
        raise ValueError("Cm reward coefficient and epoch budget must be positive")
    if not math.isfinite(args.approach_reward_coef) or args.approach_reward_coef < 0:
        raise ValueError("approach reward coefficient must be finite and nonnegative")
    if not math.isfinite(args.held_lift_reward_coef) or args.held_lift_reward_coef < 0:
        raise ValueError("held-lift reward coefficient must be finite and nonnegative")
    if not args.cmv2_checkpoint.is_file() or len(args.cmv2_sha256) != 64:
        raise ValueError("Cm reward requires an existing checkpoint and explicit SHA256")
    if ((args.contact_before is not None and args.contact_before < 0) or args.contact_after < 0 or
            args.contact_fraction < 0 or args.lift_fraction < 0 or
            not (0.0 < args.contact_fraction + args.lift_fraction <= 1.0)):
        raise ValueError("contact curriculum windows must be nonnegative")
    if args.contact_before is None and args.contact_after:
        raise ValueError("--contact-after requires --contact-before")
    if ((args.curriculum_anneal_start is None) != (args.curriculum_anneal_end is None) or
            (args.curriculum_anneal_start is not None and
             (args.contact_before is None or args.curriculum_anneal_start < 0 or
              args.curriculum_anneal_end <= args.curriculum_anneal_start))):
        raise ValueError("annealing requires a valid contact curriculum and epoch window")
    return args, passthrough


def main(argv=None) -> None:
    args, passthrough = parse_cm_reward_args(argv)
    os.environ.update({
        "REF2DEX_CM_REWARD_COEF": str(args.cm_reward_coef),
        "REF2DEX_APPROACH_REWARD_COEF": str(args.approach_reward_coef),
        "REF2DEX_HELD_LIFT_REWARD_COEF": str(args.held_lift_reward_coef),
        "REF2DEX_CMV2_CHECKPOINT": str(args.cmv2_checkpoint.resolve()),
        "REF2DEX_CMV2_SHA256": args.cmv2_sha256,
        "REF2DEX_ACTUAL_EPOCH_BUDGET": str(args.actual_epochs),
        "REF2DEX_SCRATCH_POLICY": "1",
        "REF2DEX_CM_REWARD_POSITIVE_ONLY": "1" if args.cm_reward_positive_only else "0",
    })
    if args.contact_before is not None:
        os.environ["REF2DEX_CONTACT_RESET_BEFORE"] = str(args.contact_before)
        os.environ["REF2DEX_CONTACT_RESET_AFTER"] = str(args.contact_after)
        os.environ["REF2DEX_CONTACT_RESET_FRACTION"] = str(args.contact_fraction)
        os.environ["REF2DEX_LIFT_RESET_FRACTION"] = str(args.lift_fraction)
    if args.curriculum_anneal_start is not None:
        os.environ["REF2DEX_CURRICULUM_ANNEAL_START"] = str(args.curriculum_anneal_start)
        os.environ["REF2DEX_CURRICULUM_ANNEAL_END"] = str(args.curriculum_anneal_end)
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
    # rl_games initializes the process group during A2CBase construction.
    install_horovod_facade()
    sys.path.insert(0, str(dexplore_run.parent))
    base._patch_synchronized_shutdown()
    base._install_contact_curriculum_from_env()
    import learning.dexplore_agent as agent_module
    from src.task.CmResidual.dexplore_cm_reward_agent import DExploreCmRewardAgent
    agent_module.DexploreAgent = DExploreCmRewardAgent
    sys.argv = [str(dexplore_run)] + dexplore_args
    try:
        runpy.run_path(str(dexplore_run), run_name="__main__")
    finally:
        cleanup()


if __name__ == "__main__":
    main()

"""Install the frozen CmLite reward agent before launching DExplore."""
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


def parse_cmlite_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cmlite-reward-coef", type=float, required=True)
    parser.add_argument("--approach-reward-coef", type=float, default=0.0)
    parser.add_argument("--held-lift-reward-coef", type=float, default=0.0)
    parser.add_argument("--cmlite-checkpoint", type=Path, required=True)
    parser.add_argument("--cmlite-sha256", required=True)
    parser.add_argument("--actual-epochs", type=int, required=True)
    parser.add_argument("--contact-before", type=int)
    parser.add_argument("--contact-after", type=int, default=0)
    parser.add_argument("--contact-fraction", type=float, default=1.0)
    parser.add_argument("--lift-fraction", type=float, default=0.0)
    parser.add_argument("--allow-negative-cmlite-reward", action="store_true")
    args, passthrough = parser.parse_known_args(argv)
    coefficients = (args.cmlite_reward_coef, args.approach_reward_coef,
                    args.held_lift_reward_coef)
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
    return args, passthrough


def main(argv=None):
    args, passthrough = parse_cmlite_args(argv)
    os.environ.update({
        "REF2DEX_CMLITE_REWARD_COEF": str(args.cmlite_reward_coef),
        "REF2DEX_APPROACH_REWARD_COEF": str(args.approach_reward_coef),
        "REF2DEX_HELD_LIFT_REWARD_COEF": str(args.held_lift_reward_coef),
        "REF2DEX_CMLITE_CHECKPOINT": str(args.cmlite_checkpoint.resolve()),
        "REF2DEX_CMLITE_SHA256": args.cmlite_sha256,
        "REF2DEX_CMLITE_REWARD_POSITIVE_ONLY": "0" if args.allow_negative_cmlite_reward else "1",
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

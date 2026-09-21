"""CmResidual V1.21 Cm-off entrypoint for the unmodified DExplore runner.

The explicit zero coefficient is intentionally handled before importing the
existing DExplore bootstrap.  This provides a stable, testable identity for
the future Cm runner while making accidental Cm construction impossible in the
Cm-off parity branch.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import dexplore_ddp_rank_bootstrap


def parse_cm_off_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cm-distill-coef", type=float, required=True)
    parser.add_argument("--approach-reward-coef", type=float, default=0.0)
    parser.add_argument("--actual-epochs", type=int, required=True)
    parser.add_argument("--contact-before", type=int)
    parser.add_argument("--contact-after", type=int, default=0)
    parser.add_argument("--contact-fraction", type=float, default=1.0)
    parser.add_argument("--lift-fraction", type=float, default=0.0)
    args, passthrough = parser.parse_known_args(argv)
    if args.cm_distill_coef != 0.0:
        raise ValueError("Cm-off bootstrap only accepts --cm-distill-coef 0")
    if args.actual_epochs < 1:
        raise ValueError("Cm-off bootstrap requires a positive --actual-epochs budget")
    if not math.isfinite(args.approach_reward_coef) or args.approach_reward_coef < 0:
        raise ValueError("approach reward coefficient must be finite and nonnegative")
    if ((args.contact_before is not None and args.contact_before < 0) or args.contact_after < 0 or
            args.contact_fraction < 0 or args.lift_fraction < 0 or
            not (0.0 < args.contact_fraction + args.lift_fraction <= 1.0)):
        raise ValueError("contact curriculum windows must be nonnegative")
    if args.contact_before is None and args.contact_after:
        raise ValueError("--contact-after requires --contact-before")
    return args, passthrough


def _assert_cm_not_imported() -> None:
    forbidden = tuple(name for name in sys.modules if "cmv2" in name.lower() or "cm_residual" in name.lower())
    if forbidden:
        raise RuntimeError(f"Cm-off bootstrap refuses pre-imported Cm modules: {forbidden}")


def main(argv=None) -> None:
    args, passthrough = parse_cm_off_args(argv)
    _assert_cm_not_imported()
    os.environ["REF2DEX_ACTUAL_EPOCH_BUDGET"] = str(args.actual_epochs)
    os.environ["REF2DEX_SCRATCH_POLICY"] = "1"
    os.environ["REF2DEX_APPROACH_REWARD_COEF"] = str(args.approach_reward_coef)
    if args.contact_before is not None:
        os.environ["REF2DEX_CONTACT_RESET_BEFORE"] = str(args.contact_before)
        os.environ["REF2DEX_CONTACT_RESET_AFTER"] = str(args.contact_after)
        os.environ["REF2DEX_CONTACT_RESET_FRACTION"] = str(args.contact_fraction)
        os.environ["REF2DEX_LIFT_RESET_FRACTION"] = str(args.lift_fraction)
    if args.approach_reward_coef:
        dexplore_ddp_rank_bootstrap.main(
            passthrough,
            agent_class="src.task.CmResidual.dexplore_approach_agent:DExploreApproachAgent")
    else:
        dexplore_ddp_rank_bootstrap.main(passthrough)


if __name__ == "__main__":
    main()

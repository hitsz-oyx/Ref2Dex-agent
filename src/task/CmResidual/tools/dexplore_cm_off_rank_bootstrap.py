"""CmResidual V1.21 Cm-off entrypoint for the unmodified DExplore runner.

The explicit zero coefficient is intentionally handled before importing the
existing DExplore bootstrap.  This provides a stable, testable identity for
the future Cm runner while making accidental Cm construction impossible in the
Cm-off parity branch.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import sys

import dexplore_ddp_rank_bootstrap


def parse_cm_off_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cm-distill-coef", type=float, required=True)
    parser.add_argument("--approach-reward-coef", type=float, default=0.0)
    parser.add_argument("--held-lift-reward-coef", type=float, default=0.0)
    parser.add_argument("--lift-progress-reward-coef", type=float, default=0.0)
    parser.add_argument("--grasp-link-reward-coef", type=float, default=0.0)
    parser.add_argument("--min-grasp-links", type=int, default=0)
    parser.add_argument("--actual-epochs", type=int, required=True)
    parser.add_argument("--contact-before", type=int)
    parser.add_argument("--contact-after", type=int, default=0)
    parser.add_argument("--contact-fraction", type=float, default=1.0)
    parser.add_argument("--lift-fraction", type=float, default=0.0)
    parser.add_argument("--curriculum-anneal-start", type=int)
    parser.add_argument("--curriculum-anneal-end", type=int)
    parser.add_argument("--curriculum-backtrack-start", type=int)
    parser.add_argument("--curriculum-backtrack-end", type=int)
    parser.add_argument("--save-frequency", type=int)
    parser.add_argument("--scratch-resume-checkpoint", type=Path)
    parser.add_argument("--scratch-resume-sha256")
    parser.add_argument("--learning-rate", type=float)
    args, passthrough = parser.parse_known_args(argv)
    if args.cm_distill_coef != 0.0:
        raise ValueError("Cm-off bootstrap only accepts --cm-distill-coef 0")
    if args.actual_epochs < 1:
        raise ValueError("Cm-off bootstrap requires a positive --actual-epochs budget")
    if not math.isfinite(args.approach_reward_coef) or args.approach_reward_coef < 0:
        raise ValueError("approach reward coefficient must be finite and nonnegative")
    if not math.isfinite(args.held_lift_reward_coef) or args.held_lift_reward_coef < 0:
        raise ValueError("held-lift reward coefficient must be finite and nonnegative")
    if not math.isfinite(args.lift_progress_reward_coef) or args.lift_progress_reward_coef < 0:
        raise ValueError("lift-progress reward coefficient must be finite and nonnegative")
    if not math.isfinite(args.grasp_link_reward_coef) or args.grasp_link_reward_coef < 0:
        raise ValueError("grasp-link reward coefficient must be finite and nonnegative")
    if not 0 <= args.min_grasp_links <= 5:
        raise ValueError("minimum grasp links must be in [0,5]")
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
    if ((args.curriculum_backtrack_start is None) != (args.curriculum_backtrack_end is None) or
            (args.curriculum_backtrack_start is not None and
             (args.contact_before is None or args.curriculum_backtrack_start < 0 or
              args.curriculum_backtrack_end <= args.curriculum_backtrack_start))):
        raise ValueError("backtracking requires a valid contact curriculum and epoch window")
    if (args.curriculum_anneal_start is not None and
            args.curriculum_backtrack_start is not None):
        raise ValueError("annealing and backtracking are mutually exclusive")
    if args.save_frequency is not None and args.save_frequency < 1:
        raise ValueError("save frequency must be positive")
    if ((args.scratch_resume_checkpoint is None) != (args.scratch_resume_sha256 is None) or
            (args.scratch_resume_checkpoint is not None and
             (not args.scratch_resume_checkpoint.is_file() or
              len(args.scratch_resume_sha256) != 64 or
              hashlib.sha256(args.scratch_resume_checkpoint.read_bytes()).hexdigest() !=
              args.scratch_resume_sha256))):
        raise ValueError("scratch resume requires an existing checkpoint and matching SHA256")
    if args.learning_rate is not None and not (math.isfinite(args.learning_rate) and
                                                args.learning_rate > 0):
        raise ValueError("learning rate must be finite and positive")
    return args, passthrough


def _assert_cm_not_imported() -> None:
    forbidden = tuple(name for name in sys.modules if "cmv2" in name.lower() or "cm_residual" in name.lower())
    if forbidden:
        raise RuntimeError(f"Cm-off bootstrap refuses pre-imported Cm modules: {forbidden}")


def main(argv=None, *, agent_class: str | None = None) -> None:
    args, passthrough = parse_cm_off_args(argv)
    _assert_cm_not_imported()
    os.environ["REF2DEX_ACTUAL_EPOCH_BUDGET"] = str(args.actual_epochs)
    os.environ["REF2DEX_SCRATCH_POLICY"] = "1"
    os.environ["REF2DEX_APPROACH_REWARD_COEF"] = str(args.approach_reward_coef)
    os.environ["REF2DEX_HELD_LIFT_REWARD_COEF"] = str(args.held_lift_reward_coef)
    os.environ["REF2DEX_LIFT_PROGRESS_REWARD_COEF"] = str(args.lift_progress_reward_coef)
    os.environ["REF2DEX_GRASP_LINK_REWARD_COEF"] = str(args.grasp_link_reward_coef)
    os.environ["REF2DEX_MIN_GRASP_LINKS"] = str(args.min_grasp_links)
    if args.contact_before is not None:
        os.environ["REF2DEX_CONTACT_RESET_BEFORE"] = str(args.contact_before)
        os.environ["REF2DEX_CONTACT_RESET_AFTER"] = str(args.contact_after)
        os.environ["REF2DEX_CONTACT_RESET_FRACTION"] = str(args.contact_fraction)
        os.environ["REF2DEX_LIFT_RESET_FRACTION"] = str(args.lift_fraction)
    if args.curriculum_anneal_start is not None:
        os.environ["REF2DEX_CURRICULUM_ANNEAL_START"] = str(args.curriculum_anneal_start)
        os.environ["REF2DEX_CURRICULUM_ANNEAL_END"] = str(args.curriculum_anneal_end)
    if args.curriculum_backtrack_start is not None:
        os.environ["REF2DEX_CURRICULUM_BACKTRACK_START"] = str(
            args.curriculum_backtrack_start)
        os.environ["REF2DEX_CURRICULUM_BACKTRACK_END"] = str(
            args.curriculum_backtrack_end)
    if args.save_frequency is not None:
        os.environ["REF2DEX_SAVE_FREQUENCY"] = str(args.save_frequency)
    if args.scratch_resume_checkpoint is not None:
        os.environ["REF2DEX_SCRATCH_RESUME_CHECKPOINT"] = str(
            args.scratch_resume_checkpoint.resolve())
        os.environ["REF2DEX_SCRATCH_RESUME_SHA256"] = args.scratch_resume_sha256
    if args.learning_rate is not None:
        os.environ["REF2DEX_LEARNING_RATE"] = str(args.learning_rate)
    if agent_class is not None:
        dexplore_ddp_rank_bootstrap.main(passthrough, agent_class=agent_class)
    elif (args.approach_reward_coef or args.held_lift_reward_coef or
            args.lift_progress_reward_coef or args.grasp_link_reward_coef):
        dexplore_ddp_rank_bootstrap.main(
            passthrough,
            agent_class="src.task.CmResidual.dexplore_approach_agent:DExploreApproachAgent")
    else:
        dexplore_ddp_rank_bootstrap.main(passthrough)


if __name__ == "__main__":
    main()

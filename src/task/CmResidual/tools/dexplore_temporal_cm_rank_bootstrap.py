"""Install history-conditioned Cm credit before launching DExplore."""
from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path

import dexplore_ddp_rank_bootstrap as base


def parse_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--temporal-cm-reward-coef", type=float, required=True)
    parser.add_argument("--temporal-cm-reward-scale-mm", type=float, default=20.0)
    parser.add_argument("--temporal-cm-checkpoint", type=Path, required=True)
    parser.add_argument("--temporal-cm-sha256", required=True)
    parser.add_argument("--allow-negative-temporal-reward", action="store_true")
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
    coefficients = (
        args.temporal_cm_reward_coef, args.approach_reward_coef,
        args.held_lift_reward_coef, args.lift_progress_reward_coef,
        args.grasp_link_reward_coef,
    )
    if (not all(math.isfinite(value) and value >= 0 for value in coefficients) or
            args.temporal_cm_reward_coef <= 0):
        raise ValueError("invalid temporal Cm or shared reward coefficient")
    if not math.isfinite(args.temporal_cm_reward_scale_mm) or args.temporal_cm_reward_scale_mm <= 0:
        raise ValueError("temporal Cm reward scale must be finite and positive")
    if not args.temporal_cm_checkpoint.is_file() or len(args.temporal_cm_sha256) != 64:
        raise ValueError("temporal Cm requires an existing checkpoint and SHA256")
    actual = hashlib.sha256(args.temporal_cm_checkpoint.read_bytes()).hexdigest()
    if actual != args.temporal_cm_sha256:
        raise ValueError(f"temporal Cm checkpoint SHA256 mismatch: {actual}")
    if args.actual_epochs < 1 or not 0 <= args.min_grasp_links <= 5:
        raise ValueError("invalid epoch budget or grasp-link threshold")
    if ((args.scratch_resume_checkpoint is None) != (args.scratch_resume_sha256 is None) or
            (args.scratch_resume_checkpoint is not None and
             (not args.scratch_resume_checkpoint.is_file() or
              len(args.scratch_resume_sha256) != 64 or
              hashlib.sha256(args.scratch_resume_checkpoint.read_bytes()).hexdigest() !=
              args.scratch_resume_sha256))):
        raise ValueError("scratch resume requires an existing checkpoint and matching SHA256")
    return args, passthrough


def main(argv=None):
    args, passthrough = parse_args(argv)
    os.environ.update({
        "REF2DEX_TEMPORAL_CM_REWARD_COEF": str(args.temporal_cm_reward_coef),
        "REF2DEX_TEMPORAL_CM_REWARD_SCALE_MM": str(args.temporal_cm_reward_scale_mm),
        "REF2DEX_TEMPORAL_CM_CHECKPOINT": str(args.temporal_cm_checkpoint.resolve()),
        "REF2DEX_TEMPORAL_CM_SHA256": args.temporal_cm_sha256,
        "REF2DEX_TEMPORAL_CM_POSITIVE_ONLY": "0" if args.allow_negative_temporal_reward else "1",
        "REF2DEX_ACTUAL_EPOCH_BUDGET": str(args.actual_epochs),
        "REF2DEX_SCRATCH_POLICY": "1",
        "REF2DEX_APPROACH_REWARD_COEF": str(args.approach_reward_coef),
        "REF2DEX_HELD_LIFT_REWARD_COEF": str(args.held_lift_reward_coef),
        "REF2DEX_LIFT_PROGRESS_REWARD_COEF": str(args.lift_progress_reward_coef),
        "REF2DEX_GRASP_LINK_REWARD_COEF": str(args.grasp_link_reward_coef),
        "REF2DEX_MIN_GRASP_LINKS": str(args.min_grasp_links),
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
        os.environ["REF2DEX_CURRICULUM_BACKTRACK_START"] = str(args.curriculum_backtrack_start)
        os.environ["REF2DEX_CURRICULUM_BACKTRACK_END"] = str(args.curriculum_backtrack_end)
    if args.save_frequency is not None:
        os.environ["REF2DEX_SAVE_FREQUENCY"] = str(args.save_frequency)
    if args.scratch_resume_checkpoint is not None:
        os.environ["REF2DEX_SCRATCH_RESUME_CHECKPOINT"] = str(
            args.scratch_resume_checkpoint.resolve())
        os.environ["REF2DEX_SCRATCH_RESUME_SHA256"] = args.scratch_resume_sha256
    if args.learning_rate is not None:
        os.environ["REF2DEX_LEARNING_RATE"] = str(args.learning_rate)
    base.main(
        passthrough,
        agent_class="src.task.CmResidual.dexplore_temporal_cm_agent:DExploreTemporalCmAgent",
    )


if __name__ == "__main__":
    main()

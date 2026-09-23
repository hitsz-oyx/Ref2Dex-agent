"""Rank bootstrap for frozen-Cm-weighted PPO actor updates."""
from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path

import dexplore_cm_off_rank_bootstrap as base


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cm-actor-weight-coef", type=float, required=True)
    parser.add_argument("--permute-cm-actor-weights", action="store_true")
    parser.add_argument("--cm-actor-weight-component",
                        choices=("joint", "contact_rank", "effect_rank",
                                 "effect_action_shuffled_rank"), default="joint")
    parser.add_argument("--cmlite-checkpoint", type=Path, required=True)
    parser.add_argument("--cmlite-sha256", required=True)
    args, passthrough = parser.parse_known_args(argv)
    if not math.isfinite(args.cm_actor_weight_coef) or not 0 < args.cm_actor_weight_coef <= 2:
        raise ValueError("Cm PPO actor weight coefficient must be in (0,2]")
    if (not args.cmlite_checkpoint.is_file() or len(args.cmlite_sha256) != 64 or
            hashlib.sha256(args.cmlite_checkpoint.read_bytes()).hexdigest() !=
            args.cmlite_sha256):
        raise ValueError("CmLite checkpoint SHA mismatch")
    os.environ.update({
        "REF2DEX_CM_ACTOR_WEIGHT_COEF": str(args.cm_actor_weight_coef),
        "REF2DEX_CM_ACTOR_WEIGHT_PERMUTE": "1" if args.permute_cm_actor_weights else "0",
        "REF2DEX_CM_ACTOR_WEIGHT_COMPONENT": args.cm_actor_weight_component,
        "REF2DEX_CMLITE_CHECKPOINT": str(args.cmlite_checkpoint.resolve()),
        "REF2DEX_CMLITE_SHA256": args.cmlite_sha256,
    })
    base.main(passthrough, agent_class=(
        "src.task.CmResidual.dexplore_cm_ppo_weight_agent:DExploreCmPpoWeightAgent"))


if __name__ == "__main__":
    main()

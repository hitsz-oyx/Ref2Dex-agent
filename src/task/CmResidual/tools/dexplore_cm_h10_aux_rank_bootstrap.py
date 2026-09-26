"""Pinned multi-axis H10 Cm auxiliary-representation PPO bootstrap."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import dexplore_cm_off_rank_bootstrap as base


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cm-aux-coef", type=float, required=True)
    parser.add_argument("--contact-cm-checkpoint", type=Path, required=True)
    parser.add_argument("--contact-cm-sha256", required=True)
    args, passthrough = parser.parse_known_args(argv)
    if args.cm_aux_coef not in (0.0, .002):
        raise ValueError("H10 Cm auxiliary coefficient must be 0 or .002")
    if (not args.contact_cm_checkpoint.is_file() or
            len(args.contact_cm_sha256) != 64 or
            hashlib.sha256(args.contact_cm_checkpoint.read_bytes()).hexdigest() !=
            args.contact_cm_sha256):
        raise ValueError("frozen H10 Cm checkpoint SHA mismatch")
    os.environ["REF2DEX_CM_AUX_COEF"] = str(args.cm_aux_coef)
    os.environ["REF2DEX_CONTACT_CM_CHECKPOINT"] = str(args.contact_cm_checkpoint.resolve())
    os.environ["REF2DEX_CONTACT_CM_SHA256"] = args.contact_cm_sha256
    base.main(passthrough, agent_class=(
        "src.task.CmResidual.dexplore_cm_ppo_aux_agent:DExploreCmH10AuxAgent"))


if __name__ == "__main__":
    main()

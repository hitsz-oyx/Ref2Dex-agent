"""Pinned physical-value PPO entrypoint with isolated teacher initialization."""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import dexplore_cm_off_rank_bootstrap as base


def main(argv=None):
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--physical-value-arm", choices=("plain_off", "direct_q", "cm_value", "cm_representation"), required=True)
    p.add_argument("--physical-value-checkpoint", type=Path, required=True)
    p.add_argument("--physical-value-sha256", required=True)
    a, remaining = p.parse_known_args(argv)
    if hashlib.sha256(a.physical_value_checkpoint.read_bytes()).hexdigest() != a.physical_value_sha256:
        raise ValueError("physical checkpoint drift")
    os.environ["REF2DEX_PHYSICAL_VALUE_ARM"] = a.physical_value_arm
    os.environ["REF2DEX_PHYSICAL_VALUE_CHECKPOINT"] = str(a.physical_value_checkpoint.resolve())
    base.main(remaining, agent_class="src.task.CmResidual.dexplore_physical_value_agent:DExplorePhysicalValueAgent")


if __name__ == "__main__":
    main()

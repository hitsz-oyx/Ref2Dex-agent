#!/usr/bin/env python3
"""Native process bootstrap; keep IsaacGym's required import order."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src/task/CmResidual/tools'))
import dexplore_cm_off_rank_bootstrap as base

if __name__=='__main__':
    base.main(agent_class='src.task.CmResidual.hold_plateau_agent:DExploreHoldPlateauAgent')

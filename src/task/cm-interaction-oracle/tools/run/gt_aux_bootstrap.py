"""Task-local training-only GT interaction auxiliary entrypoint."""
from pathlib import Path
import argparse
import os
import sys
ROOT=Path(__file__).resolve().parents[5]
sys.path[:0]=[str(ROOT),str(ROOT/'src/task/CmResidual/tools'),
              str(ROOT/'src/task/cm-interaction-oracle/src')]
import dexplore_cm_off_rank_bootstrap as base


def main():
    p=argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument('--gt-arm',choices=('plain','conditioned','shuffle','stopgrad'),required=True)
    args, rest=p.parse_known_args()
    os.environ['REF2DEX_GT_AUX_ARM']=args.gt_arm
    base.main(rest,agent_class='dexplore_gt_aux_agent:DExploreGtAuxAgent')


if __name__=='__main__':
    main()

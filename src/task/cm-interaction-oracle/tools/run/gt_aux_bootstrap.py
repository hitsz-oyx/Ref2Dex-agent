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
    # Training resets during agent construction, before restore() is available.
    from isaacgym import torch_utils
    sys.path.insert(0,str(ROOT/'third_party/DExplore/dexplore'))
    from env.tasks.base_dexplore_task import DexploreTask
    from oracle_y_utility import align_native_reference_tables
    original=DexploreTask._reset_ref_state_init
    def aligned_reset(task,env_ids):
        align_native_reference_tables(task)
        return original(task,env_ids)
    DexploreTask._reset_ref_state_init=aligned_reset
    base.main(rest,agent_class='dexplore_gt_aux_agent:DExploreGtAuxAgent')


if __name__=='__main__':
    main()

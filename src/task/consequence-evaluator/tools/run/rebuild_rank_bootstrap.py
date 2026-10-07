"""Native Cm-off bootstrap using CUDA PhysX and CUDA tensor views."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT/'third_party/DExplore/dexplore'),
               str(ROOT/'src/task/CmResidual/tools'), str(ROOT/'src/task/cm-interaction-oracle/src')]


def main():
    # Install the native NumPy aliases before Isaac Gym imports torch_utils.
    # This bootstrap imports no Torch at module scope.
    import dexplore_cm_off_rank_bootstrap
    # Isaac Gym must load before Torch; the actor still trains on CUDA.
    from isaacgym import torch_utils
    from env.tasks.base_dexplore_task import DexploreTask
    from oracle_y_utility import align_native_reference_tables
    original = DexploreTask._reset_ref_state_init
    def reset(task, ids):
        align_native_reference_tables(task)
        return original(task, ids)
    DexploreTask._reset_ref_state_init = reset
    argv = sys.argv[1:]
    if '--pipeline' not in argv:
        argv += ['--pipeline', 'gpu']
    dexplore_cm_off_rank_bootstrap.main(argv)


if __name__ == '__main__':
    main()

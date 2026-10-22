"""Copy every post-legacy dependency needed by the geometry evaluation worker."""
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/wam-rl-policy-v3'
MODULES=('pose_consumer','evaluation_sources','geometry_tau','stage_evaluation','grasp_repair','grasp_geometry','intent_residual','source_pd')


def copy_geometry_runtime(frozen):
    for name in MODULES:
        relative=Path('src/task/wam-rl-policy-v3/src/wam_rl_policy_v3')/(name+'.py')
        target=frozen/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,target)
    for relative in (Path('src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/staged_progress.py'),Path('src/task/wam-rl-policy-v3/tools/run/inspire_physics_host.py')):
        target=frozen/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,target)

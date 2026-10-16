import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

TASK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(TASK.parent/'shadow-finetune/src'))
from wam_rl_policy.fast_bc import batched_inputs
from shadow_finetune.data import current_inputs
from video_data_preparation.unidexgrasp_data import object_poses


def test_batched_h4_matches_original_all_features_and_ragged_geometry():
    rng=np.random.default_rng(20)
    obj=rng.normal(size=(3,4,7));obj[...,3:]=Rotation.random(12,random_state=20).as_quat().reshape(3,4,4)
    hand=rng.normal(size=(3,4,11,3));goal=rng.normal(size=(3,3))
    geometries=[dict(surface=rng.normal(size=(n,3)).astype(np.float32),normals=rng.normal(size=(n,3)).astype(np.float32)) for n in (32,45,60)]
    batch=batched_inputs(obj,hand,goal,geometries)
    for env,actual in enumerate(batch):
        expected=current_inputs(object_poses(obj[env]),hand[env],goal[env],geometries[env])
        assert actual.keys()==expected.keys()
        for key in actual:np.testing.assert_allclose(actual[key],expected[key],rtol=1e-6,atol=1e-6,err_msg=key)

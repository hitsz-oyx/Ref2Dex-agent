"""Task information must survive startup/zero WAM conditions."""
import numpy as np
from wam_rl_policy_v2.inspire_shadow import BatchedInspireInputs,CURRENT_DIM,TASK_SLICE,TAU_SLICE,VALID_INDEX


def test_same_endpoint_different_local_path_is_visible_without_tau():
    class Runtime:
        device='cpu'
        config=dict(voxel_origin_m=[-2]*3,voxel_m=.01)
    rng=np.random.default_rng(246)
    surface=rng.normal(size=(512,3)).astype(np.float32)*.02
    normals=np.zeros_like(surface);normals[:,2]=1
    runtime=Runtime();runtime.geometry=[(surface,normals,surface[:32],surface[:128],normals[:128])]
    start=np.eye(4);a=start.copy();a[0,3]=.05;b=start.copy();b[1,3]=.05
    first=BatchedInspireInputs(runtime,[np.stack((start,a,start))])
    second=BatchedInspireInputs(runtime,[np.stack((start,b,start))])
    args=(np.zeros((1,CURRENT_DIM)),start[None],np.zeros((1,11,3)),np.array([0]))
    left=first.observation(*args);right=second.observation(*args)
    assert not np.allclose(left['scheduler'][0]['goal_delta'],right['scheduler'][0]['goal_delta'])
    assert not np.array_equal(left['observation'],right['observation']), 'actor cannot see distinct local task goals while tau is unavailable'
    assert left['observation'][0,VALID_INDEX]==0
    assert not left['observation'][0,TAU_SLICE].any()
    np.testing.assert_allclose(left['observation'][0,TASK_SLICE][:3],[.003,0,0],atol=1e-6)

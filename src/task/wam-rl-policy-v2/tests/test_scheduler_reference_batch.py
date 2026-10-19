import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from wam_rl_policy_v2.goal_scheduler import _pose, _poses, OrderedGoalScheduler, transform_points


def test_reference_batch_matches_scalar_validation_and_path_geometry():
    rng=np.random.default_rng(17)
    poses=np.broadcast_to(np.eye(4),(212,4,4)).copy()
    poses[:,:3,:3]=Rotation.random(212,random_state=rng).as_matrix()
    poses[:,:3,3]=rng.normal(size=(212,3))*.02
    points=rng.normal(size=(32,3))*.02
    expected=np.stack([_pose(p) for p in poses])
    np.testing.assert_array_equal(_poses(poses),expected)
    scheduler=OrderedGoalScheduler(poses,np.arange(212)/30,points)
    np.testing.assert_allclose(scheduler.path_points,np.stack([transform_points(p,points) for p in expected]),rtol=0,atol=1e-15)
    poses[0,0,3]+=1
    assert scheduler.poses[0,0,3]!=poses[0,0,3]


@pytest.mark.parametrize('kind',['nan','reflection','scale','bottom_row'])
def test_batch_rejects_invalid_pose_inside_reference(kind):
    values=np.broadcast_to(np.eye(4),(100,4,4)).copy()
    if kind=='nan':values[63,0,3]=np.nan
    elif kind=='reflection':values[63,0,0]=-1
    elif kind=='scale':values[63,0,0]=1.01
    else:values[63,3,2]=.01
    with pytest.raises(ValueError):_pose(values[63])
    with pytest.raises(ValueError):_poses(values)

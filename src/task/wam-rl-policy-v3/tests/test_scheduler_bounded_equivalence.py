"""Compare connector optimization against the original 28-step SE(3) rule."""
import numpy as np
from scipy.spatial.transform import Rotation
from wam_rl_policy_v3.goal_scheduler import OrderedGoalScheduler,interpolate_pose


def reference(scheduler,actual,goal):
    if scheduler._within_budget(actual,goal):return goal.copy()
    low,high=0.,1.
    for _ in range(28):
        middle=(low+high)/2
        if scheduler._within_budget(actual,interpolate_pose(actual,goal,middle)):low=middle
        else:high=middle
    return interpolate_pose(actual,goal,low)


def test_bounded_matches_original_translation_rotation_and_mixed():
    rng=np.random.RandomState(241)
    for i in range(180):
        actual=np.eye(4);actual[:3,:3]=Rotation.random(random_state=rng).as_matrix();actual[:3,3]=rng.randn(3)
        goal=actual.copy()
        if i%3!=0:goal[:3,:3]=actual[:3,:3]@Rotation.from_rotvec(rng.randn(3)).as_matrix()
        if i%3!=1:goal[:3,3]+=rng.randn(3)*.2
        points=rng.randn(32,3)*(.01 if i%2 else .15)
        scheduler=OrderedGoalScheduler([actual,goal],[0,2],points)
        expected=reference(scheduler,actual,goal)
        result=scheduler._bounded(actual,goal)
        np.testing.assert_allclose(result,expected,rtol=0,atol=1e-8)
        assert scheduler._within_budget(actual,result)

"""Semantic counterexamples: geometry must not align targets to the live hand."""
import inspect
import numpy as np
from scipy.spatial.transform import Rotation
from wam_rl_policy_v3.geometry_tau import object_local_hand, geometry_future


def reference():
    rng = np.random.RandomState(19)
    local = rng.normal(size=(30, 11, 3))*.04
    objects = np.c_[rng.normal(size=(30, 3)), Rotation.random(30, random_state=rng).as_quat()]
    r = Rotation.from_quat(objects[:, 3:7]).as_matrix()
    hand = np.einsum('tki,tji->tkj', local, r)+objects[:, None, :3]
    return local, objects, hand


def test_each_future_uses_its_own_object_frame():
    local, objects, hand = reference()
    recovered = object_local_hand(hand, objects)
    np.testing.assert_allclose(recovered, local, atol=2e-15)
    pose = np.eye(4)
    future, valid = geometry_future(recovered, 3, pose)
    assert valid and future.shape == (24, 11, 3)
    np.testing.assert_allclose(future, local[4:28], atol=2e-15)


def test_live_object_equivariance_and_hand_independence():
    local, _, _ = reference()
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler('xyz', [.2, -.7, .5]).as_matrix()
    pose[:3, 3] = [.4, -.2, .3]
    future, _ = geometry_future(local, 3, pose)
    np.testing.assert_allclose(future, local[4:28]@pose[:3, :3].T+pose[:3, 3], atol=1e-15)
    assert list(inspect.signature(geometry_future).parameters) == ['local_reference', 'frame', 'live_object_pose', 'horizon']
    # Only the consumer's delta encoding may see the live hand.
    for hand in (np.zeros((11,3)), np.ones((11,3))*.31):
        tau = future-hand
        np.testing.assert_allclose(tau+hand, future, atol=1e-15)
    assert np.linalg.norm(future-np.ones_like(future)*.31) > .1


def test_clock_and_missing_horizon_are_explicit():
    local, _, _ = reference()
    a, valid = geometry_future(local, 5, np.eye(4))
    assert valid
    np.testing.assert_array_equal(a, local[6:30])
    assert geometry_future(local, 6, np.eye(4)) == (None, False)


def test_source_rigid_object_motion_is_removed():
    local = np.ones((30,11,3))*.05
    objects = np.tile([0.,0.,0.,0.,0.,0.,1.], (30,1))
    objects[:,2] = np.arange(30)*.01
    hand = local+objects[:,None,:3]
    future, valid = geometry_future(object_local_hand(hand,objects), 3, np.eye(4))
    assert valid
    np.testing.assert_allclose(future, .05, atol=1e-15)

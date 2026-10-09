import numpy as np
import torch
from consequence_evaluator.retargeter import (
    ACTIVE_FINGERS, FINGER_SCALE, commanded_targets,relative_targets,absolute_targets,
    native_control,wrist_rotation,trajectory_input,Standardizer,HandTrajectoryRetargeter)


def test_commanded_pose_roundtrip_preserves_effective_native_controls():
    rng=np.random.default_rng(13)
    q=rng.normal(0,.2,(2,25,18)).astype('float32')
    q[...,4]*=.2
    control=rng.uniform(-.03,.03,(2,24,18)).astype('float32')
    desired=commanded_targets(q[:,:24],control)
    relative=relative_targets(q[:,0],desired)
    decoded=absolute_targets(q[:,0],relative)
    np.testing.assert_allclose(decoded,desired,atol=1e-6)
    action=native_control(decoded,q[:,:24])
    indexes=np.r_[0:6,ACTIVE_FINGERS]
    np.testing.assert_allclose(action[...,indexes],control[...,indexes],atol=3e-7)
    np.testing.assert_allclose(commanded_targets(q[:,:24],action),desired,atol=1e-6)
    # The same frozen targets need DIFFERENT native increments after q changes.
    changed=q[:,:24].copy();changed[...,:3]+=.01
    next_control=native_control(decoded,changed)
    np.testing.assert_allclose(next_control[...,:3],action[...,:3]-.01,atol=1e-6)


def test_intrinsic_rotation_follows_native_urdf_chain():
    q=np.zeros((1,18));q[0,3:6]=[.2,.3,.4]
    from scipy.spatial.transform import Rotation
    expected=Rotation.from_rotvec([.2,0,0])*Rotation.from_rotvec([0,.3,0])*Rotation.from_rotvec([0,0,.4])
    np.testing.assert_allclose(wrist_rotation(q)[0],expected.as_matrix(),atol=1e-12)


def test_trajectory_input_preserves_future_time1_and_keypoint_order():
    current=np.arange(33,dtype='float32').reshape(1,11,3)
    future=current[:,None]+np.arange(1,25,dtype='float32')[None,:,None,None]
    value=trajectory_input(current,future)
    assert value.shape==(1,24,11,3)
    np.testing.assert_array_equal(value[:,0],np.ones((1,11,3)))
    std=Standardizer.fit(value)
    np.testing.assert_allclose(std.decode(std.encode(value)),value)
    model=HandTrajectoryRetargeter(32)
    result=model(torch.from_numpy(value),torch.zeros(1,36))
    assert result.shape==(1,24,12) and torch.isfinite(result).all()

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


def test_rotation_roundtrip_keeps_nearby_euler_branch_beyond_gimbal_region():
    q=np.zeros((1,25,18),dtype='float32')
    q[...,3:6]=[2.8,1.9,-2.7]
    control=np.zeros((1,24,18),dtype='float32')
    control[...,3:6]=[.01,-.02,.01]
    desired=commanded_targets(q[:,:24],control)
    decoded=absolute_targets(q[:,0],relative_targets(q[:,0],desired))
    np.testing.assert_allclose(decoded,desired,atol=1e-6)
    np.testing.assert_allclose(native_control(decoded,q[:,:24])[...,3:6],control[...,3:6],atol=1e-6)


def test_teacher_loader_aligns_commands_and_time1_geometry_without_future_state_input(tmp_path):
    import importlib.util
    import pickle
    from pathlib import Path
    source=Path(__file__).resolve().parents[1]/'tools/run/train_hand_retargeter.py'
    spec=importlib.util.spec_from_file_location('retarget_fit_contract',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    q=np.zeros((543,4,18),dtype='float32')
    q[...,0]=np.arange(543)[:,None]*.001
    hand=np.zeros((543,4,11,3),dtype='float32')
    hand[...,0]=np.arange(543)[:,None,None]*.002
    action=np.zeros((542,4,18),dtype='float32');action[...,0]=.01
    packet=dict(engineering_only=True,seed=282,role_names=['reactive_teacher'],
        role_outcomes={'reactive_teacher':{'maximum_held_frames':484}},
        replay_identity={'backend':{'name':'gpu_physx_gpu_pipeline'}},
        dof_position=q,dof_velocity=q*0,hand_keypoints=hand,actions=action,
        done=np.zeros((542,4),dtype=bool))
    path=tmp_path/'teacher.pkl'
    path.write_bytes(pickle.dumps(packet))
    first=module.teacher_data(path)
    np.testing.assert_allclose(first['hand'][0,0,:,0],.002)
    np.testing.assert_allclose(first['target'][0,:,0],.01+np.arange(24)*.001,atol=1e-7)
    assert first['ticks'].max()+24==540
    # Future measured q changes supervision, not the anchor state/geometry inputs.
    packet['dof_position'][1,0,0]+=.03
    path.write_bytes(pickle.dumps(packet))
    second=module.teacher_data(path)
    np.testing.assert_array_equal(first['state'][0],second['state'][0])
    np.testing.assert_array_equal(first['hand'][0],second['hand'][0])
    assert first['target'][0,1,0]!=second['target'][0,1,0]

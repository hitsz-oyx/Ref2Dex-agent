import numpy as np
import torch
from scipy.spatial.transform import Rotation
from consequence_evaluator.object_relative_servo import transport_wrist, bounded_transport, recover_wrist, ObjectRelativeGTExecution
from consequence_evaluator.retargeter import wrist_rotation, commanded_targets


def test_transport_preserves_object_local_wrist_pose_and_fingers():
    q=np.array([.1,.2,.9,.3,1.8,-.2]+[.5]*12,dtype='float32')
    source=np.eye(4);source[:3,:3]=Rotation.from_rotvec([.1,-.2,.05]).as_matrix();source[:3,3]=[.2,.3,.6]
    live=np.eye(4);live[:3,:3]=Rotation.from_rotvec([-.2,.1,.3]).as_matrix();live[:3,3]=[.4,.1,.8]
    out=transport_wrist(q,source,live,q)
    np.testing.assert_allclose(live[:3,:3].T@(out[:3]-live[:3,3]),source[:3,:3].T@(q[:3]-source[:3,3]),atol=1e-7)
    np.testing.assert_allclose(live[:3,:3].T@wrist_rotation(out),source[:3,:3].T@wrist_rotation(q),atol=1e-6)
    np.testing.assert_array_equal(out[6:],q[6:])


def test_identity_transport_and_current_anchor_preserve_next_step_motion():
    source=np.eye(4);q=np.zeros(18,dtype='float32');q[:3]=[.2,.3,.4]
    np.testing.assert_allclose(transport_wrist(q,source,source,q),q,atol=1e-7)
    nextq=q.copy();nextq[2]+=.01
    out=transport_wrist(nextq,source,source,q)
    assert abs(out[2]-q[2]-.01)<1e-7
    future=source.copy();future[2,3]+=.01
    cancelled=transport_wrist(nextq,future,source,q)
    np.testing.assert_allclose(cancelled[:3],q[:3],atol=1e-7)


def test_bounded_transport_preserves_nominal_motion_and_limits_feedback():
    target=np.array([.1,.2,.9,.2,.3,-.1]+[.5]*12,dtype='float32')
    corrected=target.copy();corrected[:3]+=[.3,-.2,.4];corrected[3:6]+=[1.,.3,-.2]
    result=bounded_transport(target,corrected)
    assert np.linalg.norm(result[:3]-target[:3])<=.020001
    delta=wrist_rotation(result)@wrist_rotation(target).T
    assert Rotation.from_matrix(delta).magnitude()<=.150001
    np.testing.assert_array_equal(result[6:],target[6:])
    shift=target.copy();shift[:3]+=[.1,.2,.3]
    np.testing.assert_allclose(bounded_transport(shift,shift),shift,atol=1e-7)


def test_fixed_root_inverse_recovers_wrist_independently_of_fingertips():
    template=np.random.default_rng(9).normal(size=(11,3))*.07
    target=np.array([.2,-.1,.8,.7,1.9,-.6]+[.2]*12,dtype='float32')
    points=template@wrist_rotation(target).T+target[:3]
    points[[2,4,6,8,10]]+=.4  # Arbitrary finger articulation changes tips.
    prior=target.copy();prior[:3]=0;prior[3:6]+=[.1,-.1,.1]
    actual=recover_wrist(points,template,prior)
    np.testing.assert_allclose(actual[:3],target[:3],atol=1e-7)
    np.testing.assert_allclose(wrist_rotation(actual),wrist_rotation(target),atol=1e-6)
    np.testing.assert_array_equal(actual[6:],prior[6:])


def test_cold_full_geometry_control_ignores_future_q_and_command_labels():
    # Exercise the actual execution branch with poisoned future joint labels
    # and altered future PD commands, including the native clipping seam.
    class Task:
        _dof_pos=torch.full((4,18),.2)
        _dof_vel=torch.full((4,18),.1)
        _target_states=torch.zeros(4,13)
        _target_states[:,6]=1

        def _action_to_pd_targets(self,control):
            return torch.as_tensor(commanded_targets(self._dof_pos.numpy(),control.numpy()))

    def execution(command):
        value=ObjectRelativeGTExecution.__new__(ObjectRelativeGTExecution)
        value.layout='geometry_pd_inverse';value.anchor='current'
        value.source_targets=np.full((542,18),command,dtype='float32')
        value.source_q=np.full((543,18),np.nan,dtype='float32')
        value.source_pose=np.broadcast_to(np.eye(4),(543,4,4))
        value.inverse_q=np.full((543,18),.3,dtype='float32')
        value.pd_inverse=np.tile([2.,.02],(12,1)).astype('float32')
        value.device='cpu';value.desired=[];value.live_poses=[];value.clip_counts=[];value.errors=[]
        return value

    task=Task();teacher=torch.zeros(4,18)
    first=execution(.1);second=execution(.9)
    a=first.control(0,task,None,teacher)
    b=second.control(0,task,None,teacher)
    np.testing.assert_array_equal(a[3].numpy(),b[3].numpy())
    np.testing.assert_array_equal(first.desired[0][2],second.desired[0][2])
    second.inverse_q[1,:3]+=.02
    c=second.control(0,task,None,teacher)
    assert not torch.equal(a[3,:3],c[3,:3])
    task._dof_vel[3,:3]+=.2
    d=second.control(0,task,None,teacher)
    assert not torch.equal(c[3,:3],d[3,:3])

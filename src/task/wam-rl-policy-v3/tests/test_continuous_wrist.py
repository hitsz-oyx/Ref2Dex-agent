from types import SimpleNamespace
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from wam_rl_policy_v3.continuous_wrist import normalize_wrist_positions,install_continuous_wrist_reset,require_continuous_wrist_dofs


def test_only_continuous_rotation_coordinates_change_and_fk_is_equivalent():
    q=torch.tensor([[.3,.4,.5,2.940794,-5.876947,6.753245,*([.9]*12)]],dtype=torch.float64)
    original=q.clone();fixed=normalize_wrist_positions(q)
    assert torch.equal(q,original) and torch.equal(fixed[:,:3],q[:,:3]) and torch.equal(fixed[:,6:],q[:,6:])
    assert (fixed[:,3:6].abs()<=torch.pi).all()
    np.testing.assert_allclose(Rotation.from_euler('XYZ',fixed[0,3:6]).as_matrix(),Rotation.from_euler('XYZ',q[0,3:6]).as_matrix(),atol=1e-14)
    for turns in [-4,-1,0,1,4]:
        shifted=q.clone();shifted[:,3:6]+=turns*2*torch.pi
        torch.testing.assert_close(normalize_wrist_positions(shifted),fixed)


def test_indexed_reset_normalizes_before_original_fk_without_touching_others():
    class Task:
        def _set_env_state(self,ids,q,dq):
            self.received=q.clone();self._dof_pos[ids]=q;self._dof_vel[ids]=dq
            # This stands for the real reset/FK chain consuming its arguments.
            self.fk_input=q.clone()
    install_continuous_wrist_reset(Task);install_continuous_wrist_reset(Task)
    task=Task();task._dof_pos=torch.full((2,18),.7);task._dof_vel=torch.full((2,18),.3)
    before=task._dof_pos.clone();q=torch.zeros(1,18);q[:,3:6]=torch.tensor([[-7.,8.,10.]])
    task._set_env_state(torch.tensor([1]),q,torch.zeros_like(q))
    assert torch.equal(task._dof_pos[0],before[0]) and torch.equal(task._dof_vel[0],torch.full((18,),.3))
    assert (task._dof_pos[1,3:6].abs()<=torch.pi).all()
    torch.testing.assert_close(task.fk_input,normalize_wrist_positions(q))


def test_limited_or_wrong_joint_layout_is_not_silently_wrapped():
    names=['joint'+str(i) for i in range(1,19)];limits=np.ones(18,bool);limits[3:6]=False
    require_continuous_wrist_dofs(names,limits)
    limits[4]=True
    with pytest.raises(ValueError,match='continuous'):require_continuous_wrist_dofs(names,limits)
    with pytest.raises(ValueError,match='layout'):require_continuous_wrist_dofs(list(reversed(names)),np.zeros(18,bool))

from types import SimpleNamespace
import weakref
import torch
from wam_rl_policy_v3.inspire_reset_submission import install_reset_submission


def test_multiple_logical_resets_submit_once_and_keep_index_owners_alive():
    class Gym:
        def __init__(self):self.root=[];self.dof=[]
        def set_actor_root_state_tensor_indexed(self,sim,states,ids,count):self.root.append((weakref.ref(ids),ids.tolist()))
        def set_dof_state_tensor_indexed(self,sim,states,ids,count):self.dof.append((weakref.ref(ids),ids.tolist()))
    class Task:
        def pre_physics_step(self,actions):self.action=actions
    install_reset_submission(Task,SimpleNamespace(unwrap_tensor=lambda x:x))
    task=Task();task.num_envs=4;task.device='cpu';task.sim=None;task.gym=Gym();task._motion_sampler=object()
    task._humanoid_actor_ids=torch.tensor([0,3,6,9]);task._tar_actor_ids=torch.tensor([2,5,8,11]);task._table_actor_ids=torch.tensor([1,4,7,10])
    task._root_states=torch.zeros(12,13);task._dof_state=torch.zeros(4,18,2)
    task.reset_buf=torch.ones(4);task._terminate_buf=torch.ones(4)
    # Startup reset, collector reset, and another subset can all occur before
    # any simulation. No setter may see a freed/replaced temporary index.
    task._reset_env_tensors(torch.tensor([0,1,2,3]))
    task._reset_env_tensors(torch.tensor([0,1,2,3]))
    task._reset_env_tensors(torch.tensor([1]))
    assert not task.gym.root and not task.gym.dof
    task.pre_physics_step('first')
    assert len(task.gym.root)==len(task.gym.dof)==1
    assert task.gym.root[0][0]() is task._consequence_root_indices
    assert task.gym.dof[0][0]() is task._consequence_dof_indices
    assert set(task.gym.root[0][1])==set(range(12))
    assert task.action=='first' and not task.reset_buf.any()
    task.pre_physics_step('second')
    assert len(task.gym.root)==1
    task._reset_env_tensors(torch.tensor([1,3]))
    task.pre_physics_step('third')
    assert len(task.gym.root)==len(task.gym.dof)==2
    assert set(task.gym.root[-1][1])=={3,4,5,9,10,11}
    assert task._inspire_reset_setter_submissions==2

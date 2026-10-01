import torch
from src.task.CmResidual.native_reset_transaction import NativeResetQueue

def test_reset_commits_union_once_without_stale_refresh():
    class Gym:
        def __init__(self):self.calls=[]
        def set_actor_root_state_tensor_indexed(self,sim,state,ids,count):self.calls.append(('root',ids.clone()));return True
        def set_dof_state_tensor_indexed(self,sim,state,ids,count):self.calls.append(('dof',ids.clone()));return True
        def refresh_actor_root_state_tensor(self,*args):raise AssertionError('stale refresh')
    gym=Gym();queue=NativeResetQueue(gym,lambda x:x)
    queue.set_actor_root_state_tensor_indexed(None,None,torch.tensor([0,3],dtype=torch.int32),2)
    queue.set_actor_root_state_tensor_indexed(None,None,torch.tensor([2,5],dtype=torch.int32),2)
    queue.set_actor_root_state_tensor_indexed(None,None,torch.tensor([2,5],dtype=torch.int32),2)
    queue.set_dof_state_tensor_indexed(None,None,torch.tensor([0,3],dtype=torch.int32),2)
    queue.refresh_actor_root_state_tensor(None)
    roots,dofs=queue.commit(None,torch.zeros(6,13),torch.zeros(36,2),lambda x:x)
    assert len(gym.calls)==2 and queue.refreshes_suppressed==1
    assert torch.equal(roots,torch.tensor([0,2,3,5],dtype=torch.int32))
    assert torch.equal(dofs,torch.tensor([0,3],dtype=torch.int32))

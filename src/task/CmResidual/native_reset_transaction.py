"""Combine native reset setters and defer refresh until the next physical tick."""
import torch

class NativeResetQueue:
    def __init__(self,gym,wrap):
        self.original=gym;self.wrap=wrap;self.root_ids=[];self.dof_ids=[];self.events=[];self.refreshes_suppressed=0

    def __getattr__(self,name):
        if name in ('set_actor_root_state_tensor_indexed','set_dof_state_tensor_indexed'):
            def queue(sim,states,indices,count):
                ids=self.wrap(indices).clone()
                if ids.dtype!=torch.int32 or ids.numel()!=count:raise ValueError('native reset actor index contract')
                (self.root_ids if name=='set_actor_root_state_tensor_indexed' else self.dof_ids).append(ids)
                self.events.append(dict(setter=name,count=count));return True
            return queue
        if name.startswith('refresh_'):
            def defer(*args):self.refreshes_suppressed+=1;return True
            return defer
        return getattr(self.original,name)

    def commit(self,sim,root_states,dof_states,unwrap):
        if not self.root_ids or not self.dof_ids:raise ValueError('native root and joint reset requests required')
        roots=torch.unique(torch.cat(self.root_ids));dofs=torch.unique(torch.cat(self.dof_ids))
        if self.original.set_actor_root_state_tensor_indexed(sim,unwrap(root_states),unwrap(roots),len(roots)) is False:raise RuntimeError('native root setter rejected')
        if self.original.set_dof_state_tensor_indexed(sim,unwrap(dof_states),unwrap(dofs),len(dofs)) is False:raise RuntimeError('native DOF setter rejected')
        return roots,dofs

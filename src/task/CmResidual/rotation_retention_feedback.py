"""Hold rotation after current-state acquisition; preserve translation/fingers."""
import torch
from src.task.CmResidual.natural_retention_feedback import groups

class RotationRetentionFeedback:
    def __init__(self,group,initial_height,lift_start,early_step,pd_scale):
        self.group=group;self.initial_height=initial_height;self.lift_start=lift_start;self.early_step=early_step
        self.run=torch.zeros_like(group);self.event_tick=torch.full_like(group,-1)
        self.rotation_anchor=torch.zeros((len(group),3),device=group.device)
        self.rotation_anchor_tick=torch.full_like(group,-1);self.radius=pd_scale[3:6].abs().clone()
        if (self.radius<=0).any():raise ValueError('native positive wrist rotation PD scale')

    def target(self,base,q,dq,object_root,clearance,progress,lower,upper):
        valid=(object_root[:,2]-self.initial_height>=.03)&(clearance>=.02)&(progress>=self.lift_start)
        self.run=torch.where(valid,self.run+1,torch.zeros_like(self.run))
        trigger=(self.event_tick<0)&(self.run>=5);self.event_tick[trigger]=progress[trigger]
        activate=((self.group==2)&(progress>=self.early_step))|((self.group==3)&(self.event_tick>=0))
        first=activate&(self.rotation_anchor_tick<0);self.rotation_anchor[first]=q[first,3:6];self.rotation_anchor_tick[first]=progress[first]
        goal=base.clone();held=torch.maximum(torch.minimum(self.rotation_anchor,q[:,3:6]+self.radius),q[:,3:6]-self.radius)
        goal[activate,3:6]=held[activate]
        return goal

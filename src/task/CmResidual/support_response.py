"""Prospective finite executable primitives and current support-state features."""
import torch
from src.task.CmResidual.finger_preload import preload_target
# World-axis target dx/dy in meters; shared curl-target increment in radians.
PRIMITIVES=((0.,0.,0.),(0.,0.,-.30),(0.,0.,-.15),(0.,0.,.15),
            (-.01,0.,0.),(.01,0.,0.),(0.,-.01,0.),(0.,.01,0.))
SCHEMA='support69-primitive3-h30-v1'

def assignments(motion,seed):
    generator=torch.Generator(device='cpu').manual_seed(seed+12000)
    result=torch.full((len(motion),),-1,dtype=torch.long)
    for m in range(3):
        ids=(motion.cpu()==m).nonzero().flatten()
        if len(ids)!=256:raise ValueError('768-environment balanced design required')
        result[ids]=torch.arange(8).repeat_interleave(32)[torch.randperm(256,generator=generator)]
    return result

def placement_offsets(n,seed):
    generator=torch.Generator(device='cpu').manual_seed(seed+11000)
    return (torch.rand((n,2),generator=generator)*2-1)*.01

def primitive_target(base,parameters,lower,upper):
    if parameters.shape!=(len(base),3) or not torch.isfinite(parameters).all():raise ValueError('primitive shape')
    goal=base.clone();goal[:,:2]+=parameters[:,:2]
    return preload_target(goal,parameters[:,2],lower,upper)

def support_features(context,initial_height,current_clearance):
    if context.shape[-1]!=70 or initial_height.shape!=context.shape[:1] or current_clearance.shape!=initial_height.shape:raise ValueError('support feature contract')
    q=context[:,:18];dq=context[:,18:36];obj=context[:,36:49];planned=context[:,51:69]
    x=torch.cat((q[:,3:],dq,obj[:,:3]-q[:,:3],obj[:,3:7],obj[:,7:10]-dq[:,:3],obj[:,10:13],
        context[:,49:51],planned-q,context[:,69:70],(obj[:,2]-initial_height)[:,None],current_clearance[:,None]),-1)
    if x.shape!=(len(context),69) or not torch.isfinite(x).all():raise ValueError('current-state feature finite/schema')
    return x

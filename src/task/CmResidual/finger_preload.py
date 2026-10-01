"""Predetermined extra closure respecting native dependent-joint bounds."""
import torch
DOSES=(0.,.05,.15,.30)
COUPLING={6:((7,1.05),),8:((9,1.05),),10:((11,1.05),),12:((13,1.05),),15:((16,.6),(17,.8))}

def preload_target(base,dose,lower,upper):
    target=base.clone()
    for parent,children in COUPLING.items():
        lo=max(float(lower[parent]),*(float(lower[j])/ratio for j,ratio in children))
        hi=min(float(upper[parent]),*(float(upper[j])/ratio for j,ratio in children))
        if lo>hi:raise ValueError('incompatible coupled joint bounds')
        target[:,parent]=(base[:,parent]+dose).clamp(lo,hi)
        for child,ratio in children:target[:,child]=target[:,parent]*ratio
    if ((target[:,6:]<lower[6:]-1e-6)|(target[:,6:]>upper[6:]+1e-6)).any():raise ValueError('finger target bound violation')
    if not torch.equal(target[:,:6],base[:,:6]) or not torch.equal(target[:,14],base[:,14]):raise ValueError('noncurl target changed')
    return target

def assign_doses(motion,seed):
    g=torch.Generator(device='cpu').manual_seed(seed+7000)
    assignments=torch.full((len(motion),),-1,dtype=torch.long)
    for m in range(3):
        ids=(motion.cpu()==m).nonzero().flatten()
        if len(ids)!=32:raise ValueError('frozen motion count')
        values=torch.arange(4).repeat_interleave(8)
        assignments[ids]=values[torch.randperm(32,generator=g)]
    return assignments

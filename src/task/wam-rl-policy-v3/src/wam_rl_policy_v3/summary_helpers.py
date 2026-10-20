"""Shared token summary reduction retained for runtime API compatibility."""
import torch

def type_summaries(shared,valid,scene_count,hand_count):
    """Keep scene/hand/known-goal summaries separate; never include future readouts."""
    groups=((0,scene_count),(scene_count,scene_count+hand_count),(scene_count+hand_count,shared.shape[1]))
    if shared.shape[-1]!=768 or any(stop<=start for start,stop in groups):raise ValueError('WAM scene/hand/goal token contract')
    summaries=[]
    for start,stop in groups:
        mask=valid[:,start:stop,None]
        value=torch.where(mask,shared[:,start:stop],0.).sum(1)/mask.sum(1).clamp_min(1)
        summaries.append(value)
    return torch.cat(summaries,-1).detach()

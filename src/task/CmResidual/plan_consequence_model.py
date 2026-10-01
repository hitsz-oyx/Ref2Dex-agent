"""Pre-action program-conditioned H10 physical forecaster, without V targets.

Outputs normalized signed height/clearance changes for ten ticks, ten joint
net-contact logits, last-three clearance/contact retention and clearance-loss
logits. Training labels must come from actual complete program executions.
"""
import torch
from torch import nn
from .executable_contact_options import INDEPENDENT


def plan_features(candidate_actions,rotation_anchor):
    if candidate_actions.ndim!=3 or candidate_actions.shape[1:]!=(8,18):
        raise ValueError('eight observed first commands required')
    n=len(candidate_actions);identity=torch.eye(8,device=candidate_actions.device,dtype=candidate_actions.dtype)
    identity=identity[None].expand(n,-1,-1)
    anchor=torch.zeros(n,8,3,device=candidate_actions.device,dtype=candidate_actions.dtype)
    anchor[:,6:]=rotation_anchor[:,None]
    return torch.cat((identity,candidate_actions[:,:,list(INDEPENDENT)],anchor),-1)


class PlanConsequenceModel(nn.Module):
    def __init__(self,native_observation_dim,mode='cm'):
        super().__init__()
        if mode not in ('cm','state_only','shuffled'):raise ValueError('invalid control')
        self.mode=mode
        self.history=nn.GRU(69,64,batch_first=True)
        self.context=nn.Sequential(nn.Linear(native_observation_dim,64),nn.SiLU(),nn.LayerNorm(64))
        self.base=nn.Sequential(nn.Linear(128,128),nn.SiLU(),nn.Linear(128,32))
        self.effect=nn.Sequential(nn.Linear(151,128),nn.SiLU(),nn.Linear(128,32))

    def forward(self,history,native_observation,program,base_program):
        _,h=self.history(history);state=torch.cat((h[-1],self.context(native_observation)),-1)
        if self.mode=='state_only':program=torch.zeros_like(program);base_program=torch.zeros_like(base_program)
        return self.base(state)+self.effect(torch.cat((state,program),-1))-self.effect(torch.cat((state,base_program),-1))


class StateOptionConsequenceModel(nn.Module):
    """Strong state-only control: fixed catalog heads, no action tensor input.

    Output head identity encodes the static program catalog. Report this
    explicitly; it is stronger than the unconditional state-only forecaster,
    whose predictions cannot rank candidates. It receives no first commands
    or anchor parameters, while factual training selects the observed head.
    """
    def __init__(self,native_observation_dim):
        super().__init__()
        self.history=nn.GRU(69,64,batch_first=True)
        self.context=nn.Sequential(nn.Linear(native_observation_dim,64),nn.SiLU(),nn.LayerNorm(64))
        self.heads=nn.Sequential(nn.Linear(128,128),nn.SiLU(),nn.Linear(128,8*32))

    def forward(self,history,native_observation):
        _,h=self.history(history);state=torch.cat((h[-1],self.context(native_observation)),-1)
        return self.heads(state).reshape(-1,8,32)


def physical_targets(record):
    state=record['state'];future=record['future_state'];clear=record['future_clearance'];pair=record['future_contact'].all(-1)
    if future.shape[1:]!=(10,49) or record['future_done'].any():raise ValueError('complete nonterminal H10 required')
    dz=(future[:,:,38]-state[:,None,38])/.01
    dc=(clear-record['initial_clearance'][:,None])/.01
    retain=(pair[:,-3:]&(clear[:,-3:]>=.002)).all(-1)
    lose=(clear<.002).any(-1)
    return torch.cat((dz,dc,pair.float(),retain[:,None].float(),lose[:,None].float()),-1)


def consequence_loss(prediction,target):
    return (torch.nn.functional.smooth_l1_loss(prediction[:,:20],target[:,:20])+
            torch.nn.functional.binary_cross_entropy_with_logits(prediction[:,20:],target[:,20:]))

"""Actual motor-target-conditioned one-step task physics with observed priors."""
import math
import torch
from torch import nn
from .executable_contact_options import INDEPENDENT


def transitions(record):
    if record['future_done'].any() or record['future_state'].shape[1:]!=(10,49):raise ValueError('complete source windows required')
    n=len(record['state']);pre=torch.cat((record['state'][:,None],record['future_state'][:,:-1]),1)
    clearance=torch.cat((record['initial_clearance'][:,None],record['future_clearance'][:,:-1]),1)
    hand=torch.cat((record['initial_hand_force'][:,None],record['future_hand_force'][:,:-1]),1)
    obj=torch.cat((record['initial_object_force'][:,None],record['future_object_force'][:,:-1]),1)
    contact=torch.cat((record['history'][:,-1,49:51,None].transpose(1,2),record['future_contact'][:,:-1].float()),1)
    previous=torch.cat((record['history'][:,-1,51:,None].transpose(1,2),record['actual_action'][:,:-1]),1)
    sequence=torch.cat((pre,contact,previous),-1);h=record['history'];history=[]
    for step in range(10):
        if step:h=torch.cat((h[:,1:],sequence[:,step,None]),1)
        if not torch.equal(h[:,-1,:49],pre[:,step]):raise ValueError('pre-step history alignment')
        history.append(h)
    history=torch.stack(history,1)
    weight=record['mass_kg'][:,None,None]*record['gravity_magnitude']
    raw=torch.cat((hand.flatten(-2),obj),-1)/weight
    raw=raw.sign()*torch.log1p(raw.abs())
    height=(pre[:,:,38]-record['rest_z'][:,None]).clamp_min(0)
    physical=torch.cat((raw,clearance[:,:,None],height[:,:,None]),-1)
    goal=(record['actual_pd_targets']-pre[:,:,:18])[:,:,list(INDEPENDENT)]
    future=record['future_state'];pair=record['future_contact'].all(-1);supported=pair&(record['future_clearance']>=.002)
    score=(future[:,:,38]-record['rest_z'][:,None]).clamp_min(0)*supported-height
    target=torch.stack(((future[:,:,38]-pre[:,:,38])/.002,(record['future_clearance']-clearance)/.002,pair.float(),supported.float(),(record['future_clearance']<.002).float(),score/.01),-1)
    velocity=pre[:,:,45]/30
    current_joint=contact.bool().all(-1);current_support=current_joint&(clearance>=.002);current_loss=clearance<.002
    bits=torch.stack((current_joint,current_support,current_loss),-1)
    prior=torch.cat((velocity[:,:,None].expand(-1,-1,2)/.002,(bits.float()*2-1)*math.log(99),velocity[:,:,None]/.01),-1)
    return dict(history=history.reshape(-1,10,69),physical=physical.reshape(-1,physical.shape[-1]),goal=goal.reshape(-1,12),target=target.reshape(-1,6),prior=prior.reshape(-1,6),clear=((clearance>=.002)&(pre[:,:,38]-record['rest_z'][:,None]>=.03)).reshape(-1),persist_joint=current_joint.reshape(-1).float())


class NativePDConsequenceModel(nn.Module):
    def __init__(self,physical_dim,mode='cm'):
        super().__init__();self.mode=mode
        self.history=nn.GRU(69,64,batch_first=True)
        self.physical=nn.Sequential(nn.Linear(physical_dim,64),nn.SiLU(),nn.LayerNorm(64))
        self.head=nn.Sequential(nn.Linear(140,128),nn.SiLU(),nn.Linear(128,6))

    def forward(self,history,physical,goal,prior):
        _,h=self.history(history)
        if self.mode=='state_only':goal=torch.zeros_like(goal)
        return self.head(torch.cat((h[-1],self.physical(physical),goal),-1))+prior


def loss(pred,target):
    return torch.nn.functional.smooth_l1_loss(pred[:,[0,1,5]],target[:,[0,1,5]])+torch.nn.functional.binary_cross_entropy_with_logits(pred[:,2:5],target[:,2:5])

"""Causal prediction of one rigid-transport endpoint and bounded coupling.

No evaluation outcome enters features or deployment. A predicted score
estimates the error of the model's own coefficient, not the oracle's error.
"""
import numpy as np
import torch
from torch import nn
from src.task.CmResidual.rigid_transport_capacity import VISUAL_IDS
from src.task.CmResidual.dexplore_cm_geometry import dexplore_root_pose

INPUT_SIZE=120
ARMS=('full','state_only','shuffled')
STEPS=1500
STATE_IDS=np.asarray([0,1]*7+[0],np.int64)


class CouplingPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(INPUT_SIZE,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,2))
        nn.init.zeros_(self.layers[-1].weight)
        with torch.no_grad():
            self.layers[-1].bias.copy_(torch.tensor([-2.1972245773362196,.541324854612918]))

    def forward(self,x):
        raw=self.layers(x)
        return torch.sigmoid(raw[...,0]),torch.nn.functional.softplus(raw[...,1])


@torch.no_grad()
def causal_inputs(rows,fields,object_local,joint_scale,device):
    tensor=lambda x:torch.as_tensor(x,dtype=torch.float64,device=device)
    current=dexplore_root_pose(tensor(rows['current_obj']));previous=dexplore_root_pose(tensor(rows['previous_obj']))
    rotation=current[:,:3,:3];transpose=rotation.transpose(-1,-2)
    local=tensor(object_local);lever=local-local.mean(0)
    gram=lever.square().sum()*torch.eye(3,dtype=torch.float64,device=device)-lever.T@lever
    inverse=torch.linalg.inv(gram+1e-12*torch.eye(3,dtype=torch.float64,device=device))
    def twist(flow):
        object_flow=flow@rotation[:,None]
        linear=object_flow.mean(-2)
        angular=torch.cross(lever[None,None].expand_as(object_flow),object_flow-linear[:,:,None],dim=-1).sum(-2)@inverse.T
        return torch.cat((linear/.01,angular/.05),-1)
    endpoint=twist(fields['candidates']['causal'])
    anchor=twist(fields['anchor'][:,None])[:,0]
    hand=fields['current_hand_poses'][:,VISUAL_IDS]
    hand_relative_rotation=transpose[:,None]@hand[:,:,:3,:3]
    hand_relative_position=torch.einsum('bij,bnj->bni',transpose,hand[:,:,:3,3]-current[:,None,:3,3])/.05
    relative=torch.cat((hand_relative_position,hand_relative_rotation.reshape(-1,13,9)),-1)
    scale=tensor(joint_scale)
    joint=torch.cat((tensor(rows['q'])/scale,tensor(rows['dq'])/30/scale,(tensor(rows['target'])-tensor(rows['q']))/scale),-1)
    object_pose=torch.cat((current[:,:3,3]/.1,rotation[:,:,:2].reshape(-1,6)),-1)
    past=torch.cat(((current[:,:3,3]-previous[:,:3,3])/.01,
                    ((rotation@previous[:,:3,:3].transpose(-1,-2)-torch.eye(3,dtype=torch.float64,device=device))/.05).reshape(-1,9)),-1)
    common=torch.cat((joint,object_pose,past,anchor,relative.mean(1),endpoint[:,2:].mean(1)),-1)
    assert common.shape[1]==99
    per_relative=torch.cat((torch.zeros(len(common),2,12,dtype=torch.float64,device=device),relative),1)
    kind=torch.zeros(len(common),15,3,dtype=torch.float64,device=device);kind[:,0,0]=1;kind[:,1,1]=1;kind[:,2:,2]=1
    result=torch.cat((common[:,None].expand(-1,15,-1),per_relative,endpoint,kind),-1)
    assert result.shape==(len(common),15,INPUT_SIZE) and torch.isfinite(result).all()
    return result.float()


def component_weights(arm,device):
    if arm!='state_only':return torch.ones(15,device=device)
    counts=np.bincount(STATE_IDS,minlength=2)
    return torch.tensor(15/(2*counts[STATE_IDS]),dtype=torch.float32,device=device)


def coupling_loss(coefficient,score,coefficient_label,anchor,endpoints,target,weights):
    # Fields in meters. Confidence target follows the CURRENT predicted weight;
    # detach it so score cannot drive the coefficient by changing its own label.
    direction=endpoints-anchor[:,None]
    predicted=anchor[:,None]+coefficient[...,None,None]*direction
    actual_mm=(predicted-target[:,None]).norm(dim=-1).mean(-1)*1000
    score_label=torch.log1p(actual_mm).detach()
    coefficient_weight=(direction.norm(dim=-1).mean(-1)/.01).clamp(0,1)
    return (((coefficient-coefficient_label).square()*coefficient_weight+(score-score_label).square())*weights[None]).mean()


@torch.no_grad()
def deploy(model,features,anchor,endpoints):
    coefficient,score=model(features)
    winner=score.argmin(1);batch=torch.arange(len(features),device=features.device)
    chosen=coefficient[batch,winner]
    prediction=anchor+chosen.double()[:,None,None]*(endpoints[batch,winner]-anchor)
    return dict(coefficients=coefficient,scores=score,winner=winner,prediction=prediction)


def classify(reports,baselines,near):
    e=reports['full']['episode_epe_mm']
    gates=dict(persistence=e<=.9*baselines['persistence']['episode_epe_mm'],
               state_only=e<=.9*reports['state_only']['episode_epe_mm'],
               shuffled=e<=.95*reports['shuffled']['episode_epe_mm'],
               near_state_only=near['full']['episode_epe_mm']<=.95*near['state_only']['episode_epe_mm'])
    useful=gates['persistence'] and gates['state_only']
    return gates,'PROMISING' if all(gates.values()) else ('UNCLEAR' if useful else 'UNPROMISING')

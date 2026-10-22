"""Small learned corrections in object-frame intent coordinates, above frozen R."""
from dataclasses import replace
import numpy as np
import torch
from torch import nn
from scipy.spatial.transform import Rotation

RESIDUAL_SCALE=np.array([.01]*3+[.04]*3+[.03]*6,np.float32)
RESIDUAL_CONTRACT='grasp.R-intent.bounded10mm40mrad30mrad.residual.v1'


class IntentResidualPolicy(nn.Module):
    def __init__(self):
        super().__init__();self.trunk=nn.Sequential(nn.Linear(698,128),nn.Tanh(),nn.Linear(128,64),nn.Tanh())
        self.mean=nn.Linear(64,12);self.value=nn.Linear(64,1);self.logstd=nn.Parameter(torch.full((12,),-2.5))
        nn.init.zeros_(self.mean.weight);nn.init.zeros_(self.mean.bias)
    def forward(self,x):
        h=self.trunk(x);return self.mean(h),self.value(h).squeeze(-1)
    def distribution(self,x):
        mean,value=self(x);return torch.distributions.Normal(mean,self.logstd.clamp(-4.,-1.).exp()),value


def residual_observation(observation,intent):
    pose=intent.approach_pose_object
    r=np.r_[pose[:3,3],Rotation.from_matrix(pose[:3,:3]).as_rotvec(),intent.preshape_q]
    return np.r_[np.clip(observation,-5,5),r].astype(np.float32)


def apply_intent_residual(intent,latent):
    correction=np.tanh(np.asarray(latent))*RESIDUAL_SCALE
    delta=np.eye(4);delta[:3,:3]=Rotation.from_rotvec(correction[3:6]).as_matrix();delta[:3,3]=correction[:3]
    # Rotating a keypose must not rotate its object-relative translation about
    # the object origin; corrections are independent local position/rotation.
    def changed(pose):
        pose=pose.copy();pose[:3,3]+=correction[:3];pose[:3,:3]=delta[:3,:3]@pose[:3,:3];return pose
    return replace(intent,approach_pose_object=changed(intent.approach_pose_object),grasp_pose_object=changed(intent.grasp_pose_object),
                   preshape_q=intent.preshape_q+correction[6:],closure_q=intent.closure_q+correction[6:],approach_joint_angles=intent.approach_joint_angles if np.all(correction[3:6]==0) else None)


def ppo_update(policy,optimizer,observations,latents,old_logp,rewards,values,*,bootstrap=0.,gamma=.96,lam=.95,epochs=4):
    """Fixed-policy rollout; horizon truncation bootstraps, physical terminal zero."""
    device=next(policy.parameters()).device
    obs=torch.as_tensor(np.asarray(observations),device=device);act=torch.as_tensor(np.asarray(latents),device=device)
    old=torch.as_tensor(np.asarray(old_logp),device=device);val=np.r_[values,bootstrap];advantages=np.zeros(len(rewards),np.float32);tail=0.
    for t in reversed(range(len(rewards))):
        tail=rewards[t]+gamma*val[t+1]-val[t]+gamma*lam*tail;advantages[t]=tail
    returns=torch.as_tensor(advantages+np.asarray(values),device=device,dtype=torch.float32)
    adv=torch.as_tensor(advantages,device=device);adv=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-6)
    records=[]
    for epoch in range(epochs):
        dist,value=policy.distribution(obs);logp=dist.log_prob(act).sum(-1);ratio=(logp-old).exp()
        kl=float((old-logp).mean().detach())
        if epoch and kl>.03:break
        loss=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()+.5*(value-returns).square().mean()
        optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(policy.parameters(),.5);optimizer.step()
        records.append(dict(loss=float(loss.detach()),approx_kl=kl))
    return dict(optimizer_updates=len(records),epochs=records,return_sum=float(np.sum(rewards)),target_return_mean=float(returns.mean()),bootstrap=float(bootstrap))

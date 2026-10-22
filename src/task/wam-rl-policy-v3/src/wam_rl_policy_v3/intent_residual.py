"""Small learned corrections in object-frame intent coordinates, above frozen R."""
from copy import deepcopy
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


def ppo_update(policy,optimizer,observations,latents,old_logp,rewards,values,*,bootstrap=0.,gamma=.96,lam=.95,epochs=2,target_kl=.03):
    """Update one fixed-policy rollout with an exact, adaptive KL gate.

    ``old_logp`` is the log probability of the sampled action and is useful
    for the PPO ratio, but it is not a distribution KL.  The gate therefore
    compares the pre-update Normal distribution with every candidate policy
    after the optimizer step.  An over-target candidate is rolled back and
    retried at a quarter learning rate; the first update is checked too.
    """
    if epochs < 1 or target_kl <= 0:
        raise ValueError("positive PPO epochs and target_kl required")
    device=next(policy.parameters()).device
    obs=torch.as_tensor(np.asarray(observations),device=device,dtype=torch.float32)
    act=torch.as_tensor(np.asarray(latents),device=device,dtype=torch.float32)
    old=torch.as_tensor(np.asarray(old_logp),device=device,dtype=torch.float32)
    reward_tensor=torch.as_tensor(np.asarray(rewards),device=device,dtype=torch.float32)
    val=np.r_[np.asarray(values,dtype=np.float32),bootstrap];advantages=np.zeros(len(rewards),np.float32);tail=0.
    for t in reversed(range(len(rewards))):
        tail=rewards[t]+gamma*val[t+1]-val[t]+gamma*lam*tail;advantages[t]=tail
    returns=torch.as_tensor(advantages+np.asarray(values,dtype=np.float32),device=device,dtype=torch.float32)
    adv=torch.as_tensor(advantages,device=device);adv=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-6)
    with torch.no_grad():
        old_dist,_=policy.distribution(obs)
        old_dist=torch.distributions.Normal(old_dist.loc.detach(),old_dist.scale.detach())
        replay_error=float((old_dist.log_prob(act).sum(-1)-old).abs().max())
    if replay_error > 1e-4:
        raise ValueError("behavior log probability replay drift: "+str(replay_error))
    records=[];updates=0;last_kl=0.
    for epoch in range(epochs):
        before={key:value.detach().clone() for key,value in policy.state_dict().items()}
        optimizer_before=deepcopy(optimizer.state_dict())
        accepted=False;trial_lr=float(optimizer.param_groups[0]['lr']);trial_kl=0.;trial_loss=0.
        trial_approx=0.;trial_clip=0.;trial_entropy=0.
        # Every retry starts from the same pre-step state, so the KL gate
        # controls the first optimizer update as well as later epochs.
        for _retry in range(4):
            policy.load_state_dict(before);optimizer.load_state_dict(optimizer_before)
            for group in optimizer.param_groups:group['lr']=trial_lr
            dist,value=policy.distribution(obs);logp=dist.log_prob(act).sum(-1);ratio=(logp-old).exp()
            loss=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()+.5*(value-returns).square().mean()
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(policy.parameters(),.5);optimizer.step()
            with torch.no_grad():
                candidate,_=policy.distribution(obs)
                trial_kl=float(torch.distributions.kl_divergence(old_dist,candidate).sum(-1).mean())
                candidate_logp=candidate.log_prob(act).sum(-1)
                trial_approx=float((old-candidate_logp).mean())
                candidate_ratio=(candidate_logp-old).exp()
                trial_clip=float(((candidate_ratio<.8)|(candidate_ratio>1.2)).float().mean())
                trial_entropy=float(candidate.entropy().sum(-1).mean())
            trial_loss=float(loss.detach())
            if trial_kl <= target_kl:
                accepted=True
                break
            trial_lr*=.25
        if not accepted:
            policy.load_state_dict(before);optimizer.load_state_dict(optimizer_before)
            for group in optimizer.param_groups:group['lr']=trial_lr
            records.append(dict(epoch=epoch,accepted=False,exact_kl=trial_kl,
                                approx_kl=trial_approx,clip_fraction=trial_clip,
                                entropy=trial_entropy,lr=trial_lr,loss=trial_loss,
                                reason='target_kl_reject'))
            last_kl=trial_kl
            break
        records.append(dict(epoch=epoch,accepted=True,loss=trial_loss,
                            exact_kl=trial_kl,approx_kl=trial_approx,
                            clip_fraction=trial_clip,entropy=trial_entropy,
                            lr=float(optimizer.param_groups[0]['lr'])))
        updates+=1;last_kl=trial_kl
    return dict(optimizer_updates=updates,epochs=records,target_kl=float(target_kl),
                final_exact_kl=float(last_kl),behavior_logprob_error=replay_error,
                return_sum=float(reward_tensor.sum()),return_mean=float(reward_tensor.mean()),
                reward_min=float(reward_tensor.min()),reward_max=float(reward_tensor.max()),
                target_return_mean=float(returns.mean()),advantage_std=float(adv.std(unbiased=False)),
                bootstrap=float(bootstrap))

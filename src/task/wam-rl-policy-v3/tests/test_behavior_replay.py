import torch
from wam_rl_policy_v3.recipe_ppo import behavior_replay_error


def test_replay_preserves_collection_batch_and_detects_corruption():
    class Actor:
        def __init__(self):self.shift=0.;self.sizes=[]
        def distribution(self,x):
            self.sizes.append(len(x))
            return torch.distributions.Normal(x+self.shift,torch.ones_like(x)*.05)
    actor=Actor();obs=torch.randn(5,3,24);raw=obs+.01
    dist=torch.distributions.Normal(obs,torch.ones_like(obs)*.05)
    batch=dict(observation=obs,latent=raw,logp=dist.log_prob(raw).sum(-1),mean=obs.clone(),std=dist.scale.clone())
    assert behavior_replay_error(actor,batch)==0
    assert actor.sizes==[3]*5
    batch['mean'][0,0,0]+=.1
    assert behavior_replay_error(actor,batch)>1e-4
    batch['mean']=obs.clone();actor.shift=.01
    assert behavior_replay_error(actor,batch)>1e-4

"""Trainable categorical options with a frozen, direct physical Cm prior."""
import torch
from torch import nn
from .contact_ranker import physical_history
from .contact_trajectory import TrajectoryNetwork,all_trajectories,decode_trajectories,retained_choice
from .trajectory_selector import effective_commands


class FrozenRecoveryPhysics:
    def __init__(self,checkpoint,device):
        self.saved=torch.load(checkpoint,map_location=device,weights_only=False);self.models=[]
        if not self.saved['setup_gate']['passed'] or self.saved['score_key']!='supported_change_mm':raise ValueError('frozen signed-support Cm required')
        with torch.random.fork_rng(devices=[]):
            for state in self.saved['models']['cm']:
                model=TrajectoryNetwork().to(device);model.load_state_dict(state);model.eval().requires_grad_(False);self.models.append(model)

    @torch.no_grad()
    def inputs(self,history,candidate,rest,motion,start,trigger):
        s=self.saved
        h=(physical_history(history,rest)-s['history_mean'])/s['history_scale']
        a=(candidate-s['action_mean'])/s['action_scale']
        c=torch.cat((candidate[:,4],torch.nn.functional.one_hot(motion.long(),3).float(),
                     ((start+trigger).float()/600)[:,None]),-1)
        c=(c-s['context_mean'])/s['context_scale']
        raw=torch.stack([all_trajectories(model,h,a,c) for model in self.models])
        if not all(torch.isfinite(v).all() for v in (h,a,c,raw)):raise ValueError('nonfinite physical policy input')
        prediction=decode_trajectories(raw,s['height_mean'],s['height_scale'],((history[:,-1,38]-rest)*1000)[None,:,None],s['probability_calibration']['cm'])
        recommended,_=retained_choice(prediction,s['calibration']['cm']['margin_mm'],s['release_supported'],score_key=s['score_key'])
        features=torch.cat((prediction['height_mm'].mean(0)/50,prediction['contact'].mean(0),
                            prediction['joint_contact'].mean(0)[...,None],prediction['release'].mean(0)[...,None]),-1)
        return dict(history=h.clamp(-10,10),context=c.clamp(-10,10),actions=effective_commands(a).clamp(-10,10),
                    physics=features.clamp(-10,10),recommended=recommended)


def option_prior(recommended):
    probability=torch.full((len(recommended),6),.02,device=recommended.device)
    probability.scatter_(1,recommended[:,None],.9)
    return probability.log()


class RecoveryOptionPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.history=nn.GRU(69,32,batch_first=True)
        self.context=nn.Sequential(nn.Linear(123,96),nn.SiLU())
        self.score=nn.Sequential(nn.Linear(130,64),nn.SiLU(),nn.Linear(64,1))
        self.value=nn.Sequential(nn.Linear(96,64),nn.SiLU(),nn.Linear(64,1))
        for head in (self.score,self.value):nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)

    def forward(self,history,context,actions,physics,log_prior):
        _,hidden=self.history(history)
        x=self.context(torch.cat((hidden[-1],history[:,-1],context),-1))
        bank=torch.cat((x[:,None].expand(-1,6,-1),actions,physics),-1)
        distribution=torch.distributions.Categorical(logits=self.score(bank).squeeze(-1)+log_prior)
        return distribution,self.value(x).squeeze(-1)


def policy_inputs(physical,cm_on):
    return dict(history=physical['history'],context=physical['context'],actions=physical['actions'],
                physics=physical['physics'] if cm_on else torch.zeros_like(physical['physics']),
                log_prior=option_prior(physical['recommended'] if cm_on else torch.full_like(physical['recommended'],4)))


def ppo_loss(distribution,value,selected,old_logprob,advantage,returns):
    ratio=(distribution.log_prob(selected)-old_logprob).exp()
    clipped=ratio.clamp(.8,1.2)
    actor=-torch.minimum(ratio*advantage,clipped*advantage).mean()
    value_loss=(value-returns).square().mean()
    entropy=distribution.entropy().mean()
    return actor+.5*value_loss-.01*entropy,dict(ratio=ratio,actor=actor,value=value_loss,entropy=entropy)

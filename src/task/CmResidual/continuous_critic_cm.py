"""Continuous Gaussian REQUEST policy and causal physical auxiliary V critic.

Requests and executable targets have different semantics. PPO evaluates the
request distribution; the dynamics branch consumes current state and executed
targets. The V baseline does not consume actions. No predicted reward is used.
"""
import math
import torch
from src.task.CmResidual.selective_finger_response import primitive_target

INDEPENDENT=(0,1,2,3,4,5,6,8,10,12,14,15)
RESIDUAL_SCALE=(.02,)*3+(.10,)*3+(.15,)*6
SCHEMA='hold70-continuous12-request-causal-critic-v1'


def policy_groups(motion,seed):
    rng=torch.Generator(device='cpu').manual_seed(seed+16000)
    group=torch.full((len(motion),),-1,dtype=torch.long)
    for m in range(3):
        ids=(motion.cpu()==m).nonzero().flatten()
        if len(ids)!=256:raise ValueError('balanced768env continuous comparison')
        group[ids]=torch.arange(4).repeat_interleave(64)[torch.randperm(256,generator=rng)]
    return group


class RequestActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network=torch.nn.Sequential(torch.nn.Linear(70,64),torch.nn.ReLU(),
            torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,12))
        torch.nn.init.zeros_(self.network[-1].weight)
        torch.nn.init.zeros_(self.network[-1].bias)
        self.log_std=torch.nn.Parameter(torch.full((12,),math.log(.05)))

    def forward(self,state):
        if state.ndim!=2 or state.shape[-1]!=70:raise ValueError('current normalized70 features')
        return self.network(state.clamp(-10,10)),self.log_std.clamp(-5,0)


class PhysicalCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=torch.nn.Sequential(torch.nn.Linear(70,64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU())
        self.value=torch.nn.Linear(64,1)
        self.dynamics=torch.nn.Sequential(torch.nn.Linear(76,64),torch.nn.ReLU(),torch.nn.Linear(64,6))
        torch.nn.init.zeros_(self.value.weight)
        torch.nn.init.zeros_(self.value.bias)

    def forward(self,state,executed_action,variant):
        if state.ndim!=2 or state.shape[-1]!=70 or executed_action.shape!=(len(state),12):raise ValueError('causal critic input schema')
        if variant not in ('cm','state_only','none'):raise ValueError('predeclared critic comparison')
        latent=self.encoder(state.clamp(-10,10))
        # Recorded command is not a differentiable model-generated actor reward.
        action=executed_action.detach() if variant=='cm' else torch.zeros_like(executed_action)
        dynamics=self.dynamics(torch.cat((latent,action),-1))
        return self.value(latent).squeeze(-1),dynamics


def initialized_models(seed=762):
    # Restore global CPU RNG; never reset CUDA RNG used by native simulation.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        actor=RequestActor()
        torch.random.default_generator.manual_seed(seed+1)
        critic=PhysicalCritic()
    return actor,critic


def request_log_probability(mean,log_std,requests):
    if mean.shape!=requests.shape or mean.ndim!=2 or mean.shape[-1]!=12 or log_std.shape!=(12,):raise ValueError('Gaussian request schema')
    log_std=log_std.clamp(-5,0)
    value=-.5*((requests-mean)*torch.exp(-log_std)).square()-log_std-.5*math.log(2*math.pi)
    result=value.sum(-1)
    if not torch.isfinite(result).all():raise ValueError('nonfinite request likelihood')
    return result


def actor_loss(mean,log_std,requests,old_log_probability,advantage,clip=.2,entropy_weight=.001):
    n=len(mean)
    if old_log_probability.shape!=(n,) or advantage.shape!=(n,):raise ValueError('frozen behavior likelihood and advantage required')
    log_probability=request_log_probability(mean,log_std,requests.detach())
    ratio=torch.exp(log_probability-old_log_probability.detach())
    objective=torch.minimum(ratio*advantage.detach(),ratio.clamp(1-clip,1+clip)*advantage.detach())
    entropy=(log_std.clamp(-5,0)+.5*math.log(2*math.pi*math.e)).sum()
    loss=-objective.mean()-entropy_weight*entropy
    if not torch.isfinite(loss):raise ValueError('nonfinite PPO request loss')
    return loss


def critic_loss(value,prediction,actual_return,transition,variant):
    if prediction.shape!=(len(value),6) or transition.shape!=prediction.shape or actual_return.shape!=value.shape:raise ValueError('actual return/physical transition target schema')
    if variant not in ('cm','state_only','none'):raise ValueError('critic loss variant')
    value_error=(value-actual_return.detach()).square().mean()
    physical_error=(prediction-transition.detach()).square().mean()
    loss=.5*value_error+(0. if variant=='none' else .05)*physical_error
    if not torch.isfinite(loss):raise ValueError('nonfinite physical critic loss')
    return loss


def executable_target(base,requests,current_q,lower,upper,pd_scale):
    if base.shape!=current_q.shape or base.shape!=(len(requests),18) or requests.shape!=(len(base),12) or pd_scale.shape!=(18,):raise ValueError('native target projection schema')
    delta=torch.tanh(requests)*base.new_tensor(RESIDUAL_SCALE)
    desired=base.clone()
    radius=pd_scale[:6].abs()
    desired[:,:6]=torch.maximum(torch.minimum(base[:,:6]+delta[:,:6],current_q[:,:6]+radius),current_q[:,:6]-radius)
    return primitive_target(desired,delta[:,6:],lower,upper)


def physical_transition_target(current_root,next_root):
    if current_root.shape!=next_root.shape or current_root.ndim!=2 or current_root.shape[-1]!=13:raise ValueError('one-step physical object packet')
    return torch.cat((next_root[:,:3]-current_root[:,:3],next_root[:,7:10]-current_root[:,7:10]),-1)/current_root.new_tensor((.005,)*3+(.05,)*3)

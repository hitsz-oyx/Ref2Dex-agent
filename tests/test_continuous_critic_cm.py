import math
import numpy as np
import torch
from src.task.CmResidual.continuous_critic_cm import initialized_models,request_log_probability,actor_loss,critic_loss,executable_target,physical_transition_target


def test_request_ppo_gradient_matches_independent_numpy_with_both_clip_branches():
    request=np.arange(48,dtype=np.float64).reshape(4,12)/80-.2
    advantage=np.array([1.,-1.,1.,-1.])
    def logp(mu):
        return (-.5*(request-mu)**2-.5*math.log(2*math.pi)).sum(-1)
    old=logp(0)-np.log(np.array([1.5,1.5,.5,.5]))
    def objective(mu):
        ratio=np.exp(logp(mu)-old)
        return -np.minimum(ratio*advantage,np.clip(ratio,.8,1.2)*advantage).mean()
    mu=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    loss=actor_loss(mu.expand(4,12),torch.zeros(12,dtype=torch.float64),torch.from_numpy(request),torch.from_numpy(old),torch.from_numpy(advantage),entropy_weight=0)
    loss.backward()
    eps=1e-6;finite=(objective(eps)-objective(-eps))/(2*eps)
    assert abs(float(mu.grad)-finite)<1e-8
    assert abs(float(loss)-objective(0))<1e-12
    assert np.allclose(request_log_probability(torch.zeros((4,12),dtype=torch.float64),torch.zeros(12,dtype=torch.float64),torch.from_numpy(request)).numpy(),logp(0))


def test_value_is_state_only_and_auxiliary_does_not_update_actor():
    before=torch.get_rng_state().clone();actor,critic=initialized_models()
    assert torch.equal(before,torch.get_rng_state())
    state=torch.zeros((3,70));mean,_=actor(state)
    first,p1=critic(state,mean,'cm');second,p2=critic(state,mean+1,'cm')
    assert torch.equal(first,second) and not torch.equal(p1,p2)
    _,off1=critic(state,mean,'state_only');_,off2=critic(state,mean+1,'state_only')
    assert torch.equal(off1,off2)
    loss=critic_loss(first,p1,torch.ones(3),torch.ones((3,6)),'cm');loss.backward()
    assert all(p.grad is None for p in actor.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in critic.parameters())


def test_projection_preserves_native_constraints_and_transition_is_one_step():
    base=torch.full((2,18),.1);q=base.clone();lo=torch.zeros(18);hi=torch.ones(18);hi[7]=.12
    target=executable_target(base,torch.full((2,12),10.),q,lo,hi,torch.full((18,),.01))
    assert torch.all(target[:,:6]<=q[:,:6]+.010001)
    assert torch.all(target[:,:6]>=q[:,:6]-.010001)
    for parent,children in {6:((7,1.05),),8:((9,1.05),),10:((11,1.05),),12:((13,1.05),),15:((16,.6),(17,.8))}.items():
        for child,ratio in children:assert torch.equal(target[:,child],target[:,parent]*ratio)
    assert torch.all(target[:,7]<=.120001)
    current=torch.zeros((2,13));future=current.clone();future[:,:3]=.005;future[:,7:10]=.05
    assert torch.equal(physical_transition_target(current,future),torch.ones((2,6)))

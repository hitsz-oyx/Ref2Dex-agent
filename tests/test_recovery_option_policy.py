import torch
from src.task.CmResidual.recovery_option_policy import RecoveryOptionPolicy,option_prior,policy_inputs,ppo_loss


def inputs(n=8):
    return dict(history=torch.randn(n,10,69),context=torch.randn(n,22),actions=torch.randn(n,6,12),
                physics=torch.randn(n,6,22),recommended=torch.arange(n)%6)


def test_initial_policy_executes_the_physical_prior_with_equal_on_off_entropy():
    model=RecoveryOptionPolicy();physical=inputs()
    on,_=model(**policy_inputs(physical,True));off,_=model(**policy_inputs(physical,False))
    assert torch.allclose(on.probs,option_prior(physical['recommended']).exp())
    assert torch.allclose(on.entropy(),off.entropy())
    assert (off.probs.argmax(-1)==4).all()
    assert torch.equal(policy_inputs(physical,False)['physics'],torch.zeros_like(physical['physics']))


def test_actual_option_logprob_ratio_and_finite_ppo_update_change_policy():
    model=RecoveryOptionPolicy();x=policy_inputs(inputs(16),True);optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    distribution,value=model(**x);selected=distribution.probs.argmax(-1);old=distribution.log_prob(selected).detach()
    before=distribution.probs.detach().clone()
    loss,stats=ppo_loss(distribution,value,selected,old,torch.ones(16),torch.ones(16))
    assert torch.equal(stats['ratio'],torch.ones(16))
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
    after,_=model(**x)
    assert not torch.equal(before,after.probs)
    assert after.probs[torch.arange(16),selected].mean()>before[torch.arange(16),selected].mean()


def test_clipped_surrogate_limits_large_positive_actual_choice_ratio():
    selected=torch.zeros(2,dtype=torch.long)
    distribution=torch.distributions.Categorical(probs=torch.tensor([[.8,.2],[.8,.2]]))
    old=torch.tensor([.1,.1]).log();value=torch.zeros(2)
    _,stats=ppo_loss(distribution,value,selected,old,torch.ones(2),torch.zeros(2))
    assert torch.allclose(stats['ratio'],torch.full((2,),8.))
    assert abs(float(stats['actor'])+1.2)<1e-6

import torch
from src.task.CmResidual.support_feature_policy import actor,policy_input,offpolicy_score_loss


def test_initial_policy_is_identical_with_different_physical_features():
    network = actor()
    state = torch.randn(5,69)
    a = network(policy_input(state,torch.zeros(5,8)))
    b = network(policy_input(state,torch.rand(5,8)))
    expected = torch.zeros(5,8)
    expected[:,0] = 2
    assert torch.equal(a,expected)
    assert torch.equal(b,expected)
    assert torch.equal(a.argmax(1),torch.zeros(5,dtype=torch.long))


def test_exact_importance_score_gradient_and_state_baseline_cancellation():
    # Analytic expected-reward derivative over all actions, independently of
    # the score-loss implementation and with a nonuniform behavior law.
    initial = torch.tensor([2.,-.4,.3,-.6,.1,.2,-.1,.5],dtype=torch.float64)
    reward = torch.tensor([0.,1.,.2,0.,.8,.4,1.,.1],dtype=torch.float64)
    behavior = torch.tensor([.1,.2,.05,.15,.1,.05,.2,.15],dtype=torch.float64)
    reference = initial.clone().requires_grad_(True)
    expected = -(reference.softmax(0)*reward).sum()
    correct, = torch.autograd.grad(expected,reference)
    for baseline in (0.,.4,3.):
        logits = initial.clone().requires_grad_(True)
        objective = torch.zeros((),dtype=torch.float64)
        for a in range(8):
            objective = objective+behavior[a]*offpolicy_score_loss(logits[None],
                torch.tensor([a]),reward[a:a+1],behavior[a:a+1],
                torch.tensor([baseline],dtype=torch.float64))
        actual, = torch.autograd.grad(objective,logits)
        assert torch.allclose(actual,correct,atol=1e-12,rtol=1e-12)

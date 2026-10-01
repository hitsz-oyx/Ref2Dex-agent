import torch
from src.task.CmResidual.support_gradient_cv import corrected_logit_gradient

def test_exact_correction_cancels_arbitrary_misspecified_physical_forecast():
    pi=torch.tensor([[.55,.15,.20,.10]],dtype=torch.float64).repeat(4,1)
    behavior=torch.tensor([[.1,.2,.3,.4]],dtype=torch.float64).repeat(4,1)
    action=torch.arange(4);reward=torch.tensor([1.,0.,1.,0.],dtype=torch.float64)
    wrong=torch.tensor([[.0,.9,.1,.7]],dtype=torch.float64).repeat(4,1)
    g=corrected_logit_gradient(pi,behavior,action,reward,wrong)
    expected=(g*behavior[0,:,None]).sum(0)
    truth=pi[0]*(reward-(pi[0]*reward).sum())
    torch.testing.assert_close(expected,truth,atol=1e-12,rtol=0)

def test_state_only_forecast_reduces_to_standard_scalar_baseline():
    pi=torch.tensor([[.4,.6]],dtype=torch.float64);behavior=torch.tensor([[.5,.5]],dtype=torch.float64)
    action=torch.tensor([1]);reward=torch.tensor([1.],dtype=torch.float64);q=torch.ones(1,2,dtype=torch.float64)*.3
    result=corrected_logit_gradient(pi,behavior,action,reward,q)
    torch.testing.assert_close(result,(.6/.5)*torch.tensor([[-.4,.4]],dtype=torch.float64)*.7)

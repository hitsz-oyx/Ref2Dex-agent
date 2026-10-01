import pytest
import torch
from src.task.CmResidual.randomized_effect_risk import pseudo_contrast,risk_difference


@pytest.mark.parametrize('probabilities',[[1/7]*7,[.2,.1,.15,.1,.2,.1,.15]])
def test_identified_risk_difference_equals_known_causal_risk(probabilities):
    # Enumerate every randomized treatment, retaining arbitrary common drift.
    y=torch.tensor([[.3,.5,.1],[.4,.5,.1],[.2,.6,.1],[.3,.7,.2],[.4,.3,.2],[.3,.6,.3],[.3,.4,-.1]],dtype=torch.float64)
    p=torch.tensor(probabilities,dtype=torch.float64)
    probabilities=p.expand(7,-1)
    pseudo=pseudo_contrast(y,torch.arange(7),probabilities)
    truth=torch.stack((y[1]-y[2],y[3]-y[4],y[5]-y[6]),-1)
    first=(truth+.02).expand(7,-1,-1);second=(truth-.1).expand(7,-1,-1)
    observed=(risk_difference(first,second,pseudo)*p).sum()
    exact=((first[0]-truth).square()-(second[0]-truth).square()).sum()
    assert torch.allclose(observed,exact,atol=1e-12)
    assert torch.allclose((pseudo*p[:,None,None]).sum(0),truth,atol=1e-12)


def test_positive_propensity_is_required():
    p=torch.zeros(1,7);p[:,0]=1
    with pytest.raises(ValueError):pseudo_contrast(torch.zeros(1,3),torch.zeros(1,dtype=torch.long),p)

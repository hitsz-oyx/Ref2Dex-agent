import torch
from src.task.CmResidual.direct_randomized_response import iid_assignment
from src.task.CmResidual.randomized_effect_risk import pseudo_contrast


def test_arbitrary_context_only_centering_preserves_randomized_contrast():
    # Enumerated potential responses, unequal propensities, arbitrary nuisance error.
    y=torch.arange(21,dtype=torch.float64).reshape(7,3)/10
    p=torch.tensor([.2,.1,.15,.1,.2,.1,.15],dtype=torch.float64)
    center=torch.tensor([100.,-40.,7.],dtype=torch.float64)
    z=pseudo_contrast(y-center,torch.arange(7),p.expand(7,-1))
    truth=torch.stack((y[1]-y[2],y[3]-y[4],y[5]-y[6]),-1)
    assert torch.allclose((z*p[:,None,None]).sum(0),truth,atol=1e-12)


def test_iid_assignment_is_reproducible_and_does_not_mutate_native_rng():
    torch.manual_seed(33);before=torch.get_rng_state().clone()
    arms,propensity=iid_assignment(768,494)
    assert torch.equal(before,torch.get_rng_state())
    generator=torch.Generator().manual_seed(9900+494)
    assert torch.equal(arms,torch.randint(7,(768,),generator=generator))
    assert torch.equal(propensity,torch.full((768,7),1/7))
    assert torch.equal(iid_assignment(768,494)[0],arms)

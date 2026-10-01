import torch
from src.task.CmResidual.support_feature_bank import evaluation_groups


def test_balanced_blinded_groups_without_advancing_placement_rng():
    motion=torch.arange(768)%3
    torch.manual_seed(527)
    rng=torch.random.get_rng_state().clone()
    groups=evaluation_groups(motion,541)
    assert torch.equal(rng,torch.random.get_rng_state())
    for m in range(3):
        assert torch.equal(torch.bincount(groups[motion==m],minlength=4),torch.full((4,),64))
    assert torch.equal(groups,evaluation_groups(motion,541))
    assert not torch.equal(groups,evaluation_groups(motion,542))

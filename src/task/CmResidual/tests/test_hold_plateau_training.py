import pytest
import torch
from src.task.CmResidual.hold_plateau_training import plateau_probability,phase_lift_bonus


@pytest.mark.parametrize('epoch,expected',[(0,.5),(80,.5),(140,.25),(200,0.),(300,0.)])
def test_frozen_curriculum_completes_before_final_epoch(epoch,expected):
    assert plateau_probability(epoch)==expected


def test_phase_reward_stops_before_planned_release_and_requires_both_proxies():
    reward=phase_lift_bonus(torch.full((4,),.06),torch.zeros(4),
        torch.tensor([[1.,1.],[1.,1.],[1.,0.],[1.,1.]]),
        torch.tensor([51,140,100,141]),torch.full((4,),51),torch.full((4,),140))
    assert reward.tolist()==[10.,10.,0.,0.]


def test_checkpoint_identity_is_invariant_to_compile_wrapper():
    from collections import OrderedDict
    from src.task.CmResidual.hold_plateau_training import canonical_model_state
    from src.task.CmResidual.paired_evaluation import fingerprint
    native=OrderedDict(weight=torch.tensor([1.,2.]))
    compiled=OrderedDict({'_orig_mod.weight':native['weight']})
    assert fingerprint(canonical_model_state(compiled))==fingerprint(native)
    with pytest.raises(ValueError,match='ambiguous'):
        canonical_model_state({'weight':native['weight'],'_orig_mod.weight':native['weight']})

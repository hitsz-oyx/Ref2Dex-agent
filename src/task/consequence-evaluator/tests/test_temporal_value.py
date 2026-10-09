import numpy as np
import pytest
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))

from consequence_evaluator.supervision import consecutive
from consequence_evaluator.temporal_value import episode_labels,relative_time_labels,soft_bins,TemporalValue


def recovery_trace(recover=True):
    n=160;near=np.ones(n,bool);near[62:70]=False;near[135:]=False
    held=np.ones(n,bool);held[62:70]=False;held[135:]=False
    if not recover:
        near[62:]=False;held[62:]=False
    support=np.zeros(n,bool);support[135:]=True
    height=np.full(n,.1);height[62:70]=.01;height[135:]=0
    drop=(height<.02)|(consecutive(~near)>=6)
    return dict(valid=np.ones(n,bool),near=near,supported=support,held_run=consecutive(held),
        held=held,height=height,drop=drop,place_start=135,settled=support),np.zeros((n,6))


def test_recovered_completion_is_positive_but_loss_until_restabilization_is_masked():
    trace,velocity=recovery_trace();success,mask=episode_labels(trace,velocity)
    assert success==1
    assert mask[62:114].all() and not mask[114:].any()
    assert mask[50:75].any() and not mask[115:140].any()


def test_settling_without_restabilizing_does_not_make_failure_success():
    trace,velocity=recovery_trace(False)
    assert episode_labels(trace,velocity)[0]==0


def test_confirmed_loss_masks_onset_before_the_six_frame_confirmation():
    trace,velocity=recovery_trace()
    trace['height'][62:70]=.1
    trace['drop']=consecutive(~trace['near'])>=6
    success,mask=episode_labels(trace,velocity)
    assert success==1
    assert mask[62:114].all() and not mask[61] and not mask[114]


def test_relative_labels_use_only_outcome_and_remaining_complete_horizon():
    y=relative_time_labels([1,0,1],[0,0,518],[542,542,542])
    assert np.allclose(y,[24/542,-24/542,1])
    with pytest.raises(ValueError):
        relative_time_labels([1],[519],[542])
    target=torch.tensor([-.046,0.,.046,1.])
    distribution=soft_bins(target)
    assert torch.allclose(distribution.sum(-1),torch.ones(4))
    assert torch.allclose(distribution@torch.linspace(-1,1,41),target,atol=1e-7)


def test_ha_control_cannot_read_future_and_haz_can():
    torch.manual_seed(11);model=TemporalValue(9,width=16,layers=1).eval()
    h=torch.randn(2,9);a=torch.randn(2,24,18);z=torch.randn(2,24,45)
    with torch.no_grad():
        assert torch.equal(model(h,a,z,False)['score'],model(h,a,z.flip(0),False)['score'])
        assert not torch.equal(model(h,a,z,True)['score'],model(h,a,z.flip(0),True)['score'])

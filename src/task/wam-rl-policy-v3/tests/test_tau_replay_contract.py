import numpy as np
import pytest
from wam_rl_policy_v3.tau_replay_contract import reference_tau,require_saved_clock,pd_targets
from wam_rl_policy_v3.inspire_contract import expand_action


def test_reference_delta_is_first_future_and_does_not_reanchor_on_live_hand():
    reference=np.arange(40*33).reshape(40,11,3)*.001
    tau,valid=reference_tau(reference,3)
    assert valid
    np.testing.assert_allclose(tau[0],reference[4]-reference[3],atol=1e-7)
    live=reference[3]+.07
    np.testing.assert_allclose(live+tau[-1]-reference[11],np.full((11,3),.07),atol=1e-7)
    assert not reference_tau(reference,2)[1] and not reference_tau(reference,32)[1]


def test_saved_clock_rejects_shifted_post_action_state():
    data=dict(action=np.zeros((4,12)),q=np.zeros((5,18)),tick=np.arange(5)[:,None],tau=np.zeros((5,8,11,3)),tau_valid=np.ones(5,bool))
    require_saved_clock(data);data['tick']=data['tick']+1
    with pytest.raises(ValueError,match='consecutive clock'):require_saved_clock(data)


def test_pd_reconstruction_matches_independent_torch_native_contract():
    import torch,sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'src/task/trajectory-policy/src'))
    from trajectory_policy.pd_retargeter import pd_from_action
    action=np.linspace(-.8,.8,24).reshape(2,12).astype(np.float32);q=np.arange(36).reshape(2,18)*.01
    scale=np.linspace(.5,3.,18);offset=np.linspace(-.1,.1,18)
    expected=pd_from_action(torch.tensor(expand_action(action),dtype=torch.float64),torch.tensor(q),torch.tensor(offset),torch.tensor(scale)).numpy()
    np.testing.assert_allclose(pd_targets(action,q,offset,scale),expected,atol=1e-7)

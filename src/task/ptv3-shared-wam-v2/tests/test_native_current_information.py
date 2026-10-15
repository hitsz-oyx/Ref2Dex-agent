from pathlib import Path
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/run'))
from native_current_information import NativeInformationRetargeter,current_rows,EXTRA_DIM
from ptv3_shared_wam.target_retargeter import make_retargeter
from ptv3_shared_wam.action_targets import GEOMETRY_LIFTED


def test_native_rows_use_same_tick_and_reject_clock_or_future_actor():
    data=dict(q=np.zeros((40,18)),native_current_obs=np.arange(40*721).reshape(40,721),frame_tick=np.arange(40))
    before=current_rows(data,[3])
    data['native_current_obs'][4:]=-999
    np.testing.assert_array_equal(current_rows(data,[3]),before)
    data['frame_tick'][3]=2
    with pytest.raises(ValueError,match='clock'): current_rows(data,[3])
    data['frame_tick']=np.arange(40)
    data['native_current_obs']=np.zeros((40,1442))
    with pytest.raises(ValueError,match='721D'): current_rows(data,[3])


def test_initial_parent_function_mask_and_tau_are_preserved():
    torch.manual_seed(9)
    parent=make_retargeter(GEOMETRY_LIFTED).eval()
    torch.nn.init.normal_(parent.head.weight,std=.02)
    model=NativeInformationRetargeter(parent.state_dict(),np.ones(EXTRA_DIM),np.full(EXTRA_DIM,2)).eval()
    state,tau,extra=torch.randn(3,87),torch.randn(3,24,33),torch.randn(3,EXTRA_DIM)
    for arm in ('baseline','native_current'):
        torch.testing.assert_close(model(state,tau,extra,arm),parent(state,tau),atol=1e-6,rtol=0)
    torch.nn.init.normal_(model.extra[-1].weight,std=.02)
    torch.testing.assert_close(model(state,tau,extra,'baseline'),model(state,tau,extra+99,'baseline'))
    assert not torch.allclose(model(state,tau,extra,'native_current'),model(state,tau,extra+1,'native_current'))
    assert not torch.allclose(model(state,tau,extra,'native_current'),model(state,tau+1,extra,'native_current'))

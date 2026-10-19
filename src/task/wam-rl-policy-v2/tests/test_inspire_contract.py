import numpy as np
import pytest
from wam_rl_policy_v2.inspire_contract import (
    ACTIVE, CONTRACT, NATIVE_TO_WAM, WAM_TO_NATIVE, expand_action,
    executor_input, inspire_tau,
)


def test_tau_rotates_displacements_reorders_points_and_keeps_30hz():
    trajectory=np.zeros((24,2,11,3))
    for t in range(24):
        trajectory[t,0,:,0]=(t+1)*.001+np.arange(11)*.01
    pose=np.eye(4);pose[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];pose[:3,3]=[10,20,30]
    result=inspire_tau(trajectory,pose)
    assert result.shape==(8,11,3)
    np.testing.assert_allclose(result[:,:,1],trajectory[:8,0][:,WAM_TO_NATIVE,0],atol=1e-8)
    np.testing.assert_array_equal(result[:,:,[0,2]],0)
    assert CONTRACT['tau_times_s'][-1]==8/30
    np.testing.assert_array_equal(NATIVE_TO_WAM[WAM_TO_NATIVE],np.arange(11))


def test_canonical_actions_expand_only_independent_dofs():
    active=np.linspace(-1,1,12,dtype=np.float32)
    result=expand_action(active)
    np.testing.assert_array_equal(result[list(ACTIVE)],active)
    np.testing.assert_array_equal(result[[7,9,11,13,16,17]],0)
    with pytest.raises(ValueError):expand_action(np.zeros(18))


def test_startup_masks_tau_and_clips_same_current_contract():
    obs=executor_input(np.full(721,10.),np.ones((8,11,3)),False)
    np.testing.assert_array_equal(obs[:721],5)
    np.testing.assert_array_equal(obs[721:],0)
    valid=executor_input(np.zeros(721),np.ones((8,11,3)),True)
    assert valid.shape==(986,) and valid[-1]==1

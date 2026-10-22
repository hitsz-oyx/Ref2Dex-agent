"""Additional CPU regressions for the production v3 continuous-wrist helper."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'src/task/wam-rl-policy-v3/src'))
from wam_rl_policy_v3.continuous_wrist import normalize_wrist_positions

@pytest.mark.parametrize('turns',[-5,-2,-1,1,2,5])
def test_reset_positive_negative_turns_preserve_rotation(turns):
 q=torch.zeros((1,18),dtype=torch.float64);q[0,3:6]=torch.tensor([.2+turns*2*np.pi,-.5-turns*2*np.pi,.9+turns*2*np.pi],dtype=torch.float64)
 fixed=normalize_wrist_positions(q)
 np.testing.assert_allclose(fixed[0,3:6],[.2,-.5,.9],atol=1e-13)
 np.testing.assert_allclose(np.exp(1j*q[0,3:6]),np.exp(1j*fixed[0,3:6]),atol=1e-13)

@pytest.mark.parametrize('value',[-9.,-7.,7.,9.])
def test_noncontinuous_positions_not_wrapped_and_wrong_all_axis_mutant_is_detectable(value):
 q=torch.full((1,18),value,dtype=torch.float64);fixed=normalize_wrist_positions(q)
 untouched=[0,1,2,*range(6,18)]
 assert torch.equal(fixed[:,untouched],q[:,untouched])
 wrong=torch.remainder(q+torch.pi,2*torch.pi)-torch.pi
 assert not torch.equal(wrong[:,untouched],fixed[:,untouched])

@pytest.mark.parametrize('q',[torch.zeros(18),torch.zeros((1,18),dtype=torch.int64),torch.full((1,18),float('nan'))])
def test_wrong_shape_dtype_or_nonfinite_rejected(q):
 with pytest.raises(ValueError):normalize_wrist_positions(q)

def test_batch_and_caller_input_unchanged():
 q=torch.zeros((2,18),dtype=torch.float64);q[:,3:6]=torch.tensor([[7.,-8.,9.],[-7.,8.,-9.]])
 before=q.clone();fixed=normalize_wrist_positions(q)
 assert torch.equal(before,q) and fixed.shape==q.shape
 assert not torch.equal(fixed,q)

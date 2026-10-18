import torch
import pytest
from wam_rl_policy_v2.deferred_checks import install_clean_hooks,checked_prediction


def original(values,mask):
    result=torch.where(mask[...,None],values.float(),0.)
    if not torch.isfinite(result).all():raise ValueError('nonfinite valid observation')
    return result


def test_checks_preserve_masked_nan_and_finite_values():
    clean=install_clean_hooks(original)
    values=torch.tensor([[1.,2.],[float('nan'),float('inf')]])
    mask=torch.tensor([True,False])
    with checked_prediction() as checks:
        torch.testing.assert_close(clean(values,mask),original(values,mask))
        torch.testing.assert_close(clean(values,mask),original(values,mask))
        assert len(checks)==2


def test_invalid_valid_point_rejected_before_prediction_can_escape():
    clean=install_clean_hooks(original);values=torch.tensor([[float('nan')]]);mask=torch.tensor([True])
    with pytest.raises(ValueError,match='nonfinite valid'):
        with checked_prediction():clean(values,mask)
    # The failed context cannot disable checks in subsequent normal calls.
    with pytest.raises(ValueError,match='nonfinite valid'):clean(values,mask)

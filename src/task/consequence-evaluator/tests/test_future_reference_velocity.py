"""Live tracking error is not a nominal planned-velocity signal."""
import torch
from consequence_evaluator.reference_tracking import reference_velocity, future_reference_velocity


def test_future_feedforward_ignores_live_calibration_joint_error():
    q=torch.zeros(5,18)
    q[1:,0]=torch.tensor([.01,.03,.06,.10])
    changed=q.clone();changed[0,:6]=torch.tensor([.4,-.3,.2,2.,-1.,.7])
    assert not torch.equal(reference_velocity(q,.02)[1],reference_velocity(changed,.02)[1])
    torch.testing.assert_close(future_reference_velocity(q,.02),future_reference_velocity(changed,.02),rtol=0,atol=0)
    torch.testing.assert_close(future_reference_velocity(q,.02)[1,0],torch.tensor(1.))
    torch.testing.assert_close(future_reference_velocity(q,.02)[2:],reference_velocity(q,.02)[2:],rtol=0,atol=0)


def test_future_velocity_wraps_native_angles_across_pi():
    q=torch.zeros(4,18);q[1:,3]=torch.tensor([3.13,-3.13,-3.10])
    v=future_reference_velocity(q,.02)
    assert 0<float(v[1,3])<2
    torch.testing.assert_close(v[2:],reference_velocity(q,.02)[2:],rtol=0,atol=0)

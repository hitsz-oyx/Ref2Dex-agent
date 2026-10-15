import pytest
import torch
from ptv3_shared_wam_v2.wrist_probe import override_wrist, predict_wrist


def test_override_preserves_teacher_fingers_and_passives():
    original = torch.arange(54).reshape(3, 18).float()
    wrist = torch.full((2, 6), -2.)
    result = override_wrist(original, wrist)
    assert torch.equal(result[0], original[0])
    assert torch.equal(result[1:, 6:], original[1:, 6:])
    assert torch.equal(result[1:, :6], wrist)
    assert original[1, 0] == 18


def test_current_extra_mask_and_train_normalization():
    net = torch.nn.Linear(811, 6, bias=False)
    with torch.no_grad():
        net.weight.zero_(); net.weight[:, 0] = 1; net.weight[:, 90] = 3
    stats = dict(xmean=torch.ones(811), xstd=torch.full((811,), 2.),
                 ymean=torch.ones(6), ystd=torch.full((6,), 4.))
    x = torch.full((2, 811), 3.)
    assert torch.equal(predict_wrist((net, stats, 'state_goal'), x), torch.full((2, 6), 5.))
    assert torch.equal(predict_wrist((net, stats, 'plus_current'), x), torch.full((2, 6), 17.))
    with pytest.raises(ValueError): predict_wrist((net, stats, 'state_goal'), x[:, :90])

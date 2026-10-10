import copy

import torch

from trajectory_policy.local_retargeter import LocalMotionRetargeter, SCHEMA, load_retargeter
from trajectory_policy.learned_retargeter import LearnedRetargeter


def test_local_motion_zeroes_all_tau_paths_in_state_only_checkpoint():
    torch.manual_seed(12)
    original = LearnedRetargeter(16)
    torch.manual_seed(12)
    model = LocalMotionRetargeter(16).eval()
    for key, tensor in original.state_dict().items():
        assert torch.equal(tensor, model.state_dict()[key])
    torch.nn.init.normal_(model.head.weight, std=.03)
    state = torch.randn(2, 87)
    tau = torch.randn(2, 24, 33, requires_grad=True)
    # A narrow first-frame intervention must propagate to native-action learning.
    model(state, tau)[:, 0].square().sum().backward()
    assert tau.grad[:, 0].abs().sum() > 0
    packet = dict(schema=SCHEMA, width=16, use_tau=False, model=copy.deepcopy(model.state_dict()))
    restored = load_retargeter(packet, 'cpu').eval()
    with torch.no_grad():
        assert torch.equal(restored(state, tau), restored(state, tau*100))
        native = restored.native_action(state, tau)
        assert native.shape == (2, 8, 18)
        assert (native[..., [7, 9, 11, 13, 16, 17]] == 0).all()
        assert native.abs().max() <= 1

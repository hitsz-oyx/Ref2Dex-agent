import torch

from goal_point_wam.effect_consistency import action_from_delta


def test_action_contract_reconstructs_trajectory_and_preserves_gradients():
    start = torch.randn(2, 2, 11, 3)
    delta = torch.randn(2, 24, 2, 11, 3, requires_grad=True)
    valid = torch.ones(2, 24, 2, 11, dtype=torch.bool)
    action, mask = action_from_delta(delta, start, valid)
    assert mask.all()
    torch.testing.assert_close(action[..., :3], start[:, None] + delta)
    torch.testing.assert_close(action[..., 3:6], delta)
    torch.testing.assert_close(action[:, 0, ..., 6:], delta[:, 0])
    torch.testing.assert_close(action[:, 1:, ..., 6:], delta[:, 1:] - delta[:, :-1])
    action.square().sum().backward()
    assert torch.isfinite(delta.grad).all() and delta.grad.abs().sum() > 0


def test_absent_hand_and_missing_predecessor_cannot_supply_world_action():
    delta = torch.ones(1, 24, 2, 11, 3, requires_grad=True)
    valid = torch.ones(1, 24, 2, 11, dtype=torch.bool)
    valid[:, :, 1] = False
    valid[:, 4, 0] = False
    action, mask = action_from_delta(delta, torch.zeros(1, 2, 11, 3), valid)
    assert not mask[:, :, 1].any() and action[:, :, 1].count_nonzero() == 0
    assert not mask[:, 4:6, 0].any() and action[:, 4:6, 0].count_nonzero() == 0


def test_ddp_preserves_effect_ancestor_and_repeated_backward():
    import os
    from pathlib import Path
    import torch.distributed as dist
    from goal_point_wam.effect_consistency import effect_ddp

    class TinyEffect(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = torch.nn.Linear(2, 2)
            self.world = torch.nn.Linear(2, 1).requires_grad_(False)
            self.unused_query = torch.nn.Linear(2, 2).requires_grad_(False)

        def forward(self, x):
            tau = self.decoder(x)
            return dict(trajectory=tau, effect=self.world(tau))

    root = Path(__file__).resolve().parents[4] / 'tmp/wam-effect-diagnosis'
    root.mkdir(parents=True, exist_ok=True)
    path = root / ('test-ddp-%d' % os.getpid())
    dist.init_process_group('gloo', init_method='file://' + str(path), rank=0, world_size=1)
    try:
        model = effect_ddp(TinyEffect())
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            result = model(torch.ones(1, 2))
            loss = result['effect'].square().sum()
            grads = torch.autograd.grad(loss, [result['trajectory'], model.module.decoder.weight], retain_graph=True)
            assert all(g.abs().sum() > 0 for g in grads)
            (loss + result['trajectory'].square().sum()).backward()
            assert model.module.decoder.weight.grad.abs().sum() > 0
            assert all(p.grad is None for p in model.module.world.parameters())
    finally:
        dist.destroy_process_group()

import torch

from goal_point_wam.hand_gated import GatedHandResidual


def test_zero_gate_preserves_query_and_decoder_output():
    torch.manual_seed(295)
    module = GatedHandResidual(8)
    raw, hand = torch.randn(2, 2, 11, 8), torch.randn(2, 2, 11, 8)
    valid = torch.ones(2, 2, 11, dtype=torch.bool)
    decoder = torch.nn.Sequential(torch.nn.LayerNorm(8), torch.nn.Linear(8, 3))
    new = raw + module(hand, valid)
    assert torch.equal(new, raw)
    assert torch.equal(decoder(new), decoder(raw))


def test_invalid_hand_cannot_acquire_projection_bias():
    module = GatedHandResidual(8)
    with torch.no_grad():
        module.alpha.fill_(1)
        module.projection[-1].bias.fill_(7)
    hand = torch.randn(1, 2, 11, 8, requires_grad=True)
    valid = torch.ones(1, 2, 11, dtype=torch.bool)
    valid[:, 1] = False
    residual = module(hand, valid)
    assert residual[:, 1].count_nonzero() == 0
    residual.sum().backward()
    assert hand.grad[:, 1].count_nonzero() == 0


def test_gate_wakes_before_projection_and_checkpoint_restores():
    torch.manual_seed(295)
    module = GatedHandResidual(8)
    hand = torch.randn(2, 2, 11, 8)
    valid = torch.ones(2, 2, 11, dtype=torch.bool)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-4)
    module(hand, valid).sum().backward()
    assert module.alpha.grad.abs() > 0
    assert all(p.grad.count_nonzero() == 0 for p in module.projection.parameters())
    optimizer.step()
    optimizer.zero_grad()
    module(hand, valid).sum().backward()
    assert sum(float(p.grad.abs().sum()) for p in module.projection.parameters()) > 0
    restored = GatedHandResidual(8)
    restored.load_state_dict(module.state_dict(), strict=True)
    assert torch.equal(restored(hand, valid), module(hand, valid))

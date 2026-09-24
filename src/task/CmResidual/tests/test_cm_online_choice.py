from __future__ import annotations

import torch

from src.task.CmResidual.cm_online_choice import (
    apply_wrist_z_down, select_contact_preserving_down,
)


def test_contact_aware_gate_requires_gain_and_lift_preservation():
    obj = torch.zeros(3, 13)
    obj[:, 6] = 1
    base_contact = torch.tensor([.5, .5, .5])
    down_contact = torch.tensor([.6, .6, .54])
    base_delta = torch.zeros(3, 3)
    down_delta = torch.tensor([[0., 0., -.005], [0., 0., -.02], [0., 0., 0.]])
    chosen = select_contact_preserving_down(base_contact, down_contact,
                                            base_delta, down_delta, obj)
    assert chosen.tolist() == [True, False, False]
    action = torch.zeros(3, 18)
    changed = apply_wrist_z_down(action, chosen)
    torch.testing.assert_close(changed[:, 2], torch.tensor([-.1, 0., 0.]))

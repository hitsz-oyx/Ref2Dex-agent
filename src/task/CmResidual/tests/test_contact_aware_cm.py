from __future__ import annotations

import torch

from src.task.CmResidual.contact_aware_cm import ContactAwareCm, RawContactAwareCm


def test_contact_aware_cm_shapes_and_finiteness():
    geometric = ContactAwareCm()
    raw = RawContactAwareCm(torch.zeros(67), torch.ones(67))
    for output in (geometric(torch.zeros(3, 6, 19), torch.zeros(3, 6)),
                   raw(torch.zeros(3, 67))):
        assert output["delta_local"].shape == (3, 3)
        assert output["followup_delta_local"].shape == (3, 3)
        assert output["contact_fraction"].shape == (3,)
        assert torch.isfinite(output["contact_fraction"]).all()
        assert ((output["contact_fraction"] >= 0) &
                (output["contact_fraction"] <= 1)).all()

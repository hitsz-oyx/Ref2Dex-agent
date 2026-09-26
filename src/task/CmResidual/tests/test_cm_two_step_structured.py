from __future__ import annotations

import torch

from src.task.CmResidual.cm_two_step_structured import StructuredTwoStepCm


def test_structured_second_dose_has_separate_signed_effect():
    model = StructuredTwoStepCm(torch.zeros(67), torch.ones(67), width=8)
    with torch.no_grad():
        model.followup_second.bias[0] = 1
    raw = torch.zeros(3, 69)
    raw[:, -2:] = torch.tensor([[.1, 0], [.1, .1], [-.1, -.1]])
    result = model(raw)
    torch.testing.assert_close(result["followup_delta_local"][:, 0],
                               torch.tensor([0., .02, -.02]))
    assert torch.isfinite(result["contact_fraction"]).all()

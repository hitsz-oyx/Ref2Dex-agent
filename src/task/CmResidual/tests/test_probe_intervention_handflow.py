import torch

from src.task.CmResidual.tools.probe_intervention_handflow import (
    fit_per_joint, predict_linear,
)


def test_linear_actuator_fit_recovers_action_and_velocity_effect():
    generator = torch.Generator().manual_seed(2409)
    q = torch.randn(512, 18, generator=generator)
    x = torch.randn(512, 18, generator=generator)
    velocity = torch.randn(512, 18, generator=generator)
    rows = {"q": q, "pd_target": q + x, "dof_vel": velocity,
            "next_q": q + .2 * x + .03 * velocity + .01}
    fitted = fit_per_joint(rows)
    predicted = predict_linear(rows, fitted)
    torch.testing.assert_close(predicted, rows["next_q"], atol=1e-5, rtol=0)
    torch.testing.assert_close(fitted[:, 0], torch.full((18,), .2), atol=1e-5, rtol=0)
    torch.testing.assert_close(fitted[:, 1], torch.full((18,), .03), atol=1e-5, rtol=0)

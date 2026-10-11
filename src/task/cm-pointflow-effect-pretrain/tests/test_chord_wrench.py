"""Finite CHORD contact-wrench geometry and tolerance contracts."""
import sys
from pathlib import Path

import torch

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / "src"))
from oakink_wm.chord_wrench import (  # noqa: E402
    chord_support_loss,
    contact_wrench_rays,
    support_vector,
)


def test_contact_rays_and_support_are_finite_and_masked():
    points = torch.tensor([[[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]]])
    normals = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])
    rays, valid = contact_wrench_rays(
        points, normals, torch.tensor([[True, False]]), friction=0.5, ray_count=8, body_radius=0.2
    )
    basis = torch.eye(6)
    sigma = support_vector(rays, valid, basis)
    assert rays.shape == (1, 18, 6)
    assert int(valid.sum()) == 9
    assert torch.isfinite(rays).all()
    assert torch.isfinite(sigma).all()
    assert sigma[0, 0] > 0


def test_no_contact_returns_zero_support_and_identical_support_has_unit_reward():
    rays = torch.zeros((2, 4, 6))
    valid = torch.zeros((2, 4), dtype=torch.bool)
    basis = torch.eye(6)
    sigma = support_vector(rays, valid, basis)
    assert torch.equal(sigma, torch.zeros_like(sigma))
    loss, reward = chord_support_loss(sigma, sigma)
    torch.testing.assert_close(loss, torch.zeros(()))
    torch.testing.assert_close(reward, torch.ones(2))


def test_support_tolerance_penalizes_missed_and_unintended_directions():
    reference = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    predicted = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    loss, reward = chord_support_loss(predicted, reference, tolerance=0.1)
    assert float(loss) > 0
    assert float(reward[0]) < 1.0

import torch

from src.task.CmResidual.cm_residual_policy import (
    RESIDUAL_SCALE_M,
    compose_residual_target,
    local_to_world,
    world_to_local,
)


def test_object_frame_rotation_is_not_world_frame():
    half = 2 ** -0.5
    quaternion = torch.tensor([[0.0, 0.0, half, half]])
    local = torch.tensor([[0.001, 0.0, 0.0]])
    world = local_to_world(quaternion, local)
    torch.testing.assert_close(world, torch.tensor([[0.0, 0.001, 0.0]]), atol=1e-7, rtol=0)
    torch.testing.assert_close(world_to_local(quaternion, world), local, atol=1e-7, rtol=0)


def test_zero_residual_is_exact_baseline_and_nonzero_changes_only_translation():
    baseline = torch.arange(36, dtype=torch.float64).reshape(2, 18)
    pose = torch.tensor([[0., 0., 0., 0., 0., 0., 1.],
                         [0., 0., 0., 0., 0., 0., 1.]], dtype=torch.float64)
    zero = torch.zeros(2, 3, dtype=torch.float64)
    target, details = compose_residual_target(baseline, pose, zero)
    assert torch.equal(target, baseline)
    assert not details["active"].any()
    residual = torch.tensor([[RESIDUAL_SCALE_M[0], 0., 0.], [0., -RESIDUAL_SCALE_M[1], 0.]], dtype=torch.float64)
    target, details = compose_residual_target(baseline, pose, residual)
    torch.testing.assert_close(target[:, :3] - baseline[:, :3], residual)
    torch.testing.assert_close(target[:, 3:], baseline[:, 3:])
    assert details["active"].all()


def test_residual_limit_is_fail_closed():
    baseline = torch.zeros(1, 18)
    pose = torch.tensor([[0., 0., 0., 0., 0., 0., 1.]])
    too_large = torch.tensor([[RESIDUAL_SCALE_M[0] * 1.1, 0., 0.]])
    try:
        compose_residual_target(baseline, pose, too_large)
    except ValueError as error:
        assert "authority" in str(error)
    else:
        raise AssertionError("out-of-authority residual was accepted")

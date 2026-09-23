"""V1.49 pose-effect audit is independent of old CmDecoder imports."""
import math

import torch

from src.task.CmResidual.tools.audit_original_cmv2_on_policy import (
    object_pose_delta_to_xi, rotation_error_rad,
)


def test_object_pose_delta_uses_current_object_frame():
    current = torch.eye(4).unsqueeze(0)
    current[:, :3, :3] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    current[:, :3, 3] = torch.tensor([2., 3., 4.])
    following = current.clone()
    following[:, :3, 3] = torch.tensor([2., 4., 4.])
    relative = object_pose_delta_to_xi(current, following)
    torch.testing.assert_close(relative[:, :3], torch.tensor([[1., 0., 0.]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(relative[:, 3:], torch.zeros((1, 3)), atol=1e-6, rtol=0)


def test_rotation_error_geodesic():
    zero = torch.zeros((1, 3))
    quarter_turn = torch.tensor([[0., 0., math.pi / 2]])
    torch.testing.assert_close(rotation_error_rad(zero, quarter_turn),
                               torch.tensor([math.pi / 2]), atol=1e-6, rtol=0)

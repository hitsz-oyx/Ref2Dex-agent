"""Coordinate, rotation, masking and shuffled-chunk semantic contracts."""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from oakink_wm.data import relative_effect, transform_points, local_objects, shuffle_action
from oakink_wm.model import rotation6d, geodesic


def pose(angle=0, shift=(0, 0, 0)):
    c, s = np.cos(angle), np.sin(angle)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = shift
    return T


def test_one_current_frame_matches_direct_future_points_and_global_invariance():
    anchor, now, future = pose(.4, (.2, .3, .1)), pose(-.2, (.4, .1, .2)), pose(.9, (.5, .2, .1))
    points = np.array([[.02, .03, .01], [-.03, .02, .01]])
    E = relative_effect(anchor, now, future)
    current = transform_points(points, np.linalg.inv(anchor) @ now)
    truth = transform_points(points, np.linalg.inv(anchor) @ future)
    np.testing.assert_allclose(transform_points(current, E), truth, atol=1e-12)
    G = pose(-.7, (1, 2, 3))
    np.testing.assert_allclose(relative_effect(G @ anchor, G @ now, G @ future), E, atol=1e-12)
    # Replacing C with inverse(future) would incorrectly erase the target motion.
    assert not np.allclose(truth, transform_points(points, np.linalg.inv(future) @ future))


def test_local_selection_current_distance_includes_boundary_and_excludes_invalid():
    T = np.stack([pose(), pose(shift=(.5, 0, 0)), pose(shift=(.501, 0, 0)), pose()])
    assert local_objects(T, np.array([1, 1, 1, 0], bool), 0).tolist() == [0, 1]


def test_rotation6d_is_rigid_and_identity_geodesic_gradient_is_finite():
    x = torch.tensor([[1., 0, 0, 0, 1, 0]], requires_grad=True)
    R = rotation6d(x)
    torch.testing.assert_close(R.transpose(-1, -2) @ R, torch.eye(3)[None])
    torch.testing.assert_close(torch.linalg.det(R), torch.ones(1))
    angle = geodesic(R, torch.eye(3)[None])
    angle.sum().backward()
    assert torch.isfinite(x.grad).all() and angle.item() == 0


def test_shuffle_preserves_entire_donor_chunks_masks_and_hand_groups():
    batch = dict(action=torch.arange(4*24*22*9).reshape(4, 24, 22, 9),
                 action_valid=torch.ones(4, 24, 22, dtype=torch.bool),
                 hand_presence=torch.tensor([[1, 0], [1, 0], [1, 1], [1, 1]], dtype=torch.bool))
    batch['action_valid'][:2, :, 11:] = False
    shuffled, available = shuffle_action(batch, 11)
    assert available.all()
    for i, donor in enumerate([1, 0, 3, 2]):
        assert torch.equal(shuffled['action'][i], batch['action'][donor])
        assert torch.equal(shuffled['action_valid'][i], batch['action_valid'][donor])
    singleton = {k: v[:1] for k, v in batch.items()}
    _, available = shuffle_action(singleton, 11)
    assert not available.any()  # Must not silently count an unchanged chunk as shuffled.

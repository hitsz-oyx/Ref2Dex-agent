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
    centers = np.zeros((4, 3)); centers[1, 0] = .02; centers[2, 0] = -.02
    assert local_objects(T, np.array([1, 1, 1, 0], bool), 0, centers).tolist() == [0, 2]


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


def test_real_gpu_interface_has_no_future_label_leak_and_masks_absent_actions():
    import os
    import pytest
    from oakink_wm.data import Windows, collate
    from oakink_wm.model import WorldModel, losses
    root = os.environ.get('WM30_INTERFACE_DATA')
    if not root or not torch.cuda.is_available():
        pytest.skip('Set WM30_INTERFACE_DATA for real cached GPU interface contract')
    dataset = Windows(Path(root), 'train')
    batch = {k: v.cuda() for k, v in collate([dataset[int(dataset.groups[0][0])], dataset[int(dataset.groups[1][0])]]).items()}
    torch.manual_seed(9)
    model = WorldModel().cuda().eval()
    with torch.no_grad():
        original = model(batch)
        class NoLabels(dict):
            def __getitem__(self, key):
                if key == 'effect': raise AssertionError('GT effect accessed during forward')
                return super().__getitem__(key)
        changed = NoLabels(batch)
        repeated = model(changed)
        torch.testing.assert_close(original['translation'], repeated['translation'], rtol=0, atol=1e-6)
        torch.testing.assert_close(original['rotation'], repeated['rotation'], rtol=0, atol=1e-6)
        masked = dict(batch)
        masked['action_valid'] = batch['action_valid'].clone()
        masked['action_valid'][:, :, 11:] = False
        base = model(masked)
        altered = dict(masked)
        altered['action'] = masked['action'].clone()
        altered['action'][:, :, 11:] += 1000
        ignored = model(altered)
        torch.testing.assert_close(base['translation'], ignored['translation'], rtol=0, atol=1e-6)
        altered['action'] = batch['action'] + .1
        changed_action = model(altered)
        assert not torch.allclose(base['translation'], changed_action['translation'], rtol=0, atol=1e-8)
    model.train()
    loss, _ = losses(model(batch), batch)
    loss.backward()
    grad = model.action_mlp[0].weight.grad
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert torch.isfinite(model.scene.stem[0].weight.grad).all()


def test_physical_losses_use_vector_l1_and_radius_scaled_rotation():
    from oakink_wm.model import losses
    R = torch.eye(3).expand(1, 1, 24, 3, 3)
    E = torch.eye(4).expand(1, 1, 24, 4, 4).clone()
    batch = dict(effect=E, object_valid=torch.ones(1, 1, dtype=torch.bool),
                 points=torch.zeros(1, 1, 512, 3), radius=torch.ones(1, 1)*.1)
    loss, components = losses(dict(rotation=R, translation=torch.ones(1, 1, 24, 3)*.01), batch)
    torch.testing.assert_close(components['translation'], torch.tensor(.03))
    torch.testing.assert_close(components['point'], torch.tensor(.03))
    torch.testing.assert_close(loss, torch.tensor(.06))

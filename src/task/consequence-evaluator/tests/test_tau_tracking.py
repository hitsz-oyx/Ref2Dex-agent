from pathlib import Path

import numpy as np
import torch

from consequence_evaluator.contracts import HAND_LINKS
from consequence_evaluator.reference_motion import NATIVE_DOF_NAMES
from consequence_evaluator.reference_tracking import ReferenceTracker, apply_coupling, features
from consequence_evaluator.reset_kinematics import ResetKinematics
from consequence_evaluator.tau_tracking import (
    TauTracker, fit_geometry, tau_calibration, tau_features, tau_reward)


def test_calibration_ignores_poisoned_future_joint_and_object_labels():
    packet = dict(hand_keypoints=np.ones((3, 1, 11, 3), np.float32),
                  dof_position=np.zeros((3, 1, 18), np.float32),
                  object_pose=np.zeros((3, 1, 4, 4), np.float32))
    expected = tau_calibration(packet)
    packet["dof_position"][1:] = np.nan
    packet["object_pose"][:] = np.nan
    packet["actions"] = None
    actual = tau_calibration(packet)
    assert all(np.array_equal(a, b) for a, b in zip(expected, actual))


def test_migration_preserves_controller_when_removed_object_columns_are_zero():
    q = torch.zeros(2, 18); hand = torch.zeros(2, 11, 3)
    obj = torch.eye(4).repeat(2, 1, 1); future = torch.ones(2, 24, 11, 3) * .01
    args = (q, q, hand, obj, torch.zeros(2, 6), future, q, torch.zeros(2, 12))
    oracle = ReferenceTracker()
    torch.nn.init.normal_(oracle.actor[-1].weight, std=.01)
    student = TauTracker(); student.warmstart(oracle.state_dict())
    full = features(*args[:7], obj, args[7])
    restricted = tau_features(*args)
    assert restricted.shape == (2, 897)
    assert torch.allclose(oracle.actor(full), student.actor(restricted), atol=1e-6)
    changed = list(args); changed[5] = future + .05
    assert not torch.allclose(student.actor(restricted), student.actor(tau_features(*changed)))


def test_geometric_wrist_roundtrip_uses_only_hand_and_initial_state():
    root = Path(__file__).resolve().parents[4]
    urdf = root / 'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
    q = torch.zeros(3, 18); q[:, 6:] = .3
    q[:, :3] = torch.tensor([[.1, .2, .3], [.12, .21, .35], [.14, .22, .37]])
    q[:, 3:6] = torch.tensor([[.1, .2, .3], [.15, .18, .4], [.2, .16, .45]])
    q = apply_coupling(q)
    base = torch.zeros(3, 13); base[:, 6] = 1
    fk = ResetKinematics(urdf, NATIVE_DOF_NAMES, HAND_LINKS, 'cpu')
    hand = fk.states(q, torch.zeros_like(q), base)[:, :, :3].numpy()
    result = fit_geometry(hand, q[0].numpy(), urdf, 'cpu', iterations=2)
    assert np.max(np.abs(result['q'][:, :6] - q.numpy()[:, :6])) < 1e-5
    assert np.max(np.abs(result['fitted_points'] - hand)) < 1e-5


def test_tau_reward_penalizes_object_fall_without_future_object_label():
    reference = torch.zeros(1, 11, 3); reference[:, :, 2] = .8
    obj = torch.eye(4)[None]; obj[:, 2, 3] = .8
    latent = torch.zeros(1, 12)
    good = tau_reward(reference, obj, reference, torch.tensor([True]), .5, latent)
    fallen = obj.clone(); fallen[:, 2, 3] = .5
    bad = tau_reward(reference, fallen, reference, torch.tensor([False]), .5, latent)
    assert good.item() > bad.item() + .5


def test_tau_reward_prefers_holding_when_palm_moves_below_reset():
    # The actual source palm descends while the object is lifted. Palm motion
    # must not prescribe the object's height or disable physical holding.
    reference = torch.zeros(1, 11, 3); reference[:, :, 2] = .3
    obj = torch.eye(4)[None]; obj[:, 2, 3] = .8
    latent = torch.zeros(1, 12)
    held = tau_reward(reference + .02, obj, reference, torch.tensor([True]), .5, latent)
    fallen = obj.clone(); fallen[:, 2, 3] = .5
    nominal = tau_reward(reference, fallen, reference, torch.tensor([False]), .5, latent)
    assert held.item() > nominal.item() + .5

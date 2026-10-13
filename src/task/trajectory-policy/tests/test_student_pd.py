import numpy as np
import torch

from trajectory_policy.student_pd import action_loss, native_tau, trajectory_loss, validate_student_packet
from trajectory_policy.student_reference_data import student_sample


def test_failed_future_hand_does_not_supervise_the_trajectory():
    prediction = torch.zeros(2, 24, 2, 11, 3, requires_grad=True)
    target = torch.ones_like(prediction)
    target[1] = 4.
    valid = torch.ones(2, 24, 2, 11, dtype=torch.bool)
    supervised = torch.tensor([True, False])
    loss = trajectory_loss(prediction, target, valid, supervised)
    changed = target.clone()
    changed[1] = -9.
    assert torch.allclose(loss, trajectory_loss(prediction, changed, valid, supervised))
    loss.backward()
    assert torch.count_nonzero(prediction.grad[1]) == 0
    assert torch.count_nonzero(prediction.grad[0]) > 0


def test_native_tau_uses_the_right_hand_in_native_order():
    prediction = torch.zeros(1, 24, 2, 11, 3)
    prediction[0, :, 0, 3, 0] = 1.
    prediction[0, :, 1] = 7.
    order = torch.tensor([3, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10])
    tau = native_tau(prediction, order)
    assert tau.shape == (1, 24, 33)
    assert torch.all(tau[:, :, 0] == 1.)
    assert torch.all(tau[:, :, 3:] == 0.)


def test_student_sample_keeps_the_expert_command_and_drops_the_future_hand():
    states = 8
    eye = np.broadcast_to(np.eye(4, dtype=np.float32), (states, 1, 4, 4)).copy()
    reference = dict(hand_pos=np.zeros((states, 16, 3), np.float32),
                     hand_rot=np.broadcast_to(np.eye(3, dtype=np.float32), (states, 16, 3, 3)).copy(),
                     obj=np.broadcast_to(np.eye(4, dtype=np.float32), (states, 4, 4)).copy())
    data = dict(q=np.zeros((states, 1, 18), np.float32), dq=np.zeros((states, 1, 18), np.float32),
                hand=np.zeros((states, 1, 11, 3), np.float32), obj=eye,
                velocity=np.zeros((states, 1, 6), np.float32),
                applied=np.zeros((states - 1, 1, 18), np.float32),
                student_expert_pd=np.zeros((states - 1, 18), np.float32))
    data['hand'][3, 0, 0, 0] = 0.2
    data['student_expert_pd'][2, 0] = 0.05
    canonical = dict(points=np.zeros((512, 3), np.float32), normals=np.zeros((512, 3), np.float32),
                     radius=np.float32(0.01), center=np.zeros(3, np.float32))
    sample = student_sample(data, reference, canonical, 2, 0)
    assert sample['trajectory_valid'].sum() == 0
    assert sample['tau_supervised'] == False
    assert sample['has_pd'] == True
    assert sample['reference_goal'].shape == (408,)
    assert sample['pd_target'].shape == (12,)


def test_student_packet_must_keep_its_training_base_and_retargeter():
    packet = dict(schema='ref2dex.student-pd-adapter.v1', base_checkpoint='/a/final.pt',
                  retargeter='/a/tau.pt', adapter=dict(alpha_p=0, alpha_g=0), trajectory=dict(weight=1))
    identity = validate_student_packet(packet, '/a/packet.pt', '/a/final.pt', '/a/tau.pt', lambda _path: 'abc')
    assert identity['effect_loss'] is False and identity['reference_offsets'] == [1, 16]
    packet['retargeter'] = '/a/other.pt'
    try:
        validate_student_packet(packet, '/a/packet.pt', '/a/final.pt', '/a/tau.pt', lambda _path: 'abc')
    except ValueError:
        return
    raise AssertionError('a packet trained with another R was accepted')


def test_pd_loss_ignores_rows_without_an_action_label():
    predicted = torch.zeros(2, 12, requires_grad=True)
    target = torch.zeros(2, 12)
    target[1] = 3.
    loss = action_loss(predicted, target, torch.tensor([False, True]))
    assert torch.allclose(loss, torch.tensor(9.))
    loss.backward()
    assert torch.count_nonzero(predicted.grad[0]) == 0

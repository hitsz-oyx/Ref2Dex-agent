import numpy as np

from trajectory_policy.history import measured_history


def test_actor_history_keeps_height_and_gravity_and_is_past_only():
    obj = np.broadcast_to(np.eye(4, dtype=np.float32), (4, 1, 4, 4)).copy()
    hand = np.zeros((4, 1, 11, 3), np.float32)
    q = np.zeros((4, 1, 18), np.float32)
    dq = q.copy()
    velocity = np.zeros((4, 1, 6), np.float32)
    original = measured_history(obj, hand, q, dq, velocity)
    obj[:, :, 2, 3] += .4
    hand[:, :, :, 2] += .4
    raised = measured_history(obj, hand, q, dq, velocity)
    np.testing.assert_allclose(original[:, :324], raised[:, :324], atol=1e-6)
    np.testing.assert_allclose(raised[:, 324]-original[:, 324], .4)
    np.testing.assert_array_equal(raised[:, -3:], [[0., 0., -1.]])
    dq[0, 0, 0] = .3
    changed = measured_history(obj, hand, q, dq, velocity)
    assert changed[0, 300] == np.float32(.3)

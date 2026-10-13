import socket

import pytest

from trajectory_policy.wam_batch import request_rows
from trajectory_policy.wam_ipc import SCHEMA, receive, send


def _row(env_id, tick=0):
    return dict(env_id=env_id, tick=tick, object_history=[], hand_history=[],
                planned_object_goal=[], proprio=[], robot_goal=[], retarget_state=[0.0] * 87)


def test_legacy_packet_is_one_row_and_a_batch_keeps_env_ids():
    legacy = dict(tick=3, object_history=[], hand_history=[], planned_object_goal=[],
                  proprio=[], robot_goal=[])
    rows, batched = request_rows(legacy, 'robot_goal', True, False, 3)
    assert rows == [legacy] and batched is False
    packet = dict(batch=[_row(0, 32), _row(1, 48), _row(2, 0), _row(3, 62)])
    rows, batched = request_rows(packet, 'robot_goal', True, True, 0)
    assert batched is True and [row['env_id'] for row in rows] == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        request_rows(dict(batch=[_row(0, 0), _row(0, 1)]), 'robot_goal', True, True, 0)
    with pytest.raises(ValueError):
        request_rows(dict(batch=[_row(index, 0) for index in range(33)]), 'robot_goal', True, True, 0)
    with pytest.raises(ValueError):
        request_rows(dict(tick=1, object_history=[], hand_history=[], planned_object_goal=[],
                          proprio=[], robot_goal=[]), 'robot_goal', True, False, 0)


def test_batched_tau_frame_fits_the_raised_ipc_limit():
    left, right = socket.socketpair()
    try:
        reply = dict(schema=SCHEMA, batch=True, responses=[
            dict(env_id=index, tick=index, tau=[[0.01] * 33 for _ in range(24)])
            for index in range(8)], pad='x' * 70000)
        send(left, reply)
        assert receive(right)['responses'][7]['env_id'] == 7
    finally:
        left.close()
        right.close()

"""Measured robot H4/object-goal inputs and explicit WAM-to-Inspire adapter.

The index map pairs semantic roles, not equal anatomical landmark locations.
Human wrist/first-joint offsets differ from Inspire link origins. This module
does not fit an embodiment transform or send commands to a simulator/robot.
"""
import numpy as np


WAM_SEMANTICS = ('wrist', 'thumb_mcp', 'index_mcp', 'middle_mcp', 'ring_mcp',
                 'little_mcp', 'thumb_tip', 'index_tip', 'middle_tip',
                 'ring_tip', 'little_tip')
NATIVE_LINKS = ('hand_base_link', 'thumb_proximal_base', 'thumb_tip',
                'index_proximal', 'index_tip', 'middle_proximal', 'middle_tip',
                'ring_proximal', 'ring_tip', 'pinky_proximal', 'pinky_tip')
NATIVE_TO_WAM = np.asarray([0, 1, 3, 5, 7, 9, 2, 4, 6, 8, 10])
WAM_TO_NATIVE = np.argsort(NATIVE_TO_WAM)
HAND_ORDER = ('right', 'left')


def active_hand_tau(prediction, hand_valid, side='right'):
    """Select one observed hand and reorder displacement into native links."""
    prediction = np.asarray(prediction)
    hand_valid = np.asarray(hand_valid, dtype=bool)
    if side not in HAND_ORDER or prediction.shape[-4:] != (24, 2, 11, 3):
        raise ValueError('named hand and full WAM displacement contract required')
    if hand_valid.shape != prediction.shape[:-4] + (2, 11):
        raise ValueError('current semantic point mask required')
    hand = HAND_ORDER.index(side)
    if not hand_valid[..., hand, :].all() or not np.isfinite(prediction).all():
        raise ValueError('cannot send missing/nonfinite semantic points to R')
    return prediction[..., hand, WAM_TO_NATIVE, :].reshape(*prediction.shape[:-4], 24, 33)


def robot_goal_sample(object_history, hand_history, planned_object_goal, canonical,
                      goal_points=32, sample_id=0, fps=30.):
    """Use only measured past/current H4 and a decision-known object goal.

    Raw inputs use simulator world coordinates; one current-object inverse
    transforms all history and the known goal. No future hand, future measured
    object or q/action is accepted by this scene/goal constructor. Dummy future
    fields are shape placeholders and are not passed through the world branch.
    """
    obj = np.asarray(object_history)
    hand = np.asarray(hand_history)
    goal = np.asarray(planned_object_goal)
    if obj.shape != (4, 4, 4) or hand.shape != (4, 11, 3) or goal.shape != (4, 4):
        raise ValueError('world H4 and one planned object pose required')
    if any(not np.isfinite(x).all() for x in (obj, hand, goal)):
        raise ValueError('nonfinite robot measurements')
    for pose in (obj, goal[None]):
        rotation = pose[..., :3, :3]
        if (not np.allclose(rotation.swapaxes(-1, -2) @ rotation, np.eye(3), atol=1e-5)
                or not np.allclose(np.linalg.det(rotation), 1., atol=1e-5)
                or not np.allclose(pose[..., 3, :], [0., 0., 0., 1.], atol=1e-6)):
            raise ValueError('rigid measured/goal poses required')
    anchor = np.linalg.inv(obj[-1]).astype(np.float32)
    poses = anchor @ obj
    hh = (hand - obj[-1, :3, 3]) @ obj[-1, :3, :3]
    hh = hh[:, NATIVE_TO_WAM]
    points = np.asarray(canonical['points'], dtype=np.float32)
    normals = np.asarray(canonical['normals'], dtype=np.float32)
    if points.shape != (512, 3) or normals.shape != points.shape:
        raise ValueError('frozen 512-point robot geometry required')
    ph = np.einsum('tij,pj->tpi', poses[:, :3, :3], points) + poses[:, None, :3, 3]
    dynamics = lambda a: np.concatenate(((a[-1] - a[-2]) * fps,
        (a[-1] - 2 * a[-2] + a[-3]) * fps ** 2, a[-1] - a[0]), -1)
    static = float(np.max(np.linalg.norm(ph[-1] - ph[0], axis=-1)) < .002)
    obj_extra = np.tile([1., static, 0., 0., 0., 0.], (512, 1))
    obj_features = np.concatenate((normals, dynamics(ph), obj_extra), -1)
    hand_extra = np.stack((np.zeros(11), np.zeros(11), np.ones(11),
                          np.zeros(11), np.arange(11) / 10., np.zeros(11)), -1)
    hand_features = np.concatenate((np.zeros((11, 3)), dynamics(hh), hand_extra), -1)
    start = np.zeros((2, 11, 3), np.float32)
    start[0] = hh[-1]
    valid = np.zeros((2, 11), bool)
    valid[0] = True
    hd = np.zeros((2, 11, 9), np.float32)
    hd[0] = dynamics(hh)
    indices = np.linspace(0, 511, goal_points, dtype=np.int64)
    endpoint = anchor @ goal
    goal_xyz = points[indices] @ endpoint[:3, :3].T + endpoint[:3, 3]
    return dict(xyz=np.concatenate((points, hh[-1])).astype(np.float32),
        features=np.concatenate((obj_features, hand_features)).astype(np.float32),
        scene_object=np.r_[np.zeros(512, np.int64), np.full(11, -1, np.int64)],
        points=points[None], radius=np.asarray([canonical['radius']], np.float32),
        object_features=np.r_[canonical['center'], np.eye(3).ravel(),
                             canonical['radius'], 1., 0.][None].astype(np.float32),
        action=np.zeros((24, 22, 9), np.float32), action_valid=np.zeros((24, 22), bool),
        effect=np.tile(np.eye(4, dtype=np.float32), (1, 24, 1, 1)), category=0,
        sample_id=np.asarray([sample_id, 0, 0], np.int64), hand_presence=np.asarray([True, False]),
        start_hand=start, start_hand_history=hd, start_hand_valid=valid,
        start_object_points=points[indices], goal_object_points=goal_xyz.astype(np.float32),
        object_goal_valid=np.ones(goal_points, bool), goal_object_slot=np.int64(0),
        trajectory_target=np.zeros((24, 2, 11, 3), np.float32),
        trajectory_valid=np.zeros((24, 2, 11), bool), source_id=np.int64(0))

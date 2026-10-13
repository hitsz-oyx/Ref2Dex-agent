"""Demonstration hand/object reference plus expert PD at a visited state.

Student rows keep the expert's current PD and do not admit that state's future
hand as a trajectory label. Nominal teacher rows may, because that hand is the
executed expert trajectory.
"""
import numpy as np

from .pd_retargeter import encode_target
from .reference_actor import reference_goal
from .retargeter_data import ACTIVE, features
from .wam_consequence import actual_consequence_sample
from .wam_shadow import robot_goal_sample


def measured_proprio(q, dq, previous):
    q, dq, previous = [np.asarray(x, np.float32) for x in (q, dq, previous)]
    if q.shape != (18,) or dq.shape != (18,) or previous.shape != (18,):
        raise ValueError('native18 q, dq and previous command required')
    prop = np.concatenate((q, dq, previous[list(ACTIVE)]))
    if not np.isfinite(prop).all():
        raise ValueError('finite measured proprio required')
    return prop.astype(np.float32)


def _bind(sample, data, reference, tick, role, pd_target, supervised):
    tick, role = int(tick), int(role)
    q, dq = data['q'][tick, role], data['dq'][tick, role]
    previous = data['applied'][tick - 1, role] if tick else np.zeros(18, np.float32)
    state, _ = features(q, dq, data['hand'][tick, role], data['obj'][tick, role],
                        data['velocity'][tick, role],
                        np.broadcast_to(data['hand'][tick, role], (24, 11, 3)))
    goal = reference_goal(data['obj'][tick, role][None], reference, [tick])[0]
    label = encode_target(np.asarray(pd_target, np.float32)[None], np.asarray(q, np.float32)[None])[0]
    if sample['trajectory_valid'].any() and not supervised:
        raise ValueError('a student state cannot supervise its future hand')
    sample.update(proprio=measured_proprio(q, dq, previous), reference_goal=goal,
                  robot_mask=np.bool_(True), native_state=state, pd_target=label,
                  has_pd=np.bool_(True), tau_supervised=np.bool_(supervised))
    return sample


def student_sample(data, reference, canonical, tick, role=2):
    """Expert PD at this measured state. The future hand is not a label."""
    tick, role = int(tick), int(role)
    if 'student_expert_pd' not in data:
        raise ValueError('recorded expert PD required')
    past = np.maximum(tick + np.arange(-3, 1), 0)
    goal_pose = reference['obj'][min(tick + 16, len(reference['obj']) - 1)]
    sample = robot_goal_sample(data['obj'][past, role], data['hand'][past, role], goal_pose,
                               canonical, sample_id=tick)
    if sample['trajectory_valid'].any():
        raise ValueError('student scene sample leaked a trajectory label')
    return _bind(sample, data, reference, tick, role, data['student_expert_pd'][tick], False)


def nominal_sample(data, reference, canonical, tick, role):
    """Teacher execution: its future hand is the expert trajectory."""
    tick, role = int(tick), int(role)
    goal_pose = reference['obj'][min(tick + 16, len(reference['obj']) - 1)]
    sample = actual_consequence_sample(data['obj'], data['hand'], tick, role, goal_pose,
                                       canonical, sample_id=tick * 32 + role)
    future = sample['action'].reshape(24, 2, 11, 9)[..., :3]
    sample['trajectory_target'] = (future - sample['start_hand'][None]).astype(np.float32)
    sample['trajectory_valid'] = sample['action_valid'].reshape(24, 2, 11) & sample['start_hand_valid'][None]
    sample['trajectory_target'][~sample['trajectory_valid']] = 0.
    return _bind(sample, data, reference, tick, role, data['pd_targets'][tick, role], True)


def reference_collate(samples):
    import torch
    from goal_point_wam.data import goal_collate
    batch = goal_collate(samples)
    for key, dtype in (('proprio', torch.float32), ('reference_goal', torch.float32),
                       ('native_state', torch.float32), ('pd_target', torch.float32)):
        batch[key] = torch.as_tensor(np.stack([s[key] for s in samples]), dtype=dtype)
    for key in ('robot_mask', 'has_pd', 'tau_supervised'):
        batch[key] = torch.as_tensor(np.stack([bool(s[key]) for s in samples]))
    return batch

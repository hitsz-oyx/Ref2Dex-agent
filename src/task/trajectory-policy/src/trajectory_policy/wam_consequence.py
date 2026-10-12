"""Own-row executed hand motion and actual rigid consequences for WAM world.

Future hand is a world execution condition, never an expert actor target.
The 9D hand-motion proxy is not a native-command causal dynamics contract.
"""
import numpy as np

from .wam_shadow import NATIVE_TO_WAM, robot_goal_sample

HORIZON = 24
EXECUTORS = {'official_teacher': 0, 'gt_tau_R': 1, 'pure_wam': 2,
             'wam_command_replay': 3, 'expert_takeover': 4}


def actual_consequence_sample(object_states, hand_states, tick, role,
                              planned_object_goal, canonical, sample_id=0):
    """Current own H4 -> executed future9D / rigid effect, fixed current anchor.

    Inputs deliberately exclude native q/commands and privileged expert obs.
    Control t causes state t+1; sample t uses states t+1..t+24. Startup H4
    repeats the available initial state, as in robot_goal_sample. All robot
    links are measured; only the right hand exists in this native contract.
    This helper grants no scientific training or execution admission.
    """
    objects = np.asarray(object_states)
    hands = np.asarray(hand_states)
    if (objects.ndim != 4 or objects.shape[-2:] != (4, 4)
            or hands.shape != (*objects.shape[:2], 11, 3)
            or not isinstance(tick, (int, np.integer)) or not isinstance(role, (int, np.integer))
            or not 0 <= tick < len(objects)-HORIZON or not 0 <= role < objects.shape[1]):
        raise ValueError('own-row complete measured H4/future24 window required')
    past = np.maximum(tick+np.arange(-3, 1), 0)
    future = tick+np.arange(1, HORIZON+1)
    history_obj, history_hand = objects[past, role], hands[past, role]
    sample = robot_goal_sample(history_obj, history_hand, planned_object_goal,
                              canonical, sample_id=sample_id, fps=30.)
    future_obj = np.asarray(objects[future, role], dtype=np.float64)
    future_hand = np.asarray(hands[future, role], dtype=np.float64)
    if not np.isfinite(future_obj).all() or not np.isfinite(future_hand).all():
        raise ValueError('finite actual future observations required')
    rotations = future_obj[:, :3, :3]
    if (not np.allclose(rotations.swapaxes(-1, -2) @ rotations, np.eye(3), atol=1e-5)
            or not np.allclose(np.linalg.det(rotations), 1., atol=1e-5)
            or not np.allclose(future_obj[:, 3], [0., 0., 0., 1.], atol=1e-6)):
        raise ValueError('actual future object poses must be SE3')
    current_obj = np.asarray(objects[tick, role], dtype=np.float64)
    rotation, center = current_obj[:3, :3], current_obj[:3, 3]
    current_hand = (np.asarray(hands[tick, role], dtype=np.float64)-center) @ rotation
    local_future = (future_hand-center) @ rotation
    current_hand, local_future = current_hand[NATIVE_TO_WAM], local_future[:, NATIVE_TO_WAM]
    displacement = local_future-current_hand[None]
    increment = np.diff(np.concatenate((current_hand[None], local_future)), axis=0)
    action = np.zeros((HORIZON, 2, 11, 9), np.float32)
    action[:, 0] = np.concatenate((local_future, displacement, increment), axis=-1)
    action_valid = np.zeros((HORIZON, 2, 11), bool)
    action_valid[:, 0] = True
    # The one native object is also the current anchor: C*T_now=I.
    effect = (np.linalg.inv(current_obj) @ future_obj)[None].astype(np.float32)
    sample.update(action=action.reshape(HORIZON, 22, 9),
                  action_valid=action_valid.reshape(HORIZON, 22), effect=effect)
    # NEVER call GoalWindows: actual consequences are not expert actor labels.
    sample['trajectory_target'].fill(0.)
    sample['trajectory_valid'].fill(False)
    return sample


def future_control_metadata(expert_applied, tick, role):
    """Metadata only; retain actual control provenance outside world inputs."""
    mask = np.asarray(expert_applied)
    if (mask.ndim != 2 or mask.shape[1] != 2 or mask.dtype != bool
            or not 0 <= tick <= len(mask)-HORIZON or role not in range(4)):
        raise ValueError('native learner expert mask and complete24 controls required')
    source = mask[tick:tick+HORIZON].copy()
    if role == 0:
        own = np.ones(HORIZON, bool)
        executor = np.full(HORIZON, EXECUTORS['official_teacher'], np.int64)
    elif role == 1:
        own = np.zeros(HORIZON, bool)
        executor = np.full(HORIZON, EXECUTORS['gt_tau_R'], np.int64)
    else:
        own = source[:, role-2].copy()
        executor = np.where(own, EXECUTORS['expert_takeover'],
            EXECUTORS['pure_wam'] if role == 2 else EXECUTORS['wam_command_replay']).astype(np.int64)
    return dict(control_ticks=np.arange(tick, tick+HORIZON),
                source_expert_applied=source, own_expert_applied=own, executor_code=executor)

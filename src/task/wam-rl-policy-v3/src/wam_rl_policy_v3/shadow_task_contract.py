"""Trainable UniDex task10 contract; legacy rollouts have no net-contact log."""
import numpy as np
from scipy.spatial.transform import Rotation

from .executor_contract import ExecutorLayout

CURRENT_SCHEMA = 'ref2dex.shadow-task-current-bc.v1'
SCHEDULED_SCHEMA = 'ref2dex.shadow-task-scheduled-bc.v1'
FEEDBACK = 'same-execution-motion-only.v1'


def conditioning(with_tau=True):
    packet = ExecutorLayout('unidex.shadow.legacy300.v1', 300, 24, 60, with_tau).packet()
    packet.update(schema='ref2dex.shadow-task-conditioning.v1',
                  feedback=FEEDBACK, goal_source='native_position_goal_TaskSpec.shadow_goal',
                  wam_goal='same_TaskController_task10_local_SE3',
                  goal_semantics='spatial_waypoint', goal_reach_time_guaranteed=False)
    return packet


def measured_velocity(pose, observation):
    """Undo UniDex observation yaw and its 0.2 angular-velocity scale."""
    pose, observation = np.asarray(pose), np.asarray(observation)
    if pose.shape != (4, 4) or observation.shape != (300,):
        raise ValueError('current object pose and original current300 required')
    yaw = pose[:3, :3] @ Rotation.from_quat(observation[194:198]).as_matrix().T
    return np.r_[yaw @ observation[198:201], yaw @ (observation[201:204] / .2)]


def task_point_delta(task, points):
    task = np.asarray(task)
    return (np.asarray(points) @ Rotation.from_rotvec(task[3:6]).as_matrix().T
            + task[:3] - points)

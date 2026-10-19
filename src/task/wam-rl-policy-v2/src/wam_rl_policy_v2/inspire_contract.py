"""DExplore Inspire contracts, independent of the Shadow 60 Hz interface."""
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation

ACTIVE = (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15)
NATIVE_TO_WAM = np.array([0, 1, 3, 5, 7, 9, 2, 4, 6, 8, 10])
WAM_TO_NATIVE = np.argsort(NATIVE_TO_WAM)
CONTRACT = dict(schema='ref2dex.inspire-scheduled-current.v1', observation_dim=986,
    current_dim=721, reference_delta=0, control_hz=30, history_offsets=[-3,-2,-1,0],
    tau_shape=[8,11,3], tau_times_s=[i/30 for i in range(1,9)],
    tau_frame='world', tau_representation='displacement_from_current_hand',
    hand_order='native_inspire_base_then_finger_base_tip', hand_slot='right',
    unit='metres', active_action_indices=list(ACTIVE), native_action_dim=18,
    startup='zero_tau_invalid_until_four_measured_frames', current_clip=5.)


def expand_action(active):
    active = np.asarray(active, dtype=np.float32)
    if active.shape[-1:] != (12,) or not np.isfinite(active).all():
        raise ValueError('finite 12 independent commands required')
    out = np.zeros(active.shape[:-1]+(18,), np.float32)
    out[..., ACTIVE] = np.clip(active, -1, 1)
    return out


def inspire_tau(trajectory, pose):
    trajectory, pose = np.asarray(trajectory), np.asarray(pose)
    if trajectory.shape != (24,2,11,3) or pose.shape != (4,4):
        raise ValueError('v4 24x2x11x3 and measured object SE3 required')
    # Eight 30 Hz samples, no Shadow 30->60 Hz interpolation.
    return (trajectory[:8,0][:,WAM_TO_NATIVE] @ pose[:3,:3].T).astype(np.float32)


def executor_input(current, tau, valid):
    current, tau = np.asarray(current), np.asarray(tau)
    if current.shape != (721,) or tau.shape != (8,11,3):
        raise ValueError('Inspire current721 / tau8x11x3 required')
    out = np.r_[np.clip(current,-5,5), tau.reshape(-1) if valid else np.zeros(264), float(valid)].astype(np.float32)
    if not np.isfinite(out).all():
        raise ValueError('nonfinite executor input')
    return out


class InspireConditioner:
    """One ordered measured episode; scheduler consumes the known object plan."""
    def __init__(self, runtime, reference):
        from .goal_scheduler import OrderedGoalScheduler
        self.runtime = runtime
        self.scheduler = OrderedGoalScheduler(reference, np.arange(len(reference))/30,
            runtime.geometry[0][2], interaction_points=runtime.geometry[0][3], goal_type='se3_pose')
        self.poses, self.hands = deque(maxlen=4), deque(maxlen=4)
        self.last_tick = -1

    def prepare(self, tick, pose, hand):
        from .wam_v4_runtime import v4_online_samples
        if tick != self.last_tick+1:
            raise ValueError('new episode needs fresh conditioner; no skipped/repeated frames')
        self.last_tick = tick
        pose, hand = np.asarray(pose), np.asarray(hand)
        if pose.shape != (4,4) or hand.shape != (11,3):
            raise ValueError('measured pose / native hand11 required')
        self.poses.append(pose.copy()); self.hands.append(hand[NATIVE_TO_WAM].copy())
        goal = self.scheduler.update(pose, tick/30, hand_points=hand[NATIVE_TO_WAM])
        if len(self.poses)<4:
            return None, goal
        poses = np.stack(self.poses)
        objects = np.concatenate((poses[:,:3,3],Rotation.from_matrix(poses[:,:3,:3]).as_quat()),axis=-1)[None]
        samples = v4_online_samples(objects, np.stack(self.hands)[None],self.runtime.geometry, goal_delta=goal['goal_delta'][None])
        if not self.runtime.workspace_valid(samples).all():
            raise ValueError('Inspire measured state outside WAM workspace')
        return samples[0], goal

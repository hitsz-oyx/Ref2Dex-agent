"""Original Inspire motion reference in the collector's 11-link convention."""
import numpy as np
import torch

from .contracts import HAND_LINKS
from .reset_kinematics import ResetKinematics

REFERENCE_SCHEMA = 'ref2dex.consequence-reference-progress.reference.v1'
# Native Gym DOF order, not XML joint order. The six wrist DOFs carry the
# world pose; the Inspire actor root is identity (_set_env_state).
NATIVE_DOF_NAMES = ('joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6',
                    'index_proximal_joint', 'index_intermediate_joint',
                    'middle_proximal_joint', 'middle_intermediate_joint',
                    'pinky_proximal_joint', 'pinky_intermediate_joint',
                    'ring_proximal_joint', 'ring_intermediate_joint',
                    'thumb_proximal_yaw_joint', 'thumb_proximal_pitch_joint',
                    'thumb_intermediate_joint', 'thumb_distal_joint')
QPOS_START = 245 + 32 * 4


def reconstruct_points(q, root, urdf):
    q = torch.as_tensor(np.asarray(q), dtype=torch.float32)
    root = torch.as_tensor(np.asarray(root), dtype=torch.float32)
    fk = ResetKinematics(urdf, NATIVE_DOF_NAMES, HAND_LINKS, 'cpu')
    with torch.inference_mode():
        return fk.states(q, torch.zeros_like(q), root)[:, :, :3].numpy()


def original_reference(motion, urdf):
    """No human point substitution, simulation, PD conversion or phase labels."""
    data = torch.load(motion, map_location='cpu', weights_only=True)
    if (not isinstance(data, torch.Tensor) or data.ndim != 2
            or data.shape[1] < QPOS_START + 18 or len(data) < 25):
        raise ValueError('original retargeted Inspire reference tensor required')
    data = data.float().numpy()
    q = data[:, QPOS_START:QPOS_START + 18]
    if not np.isfinite(q).all():
        raise ValueError('nonfinite reference robot joints')
    root = np.zeros((len(q), 13), np.float32)
    root[:, 6] = 1
    points = reconstruct_points(q, root, urdf)
    from src.task.CmResidual.object_frame_kinematics import quaternion_matrix
    quaternion = torch.as_tensor(data[:, 201:205])
    if not torch.isfinite(quaternion).all() or torch.any(quaternion.norm(dim=1) < 1e-6):
        raise ValueError('invalid reference object quaternion')
    poses = np.tile(np.eye(4), (len(q), 1, 1))
    poses[:, :3, 3] = data[:, 198:201]
    poses[:, :3, :3] = quaternion_matrix(quaternion).numpy()
    return dict(object_pose=poses.astype(np.float32), hand_keypoints=points,
                timestamps=np.arange(len(q), dtype=np.float64) / 30, q=q.copy())

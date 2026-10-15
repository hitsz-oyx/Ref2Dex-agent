"""Explicit 15 -> 30Hz HOI4D resampling, preserving observed vs synthetic frames."""
import numpy as np
from scipy.spatial.transform import Rotation

from .windows import validate_sequence


def resample_native15(sequence, row):
    """Only the author-consistent 300-frame/20-second subset; no extrapolation."""
    ids = np.asarray(sequence['native_frame_id'])
    n = len(ids)
    if (n != 300 or row.get('native_frames') != 300 or row.get('action_duration_s') != 20
            or row.get('clock_status') != 'CONSISTENT_WITH_NOMINAL15'
            or ids.dtype.kind not in 'iu' or not (np.diff(ids) == 1).all()):
        raise ValueError('HOI4D consistent native15 clock subset required')
    # Geometry validation is independent of the native sampling interval.
    geometry = dict(time=np.arange(n)/30, hand=sequence['hand'], hand_valid=sequence['hand_valid'],
        object_poses=sequence['object_poses'], object_pose_valid=sequence['object_pose_valid'], segment=np.zeros(n,np.int64))
    validate_sequence(geometry)
    poses, hand = sequence['object_poses'], sequence['hand']
    known, valid = sequence['object_pose_valid'], sequence['hand_valid']
    count = 2*n-1
    result = dict(time=np.arange(count,dtype=np.float64)/30,
        hand=np.full((count,2,11,3),np.nan,np.float32), hand_valid=np.zeros((count,2,11),bool),
        object_poses=np.full((count,poses.shape[1],4,4),np.nan,np.float32),
        object_pose_valid=np.zeros((count,poses.shape[1]),bool), segment=np.zeros(count,np.int64),
        native_source_frame_ids=ids.copy(), native_frame_position=ids[0]+np.arange(count)/2,
        observed_frame=np.arange(count)%2 == 0, interpolated_frame=np.arange(count)%2 == 1)
    result['hand'][::2] = hand
    result['hand_valid'][::2] = valid
    result['object_poses'][::2] = poses
    result['object_pose_valid'][::2] = known
    midpoint_valid = valid[:-1] & valid[1:]
    result['hand'][1::2] = np.where(midpoint_valid[...,None], (hand[:-1]+hand[1:])/2, np.nan)
    result['hand_valid'][1::2] = midpoint_valid
    for slot in range(poses.shape[1]):
        pairs = np.flatnonzero(known[:-1,slot] & known[1:,slot])
        if not len(pairs):
            continue
        q0 = Rotation.from_matrix(poses[pairs,slot,:3,:3]).as_quat()
        q1 = Rotation.from_matrix(poses[pairs+1,slot,:3,:3]).as_quat()
        q1 = np.where((q0*q1).sum(-1,keepdims=True) < 0, -q1, q1)
        # At t=1/2, the normalized short-arc quaternion sum is exact SLERP.
        middle = q0+q1
        middle /= np.linalg.norm(middle,axis=-1,keepdims=True)
        output_ids = 2*pairs+1
        result['object_poses'][output_ids,slot] = np.eye(4,dtype=np.float32)
        result['object_poses'][output_ids,slot,:3,:3] = Rotation.from_quat(middle).as_matrix()
        result['object_poses'][output_ids,slot,:3,3] = (poses[pairs,slot,:3,3]+poses[pairs+1,slot,:3,3])/2
        result['object_pose_valid'][output_ids,slot] = True
    validate_sequence(result)
    return result


def validate_resampled_clock(sequence, source_fps):
    """Verify the full mapping; frame repetition cannot masquerade as new capture."""
    ids = np.asarray(sequence['native_source_frame_ids'])
    position = np.asarray(sequence['native_frame_position'])
    n = len(sequence['time'])
    observed, synthetic = sequence['observed_frame'], sequence['interpolated_frame']
    if (source_fps != 15 or ids.dtype.kind not in 'iu' or not (np.diff(ids) == 1).all()
            or n != 2*len(ids)-1 or position.shape != (n,)
            or not np.allclose(position,ids[0]+np.arange(n)/2,atol=1e-10,rtol=0)
            or not np.allclose(sequence['time'],(position-position[0])/15,atol=1e-10,rtol=0)
            or observed.dtype.kind != 'b' or synthetic.dtype.kind != 'b'
            or not np.array_equal(observed,np.arange(n)%2 == 0)
            or not np.array_equal(synthetic,~observed)):
        raise ValueError('invalid native15 -> output30 clock/provenance mapping')

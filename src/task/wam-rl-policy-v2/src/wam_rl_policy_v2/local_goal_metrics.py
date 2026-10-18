"""Measured fixed-horizon outcome of a requested spatial goal; not success certification."""
import numpy as np
from scipy.spatial.transform import Rotation


def local_goal_outcomes(poses,goals,*,native_hz=60,horizon_s=.8):
    poses=np.asarray(poses,dtype=np.float64);goals=np.asarray(goals,dtype=np.float64)
    if poses.shape!=(len(goals)+1,4,4) or goals.shape[1:]!=(4,4) or not np.isfinite(poses).all() or not np.isfinite(goals).all():
        raise ValueError('N+1 measured poses and N requested goals required')
    offset=int(round(native_hz*horizon_s))
    if offset<1 or not np.isclose(offset/native_hz,horizon_s):raise ValueError('horizon must align with native clock')
    count=max(0,len(poses)-offset)
    ids=np.arange(count,dtype=np.int64)
    current=poses[:count];future=poses[offset:offset+count];target=goals[:count]
    initial=np.linalg.norm(target[:,:3,3]-current[:,:3,3],axis=-1)
    final=np.linalg.norm(target[:,:3,3]-future[:,:3,3],axis=-1)
    rotation=(Rotation.from_matrix(np.swapaxes(target[:,:3,:3],-1,-2)@future[:,:3,:3]).magnitude()
              if count else np.empty(0))
    return dict(ticks=ids,horizon_steps=offset,horizon_s=horizon_s,initial_position_error_m=initial,
        endpoint_position_error_m=final,endpoint_rotation_error_rad=rotation,
        position_error_reduction_m=initial-final,censored_tail_goals=len(goals)-count)

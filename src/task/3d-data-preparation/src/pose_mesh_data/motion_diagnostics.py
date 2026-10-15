"""Read-only physical motion diagnostics across heterogeneous native sources."""
import numpy as np


def transition_diagnostics(data, queries):
    """Yield metric arrays, masks and original end/start frames.

    Use recorded frames only (HOI4D is 15Hz even inside a 30Hz view). Never
    bridge unknown observations, a segment boundary, or a missing source tick.
    Surface displacement uses fixed query identities and includes rotation.
    """
    time=np.asarray(data['time'],np.float64);poses=np.asarray(data['object_poses'],np.float64)
    known=np.asarray(data['object_pose_valid'],bool);hand=np.asarray(data['hand'],np.float64)
    valid=np.asarray(data['hand_valid'],bool)
    observed=np.asarray(data.get('observed_frame',np.ones(len(time),bool)),bool)
    ids=np.flatnonzero(observed)
    if len(ids)<3:raise ValueError('at least three recorded observations required')
    dt=np.diff(time[ids]);segment=np.asarray(data['segment'])
    nominal=2/30 if not observed.all() else 1/30
    links=(segment[ids[1:]]==segment[ids[:-1]]) & np.isclose(dt,nominal,atol=1e-5,rtol=0)
    if not np.isfinite(time).all() or (dt<=0).any():raise ValueError('finite increasing source clock required')
    start,end=ids[:-1],ids[1:]
    objmask=known[start]&known[end]&links[:,None]
    handmask=valid[start]&valid[end]&links[:,None,None]
    translation=poses[ids,:,:3,3]
    velocity=np.diff(translation,axis=0)/dt[:,None,None]
    hand_velocity=np.diff(hand[ids],axis=0)/dt[:,None,None,None]
    relative_trace=np.einsum('toij,toij->to',poses[start,:,:3,:3],poses[end,:,:3,:3])
    omega=np.arccos(np.clip((relative_trace-1)/2,-1,1))/dt[:,None]
    surface=np.empty(objmask.shape,np.float64)
    q=np.asarray(queries,np.float64)
    if q.shape!=(poses.shape[1],32,3):raise ValueError('32 fixed query identities per object required')
    for offset in range(0,len(start),256):
        a,b=start[offset:offset+256],end[offset:offset+256]
        change=poses[b,:,:3,:3]-poses[a,:,:3,:3]
        delta=np.einsum('toij,opj->topi',change,q)+(poses[b,:,:3,3]-poses[a,:,:3,3])[:,:,None]
        surface[offset:offset+256]=np.linalg.norm(delta,axis=-1).max(-1)
    yield 'object_step_m',np.linalg.norm(np.diff(translation,axis=0),axis=-1),objmask,start,end
    yield 'surface_step_m',surface,objmask,start,end
    yield 'object_speed_mps',np.linalg.norm(velocity,axis=-1),objmask,start,end
    yield 'angular_speed_radps',omega,objmask,start,end
    yield 'hand_step_m',np.linalg.norm(np.diff(hand[ids],axis=0),axis=-1),handmask,start,end
    yield 'hand_speed_mps',np.linalg.norm(hand_velocity,axis=-1),handmask,start,end
    accdt=(dt[1:]+dt[:-1])/2
    acc=np.linalg.norm(np.diff(velocity,axis=0)/accdt[:,None,None],axis=-1)
    handacc=np.linalg.norm(np.diff(hand_velocity,axis=0)/accdt[:,None,None,None],axis=-1)
    yield 'object_acceleration_mps2',acc,objmask[1:]&objmask[:-1],start[:-1],end[1:]
    yield 'hand_acceleration_mps2',handacc,handmask[1:]&handmask[:-1],start[:-1],end[1:]


def window_coverage(ticks,frames):
    changes=np.zeros(frames+1,np.int64);ticks=np.asarray(ticks,np.int64)
    if len(ticks):
        if (ticks<3).any() or (ticks+24>=frames).any():raise ValueError('H4/K24 index exceeds capture')
        np.add.at(changes,ticks-3,1);np.add.at(changes,ticks+25,-1)
    return np.cumsum(changes[:-1])>0

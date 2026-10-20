"""Measured rollout admission; no comparison to DExplore training references."""
import numpy as np

QUALITY_SCHEMA='ref2dex.teacher-measured-rollout-admission.v2'
TABLE_SUPPORT_CONTRACT='ref2dex.table-support.thin-axis-world-up.v1'


def failure_masks(bad_frames,*,preceding_frames=50,history=4,horizon=24):
    bad=np.asarray(bad_frames,bool);n=len(bad)
    unsafe=np.zeros(n,bool)
    for t in np.flatnonzero(bad):unsafe[max(0,t-preceding_frames):t+1]=True
    prefix=np.r_[0,np.cumsum(unsafe)]
    commands=~(unsafe[:-1]|unsafe[1:])
    ticks=np.arange(max(0,n-horizon))
    windows=(prefix[ticks+horizon+1]-prefix[np.maximum(0,ticks-history+1)])==0
    return commands,windows,unsafe


def measured_failures(poses,coupled,lost,supported,support_gap,footprint):
    from scipy.spatial.transform import Rotation
    poses=np.asarray(poses);coupled=np.asarray(coupled,bool);lost=np.asarray(lost,bool)
    had=np.maximum.accumulate(coupled)
    jump=np.r_[False,np.linalg.norm(np.diff(poses[:,:3,3],axis=0),axis=-1)>.08]
    rotation=Rotation.from_matrix(poses[:-1,:3,:3].transpose(0,2,1)@poses[1:,:3,:3]).magnitude()
    jump|=np.r_[False,rotation>.7]
    penetration=np.asarray(footprint,bool)&(np.asarray(support_gap)<-.02)
    drop=had&lost&~np.asarray(supported,bool)
    return dict(unwitnessed_large_step=jump,table_penetration=penetration,unsupported_loss_after_coupling=drop)


def table_support(poses,points,table_pose,table_bounds,object_force):
    from scipy.spatial.transform import Rotation
    p=np.asarray(poses);points=np.asarray(points);table=np.asarray(table_pose)
    bounds=np.asarray(table_bounds)
    rotation=Rotation.from_quat(table[3:7]).as_matrix();gaps=[];footprints=[]
    # Tabletop meshes may lie in local XY or XZ. The thin dimension is the
    # normal; select its face pointing against gravity, including negative axes.
    normal_axis=int(np.argmin(bounds[1]-bounds[0]))
    sign=1. if rotation[2,normal_axis]>=0 else -1.
    normal=sign*rotation[:,normal_axis]
    if normal[2]<.5:raise ValueError('tabletop thin-axis normal must face world up')
    tangent_axes=[axis for axis in range(3) if axis!=normal_axis]
    top=bounds[1,normal_axis] if sign>0 else -bounds[0,normal_axis]
    for lo in range(0,len(p),128):
        batch=p[lo:lo+128]
        world=points[None]@batch[:,:3,:3].transpose(0,2,1)+batch[:,None,:3,3]
        local=(world-table[:3])@rotation;center=local.mean(1)
        gaps.extend((sign*local[:,:,normal_axis]).min(1)-top)
        footprints.extend(((center[:,tangent_axes]>=bounds[0,tangent_axes])
                          &(center[:,tangent_axes]<=bounds[1,tangent_axes])).all(1))
    gaps=np.asarray(gaps);footprints=np.asarray(footprints)
    # This is a conservative geometry/net-force proxy, not pairwise contact.
    normal_force=np.asarray(object_force)@normal
    supported=footprints&(np.abs(gaps)<=.02)&(normal_force>1e-4)
    return supported,gaps,footprints

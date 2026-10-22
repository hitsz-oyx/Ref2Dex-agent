"""Physical checks independent of any neural encoder or predicted trajectory."""
import numpy as np
from scipy.spatial.transform import Rotation
from .teacher_quality import table_support

PHYSICAL_SAFETY_POLICY='measured.table20mm.prismatic100mm.revolute100mrad.v1'


def physical_safety(snapshot,index,points,table_bounds,joint_limits=None):
    obj=np.asarray(snapshot['objects'])[index];q=np.asarray(snapshot['q'])[index];dq=np.asarray(snapshot['dq'])[index]
    if not np.isfinite(np.r_[obj,q,dq]).all():raise ValueError('nonfinite physical state; abort native run')
    pose=np.eye(4);pose[:3,:3]=Rotation.from_quat(obj[3:7]).as_matrix();pose[:3,3]=obj[:3]
    _,gap,footprint=table_support(pose[None],points,np.asarray(snapshot['table_pose'])[index],table_bounds,np.asarray(snapshot['object_contact_force'])[index:index+1])
    reason='physical_table_penetration' if footprint[0] and gap[0]<-.02 else None
    violation=[]
    if joint_limits is not None:
        mask=np.asarray(joint_limits['has_limits'],bool);lower=np.asarray(joint_limits['lower']);upper=np.asarray(joint_limits['upper'])
        if mask.shape!=(18,) or lower.shape!=(18,) or upper.shape!=(18,):raise ValueError('native joint safety bounds required')
        # DOFs0:3 are prismatic (metres); the remaining DOFs are radians.
        # The equal numeric margins have different physical units.
        margins=np.full(18,.10)
        violation=np.flatnonzero(mask&((q<lower-margins)|(q>upper+margins))).tolist()
        if violation:reason='physical_joint_limit'
    return dict(abort=reason is not None,reason=reason,table_gap_m=float(gap[0]),inside_table_footprint=bool(footprint[0]),
        joint_violation_indices=violation,joint_bounds_available=joint_limits is not None,
        prismatic_margin_m=.10,revolute_margin_rad=.10,
        scope='CAD/table penetration and actual limited-joint positions; pairwise hand-object penetration not measured')

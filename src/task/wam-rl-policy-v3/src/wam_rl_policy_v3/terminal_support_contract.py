"""Necessary-condition proof for a fixed table, not an execution certificate."""
import numpy as np
from scipy.spatial.transform import Rotation
from .goal_scheduler import SchedulerConfig
from .teacher_quality import table_support


def terminal_support_compatibility(spec,points,table_pose,table_bounds,*,config=None):
    config=config or SchedulerConfig();points=np.asarray(points);table=np.asarray(table_pose);bounds=np.asarray(table_bounds)
    goal=spec.object_path[-1];rotation=Rotation.from_quat(table[3:7]).as_matrix()
    normal_axis=int(np.argmin(bounds[1]-bounds[0]));tangent=[i for i in range(3) if i!=normal_axis]
    _,gap,footprint=table_support(goal[None],points,table,bounds,np.array([[0.,0.,100.]]))
    # Any point can move at most translation_tol + 2*r*sin(angle_tol/2).
    # Ignoring the additional mean-point tolerance enlarges the allowed set;
    # disjointness of this superset remains a necessary-condition proof.
    angle=min(config.rotation_tolerance_rad,np.pi) if spec.goal_type=='se3_pose' else np.pi
    displacement_bound=float(config.position_tolerance_m+2*np.linalg.norm(points,axis=-1).max()*np.sin(angle/2))
    center=((points@goal[:3,:3].T+goal[:3,3])-table[:3])@rotation;center=center.mean(0)
    outside=np.maximum(np.maximum(bounds[0,tangent]-center[tangent],center[tangent]-bounds[1,tangent]),0)
    gap_impossible=bool(abs(gap[0])>.02+displacement_bound+1e-9)
    footprint_impossible=bool(outside.max()>displacement_bound+1e-9)
    applies=bool(spec.terminal_mode.value=='RELEASE' and spec.require_support)
    return dict(applicable=applies,geometrically_disproved=bool(applies and (gap_impossible or footprint_impossible)),
        goal_gap_m=float(gap[0]),goal_inside_table_footprint=bool(footprint[0]),point_displacement_upper_bound_m=displacement_bound,
        possible_gap_interval_m=[float(gap[0]-displacement_bound),float(gap[0]+displacement_bound)],required_support_gap_interval_m=[-.02,.02],
        outside_footprint_by_axis_m=outside.tolist(),gap_sets_disjoint=gap_impossible,footprint_sets_disjoint=footprint_impossible,
        position_tolerance_m=config.position_tolerance_m,rotation_tolerance_rad=config.rotation_tolerance_rad,
        scope='Fixed measured table and current geometric support predicate; no assertion that other tasks are physically executable, no labels rewritten.')

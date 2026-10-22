"""The three-arm task is a new synthetic instruction, not a corpus relabeling."""
import numpy as np

TASK_CONTRACT='synthetic-grasp-lift8cm-position-HOLD1s-timeout20s.v2'
SOURCE_TASK_CONTRACT='source-declared-full-task.stage-observation-budget20s.v1'
ENCODER_POLICY='voxel-input-mask-only.same-executor-zero-tau-valid-false.v2'
FALLBACK_POLICY='same-executor-observed-state.tau-zero.valid-false.v1'
ENCODER_CONFIG=dict(voxel_origin_m=[-2.,-2.,-2.],voxel_m=.005,max_grid=65535)


def encoder_domain_available(points,canonical_hand,pose):
    local=(np.asarray(canonical_hand)-pose[:3,3])@pose[:3,:3]
    xyz=np.r_[points,local].astype(np.float32)
    grid=np.floor((xyz-np.asarray(ENCODER_CONFIG['voxel_origin_m'],np.float32))/np.float32(ENCODER_CONFIG['voxel_m']))
    return bool(np.isfinite(grid).all() and grid.min()>=0 and grid.max()<=65535)


def grasp_lift_task(initial_pose):
    from wam_rl_policy_v3.task_spec import TaskSpec,TaskEvent,EventType,TerminalMode
    path=np.repeat(np.asarray(initial_pose)[None],2,0);path[1,2,3]+=.08
    return TaskSpec(path,np.array([0,60]),30.,61,TerminalMode.HOLD,(TaskEvent(0,EventType.CONTACT),),
        goal_type='position_only',require_support=False)


class LiftHoldWitness:
    """Independent narrow calibration task: >=5cm, coupled force-bearing hold1s.

    This explicitly separate task does not claim the synthetic 8cm endpoint or
    a corpus full RELEASE task. Duration is elapsed time, not sample occupancy.
    """
    contract='calibration.lift5cm-force-coupled-unsupported-elapsed1s.v1'
    def __init__(self,z):self.z=float(z);self.since=None;self.longest=0.;self.ever_success=False;self.final_success=False;self.last_time=None
    def update(self,z,now,feedback,contact_forces,*,physical_abort=False):
        if self.last_time is not None and now<=self.last_time:raise ValueError('strictly increasing witness time required')
        self.last_time=float(now)
        valid=bool(z-self.z>=.05 and feedback.grasp_established and not feedback.grasp_lost and feedback.force_valid
            and feedback.supported is False and (np.linalg.norm(contact_forces,axis=-1)>=.1).sum()>=2 and not physical_abort)
        if valid:
            if self.since is None:self.since=float(now)
            self.longest=max(self.longest,float(now)-self.since)
        else:self.since=None
        self.final_success=bool(valid and now-self.since>=1.-1e-10)
        self.ever_success|=self.final_success
        return self.final_success

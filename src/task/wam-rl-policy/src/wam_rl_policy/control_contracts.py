"""Versioned Shadow control boundary: intent, geometry, predictions, and actions differ."""
from dataclasses import dataclass
import numpy as np
from .goal_scheduler import rigid_goal_delta

ROBOT_ID='unidexgrasp2.shadow.v1'
HAND_LAYOUT='palm-thumb/index/middle/ring/little-bases-then-tips.v1'
ACTION_SPACE='unidexgrasp2.canonical24.yaw-wrist6-fingers18.v1'
SPATIAL_GOAL='spatial_path_waypoint'
PHASES=('unspecified','approach','grasp','hold','transport','place','release','recover')


def frozen_array(value,shape,name):
    value=np.asarray(value,dtype=np.float64).copy()
    if value.shape!=shape or not np.isfinite(value).all():raise ValueError('finite '+name+' shape '+str(shape)+' required')
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class TaskIntent:
    task_id: str
    phase: str = 'unspecified'

    def __post_init__(self):
        if not self.task_id or self.phase not in PHASES:raise ValueError('explicit task identity and supported phase vocabulary required')


@dataclass(frozen=True)
class ExecutionFeedback:
    """External evidence, never inferred from path progress or predicted C."""
    execution_ready: object = None
    lost_control: bool = False
    task_success: object = None
    source: str = 'unknown'

    def __post_init__(self):
        for name in ('execution_ready','task_success'):
            if getattr(self,name) is not None and type(getattr(self,name)) is not bool:raise ValueError('optional boolean '+name+' required')
        if type(self.lost_control) is not bool:raise ValueError('boolean lost_control required')
        if (self.execution_ready is not None or self.task_success is not None or self.lost_control) and self.source=='unknown':
            raise ValueError('external feedback must name its evidence source')


@dataclass(frozen=True)
class MeasuredShadowState:
    episode_id: str
    tick: int
    time_s: float
    current: np.ndarray
    object_pose: np.ndarray
    hand: np.ndarray
    hand_valid: np.ndarray
    robot_id: str = ROBOT_ID
    hand_layout: str = HAND_LAYOUT

    def __post_init__(self):
        if not self.episode_id or type(self.tick) is not int or self.tick<0 or not np.isfinite(self.time_s):raise ValueError('episode, nonnegative tick and finite time required')
        if self.robot_id!=ROBOT_ID or self.hand_layout!=HAND_LAYOUT:raise ValueError('robot/hand-layout adapter mismatch')
        for name,shape in (('current',(300,)),('object_pose',(4,4)),('hand',(11,3))):
            object.__setattr__(self,name,frozen_array(getattr(self,name),shape,name))
        valid=np.asarray(self.hand_valid)
        if valid.shape!=(11,) or valid.dtype!=np.bool_ or not valid.all():
            raise ValueError('current Shadow adapter requires all 11 measured points; missing points cannot be imputed')
        valid=valid.copy();valid.setflags(write=False);object.__setattr__(self,'hand_valid',valid)
        pose=self.object_pose;r=pose[:3,:3]
        if not np.allclose(pose[3],[0,0,0,1]) or not np.allclose(r.T@r,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(r),1.,atol=1e-5):raise ValueError('proper world-from-object SE(3) required')


def goal_envelope(decision,state,queries):
    """Do not relabel a spatial waypoint as a timed reference endpoint."""
    if decision.get('goal_semantics')!=SPATIAL_GOAL:raise ValueError('this bridge requires explicit spatial waypoint semantics')
    if not np.isclose(decision['horizon_s'],.8):raise ValueError('WAM v4 forecast horizon must be 24/30 seconds')
    points=frozen_array(queries,(32,3),'fixed flow queries')
    delta=rigid_goal_delta(state.object_pose,decision['goal_pose'],points)
    if not np.allclose(delta,decision['goal_delta'],atol=1e-7,rtol=1e-6):raise ValueError('goal delta disagrees with current anchor and point identities')
    return dict(schema='ref2dex.local-goal.v1',episode_id=state.episode_id,tick=state.tick,
        anchor_time_s=state.time_s,frame='current_object_anchor',units='metres',query_count=32,
        semantics=SPATIAL_GOAL,forecast_steps=24,forecast_hz=30,forecast_horizon_s=.8,
        reference_endpoint_time_s=None,reachability_certified=False,
        training_condition_semantics='observed_future_endpoint_24_at_30hz',
        deployment_distribution_shift='spatial_waypoint_not_observed_future_endpoint',
        goal_pose=frozen_array(decision['goal_pose'],(4,4),'goal pose'),goal_delta=frozen_array(delta,(32,3),'goal delta'))


def trajectory_contract(state,history_valid):
    return dict(schema='ref2dex.executor-tau.v1',episode_id=state.episode_id,anchor_tick=state.tick,
        anchor_time_s=state.time_s,frame='world',units='metres',representation='displacement_from_current_hand',
        hz=60,target_offsets_s=(np.arange(1,9)/60).tolist(),hand_slot='right',hand_layout=HAND_LAYOUT,
        valid=np.full((8,11),bool(history_valid)),validity_semantics='observed_support_not_future_success',source='WAMv4.24x30hz.current_object_anchor',
        interpolation='linear_with_tau0_zero',interaction_role='prediction_not_contact_certificate')


def scheduled_conditioning_contract():
    """Serializable training/deployment identity, independent of run paths."""
    return dict(schema='ref2dex.scheduled-shadow-conditioning.v1',robot_id=ROBOT_ID,
        hand_layout=HAND_LAYOUT,action_space=ACTION_SPACE,input_dim=565,current_dim=300,
        goal_semantics=SPATIAL_GOAL,goal_reach_time_guaranteed=False,
        wam_hz=30,wam_steps=24,tau_frame='world',tau_representation='displacement_from_current_hand',
        tau_hz=60,tau_steps=8,tau_hand_slot='right',tau_units='metres',
        history_indices=[-6,-4,-2,0],native_hz=60,startup='zero_tau_with_validity_zero',
        phase_conditioning=False,interaction_conditioning=False)

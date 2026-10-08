"""Ref4 outcome supervision, independent of the old H-matched local-event schema.

S is an observed fixed-controller suffix outcome, not an optimal Q label.
Geometry-only proximity/persistence are task proxies, not collision-force GT.
"""
import numpy as np

from .contracts import K
from .supervision import consecutive

RAW_SCHEMA = 'ref2dex.consequence-value.episodes.v1'
DATA_SCHEMA = 'ref2dex.consequence-value.windows.v1'
RULE = 'full-reference-grasp-lift-controlled-place-stage-margin-v1'
PARAMETERS = dict(lift_m=.03, drop_height_m=.02, surface_gap_m=.01,
                  stable_frames=45, lost_geometry_frames=6, grasp_frames=3,
                  support_gap_m=.02,settled_frames=15,settled_linear_mps=.05,
                  settled_angular_rps=.3,unheld_fall_mps=.25,
                  preference_progress_deadzone=.05,
                  preference_margin_deadzone=[.005,.125,.002])


def task_trace(packet, diagnostics):
    steps = len(packet['action'])
    pose = np.asarray(packet['object_pose'])
    gap = np.asarray(diagnostics['surface_gap'])
    valid = np.asarray(diagnostics['contact_valid'])
    if (pose.shape != (steps+1,4,4) or gap.shape != (steps+1,) or valid.shape != (steps+1,)
            or valid.dtype != np.bool_ or valid[0] or not valid[1:].all()
            or not np.isfinite(pose).all() or not np.isfinite(gap).all() or (gap < 0).any()
            or not np.isclose(float(diagnostics['initial_height']),pose[0,2,3],atol=1e-5)):
        raise ValueError('complete measured physical diagnostics required')
    height = pose[:,2,3]-pose[0,2,3]
    near = valid & (gap <= PARAMETERS['surface_gap_m'])
    near_run = consecutive(near)
    held = near & (height >= PARAMETERS['lift_m'])
    held_run = consecutive(held)
    velocity = np.asarray(diagnostics['object_velocity'])
    support = np.asarray(diagnostics['support_gap'])
    footprint = np.asarray(diagnostics['table_footprint'])
    reference = np.asarray(diagnostics['reference_object_pose'])
    if (velocity.shape != (steps+1,6) or support.shape != (steps+1,)
            or footprint.shape != (steps+1,) or reference.shape != pose.shape
            or not np.isfinite(velocity).all() or not np.isfinite(support).all()
            or not np.isfinite(reference).all() or (footprint!=footprint.astype(bool)).any()):
        raise ValueError('complete reference, velocity and measured table support required')
    reference_height=reference[:,2,3]-reference[0,2,3]
    elevated=np.flatnonzero(reference_height>=PARAMETERS['lift_m'])
    if not len(elevated) or abs(reference_height[-1])>PARAMETERS['support_gap_m']:
        raise ValueError('fixed full reference must lift and return to support')
    place_start=int(elevated[-1])+1
    supported=valid&footprint.astype(bool)&(np.abs(support)<=PARAMETERS['support_gap_m'])
    held &= ~supported
    held_run=consecutive(held)
    settled=supported&(np.linalg.norm(velocity[:,:3],axis=-1)<=PARAMETERS['settled_linear_mps'])
    settled &= np.linalg.norm(velocity[:,3:],axis=-1)<=PARAMETERS['settled_angular_rps']
    stage = np.zeros(steps+1,np.int64)
    stage[near_run >= PARAMETERS['grasp_frames']] = 1
    stage[held] = 2
    stage[held_run >= PARAMETERS['stable_frames']] = 3
    lost = consecutive(valid & ~near)
    drop = (height < PARAMETERS['drop_height_m']) | (lost >= PARAMETERS['lost_geometry_frames'])
    completion=np.flatnonzero(held_run>=PARAMETERS['stable_frames'])
    controlled=False
    if len(completion):
        first=int(completion[0])
        before_place=slice(first+1,place_start)
        premature=bool(drop[before_place].any())
        unsupported_lost=consecutive(valid&~near&~supported)>=PARAMETERS['lost_geometry_frames']
        falling=valid&~near&~supported&(velocity[:,2]<-PARAMETERS['unheld_fall_mps'])
        unsafe=unsupported_lost|falling
        controlled=not premature and not unsafe[max(first+1,place_start):].any()
        stage[place_start:] = np.where(near[place_start:]|supported[place_start:],4,stage[place_start:])
        if controlled:
            placed=(consecutive(settled)>=PARAMETERS['settled_frames'])&(np.arange(steps+1)>=place_start)
            stage[placed]=5
    success=bool(controlled and consecutive(settled)[-1]>=PARAMETERS['settled_frames'])
    return dict(height=height,gap=gap,valid=valid,near=near,held=held,held_run=held_run,
                stage=stage,progress=(stage/5).astype('float32'),drop=drop,
                support_gap=support,supported=supported,settled=settled,
                place_start=place_start,task_success=success)


def suffix_success(trace, tick):
    """Observed full-reference outcome after this plan and fixed continuation.

    Completed past task stages remain completed. Success also requires the
    observed future placement/end, rather than a new45frame hold after every
    late decision. Incomplete24step windows are unknown, never negative.
    """
    if len(trace['held'])-tick-1 < K:
        return None
    return int(trace['task_success'])


def validate_plan_execution(packet, record,residual_bound=.2):
    """Every accepted plan is preknown, executes24steps and has fixed continuation."""
    if not .2<=residual_bound<=.5:raise ValueError('registered bounded intervention required')
    steps = len(packet['action'])
    plans,known = packet['residual_plan'],packet['plan_known']
    if (plans.shape != (steps,K,18) or known.shape != (steps,) or known.dtype != np.bool_
            or not np.isfinite(plans).all() or np.abs(plans).max() > residual_bound+1e-6):
        raise ValueError('invalid decision-known residual schedule')
    trigger = int(record['perturbation_tick'])
    if trigger >= 0 and trigger+K > steps:
        raise ValueError('intervention did not execute its entire24step plan')
    if trigger < 0 and np.abs(plans).max() > 1e-7:
        raise ValueError('nonzero plan without a recorded intervention')
    for tick in np.flatnonzero(known[:max(0,steps-K+1)]):
        if not known[tick:tick+K].all():
            raise ValueError('future requested plan became unknown')
        if not np.allclose(plans[tick],plans[tick:tick+K,0],atol=1e-7,rtol=0):
            raise ValueError('recorded24plan differs from subsequently requested controls')
        if trigger > tick:
            raise ValueError('pre-trigger plan cannot promise fixed continuation')


def label_window(trace, tick):
    success = suffix_success(trace,tick)
    if success is None or tick+K >= len(trace['height']):
        return None
    future = slice(tick+1,tick+K+1)
    tail = slice(tick+K-7,tick+K+1)
    progress = trace['progress'][future].copy()
    stable_height = np.where(trace['near'][tail],trace['height'][tail],0)
    margin = np.array([np.median(stable_height),np.mean(trace['held'][tail]),
                       -np.median(trace['gap'][tail])],np.float32)
    return dict(success=int(success),progress=progress,progress_summary=float(progress.mean()),
                stage=trace['stage'][future].copy(),margin=margin,
                progress_mask=trace['valid'][future].copy())


def preference(first, second):
    """Lexicographic S/P/M with explicit deadzones; zero means tie/abstain."""
    if first['success'] != second['success']:
        return 1 if first['success'] > second['success'] else -1
    delta = first['progress_summary']-second['progress_summary']
    if abs(delta) > PARAMETERS['preference_progress_deadzone']:
        return 1 if delta > 0 else -1
    for a,b,deadzone in zip(first['margin'],second['margin'],PARAMETERS['preference_margin_deadzone']):
        if abs(a-b) > deadzone:
            return 1 if a > b else -1
    return 0

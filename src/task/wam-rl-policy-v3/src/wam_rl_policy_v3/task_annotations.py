"""Construct provisional tasks from same-execution teacher object/contact rollouts.

Measured contact endings are release hypotheses, not verified planned intent.
An accepted complete task declaration is kept separate from data availability.
Online feedback still comes from that later execution, never this saved future.
The reference-label helper below is retained only for explicit historical audits.
"""
import numpy as np
from .task_spec import TaskSpec,TaskEvent,EventType,TerminalMode

REFERENCE_CONTACT_COLUMNS=np.array([168+i for i in (3,6,9,12,15)])
ANNOTATION_SCHEMA='ref2dex.teacher-rollout-planned-events.v1'
GOAL_SOURCE='same_execution_teacher_object_rollout'
PREFIX_TASK_SEMANTICS='ref2dex.declared-task-prefix-mask.v1'


def task_with_safe_prefix(declared_task,safe_prefix_commands,*,unsafe_future=None):
    """Keep the full declaration; prefix limits usable data, never task intent.

    Measured force/contact-loss events remain an unverified task declaration.
    This helper does not certify that a RELEASE was intended or executable.
    """
    if not isinstance(declared_task,TaskSpec):raise TypeError('complete declared TaskSpec required')
    if type(safe_prefix_commands) is not int or not 0<=safe_prefix_commands<declared_task.source_length:
        raise ValueError('safe prefix control boundary outside original task source')
    events=declared_task.packet()['events'];end=int(declared_task.source_frames[-1])
    censored=safe_prefix_commands<declared_task.source_length-1
    availability=dict(schema=PREFIX_TASK_SEMANTICS,safe_prefix_commands=safe_prefix_commands,
        source_length=declared_task.source_length,declared_endpoint_frame=end,
        observed_prefix_last_frame=safe_prefix_commands,task_censored=censored,
        unsafe_future=unsafe_future,complete_positive_eligible=False,terminal_intent_verified=False,
        declared_events=events,events_beyond_observed_prefix=[e for e in events if e['source_frame']>safe_prefix_commands],
        events_without_available_command_state=[e for e in events if e['source_frame']>=safe_prefix_commands],
        event_labels_negative_beyond_prefix=False,
        declaration_provenance='Preserved complete task; measured force/proximity release inference is not verified planned intent.',
        interpretation='Prefix is a control availability/censoring boundary, never a replacement terminal, endpoint or event stream.')
    return declared_task,availability


def planned_contact_events(reference_current,*,minimum_contact_frames=3,merge_gap_frames=2):
    reference=np.asarray(reference_current)
    if (reference.ndim!=2 or len(reference)<2 or reference.shape[1]<=REFERENCE_CONTACT_COLUMNS.max()
            or minimum_contact_frames<1 or merge_gap_frames<0):raise ValueError('valid reference contact fields and frame limits required')
    labels=reference[:,REFERENCE_CONTACT_COLUMNS]
    if not np.isfinite(labels).all() or not np.isin(labels,[0,1]).all():raise ValueError('explicit binary DExplore contact annotations required')
    active=(labels>0).any(-1)
    return _events_from_mask(active,minimum_contact_frames,merge_gap_frames)


def _events_from_mask(active,minimum_contact_frames=3,merge_gap_frames=2):
    starts=np.flatnonzero(np.diff(np.r_[False,active,False].astype(np.int8))==1)
    ends=np.flatnonzero(np.diff(np.r_[False,active,False].astype(np.int8))==-1)
    intervals=[]
    for start,end in zip(starts,ends):
        if intervals and start-intervals[-1][1]<=merge_gap_frames:intervals[-1][1]=int(end)
        else:intervals.append([int(start),int(end)])
    intervals=[(start,end) for start,end in intervals if end-start>=minimum_contact_frames]
    events=[]
    for start,end in intervals:
        events.append(TaskEvent(start,EventType.CONTACT))
        if end<len(active):events.append(TaskEvent(end,EventType.RELEASE))
    terminal=(TerminalMode.UNKNOWN if not intervals else
              TerminalMode.HOLD if intervals[-1][1]==len(active) else TerminalMode.RELEASE)
    return tuple(events),terminal


def reference_task_spec(reference_pose,reference_current,*,source_fps=30,require_support=True):
    """Historical reference audit only; never the teacher-rollout task builder."""
    if len(reference_pose)!=len(reference_current):raise ValueError('pose/contact source frames must align')
    events,terminal=planned_contact_events(reference_current)
    return TaskSpec.from_reference(reference_pose,source_fps=source_fps,terminal_mode=terminal,
        events=events,require_support=require_support)


def teacher_rollout_task_spec(object_pose,contact_forces,force_valid,*,source_fps=30,terminal_mode=None,require_support=True):
    from dataclasses import replace
    forces=np.asarray(contact_forces);valid=np.asarray(force_valid)
    n=len(object_pose)
    if (forces.shape!=(n,5,3) or valid.shape!=(n,) or valid.dtype!=np.bool_
            or not np.isfinite(forces).all()):raise ValueError('aligned same-execution contact force/validity required')
    contact=(np.linalg.norm(forces,axis=-1)>=.1).sum(-1)>=2
    events,inferred=_events_from_mask(contact&valid)
    # Keep the actual initial anchor and terminal pose. Consecutive stationary
    # duplicates are simplified; return-to-start and original event frames stay.
    spec=TaskSpec.from_reference(object_pose,source_fps=source_fps,terminal_mode=inferred if terminal_mode is None else terminal_mode,
        events=events,trim_interaction=False,require_support=require_support)
    return replace(spec,source_role=GOAL_SOURCE)


def task_from_teacher_archive(data,contact_body_ids,*,source_fps=30,terminal_mode=None,object_points=None):
    """Named measured fields only: reference_obj/reference_current are ignored."""
    n=len(data['obj']);tick=np.asarray(data['frame_tick'])
    if not np.array_equal(tick,np.arange(n)):raise ValueError('contiguous same-episode rollout clock required')
    valid=np.asarray(data['contact_force_valid']).copy() if 'contact_force_valid' in data else tick>0
    if object_points is not None:
        points=np.asarray(object_points);hand=np.asarray(data['hand'])[:,[2,4,6,8,10]]
        if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise ValueError('finite object CAD points required')
        near=[]
        for lo in range(0,n,128):
            pose=np.asarray(data['obj'][lo:lo+128]);local=(hand[lo:lo+128]-pose[:,None,:3,3])@pose[:,:3,:3]
            gap=np.linalg.norm(local[:,:,None]-points[None,None],axis=-1).min(-1)
            near.extend((gap<=.018).sum(-1)>=2)
        valid&=np.asarray(near)
    return teacher_rollout_task_spec(data['obj'],data['body_contact_forces'][:,contact_body_ids],valid,
        source_fps=source_fps,terminal_mode=terminal_mode)

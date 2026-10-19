"""Convert known reference contact annotations into ordered task instructions.

These annotations describe intent, never measured execution feedback. In the
current DExplore export the five selected contact labels are binary. Other
encodings require an explicit adapter rather than an implicit threshold.
"""
import numpy as np
from .task_spec import TaskSpec,TaskEvent,EventType,TerminalMode

REFERENCE_CONTACT_COLUMNS=np.array([168+i for i in (3,6,9,12,15)])
ANNOTATION_SCHEMA='ref2dex.dexplore-planned-events.v1'


def planned_contact_events(reference_current,*,minimum_contact_frames=3,merge_gap_frames=2):
    reference=np.asarray(reference_current)
    if (reference.ndim!=2 or len(reference)<2 or reference.shape[1]<=REFERENCE_CONTACT_COLUMNS.max()
            or minimum_contact_frames<1 or merge_gap_frames<0):raise ValueError('valid reference contact fields and frame limits required')
    labels=reference[:,REFERENCE_CONTACT_COLUMNS]
    if not np.isfinite(labels).all() or not np.isin(labels,[0,1]).all():raise ValueError('explicit binary DExplore contact annotations required')
    active=(labels>0).any(-1)
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
        if end<len(reference):events.append(TaskEvent(end,EventType.RELEASE))
    terminal=(TerminalMode.UNKNOWN if not intervals else
              TerminalMode.HOLD if intervals[-1][1]==len(reference) else TerminalMode.RELEASE)
    return tuple(events),terminal


def dexplore_task_spec(reference_pose,reference_current,*,source_fps=30,require_support=True):
    if len(reference_pose)!=len(reference_current):raise ValueError('pose/contact source frames must align')
    events,terminal=planned_contact_events(reference_current)
    return TaskSpec.from_reference(reference_pose,source_fps=source_fps,terminal_mode=terminal,
        events=events,require_support=require_support)

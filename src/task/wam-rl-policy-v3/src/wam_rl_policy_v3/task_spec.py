"""Known object instructions and planned events, independent of measured execution."""
from dataclasses import dataclass
from enum import Enum, IntEnum
import numpy as np
from scipy.spatial.transform import Rotation
from .goal_scheduler import _poses

TASK_SCHEMA='ref2dex.object-task.v1'


class Phase(IntEnum):
    ACQUIRE=0
    TRACK=1
    RELEASE=2


class EventType(str,Enum):
    CONTACT='CONTACT'
    RELEASE='RELEASE'


class TerminalMode(str,Enum):
    HOLD='HOLD'
    RELEASE='RELEASE'
    UNKNOWN='UNKNOWN'


@dataclass(frozen=True)
class TaskEvent:
    source_frame:int
    kind:EventType

    def __post_init__(self):
        if type(self.source_frame) is not int or self.source_frame<0:raise ValueError('nonnegative source event frame required')
        object.__setattr__(self,'kind',EventType(self.kind))


@dataclass(frozen=True)
class TaskSpec:
    object_path:np.ndarray
    source_frames:np.ndarray
    source_fps:float
    source_length:int
    terminal_mode:TerminalMode
    events:tuple=()
    goal_type:str='se3_pose'
    require_support:bool=True
    source_role:str='explicit_task_instruction'
    annotation_invalid:bool=False
    annotation_invalid_reasons:tuple=()

    def require_success_path(self):
        if self.terminal_mode==TerminalMode.UNKNOWN:
            raise ValueError('UNKNOWN task has no defined success path; reject at executable-task admission')
        if self.annotation_invalid:
            raise ValueError('annotation_invalid task is ineligible for complete-task scoring: '+','.join(self.annotation_invalid_reasons))

    def __post_init__(self):
        poses=_poses(self.object_path);frames=np.asarray(self.source_frames)
        if (len(poses)<2 or frames.shape!=(len(poses),) or frames.dtype.kind not in 'iu'
                or (frames<0).any() or (np.diff(frames)<=0).any()
                or type(self.source_length) is not int or self.source_length<=int(frames[-1])
                or not np.isfinite(self.source_fps) or self.source_fps<=0):raise ValueError('ordered source-frame identities and positive fps required')
        if self.goal_type not in ('position_only','se3_pose'):raise ValueError('explicit task goal constraints required')
        if self.source_role not in ('explicit_task_instruction','same_execution_teacher_object_rollout'):raise ValueError('explicit task source role required')
        events=tuple(self.events)
        if any(not isinstance(e,TaskEvent) or e.source_frame>=self.source_length for e in events):raise ValueError('event outside original source')
        if any(a.source_frame>=b.source_frame for a,b in zip(events,events[1:])):raise ValueError('events must be ordered')
        if events and (events[0].kind!=EventType.CONTACT or any(a.kind==b.kind for a,b in zip(events,events[1:]))):
            raise ValueError('planned contact/release events must alternate starting with contact')
        terminal=TerminalMode(self.terminal_mode)
        if terminal==TerminalMode.HOLD and events and events[-1].kind==EventType.RELEASE:raise ValueError('HOLD cannot end with a planned release')
        poses.setflags(write=False);frames=frames.astype(np.int64,copy=True);frames.setflags(write=False)
        object.__setattr__(self,'object_path',poses);object.__setattr__(self,'source_frames',frames)
        object.__setattr__(self,'events',events);object.__setattr__(self,'terminal_mode',terminal)
        reasons=tuple(self.annotation_invalid_reasons)
        if terminal==TerminalMode.UNKNOWN and 'terminal_semantics_undefined' not in reasons:reasons+=('terminal_semantics_undefined',)
        if any(not isinstance(reason,str) or not reason for reason in reasons):raise ValueError('explicit annotation invalid reasons required')
        object.__setattr__(self,'annotation_invalid',bool(self.annotation_invalid or reasons))
        object.__setattr__(self,'annotation_invalid_reasons',reasons)

    @classmethod
    def from_reference(cls,poses,*,source_fps,terminal_mode,events=(),goal_type='se3_pose',
                       trim_interaction=True,translation_epsilon_m=1e-4,rotation_epsilon_rad=1e-3,require_support=True):
        poses=_poses(poses);events=tuple(events)
        if len(poses)<2 or translation_epsilon_m<0 or rotation_epsilon_rad<0:raise ValueError('valid reference and simplification tolerances required')
        first,last=0,len(poses)-1
        if trim_interaction and events:
            first=events[0].source_frame
            if events[-1].kind==EventType.RELEASE:last=events[-1].source_frame
        if not 0<=first<=last<len(poses):raise ValueError('event outside reference')
        if first==last:first=max(0,first-1);last=min(len(poses)-1,last+1)
        keep=[first];mandatory={e.source_frame for e in events}
        for i in range(first+1,last):
            prior=poses[keep[-1]]
            distance=np.linalg.norm(poses[i,:3,3]-prior[:3,3])
            angle=Rotation.from_matrix(prior[:3,:3].T@poses[i,:3,:3]).magnitude()
            if i in mandatory or distance>translation_epsilon_m or (goal_type=='se3_pose' and angle>rotation_epsilon_rad):keep.append(i)
        keep.append(last)
        return cls(poses[keep],np.array(keep),source_fps,len(poses),terminal_mode,events,goal_type,require_support)

    @classmethod
    def shadow_goal(cls,start,goal_position):
        path=np.repeat(np.asarray(start)[None],2,axis=0).copy();path[1,:3,3]=goal_position
        return cls.from_reference(path,source_fps=60,terminal_mode=TerminalMode.HOLD,
            goal_type='position_only',trim_interaction=False,require_support=False)

    def packet(self):
        return dict(schema=TASK_SCHEMA,object_path=self.object_path.tolist(),source_frames=self.source_frames.tolist(),
            source_fps=self.source_fps,source_length=self.source_length,terminal_mode=self.terminal_mode.value,
            events=[dict(source_frame=e.source_frame,kind=e.kind.value) for e in self.events],
            goal_type=self.goal_type,require_support=self.require_support,source_role=self.source_role,
            annotation_invalid=self.annotation_invalid,annotation_invalid_reasons=list(self.annotation_invalid_reasons))

    @classmethod
    def from_packet(cls,packet):
        if packet.get('schema')!=TASK_SCHEMA:raise ValueError('unsupported TaskSpec schema')
        return cls(np.asarray(packet['object_path']),np.asarray(packet['source_frames']),packet['source_fps'],
            packet['source_length'],packet['terminal_mode'],tuple(TaskEvent(**e) for e in packet['events']),
            packet['goal_type'],packet['require_support'],packet.get('source_role','explicit_task_instruction'),
            packet.get('annotation_invalid',False),tuple(packet.get('annotation_invalid_reasons',())))

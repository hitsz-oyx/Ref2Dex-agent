"""Measured-feedback task phases around the existing ordered geometric tracker."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .goal_scheduler import OrderedGoalScheduler,SchedulerConfig,_pose
from .task_spec import TaskSpec,Phase,EventType,TerminalMode
from .task_feedback import InteractionFeedback

SCHEDULER_SCHEMA='ref2dex.measured-task-scheduler.v1'


@dataclass(frozen=True)
class TaskSchedulerConfig:
    acquire_timeout_s:float=15.
    release_timeout_s:float=10.
    episode_timeout_s:float=60.
    terminal_hold_s:float=1.
    probe_distance_m:float=.003

    def __post_init__(self):
        if any(not np.isfinite(v) or v<=0 for v in self.__dict__.values()):raise ValueError('positive task scheduler limits required')


@dataclass(frozen=True)
class PlannedEvent:
    kind:EventType|None
    delta_s:float
    valid:bool

    def wam_condition(self):
        # WAM ticks always mean 30 Hz, independent of native action frequency.
        # Matches WAM-v5's next-contact-release-plan.frame30.v1 encoding.
        kind={None:0,EventType.CONTACT:1,EventType.RELEASE:2}[self.kind]
        return dict(event_type=kind,event_delta_steps=self.delta_s*30,event_valid=self.valid,event_hz=30)


@dataclass(frozen=True)
class SchedulerDecision:
    phase:Phase
    local_goal_pose:np.ndarray
    terminal_goal_pose:np.ndarray
    path_progress:float
    next_event:PlannedEvent
    path_complete:bool
    task_complete:bool
    terminated:bool
    failure_reason:str|None
    probe_active:bool
    recovering:bool
    transitions:tuple
    source_cursor_frame:float
    goal_type:str


class TaskScheduler:
    def __init__(self,spec:TaskSpec,object_points,*,config=None,path_config=None,interaction_points=None):
        self.spec=spec;self.config=config or TaskSchedulerConfig()
        self.path=OrderedGoalScheduler(spec.object_path,spec.source_frames/spec.source_fps,object_points,
            config=path_config or SchedulerConfig(),goal_type=spec.goal_type,interaction_points=interaction_points)
        self.reset()

    def reset(self):
        self.path.reset();self.phase=Phase.ACQUIRE;self.start_time=self.last_time=self.phase_since=None
        self.path_decision=None;self.settle_since=None;self.probe_origin=None
        self.failure_reason=None;self.complete=False;self.recovering=False;self.paid_events=set()
        self.plan_cursor_frame=float(self.spec.source_frames[0])
        self.confirmed_progress=0.;self.confirmed_pointer=0;self.confirmed_path_complete=False
        self.acquire_source_origin=0.;self.event_index=0
        self.last_grasp=False;self.last_release=False

    def _transition(self,event,events):
        if event not in self.paid_events:self.paid_events.add(event);events.append(event)

    def _event(self,now):
        # Loss of grasp is a recovery instruction, never completion of the
        # intended future release. Keep the immutable plan cursor underneath.
        if self.phase==Phase.ACQUIRE and self.recovering:
            return PlannedEvent(EventType.CONTACT,0.,True)
        if self.event_index>=len(self.spec.events):
            # An exhausted explicit plan means known NONE. A task without
            # planned annotations cannot make that assertion.
            return PlannedEvent(None,0.,bool(self.spec.events))
        planned=self.spec.events[self.event_index];kind=planned.kind
        if self.phase==Phase.ACQUIRE and kind==EventType.CONTACT:
            elapsed=now-self.phase_since
            delta=max(0.,(planned.source_frame-self.acquire_source_origin)/self.spec.source_fps-elapsed)
        elif self.phase==Phase.RELEASE and kind==EventType.RELEASE:delta=0.
        else:
            delta=max(0.,(planned.source_frame-self.plan_cursor_frame)/self.spec.source_fps)
        return PlannedEvent(kind,float(delta),True)

    def _release_expected(self,actual,planned):
        if self.phase==Phase.RELEASE:return True
        if self.phase!=Phase.TRACK or self.recovering or self.plan_cursor_frame<planned.source_frame:
            return False
        # Intermediate planned releases also need confirmed path context and
        # observed arrival at the corresponding source pose, not elapsed time.
        indices=np.flatnonzero(self.spec.source_frames==planned.source_frame)
        return bool(len(indices) and self.path._reached(actual,self.spec.object_path[indices[0]]))

    def update(self,pose,now,feedback:InteractionFeedback,*,hand_points=None):
        if self.complete or self.failure_reason is not None:raise ValueError('terminal task requires reset before another control decision')
        actual=_pose(pose);now=float(now)
        if not np.isfinite(now) or (self.last_time is not None and now<=self.last_time):raise ValueError('strictly increasing task time required')
        if not isinstance(feedback,InteractionFeedback):raise ValueError('measured InteractionFeedback required')
        if feedback.grasp_established and feedback.released:raise ValueError('contradictory measured grasp/release feedback')
        if self.start_time is None:self.start_time=self.phase_since=now
        dt=0. if self.last_time is None else now-self.last_time;self.last_time=now;events=[]
        paused_path_budget=self.phase==Phase.ACQUIRE
        if self.event_index<len(self.spec.events):
            planned=self.spec.events[self.event_index];kind=planned.kind
            observed=((kind==EventType.CONTACT and feedback.grasp_established and not self.last_grasp)
                or (kind==EventType.RELEASE and feedback.released and not self.last_release
                    and self._release_expected(actual,planned)))
            if observed:self.event_index+=1
        self.last_grasp=feedback.grasp_established;self.last_release=feedback.released
        if not self.complete:
            if not feedback.workspace_ok:self.failure_reason='workspace'
            elif now-self.start_time>=self.config.episode_timeout_s:self.failure_reason='episode_timeout'
        goal=actual.copy();probe=False
        if not self.complete and self.failure_reason is None:
            if self.phase==Phase.TRACK and feedback.grasp_lost:
                self.phase=Phase.ACQUIRE;self.phase_since=now;self.recovering=True
                self.acquire_source_origin=self.plan_cursor_frame
                self.settle_since=None;self.probe_origin=None
                # A recovery gap supplies no interpolation witness.
                self.path.last_pose=None;self.path.interaction.reset()
            if self.phase==Phase.ACQUIRE:
                if feedback.grasp_established:
                    self.phase=Phase.TRACK;self.phase_since=now;self.probe_origin=None;self.recovering=False
                    self._transition('grasp_acquired',events)
                elif now-self.phase_since>=self.config.acquire_timeout_s:self.failure_reason='acquire_timeout'
                elif feedback.grasp_candidate:
                    if self.probe_origin is None:self.probe_origin=actual.copy()
                    target=self.spec.object_path[min(1,len(self.spec.object_path)-1)]
                    direction=target[:3,3]-self.probe_origin[:3,3]
                    if np.linalg.norm(direction)<1e-8:direction=np.array([0.,0.,1.])
                    goal=self.probe_origin.copy();goal[:3,3]+=direction/np.linalg.norm(direction)*self.config.probe_distance_m
                    goal=self.path._bounded(actual,goal);probe=True
                else:self.probe_origin=None
            if self.phase in (Phase.ACQUIRE,Phase.TRACK) and self.failure_reason is None:
                # Observe geometry from the first real state, before the
                # motion-coupling dwell confirms grasp. Path milestones are
                # geometric evidence, not grasp or task-stage certificates.
                # Preserve real observation dt for crossing bounds, while
                # excluding acquisition waiting from path timeout budgets.
                if paused_path_budget and self.path.start_time is not None:
                    self.path.start_time+=dt;self.path.last_progress_time+=dt
                candidate=feedback.contact_observed or feedback.grasp_candidate or feedback.grasp_established
                if self.phase==Phase.ACQUIRE and not candidate:
                    # Candidate motion is provisional until grasp confirmation.
                    # Dropping that evidence discards the unconfirmed suffix.
                    self.path.pointer=self.confirmed_pointer;self.path.complete=self.confirmed_path_complete
                    self.path.settle_since=None;self.path.last_pose=None
                    self.path.progress_anchor_arc=float(self.path.arc[self.path.milestones[self.confirmed_pointer]])
                self.path_decision=self.path.update(actual,now,hand_points=hand_points,
                    execution_ready=bool(self.phase==Phase.TRACK or (candidate and not self.recovering)))
                if self.phase==Phase.TRACK:
                    self.confirmed_progress=self.path_decision['confirmed_progress']
                    self.confirmed_pointer=self.path.pointer;self.confirmed_path_complete=self.path.complete
                    self.plan_cursor_frame=max(self.plan_cursor_frame,float(self.spec.source_frames[self.path_decision['projected_index']]))
            if self.phase==Phase.TRACK:
                goal=self.path_decision['goal_pose'].copy()
                if self.path_decision['timed_out']:self.failure_reason='path_'+self.path_decision['reason']
                at_end=self.path._reached(actual,self.path.poses[-1])
                if self.path_decision['complete'] and at_end:
                    self._transition('path_arrived',events)
                    if self.spec.terminal_mode==TerminalMode.RELEASE:
                        self.phase=Phase.RELEASE;self.phase_since=now;self.settle_since=None
                    elif self.spec.terminal_mode==TerminalMode.HOLD:
                        valid=feedback.grasp_established and not feedback.grasp_lost and feedback.stable
                        self._settle(valid,now,events)
                else:self.settle_since=None
            if self.phase==Phase.RELEASE:
                goal=self.path.poses[-1].copy()
                if self.spec.goal_type=='position_only':goal[:3,:3]=actual[:3,:3]
                at_end=self.path._reached(actual,goal)
                valid=at_end and feedback.released and feedback.stable and (feedback.supported is True or not self.spec.require_support)
                self._settle(valid,now,events)
                if not self.complete and now-self.phase_since>=self.config.release_timeout_s:self.failure_reason='release_timeout'
        if self.failure_reason is not None:goal=actual.copy();self.settle_since=None;probe=False
        if self.complete:
            goal=self.path.poses[-1].copy()
            if self.spec.goal_type=='position_only':goal[:3,:3]=actual[:3,:3]
        progress=self.confirmed_progress
        terminal=self.path.poses[-1].copy()
        if self.spec.goal_type=='position_only':terminal[:3,:3]=actual[:3,:3]
        return SchedulerDecision(self.phase,goal,terminal,progress,self._event(now),
            bool(self.confirmed_path_complete),self.complete,bool(self.complete or self.failure_reason),self.failure_reason,
            probe,self.recovering,tuple(events),self.plan_cursor_frame,self.spec.goal_type)

    def _settle(self,valid,now,events):
        if not valid:self.settle_since=None;return
        if self.settle_since is None:self.settle_since=now
        if now-self.settle_since>=self.config.terminal_hold_s:
            self.complete=True;self._transition('task_succeeded',events)

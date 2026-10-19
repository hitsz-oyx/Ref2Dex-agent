"""The same causal task boundary is used by archive replay and native control."""
from .task_feedback import FeedbackObserver
from .task_scheduler import TaskScheduler
from .executor_contract import task_condition
from .goal_scheduler import rigid_goal_delta

EVENT_SCHEMA='ref2dex.wam-planned-contact-release.v1'


class TaskController:
    def __init__(self,spec,object_points,*,scheduler_config=None,path_config=None,interaction_points=None):
        self.scheduler=TaskScheduler(spec,object_points,config=scheduler_config,path_config=path_config,interaction_points=interaction_points)
        self.observer=FeedbackObserver(object_points if interaction_points is None else interaction_points)

    def reset(self):self.scheduler.reset();self.observer.reset()

    def observe(self,pose,canonical_hand,velocity,now,*,contact_forces=None,force_valid=False,supported=None,workspace_ok=True):
        feedback=self.observer.update(pose,canonical_hand,velocity,now,contact_forces=contact_forces,
            force_valid=force_valid,supported=supported,workspace_ok=workspace_ok)
        decision=self.scheduler.update(pose,now,feedback,hand_points=canonical_hand)
        return dict(decision=decision,feedback=feedback,task=task_condition(pose,decision))


def wam_goal(runtime,pose,decision,query_points,*,require_events=False):
    """Never silently pretend a v4/v5 checkpoint has learned an event token."""
    capable=getattr(runtime,'event_conditioning_schema',None)==EVENT_SCHEMA
    if require_events and not capable:raise ValueError('checkpoint has no trained planned-event input; event-conditioned WAM is required')
    inputs=dict(goal_delta=rigid_goal_delta(pose,decision.local_goal_pose,query_points))
    planned=decision.next_event.wam_condition()
    if capable:inputs.update(planned)
    return inputs,dict(event_consumed=capable,planned_event=planned,goal_semantics='spatial_waypoint',horizon_s=.8)

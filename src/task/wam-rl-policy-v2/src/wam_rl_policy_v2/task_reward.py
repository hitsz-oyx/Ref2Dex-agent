"""Potential shaping over (measured state, task phase), with terminal potential zero."""
from dataclasses import dataclass
import numpy as np
from .task_spec import Phase


@dataclass(frozen=True)
class RewardFeatures:
    palm_gap_m:float=0.
    mean_tip_gap_m:float=0.
    tracking_error_m:float=0.
    terminal_error_m:float=0.
    object_speed_mps:float=0.
    contact_force_n:float=0.
    hand_clearance_m:float=0.


def potential(decision,features):
    values=np.array(list(features.__dict__.values()),float)
    if not np.isfinite(values).all() or (values<0).any():raise ValueError('finite nonnegative measured reward features required')
    if decision.phase==Phase.ACQUIRE:return -min(features.palm_gap_m,.5)-min(features.mean_tip_gap_m,.5)
    if decision.phase==Phase.TRACK:return decision.path_progress-min(features.tracking_error_m,.2)
    return (-min(features.terminal_error_m,.2)-min(features.object_speed_mps,1.)
        -.01*min(features.contact_force_n,20.)+min(features.hand_clearance_m,.05))


class TaskReward:
    def __init__(self,*,event_bonus=.2,success_bonus=1.,failure_cost=1.,time_cost_per_s=.01):
        if min(event_bonus,success_bonus,failure_cost,time_cost_per_s)<0:raise ValueError('nonnegative task reward parameters required')
        self.event_bonus=event_bonus;self.success_bonus=success_bonus;self.failure_cost=failure_cost
        self.time_cost_per_s=time_cost_per_s;self.reset()

    def reset(self):self.paid=set();self.terminal=False

    def step(self,previous,current,before,after,*,gamma,dt_s,effort_cost=0.):
        if self.terminal:raise ValueError('reset reward after task termination')
        if not 0<gamma<=1 or not np.isfinite([dt_s,effort_cost]).all() or min(dt_s,effort_cost)<0:
            raise ValueError('valid discount, control duration and normalized effort cost required')
        # Phase on each side belongs to that state; never use the old phase on both sides.
        shaping=gamma*(0. if current.terminated else potential(current,after))-potential(previous,before)
        events=0.
        for event in current.transitions:
            if event in self.paid:continue
            self.paid.add(event)
            if event in ('grasp_acquired','path_arrived'):events+=self.event_bonus
        sparse=self.success_bonus if current.task_complete else -self.failure_cost if current.failure_reason else 0.
        self.terminal=current.terminated
        parts=dict(shaping=shaping,event=events,sparse=sparse,time=-self.time_cost_per_s*dt_s,effort=-effort_cost)
        return float(sum(parts.values())),parts

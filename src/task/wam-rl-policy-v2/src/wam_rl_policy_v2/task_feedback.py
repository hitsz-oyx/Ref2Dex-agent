"""Measured interaction evidence; planned events never enter this observer."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .interaction_state import InteractionObserver


@dataclass(frozen=True)
class InteractionFeedback:
    grasp_candidate:bool=False
    grasp_established:bool=False
    grasp_lost:bool=False
    released:bool=False
    stable:bool=False
    supported:bool|None=None
    workspace_ok:bool=True
    force_valid:bool=False
    source:str='measured_motion_heuristic'


class FeedbackObserver:
    """Use force persistence for candidates, observed coupling for acquisition.

    A static force-bearing candidate requests a bounded acquisition probe from
    TaskScheduler. Proximity/force alone is not a force-closure certificate.
    """
    def __init__(self,object_points,*,contact_force_n=.1,dwell_s=.12,release_gap_m=.025):
        if min(contact_force_n,dwell_s,release_gap_m)<=0:raise ValueError('positive interaction thresholds required')
        self.motion=InteractionObserver(object_points);self.contact_force_n=contact_force_n
        self.dwell_s=dwell_s;self.release_gap_m=release_gap_m;self.reset()

    def reset(self):
        self.motion.reset();self.candidate_since=self.release_since=self.no_force_since=None
        self.last_time=None;self.had_grasp=False

    def update(self,pose,hand,velocity,now,*,contact_forces=None,force_valid=False,supported=None,workspace_ok=True):
        if self.last_time is not None and now<=self.last_time:raise ValueError('strictly increasing feedback time required')
        velocity=np.asarray(velocity)
        if velocity.shape!=(6,) or not np.isfinite(velocity).all():raise ValueError('finite object linear/angular velocity required')
        observed=self.motion.update(pose,hand,now);self.last_time=now
        count=None
        if force_valid:
            forces=np.asarray(contact_forces)
            if forces.shape!=(5,3) or not np.isfinite(forces).all():raise ValueError('five measured contact forces in newtons required')
            count=int((np.linalg.norm(forces,axis=-1)>=self.contact_force_n).sum())
        if count is not None and count>=2:
            if self.candidate_since is None:self.candidate_since=now
        else:self.candidate_since=None
        candidate=self.candidate_since is not None and now-self.candidate_since>=self.dwell_s
        established=bool(observed['coupled'] and (count is None or count>=2))
        self.had_grasp|=established
        if count==0:
            if self.no_force_since is None:self.no_force_since=now
        else:self.no_force_since=None
        lost=bool(observed['lost_control'] or (self.had_grasp and self.no_force_since is not None and now-self.no_force_since>=self.dwell_s))
        far=observed['fingertip_gap_m'] is not None and observed['fingertip_gap_m']>=self.release_gap_m
        if count==0 and far:
            if self.release_since is None:self.release_since=now
        else:self.release_since=None
        released=self.release_since is not None and now-self.release_since>=self.dwell_s
        stable=np.linalg.norm(velocity[:3])<=.02 and np.linalg.norm(velocity[3:])<=.2
        return InteractionFeedback(bool(candidate),established,lost,bool(released),bool(stable),supported,
            bool(workspace_ok),bool(force_valid),'measured_force_and_motion_heuristic' if force_valid else 'measured_motion_heuristic')

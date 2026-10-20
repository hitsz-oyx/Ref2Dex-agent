"""Measured-state -> scheduler -> frozen WAM -> BC/PPO actor -> isolated native action.

This is a Shadow-specific synchronous control boundary, not a robot-agnostic
IK solver. The caller owns simulation stepping and task-success measurement.
"""
from dataclasses import asdict
import numpy as np
import torch
from .action_contract import step_with_isolated_action
from .control_contracts import (ACTION_SPACE,ExecutionFeedback,MeasuredShadowState,
    TaskIntent,goal_envelope,trajectory_contract)
from .scheduled_executor import ScheduledCurrentExecutorInputs


class ScheduledShadowControl:
    def __init__(self,runtime,scheduler,actor,intent):
        if not isinstance(intent,TaskIntent):raise TypeError('TaskIntent required')
        if not getattr(actor,'current_model',False):raise ValueError('scheduled565 BC/PPO actor required')
        self.bridge=ScheduledCurrentExecutorInputs(runtime,scheduler)
        self.actor,self.intent=actor,intent
        self.episode_id=None;self.last_tick=None;self.pending=None

    def reset(self,episode_id):
        if not isinstance(episode_id,str) or not episode_id:raise ValueError('episode identity required')
        self.bridge.reset();self.episode_id=episode_id;self.last_tick=None;self.pending=None

    @torch.no_grad()
    def compute(self,state,*,feedback=None,phase=None,sample=False):
        if not isinstance(state,MeasuredShadowState):raise TypeError('MeasuredShadowState required')
        if state.episode_id!=self.episode_id:raise ValueError('explicit reset required for episode change')
        if self.last_tick is not None and state.tick!=self.last_tick+1:raise ValueError('consecutive native ticks required')
        feedback=feedback if feedback is not None else ExecutionFeedback()
        if not isinstance(feedback,ExecutionFeedback):raise TypeError('ExecutionFeedback required')
        intent=self.intent if phase is None else TaskIntent(self.intent.task_id,phase)
        decision=self.bridge.observation(state.current,state.object_pose,state.hand,state.time_s,
            execution_ready=feedback.execution_ready,lost_control=feedback.lost_control)
        self.last_tick=state.tick;self.pending=None
        if decision['scheduler']['timed_out']:raise RuntimeError('scheduler timeout: caller must reset or recover explicitly')
        if np.asarray(decision['tau']).shape!=(8,11,3) or not np.isfinite(decision['tau']).all():raise ValueError('finite world tau8x11x3 required')
        goal=goal_envelope(decision['scheduler'],state,self.bridge.scheduler.points)
        tau_meta=trajectory_contract(state,decision['history_valid'])
        device=next(self.actor.parameters()).device
        obs=torch.as_tensor(decision['observation'],device=device).unsqueeze(0)
        if sample:
            distribution=self.actor.distribution(obs);raw=distribution.sample()
            log_prob=distribution.log_prob(raw).sum(-1)
        else:raw=self.actor(obs);log_prob=None
        if raw.shape!=(1,24) or not torch.isfinite(raw).all():raise ValueError('finite native24 actor output required')
        if log_prob is not None and not torch.isfinite(log_prob).all():raise ValueError('finite PPO log probability required')
        canonical=raw.clamp(-1,1).detach().clone()
        key=(state.episode_id,state.tick,float(state.time_s))
        self.pending=(key,canonical.clone())
        return dict(schema='ref2dex.shadow-control-step.v1',episode_id=key[0],tick=key[1],time_s=key[2],
            task_intent=asdict(intent),phase_consumed_by_wam=False,phase_consumed_by_actor=False,
            feedback=asdict(feedback),path_complete=bool(decision['scheduler']['complete']),
            task_success=feedback.task_success,scheduler=decision['scheduler'],goal=goal,
            tau=decision['tau'].copy(),tau_contract=tau_meta,
            interaction=decision['interaction'],interaction_valid=decision['interaction_valid'],
            executor_observation=decision['observation'].copy(),action_space=ACTION_SPACE,
            raw_action=raw[0].cpu().numpy().copy(),canonical_action=canonical[0].cpu().numpy().copy(),
            action_log_prob=float(log_prob[0]) if log_prob is not None else None,
            action_transform='clamp[-1,1]; simulator owns subsequent wrist frame transform')

    def execute(self,command,sim_step):
        """Execute at most once; simulator cannot mutate the recorded command.

        For PPO, retain raw_action and action_log_prob from compute(). Never
        recompute probability using canonical_action after clipping.
        """
        key=(command['episode_id'],command['tick'],float(command['time_s']))
        if self.pending is None or key!=self.pending[0]:raise ValueError('stale or already executed command')
        canonical=self.pending[1];self.pending=None
        step_with_isolated_action(sim_step,canonical)

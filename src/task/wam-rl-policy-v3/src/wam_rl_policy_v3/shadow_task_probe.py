"""Bounded UniDex evaluation of task10/WAM goals with the trained Shadow actor.

This is a model-only, zero-column migration for evaluation. It does not resume
an optimizer or claim that the new goal columns have been learned.
"""
from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np
import torch

from .bc_mlp import CurrentMLP
from .executor_contract import ExecutorLayout,expand_named_model,task_condition
from .official_bridge import ScheduledOfficialEnv
from .task_runtime import TaskController
from .task_spec import TaskSpec
from .measured_execution import task_wam_goal

LAYOUT=ExecutorLayout('unidex.shadow.legacy300.v1',300,24,60)


def expand_official_state(weights):
    parent=CurrentMLP(565)
    parent.load_state_dict({k[6:]:v for k,v in weights.items() if k.startswith('actor.')},strict=True)
    actor=expand_named_model(parent,dict(robot=300,tau=264,tau_valid=1),LAYOUT.blocks,clip_dim=300)
    result={k:v.clone() for k,v in weights.items() if not k.startswith('actor.')}
    result.update({'actor.'+k:v for k,v in actor.state_dict().items()})
    source=weights['critic.0.weight']
    if source.shape[1]!=565:raise ValueError('legacy Shadow critic565 required')
    target=source.new_zeros((len(source),575));target[:,:300]=source[:,:300];target[:,310:]=source[:,300:]
    result['critic.0.weight']=target
    return result


class TaskGoalScheduler:
    def __init__(self,state,index,geometry,*,force_feedback=True):
        self.force_feedback=force_feedback
        self.points=geometry[2]
        self.controller=TaskController(TaskSpec.shadow_goal(state['pose'][index],state['goal'][index]),
            self.points,interaction_points=geometry[0])
        self.latest=None;self.max_goal_error=0.
        self.feedback(state,index,False)

    def reset(self):
        self.controller.reset();self.latest=None

    def feedback(self,state,index,valid):
        self.velocity=state['velocity'][index]
        self.forces=state['contact_forces'][index] if self.force_feedback else None
        self.force_valid=bool(valid and self.force_feedback)

    def update(self,pose,now,*,hand_points):
        if self.latest is not None and self.latest['decision'].terminated:
            old=self.latest['decision'];goal=pose.copy()
            if old.task_complete:goal[:3,3]=old.terminal_goal_pose[:3,3]
            decision=replace(old,local_goal_pose=goal)
            task=task_condition(pose,decision)
        else:
            result=self.controller.observe(pose,hand_points,self.velocity,now,
                contact_forces=self.forces,force_valid=self.force_valid)
            decision=result['decision'];task=result['task']
        delta,error=task_wam_goal(pose,decision.local_goal_pose,task,self.points)
        if not self.force_feedback:
            from .shadow_task_contract import task_point_delta
            delta=task_point_delta(task,self.points)
        self.max_goal_error=max(self.max_goal_error,error)
        self.latest=dict(decision=decision,task=task)
        return dict(goal_pose=decision.local_goal_pose,goal_delta=delta,task=task,
            state=decision.phase.name,confirmed_progress=decision.path_progress)


class TaskScheduledOfficialEnv(ScheduledOfficialEnv):
    def __init__(self,*args,force_feedback=True,**kwargs):
        self.force_feedback=force_feedback
        super().__init__(*args,**kwargs)
        from gym import spaces
        self.num_observations=575
        self.observation_space=spaces.Box(low=-np.inf,high=np.inf,shape=(575,),dtype=np.float32)
        self.phase_counts=np.zeros((self.num_envs,3),np.int64)
        self.max_progress=np.zeros(self.num_envs);self.first_done=np.zeros(self.num_envs,bool)
        self.evidence={name:np.zeros(self.num_envs,np.int64) for name in (
            'observed','near2','force2','contact_candidate','moving','near_force_moving',
            'speed_limit_reject','relative_ratio_reject','motion_coupled')}

    def scheduler(self,state,index):return TaskGoalScheduler(state,index,self.runtime.geometry[index],force_feedback=self.force_feedback)

    def augment(self,native,reset_ids):
        state=self.snapshot()
        if self.inputs is not None:
            for i,scheduler in enumerate(self.inputs.schedulers):scheduler.feedback(state,i,self.ticks[i]>0)
        legacy=super().augment(native,reset_ids,state=state)
        tasks=[]
        for i,scheduler in enumerate(self.inputs.schedulers):
            tasks.append(scheduler.latest['task']);decision=scheduler.latest['decision']
            if not self.first_done[i]:
                self.phase_counts[i,int(decision.phase)]+=1
                self.max_progress[i]=max(self.max_progress[i],decision.path_progress)
                evidence=scheduler.controller.observer.last_evidence
                if evidence is not None and not decision.terminated:
                    self.evidence['observed'][i]+=1
                    near=evidence['near_fingers']>=2
                    force=evidence['force_count'] is not None and evidence['force_count']>=2
                    speed=evidence['object_speed_mps'];relative=evidence['relative_hand_speed_mps']
                    config=scheduler.controller.observer.motion.config
                    moving=speed is not None and speed>config.minimum_object_speed_mps
                    for key,value in [('near2',near),('force2',force),('contact_candidate',near and force),
                            ('moving',moving),('near_force_moving',near and force and moving),('motion_coupled',evidence['coupled'])]:
                        self.evidence[key][i]+=int(value)
                    if near and force and moving and relative is not None:
                        self.evidence['speed_limit_reject'][i]+=int(relative>=config.relative_speed_limit_mps)
                        self.evidence['relative_ratio_reject'][i]+=int(relative>config.coupling_noise_mps+config.coupling_relative_ratio*speed)
        return torch.cat((legacy[:,:300],legacy.new_tensor(np.stack(tasks)),legacy[:,300:]),1)

    def step(self,*args,**kwargs):
        result=super().step(*args,**kwargs)
        self.first_done|=result[2].detach().cpu().numpy().astype(bool)
        return result


class TaskCurrentOfficialEnv(ScheduledOfficialEnv):
    """Actual Current310 goal BC: causal task10, with no neural tau input."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,native_only=True,**kwargs)
        from gym import spaces
        self.num_observations=310
        self.observation_space=spaces.Box(low=-np.inf,high=np.inf,shape=(310,),dtype=np.float32)
        self.schedulers=None
        self.phase_counts=np.zeros((self.num_envs,3),np.int64)
        self.max_progress=np.zeros(self.num_envs)
        self.first_done=np.zeros(self.num_envs,bool)

    def augment(self,native,reset_ids):
        state=self.snapshot();times=self.ticks.copy()/60
        if self.schedulers is None:
            self.schedulers=[TaskGoalScheduler(state,i,self.runtime.geometry[i],force_feedback=False)
                             for i in range(self.num_envs)]
        for i in reset_ids:
            self.schedulers[i]=TaskGoalScheduler(state,i,self.runtime.geometry[i],force_feedback=False)
            times[i]=0.
        current=super().augment(native,reset_ids,state=state)
        tasks=[]
        for i,scheduler in enumerate(self.schedulers):
            scheduler.feedback(state,i,False)
            result=scheduler.update(state['pose'][i],float(times[i]),hand_points=state['hand'][i])
            tasks.append(result['task']);decision=scheduler.latest['decision']
            if not self.first_done[i]:
                self.phase_counts[i,int(decision.phase)]+=1
                self.max_progress[i]=max(self.max_progress[i],decision.path_progress)
        return torch.cat((current,current.new_tensor(np.stack(tasks))),1)

    def step(self,*args,**kwargs):
        result=super().step(*args,**kwargs)
        self.first_done|=result[2].detach().cpu().numpy().astype(bool)
        return result


def run(task,snapshot,sim_step,actor,runtime,args):
    if not args.evaluate or not args.official_ppo or not args.rl_checkpoint or args.ppo_updates:
        raise ValueError('task10 Shadow probe is model-only evaluation of an existing PPO checkpoint')
    if args.goal_type!='position_only':raise ValueError('UniDex task probe requires its native position-only goal')
    import importlib.util
    import yaml
    from tasks.hand_base.vec_task import VecTaskPython
    from .official_bridge import prepare_official
    from .official_evaluation import evaluate
    root=Path(os.environ.get('REF2DEX_RUN_ROOT',Path(__file__).resolve().parents[5]))
    source=prepare_official(root,args.output,masked_resets=True,correct_distribution_kl=True)
    spec=importlib.util.spec_from_file_location('shadow_task_official_actor',source/'algorithms/rl/ppo/module.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cfg=yaml.safe_load((source/'cfg/ppo/config.yaml').read_text())
    cfg['policy'].update(backbone_type='pn',freeze_backbone=False,obs_dim=575)
    weights=torch.load(args.rl_checkpoint,map_location='cpu',weights_only=False)
    from .control_contracts import scheduled_conditioning_contract
    if (weights.get('schema')!='ref2dex.scheduled-official-ppo.v1'
            or weights.get('conditioning')!=scheduled_conditioning_contract()):raise ValueError('trained Shadow565 PPO with its declared conditioning required')
    env=TaskScheduledOfficialEnv(VecTaskPython(task,'cuda:0',cfg['clip_observations'],cfg['clip_actions']),
        snapshot,runtime,protocol=args.execution_protocol or 'corrected',goal_type='position_only',termination_policy=args.termination_policy)
    policy=module.ActorCritic((575,),(0,),(24,),cfg['learn']['init_noise_std'],cfg['policy']).to(runtime.device)
    policy.actor=expand_named_model(actor.actor,dict(robot=300,tau=264,tau_valid=1),LAYOUT.blocks,clip_dim=300).to(runtime.device)
    policy.load_state_dict(expand_official_state(weights['model']),strict=True)
    (args.output/'task-migration.json').write_text(json.dumps(dict(
        source_checkpoint=str(args.rl_checkpoint),iteration=weights['iteration'],layout=LAYOUT.packet(),
        mode='model-only evaluation; new task columns zero; optimizer not loaded',
        task_goal='same TaskController decision feeds task10 and WAM',
        contact_source='Isaac Gym net_contact_force_tensor at five distal bodies; world force N',
        terminal_policy='keep final task goal until native episode ends',
        geometry=runtime.geometry_identity),indent=2)+'\n')
    reports=evaluate(env,policy,args,weights['iteration'])
    (args.output/'task-diagnostics.json').write_text(json.dumps(dict(phase_counts=env.phase_counts.tolist(),
        max_progress=env.max_progress.tolist(),goal_task_error_m=max(s.max_goal_error for s in env.inputs.schedulers),
        evidence_frames={k:v.tolist() for k,v in env.evidence.items()},
        task_columns_learned=False),indent=2)+'\n')
    return reports


def run_bc(task,snapshot,sim_step,actor,runtime,args):
    """Evaluate the exact BC mean without creating PPO or loading its weights."""
    if not args.bc_evaluation or not args.evaluate or args.rl_checkpoint or args.ppo_updates:
        raise ValueError('BC-only evaluation forbids an RL checkpoint or updates')
    import yaml
    from tasks.hand_base.vec_task import VecTaskPython
    from .official_evaluation import evaluate
    from .shadow_task_contract import CURRENT_SCHEMA as TASK_CURRENT,SCHEDULED_SCHEMA as TASK_SCHEDULED
    packet=torch.load(args.policy,map_location='cpu',weights_only=False)
    trained_task=packet.get('schema') in (TASK_CURRENT,TASK_SCHEDULED)
    task_current=packet.get('schema')==TASK_CURRENT
    if trained_task and not args.task_controller:
        raise ValueError('trained task10 policy requires its actual TaskController environment')
    if trained_task and not task_current:
        if not args.wam_checkpoint or str(args.wam_checkpoint.resolve())!=packet['config']['wam']:
            raise ValueError('trained575 WAM checkpoint differs from its BC cache')
    if trained_task and (not args.wam_surface_manifest or str(args.wam_surface_manifest.resolve())!=packet['config']['dense']):
        raise ValueError('trained task BC geometry differs from its declared source')
    root=Path(os.environ.get('REF2DEX_RUN_ROOT',Path(__file__).resolve().parents[5]))
    cfg=yaml.safe_load((root/'third_party/UniDexGrasp2-official-ppo/dexgrasp/cfg/ppo/config.yaml').read_text())
    env_type=TaskCurrentOfficialEnv if task_current else TaskScheduledOfficialEnv if args.task_controller else ScheduledOfficialEnv
    extra=dict(force_feedback=False) if trained_task and not task_current else {}
    env=env_type(VecTaskPython(task,'cuda:0',cfg['clip_observations'],cfg['clip_actions']),snapshot,runtime,
        protocol=args.execution_protocol or 'corrected',goal_type=args.goal_type,termination_policy=args.termination_policy,**extra)
    if args.task_controller and not trained_task:
        actor.actor=expand_named_model(actor.actor,dict(robot=300,tau=264,tau_valid=1),LAYOUT.blocks,clip_dim=300).to(runtime.device)
    report=evaluate(env,actor,args,None,checkpoint=args.policy,policy_kind='bc')
    declaration=dict(policy_kind='bc',bc_checkpoint=str(args.policy),bc_step=packet.get('step'),
        input_dim=env.num_observations,ppo_checkpoint=None,optimizer_constructed=False,optimizer_steps=0,
        task_columns_learned=bool(trained_task) if args.task_controller else None,
        zero_expansion=bool(args.task_controller and not trained_task),
        task_learning=packet.get('task_learning'),
        feedback=packet.get('config',{}).get('conditioning',{}).get('feedback'))
    if args.task_controller:
        schedulers=env.schedulers if task_current else env.inputs.schedulers
        declaration.update(phase_counts=env.phase_counts.tolist(),max_progress=env.max_progress.tolist(),
            goal_task_error_m=max(s.max_goal_error for s in schedulers))
    (args.output/'bc-evaluation.json').write_text(json.dumps(declaration,indent=2)+'\n')
    return report

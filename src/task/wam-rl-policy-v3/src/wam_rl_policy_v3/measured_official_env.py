"""Measured686 boundary for the pinned Shadow PPO; same-episode step, eager indexed reset."""
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import json,time
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .task_runtime import TaskController
from .task_spec import TaskSpec
from .measured_reward_bridge import MeasuredRewardTracker,reward_features
from .measured_execution import online_support,task_wam_goal
from .robot_adapter import InspireAdapter
from .inspire_shadow import current_state,tau_world
from .inspire_contract import expand_action
from .inspire_state_codec import unpack_state
from .wam_v4_runtime import v4_online_samples,collate_v4
from .task_admission import load_executable_tasks

REWARD_COMPONENTS=('shaping','event','sparse','time','effort','external_terminal')


def check_indexed_reset(before,after,ids):
    ids=np.asarray(ids,int);untouched=np.setdiff1d(np.arange(len(before['episodes'])),ids)
    old=np.asarray(before['episodes']);new=np.asarray(after['episodes'])
    if not(np.all(new[ids]==old[ids]+1) and np.array_equal(new[untouched],old[untouched]) and np.all(np.asarray(after['tick'])[ids]==0) and np.array_equal(np.asarray(before['tick'])[untouched],np.asarray(after['tick'])[untouched])):
        raise ValueError('partial reset corrupted episode or tick ownership')


class MeasuredOfficialEnv:
    def __init__(self,connection,hello,manifest,runtime,model,*,reset_probe=False,batch=256,shared_tf32=False):
        from gym import spaces
        self.connection=connection;self.native=unpack_state(hello['state']);self.runtime=runtime;self.model=model
        self.device=runtime.device;self.num_envs=len(self.native['q']);self.num_environments=self.num_envs
        self.num_observations=686;self.num_actions=12;self.num_states=0;self.rl_device=str(self.device)
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(686,),dtype=np.float32)
        self.state_space=spaces.Box(-np.inf,np.inf,shape=(0,),dtype=np.float32);self.action_space=spaces.Box(-1,1,shape=(12,),dtype=np.float32)
        self.task=SimpleNamespace(cfg={'seed':246},max_episode_length=1800,random_time=False)
        self.batch=batch;self.shared_tf32=shared_tf32;self.inference_audit=None;self.reset_probe=reset_probe;self.steps=0;self.reset_count=0;self.completed_episodes=0;self.successes=0;self.workspace_failures=0
        self.reset_audits=[];self.source_mismatches=0;self.max_goal_error=0.;self.tau_valid_count=0;self.timings={}
        entries,self.task_admission=load_executable_tasks(manifest);self.records={};self.shape_features={}
        if not entries:raise ValueError('no executable TRAIN tasks: all eligible tasks UNKNOWN or empty')
        for row,spec,side in entries:
            with np.load(Path(row['source'])/'canonical.npz') as z:points=z['points'].astype(np.float32)
            with np.load(side/'admission.npz') as z:bounds=z['table_bounds'].copy()
            self.records[row['sequence']]=(spec,points,bounds,row['object'])
        with torch.inference_mode():
            keys=list(self.records)
            for lo in range(0,len(keys),64):
                part=keys[lo:lo+64];features=model.shape(torch.as_tensor(np.stack([self.records[k][1] for k in part]),device=self.device))
                self.shape_features.update(zip(part,features.unbind()))
        n=self.num_envs
        self.controllers=[None]*n;self.trackers=[None]*n;self.latest=[None]*n
        self.histories=[deque(maxlen=4) for _ in range(n)];self.hands=[deque(maxlen=4) for _ in range(n)]
        self.previous=np.zeros((n,12),np.float32);self.observations=torch.zeros((n,686),device=self.device)
        self.workspace=np.zeros(n,bool);self.initial_z=np.zeros(n);self.max_lift=np.zeros(n)
        self.streak=np.zeros(n,int);self.longest=np.zeros(n,int);self.returns=np.zeros(n)
        self.reward_components=np.zeros((n,len(REWARD_COMPONENTS)),np.float64);self.phase_controls=np.zeros((n,3),np.int64)
        self.reward_totals=np.zeros(len(REWARD_COMPONENTS),np.float64);self.phase_totals=np.zeros(3,np.int64)
        self.transition_count=0;self.reward_total=0.;self.completed_controls=0;self.completed_returns=0.
        self.episode_sink=None
        self._refresh(range(n),reset=True)

    def rpc(self,op,**kwargs):
        from trajectory_policy.wam_ipc import send,receive
        send(self.connection,dict(op=op,**kwargs));packet=receive(self.connection)
        if 'state' in packet:packet['state']=unpack_state(packet['state'])
        return packet

    def _refresh(self,indices,*,reset=False):
        started=time.perf_counter();ids=list(indices);native=self.native
        if not ids:return
        # Convert the extended JSON fields once per snapshot, rather than
        # converting the full environment array inside every environment loop.
        for key,dtype in [('dof_force',np.float32),('contact_forces',np.float32),
                          ('dof_force_valid',np.bool_),('contact_force_valid',np.bool_),
                          ('object_contact_force',np.float64),('table_pose',np.float64)]:
            native[key]=np.asarray(native[key],dtype=dtype)
        objects=np.asarray(native['objects'],np.float32);hand=InspireAdapter.canonical_hand(np.asarray(native['hand'],np.float64))
        poses=np.repeat(np.eye(4)[None],self.num_envs,0);poses[:,:3,:3]=Rotation.from_quat(objects[:,3:7]).as_matrix();poses[:,:3,3]=objects[:,:3]
        measured=[];tasks=[];shape=[];samples=[];sample_ids=[];rotations=[]
        tau=np.zeros((len(ids),8,11,3),np.float32);valid=np.zeros(len(ids),bool)
        for local,i in enumerate(ids):
            name=native['sequences'][i]
            if name not in self.records:raise ValueError('native sampled motion outside admitted TRAIN')
            spec,points,bounds,obj=self.records[name]
            if self.runtime.env_objects[i]['object_code']!='dexplore/'+obj:raise ValueError('reset changed bound object geometry')
            if reset:
                self.controllers[i]=TaskController(spec,points);self.trackers[i]=MeasuredRewardTracker()
                self.histories[i].clear();self.hands[i].clear();self.previous[i]=0
                self.initial_z[i]=objects[i,2];self.max_lift[i]=0;self.streak[i]=0;self.longest[i]=0;self.returns[i]=0
                self.reward_components[i]=0;self.phase_controls[i]=0
            self.workspace[i]=False
            dof_valid=bool(np.asarray(native['dof_force_valid'])[i]);contact_valid=bool(np.asarray(native['contact_force_valid'])[i])
            force=np.asarray(native['dof_force'],np.float32)[i];contact=np.asarray(native['contact_forces'],np.float32)[i]
            supported=online_support(native,i,poses[i],points,bounds)
            decision=self.controllers[i].observe(poses[i],hand[i],objects[i,7:13],float(native['tick'][i])/30,
                contact_forces=contact,force_valid=contact_valid,supported=supported)['decision']
            # The same independently packed task10 generates the WAM local goal.
            from .executor_contract import task_condition
            task=task_condition(poses[i],decision)
            self.histories[i].append(np.r_[poses[i,:3,3],objects[i,3:7]]);self.hands[i].append(hand[i].copy())
            if len(self.histories[i])==4 and not decision.terminated:
                delta,error=task_wam_goal(poses[i],decision.local_goal_pose,task,self.runtime.geometry[i][2]);self.max_goal_error=max(self.max_goal_error,error)
                sample=v4_online_samples(np.asarray(self.histories[i])[None],np.asarray(self.hands[i])[None],[self.runtime.geometry[i]],goal_delta=delta[None])[0]
                if self.runtime.workspace_valid([sample])[0]:samples.append(sample);sample_ids.append(local);rotations.append(poses[i,:3,:3].copy())
                else:self.workspace[i]=True
            features=reward_features(poses[i],hand[i],points,decision,objects[i,7:13],contact)
            if reset:self.trackers[i].initialize(decision,features)
            self.latest[i]=(decision,features)
            base=current_state(np.asarray(native['q'])[i:i+1],np.asarray(native['dq'])[i:i+1],np.asarray(native['physical'])[i:i+1],objects[i:i+1],spec.object_path[-1,:3,3][None],self.previous[i:i+1])[0]
            base=np.clip(base,-5,5)
            measured.append(np.r_[base,np.where(dof_valid,force,0),np.where(contact_valid,contact.reshape(-1),0),float(dof_valid),float(contact_valid)])
            tasks.append(task);shape.append(self.shape_features[name])
        packed=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',enabled=False):
            for lo in range(0,len(samples),self.batch):
                part=samples[lo:lo+self.batch];batch={k:v.to(self.device) for k,v in collate_v4(part).items()}
                prediction=self.runtime.predict_hand(batch,shared_tf32=self.shared_tf32)['trajectory']
                if self.shared_tf32 and self.inference_audit is None:
                    reference=self.runtime.predict_hand(batch,shared_tf32=False)['trajectory']
                    difference=float((prediction-reference).abs().max())
                    self.inference_audit=dict(batch=len(part),full_trajectory_max_error_m=difference,limit_m=.001)
                    if difference>.001:raise ValueError('optimized WAM differs by more than1mm')
                trajectory=prediction.float().cpu().numpy()
                predicted=tau_world(trajectory,np.asarray(rotations[lo:lo+self.batch]))
                chosen=sample_ids[lo:lo+self.batch];tau[chosen]=predicted;valid[chosen]=True
            raw=torch.as_tensor(np.asarray(measured,np.float32),device=self.device)
            robot=torch.cat((raw[:,:345],torch.stack(shape),raw[:,345:]),-1)
            obs=torch.cat((robot,torch.as_tensor(np.asarray(tasks),device=self.device),torch.as_tensor(tau.reshape(len(ids),264),device=self.device),torch.as_tensor(valid[:,None],dtype=torch.float32,device=self.device)),-1)
            if obs.shape!=(len(ids),686) or not torch.isfinite(obs).all():raise ValueError('bad measured PPO observation')
            self.observations[ids]=obs
        self.tau_valid_count+=int(valid.sum());self.timings=dict(causal_pack_s=packed-started,wam_and_transfer_s=time.perf_counter()-packed,refreshed=len(ids),valid_tau=int(valid.sum()))

    def reset(self):
        before=self.native;self.native=self.rpc('reset',ids=list(range(self.num_envs)))['state'];check_indexed_reset(before,self.native,range(self.num_envs))
        self.reset_count+=self.num_envs;self._refresh(range(self.num_envs),reset=True);return self.observations.clone()

    def get_state(self):return torch.zeros((self.num_envs,0),device=self.device)

    def training_statistics(self):
        out=dict(transitions=self.transition_count,reward_sum=self.reward_total,
                 completed_episode_controls=self.completed_controls,completed_episode_return_sum=self.completed_returns)
        out.update({'phase_'+name+'_controls':int(value) for name,value in zip(('acquire','track','release'),self.phase_totals)})
        out.update({'reward_component_'+name:float(value) for name,value in zip(REWARD_COMPONENTS,self.reward_totals)})
        return out

    def step(self,actions,id=0):
        started=time.perf_counter();source=list(self.native['episodes']);tick=np.asarray(self.native['tick']).copy()
        canonical=actions.detach().clamp(-1,1).cpu().numpy().copy().astype(np.float32)
        self.native=self.rpc('step',action=expand_action(canonical).tolist(),episodes=source)['state']
        if self.native['episodes']!=source or not np.array_equal(np.asarray(self.native['tick']),tick+1):
            self.source_mismatches+=1;raise ValueError('action executed in wrong episode or time')
        self.previous[:]=canonical;self._refresh(range(self.num_envs));self.steps+=1
        reward=np.zeros(self.num_envs,np.float32);done=np.zeros(self.num_envs,bool);results={}
        for i,(decision,features) in enumerate(self.latest):
            r,parts=self.trackers[i].step(decision,features,canonical[i]);reward[i]=r
            done[i]=decision.terminated or self.workspace[i]
            forced=self.reset_probe and ((self.steps==2 and i==0) or (self.steps==4 and i==1))
            if forced:done[i]=True
            if (self.workspace[i] or forced) and not decision.terminated:
                # External terminal must also zero the next potential, not bootstrap through reset.
                from .task_reward import potential
                reward[i]-=self.trackers[i].gamma*potential(decision,features)+self.trackers[i].reward.failure_cost
            self.returns[i]+=reward[i]
            component=np.array([parts[k] for k in REWARD_COMPONENTS[:-1]]+[float(reward[i])-float(r)],np.float64)
            self.reward_components[i]+=component;self.reward_totals+=component
            self.phase_controls[i,int(decision.phase)]+=1;self.phase_totals[int(decision.phase)]+=1
            self.transition_count+=1;self.reward_total+=float(reward[i])
            lift=float(self.native['objects'][i][2]-self.initial_z[i]);self.max_lift[i]=max(self.max_lift[i],lift)
            self.streak[i]=self.streak[i]+1 if lift>=.05 else 0;self.longest[i]=max(self.longest[i],self.streak[i])
            if done[i]:
                result=dict(episode_id=source[i],sequence=self.native['sequences'][i],task_success=bool(decision.task_complete and not self.workspace[i] and not forced),
                    failure_reason='engineering_reset' if forced else 'wam_workspace' if self.workspace[i] else decision.failure_reason,
                    confirmed_progress=float(decision.path_progress),return_sum=float(self.returns[i]),controls=int(self.native['tick'][i]),max_lift_m=float(self.max_lift[i]),ever_lift5cm30=bool(self.longest[i]>=30),
                    terminal_mode=self.controllers[i].scheduler.spec.terminal_mode.value,
                    reward_components=dict(zip(REWARD_COMPONENTS,self.reward_components[i].tolist())),
                    phase_controls=dict(zip(('acquire','track','release'),self.phase_controls[i].tolist())))
                results[i]=result;self.completed_episodes+=1;self.successes+=int(result['task_success']);self.workspace_failures+=int(self.workspace[i])
                self.completed_controls+=result['controls'];self.completed_returns+=result['return_sum']
                if self.episode_sink is not None:self.episode_sink(dict(result,environment=i))
        terminal=self.observations.clone();terminal_episodes=list(self.native['episodes']);ids=np.flatnonzero(done)
        if len(ids):
            before=self.native;self.native=self.rpc('reset',ids=ids.tolist())['state'];check_indexed_reset(before,self.native,ids)
            self._refresh(ids,reset=True);self.reset_count+=len(ids)
            if len(self.reset_audits)<16:self.reset_audits.append(dict(step=self.steps,ids=ids.tolist(),execution_episode=terminal_episodes,next_observation_episode=list(self.native['episodes']),next_tick=np.asarray(self.native['tick']).tolist()))
        info=dict(episode_results=results,action_source_episode=source,execution_episode=terminal_episodes,next_observation_episode=list(self.native['episodes']),
            terminal_observation=terminal[ids],transition_valid=torch.ones(self.num_envs,dtype=torch.bool,device=self.device))
        self.timings['step_total_s']=time.perf_counter()-started
        return self.observations.clone(),torch.as_tensor(reward,device=self.device),torch.as_tensor(done.astype(np.int64),device=self.device),info

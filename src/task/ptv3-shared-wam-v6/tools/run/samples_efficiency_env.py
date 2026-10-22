"""Three-arm causal grasp/lift environment; shared reward, task and resets."""
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import json,time
import numpy as np
import torch
from dataclasses import replace
from scipy.spatial.transform import Rotation
from wam_rl_policy_v3.task_runtime import TaskController
from wam_rl_policy_v3.task_spec import TaskSpec
from wam_rl_policy_v3.measured_reward_bridge import MeasuredRewardTracker,reward_features
from wam_rl_policy_v3.measured_execution import online_support,task_wam_goal
from wam_rl_policy_v3.robot_adapter import InspireAdapter
from wam_rl_policy_v3.inspire_shadow import current_state,tau_world
from wam_rl_policy_v3.inspire_contract import expand_action
from wam_rl_policy_v3.inspire_state_codec import unpack_state
from wam_rl_policy_v3.wam_v4_runtime import v4_online_samples,collate_v4
from wam_rl_policy_v3.physical_safety import physical_safety,PHYSICAL_SAFETY_POLICY
from ptv3_shared_wam_v6.execution_contract import encoder_domain_available,grasp_lift_task,TASK_CONTRACT,FALLBACK_POLICY
from ptv3_shared_wam_v6.online_runtime import EncodingUnavailable


def check_indexed_reset(before,after,ids):
    ids=np.asarray(ids,int);untouched=np.setdiff1d(np.arange(len(before['episodes'])),ids)
    old=np.asarray(before['episodes']);new=np.asarray(after['episodes'])
    if not(np.all(new[ids]==old[ids]+1) and np.array_equal(new[untouched],old[untouched]) and np.all(np.asarray(after['tick'])[ids]==0) and np.array_equal(np.asarray(before['tick'])[untouched],np.asarray(after['tick'])[untouched])):
        raise ValueError('partial reset corrupted episode or tick ownership')


class SampleEfficiencyEnv:
    def __init__(self,connection,hello,manifest,runtime,model,*,reset_probe=False,batch=256,shared_tf32=False,arm="A",forecast=None,geometry_cache=None):
        from gym import spaces
        self.arm=arm;self.forecast=forecast
        self.connection=connection;self.native=unpack_state(hello['state']);self.runtime=runtime;self.model=model
        self.device=runtime.device;self.num_envs=len(self.native['q']);self.num_environments=self.num_envs
        self.num_observations=846;self.num_actions=12;self.num_states=0;self.rl_device=str(self.device)
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(846,),dtype=np.float32)
        self.state_space=spaces.Box(-np.inf,np.inf,shape=(0,),dtype=np.float32);self.action_space=spaces.Box(-1,1,shape=(12,),dtype=np.float32)
        self.task=SimpleNamespace(cfg={'seed':246},max_episode_length=600,random_time=False)
        self.batch=batch;self.shared_tf32=shared_tf32;self.inference_audit=None;self.reset_probe=reset_probe;self.steps=0;self.reset_count=0;self.completed_episodes=0;self.successes=0;self.workspace_failures=0
        self.reset_audits=[];self.source_mismatches=0;self.max_goal_error=0.;self.tau_valid_count=0;self.timings={}
        self.joint_limits=hello.get('audit',{}).get('physical_joint_limits')
        self.encoder_unavailable_steps=0;self.wam_covered_controls=0;self.decision_controls=0;self.physical_safety_aborts=0
        self.encoder_error_count=0;self.encoder_errors=[]
        audit_path=Path(manifest);audit=json.loads(audit_path.read_text());self.records={};self.shape_features={}
        if geometry_cache is None:raise ValueError('all arms require the same dense2048 CAD source')
        geometry_cache=Path(geometry_cache)
        geometry_manifest=json.loads((geometry_cache/'surface-manifest.json').read_text())
        surfaces={}
        for row in geometry_manifest['objects']:
            with np.load(geometry_cache/row['path']) as z:surfaces[row['key']]=z['surface'].astype(np.float32)
        if audit.get('schema')!='ref2dex.teacher-measured-rollout-admission.v2' or audit.get('status')!='COMPLETED':raise ValueError('corrected completed corpus required')
        for row in audit['records']:
            if row['split']!='train' or row['engineering']!='PASS' or row['training_prefix_commands']<=0:continue
            side=audit_path.parent/row['sidecar']
            points=surfaces['dexplore/'+row['object']+'@1']
            if points.shape!=(2048,3):raise ValueError('shared CAD must contain genuine 2048 points')
            with np.load(side/'admission.npz') as z:bounds=z['table_bounds'].copy()
            spec=TaskSpec.from_packet(json.loads((side/'task.json').read_text()))
            raw=side/'raw-task.json'
            if raw.exists() and spec.packet()!=TaskSpec.from_packet(json.loads(raw.read_text())).packet():raise ValueError('source task must preserve complete declaration before synthetic-task initialization')
            self.records[row['sequence']]=(spec,points,bounds,row['object'])
        with torch.inference_mode():
            keys=list(self.records)
            for lo in range(0,len(keys),64):
                part=keys[lo:lo+64];features=model.shape(torch.as_tensor(np.stack([self.records[k][1] for k in part]),device=self.device))
                self.shape_features.update(zip(part,features.unbind()))
        n=self.num_envs
        self.controllers=[None]*n;self.trackers=[None]*n;self.latest=[None]*n
        self.histories=[deque(maxlen=4) for _ in range(n)];self.hands=[deque(maxlen=4) for _ in range(n)]
        self.previous=np.zeros((n,12),np.float32);self.observations=torch.zeros((n,846),device=self.device)
        self.targets=[None]*n;self.episode_results=[];self.forecast_times=[]
        self.encoder_available=np.zeros(n,bool);self.physical_abort=np.zeros(n,bool);self.safety=[None]*n
        self.ever_success=np.zeros(n,bool)
        self.workspace=np.zeros(n,bool);self.initial_z=np.zeros(n);self.max_lift=np.zeros(n)
        self.streak=np.zeros(n,int);self.longest=np.zeros(n,int);self.returns=np.zeros(n)
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
        measured=[];tasks=[];shape=[];samples=[];sample_ids=[];rotations=[];shared_history=[]
        tau=np.zeros((len(ids),8,11,3),np.float32);valid=np.zeros(len(ids),bool)
        for local,i in enumerate(ids):
            name=native['sequences'][i]
            if name not in self.records:raise ValueError('native sampled motion outside admitted TRAIN')
            spec,points,bounds,obj=self.records[name]
            
            if reset:
                from wam_rl_policy_v3.task_scheduler import TaskSchedulerConfig
                # A separate explicit task, never a replacement/relabeling of
                # the source declaration retained in self.records.
                self.targets[i]=grasp_lift_task(poses[i]);self.ever_success[i]=False
                self.controllers[i]=TaskController(self.targets[i],points,scheduler_config=TaskSchedulerConfig(
                    acquire_timeout_s=15.,episode_timeout_s=20.,terminal_hold_s=1.))
                self.trackers[i]=MeasuredRewardTracker()
                self.histories[i].clear();self.hands[i].clear();self.previous[i]=0
                self.initial_z[i]=objects[i,2];self.max_lift[i]=0;self.streak[i]=0;self.longest[i]=0;self.returns[i]=0
            self.workspace[i]=False
            dof_valid=bool(np.asarray(native['dof_force_valid'])[i]);contact_valid=bool(np.asarray(native['contact_force_valid'])[i])
            force=np.asarray(native['dof_force'],np.float32)[i];contact=np.asarray(native['contact_forces'],np.float32)[i]
            supported=online_support(native,i,poses[i],points,bounds)
            self.safety[i]=physical_safety(native,i,points,bounds,self.joint_limits)
            self.physical_abort[i]=self.safety[i]['abort']
            if reset and self.physical_abort[i]:raise ValueError('unsafe physical initial state: '+str(self.safety[i]))
            decision=self.controllers[i].observe(poses[i],hand[i],objects[i,7:13],float(native['tick'][i])/30,
                contact_forces=contact,force_valid=contact_valid,supported=supported,workspace_ok=not self.physical_abort[i])['decision']
            if self.physical_abort[i]:decision=replace(decision,failure_reason=self.safety[i]['reason'])
            # The same independently packed task10 generates the WAM local goal.
            from wam_rl_policy_v3.executor_contract import task_condition
            task=task_condition(poses[i],decision)
            self.histories[i].append(np.r_[poses[i,:3,3],objects[i,3:7]]);self.hands[i].append(hand[i].copy())
            shared_history.append(history_features(self.histories[i],self.hands[i]))
            domain=encoder_domain_available(points,hand[i],poses[i])
            self.workspace[i]=not domain  # legacy diagnostic only, never done/success
            self.encoder_available[i]=bool(domain and len(self.histories[i])==4 and not decision.terminated)
            if self.encoder_available[i] and self.arm != 'A':
                sample_ids.append(local)
                samples.append(dict(objects=np.asarray(self.histories[i]),hands=np.asarray(self.hands[i]),
                    goal=decision.local_goal_pose.copy(),event=decision.next_event.wam_condition(),
                    object_code='dexplore/'+obj,points=points,pose=poses[i].copy()))
            features=reward_features(poses[i],hand[i],points,decision,objects[i,7:13],contact)
            if reset:self.trackers[i].initialize(decision,features)
            self.latest[i]=(decision,features)
            base=current_state(np.asarray(native['q'])[i:i+1],np.asarray(native['dq'])[i:i+1],np.asarray(native['physical'])[i:i+1],objects[i:i+1],self.targets[i].object_path[-1,:3,3][None],self.previous[i:i+1])[0]
            base=np.clip(base,-5,5)
            measured.append(np.r_[base,np.where(dof_valid,force,0),np.where(contact_valid,contact.reshape(-1),0),float(dof_valid),float(contact_valid)])
            tasks.append(task);shape.append(self.shape_features[name])
        packed=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',enabled=False):
            if samples:
                begin=time.perf_counter()
                try:
                    if self.arm=='B':
                        predicted=np.stack([geometric_reference(s['hands'][-1],s['pose'],s['points'],s['goal'])[:8] for s in samples])
                    else:
                        # Preserve the policy exploration/minibatch RNG stream.
                        with torch.random.fork_rng(devices=[self.device.index or 0] if self.device.type=='cuda' else []):
                            predicted=self.forecast.predict(np.stack([s['objects'] for s in samples]),
                                np.stack([s['hands'] for s in samples]),np.stack([s['goal'] for s in samples]),
                                [s['event'] for s in samples],[s['object_code'] for s in samples])[:,:8]
                    if predicted.shape!=(len(samples),8,11,3):raise ValueError('forecast adapter layout mismatch')
                    if not np.isfinite(predicted).all():raise EncodingUnavailable('nonfinite forecast')
                except EncodingUnavailable as error:
                    self.encoder_error_count+=len(samples)
                    self.encoder_available[np.asarray(ids)[sample_ids]]=False
                    if len(self.encoder_errors)<8:self.encoder_errors.append(dict(control=self.steps,message=str(error)[:512],samples=len(samples)))
                else:
                    from wam_rl_policy_v3.inspire_contract import WAM_TO_NATIVE
                    tau[sample_ids]=predicted[:,:,WAM_TO_NATIVE];valid[sample_ids]=True
                self.forecast_times.append(dict(batch=len(samples),seconds=time.perf_counter()-begin,valid_predictions=int(valid[sample_ids].sum())))
                if len(self.forecast_times)>4096:self.forecast_times.pop(0)
            raw=torch.as_tensor(np.asarray(measured,np.float32),device=self.device)
            robot=torch.cat((raw[:,:345],torch.stack(shape),raw[:,345:]),-1)
            obs=torch.cat((robot,torch.as_tensor(np.asarray(tasks),device=self.device),torch.as_tensor(tau.reshape(len(ids),264),device=self.device),torch.as_tensor(valid[:,None],dtype=torch.float32,device=self.device),torch.as_tensor(np.asarray(shared_history),device=self.device)),-1)
            if obs.shape!=(len(ids),846) or not torch.isfinite(obs).all():raise ValueError('bad measured PPO observation')
            self.observations[ids]=obs
        self.tau_valid_count+=int(valid.sum());self.timings=dict(causal_pack_s=packed-started,wam_and_transfer_s=time.perf_counter()-packed,refreshed=len(ids),valid_tau=int(valid.sum()))

    def reset(self):
        before=self.native;self.native=self.rpc('reset',ids=list(range(self.num_envs)))['state'];check_indexed_reset(before,self.native,range(self.num_envs))
        self.reset_count+=self.num_envs;self._refresh(range(self.num_envs),reset=True);return self.observations.clone()

    def get_state(self):return torch.zeros((self.num_envs,0),device=self.device)

    def step(self,actions,id=0):
        started=time.perf_counter();source=list(self.native['episodes']);tick=np.asarray(self.native['tick']).copy()
        # The policy consumes zero tau/valid=false on unavailable input. It is
        # the predeclared state-only mode of the SAME executor: no hidden action
        # substitution that would corrupt PPO's stored behavior logprob.
        self.decision_controls+=self.num_envs
        self.encoder_unavailable_steps+=int((~self.encoder_available).sum())
        self.wam_covered_controls+=int(self.observations[:,685].sum().item()) if self.arm=='C' else 0
        canonical=actions.detach().clamp(-1,1).cpu().numpy().copy().astype(np.float32)
        self.native=self.rpc('step',action=expand_action(canonical).tolist(),episodes=source)['state']
        if self.native['episodes']!=source or not np.array_equal(np.asarray(self.native['tick']),tick+1):
            self.source_mismatches+=1;raise ValueError('action executed in wrong episode or time')
        self.previous[:]=canonical;self._refresh(range(self.num_envs));self.steps+=1
        reward=np.zeros(self.num_envs,np.float32);done=np.zeros(self.num_envs,bool);post_success=np.zeros(self.num_envs,bool);results={}
        for i,(decision,features) in enumerate(self.latest):
            r,_=self.trackers[i].step(decision,features,canonical[i]);reward[i]=r
            self.ever_success[i]|=bool(decision.task_complete and not self.physical_abort[i])
            done[i]=decision.terminated or self.physical_abort[i]
            forced=self.reset_probe and ((self.steps==2 and i==0) or (self.steps==4 and i==1))
            if forced:done[i]=True
            if forced and not decision.terminated:
                # External terminal must also zero the next potential, not bootstrap through reset.
                from wam_rl_policy_v3.task_reward import potential
                reward[i]-=self.trackers[i].gamma*potential(decision,features)+self.trackers[i].reward.failure_cost
            self.returns[i]+=reward[i]
            lift=float(self.native['objects'][i][2]-self.initial_z[i]);self.max_lift[i]=max(self.max_lift[i],lift)
            self.streak[i]=self.streak[i]+1 if lift>=.05 else 0;self.longest[i]=max(self.longest[i],self.streak[i])
            if done[i]:
                success=bool(decision.task_complete and not self.physical_abort[i] and not forced)
                post_success[i]=success
                source_spec=self.records[self.native['sequences'][i]][0]
                result=dict(episode_id=source[i],sequence=self.native['sequences'][i],task_success=success,
                    ever_success=bool(self.ever_success[i]),final_success=success,task_contract=TASK_CONTRACT,source_full_task_success=None,
                    source_terminal_mode=source_spec.terminal_mode.value,source_annotation_invalid=source_spec.annotation_invalid,
                    physical_safety_abort=bool(self.physical_abort[i]),physical_safety_evidence=self.safety[i],
                    failure_reason='engineering_reset' if forced else decision.failure_reason,
                    confirmed_progress=float(decision.path_progress),return_sum=float(self.returns[i]),controls=int(self.native['tick'][i]),max_lift_m=float(self.max_lift[i]),ever_lift5cm30=bool(self.longest[i]>=30))
                results[i]=result;self.episode_results.append(result);self.completed_episodes+=1;self.successes+=int(result['task_success']);self.physical_safety_aborts+=int(self.physical_abort[i])
        terminal=self.observations.clone();terminal_episodes=list(self.native['episodes']);ids=np.flatnonzero(done)
        availability=self.encoder_available.copy();physical_abort=self.physical_abort.copy();ever_success=self.ever_success.copy()
        if len(ids):
            before=self.native;self.native=self.rpc('reset',ids=ids.tolist())['state'];check_indexed_reset(before,self.native,ids)
            self._refresh(ids,reset=True);self.reset_count+=len(ids)
            if len(self.reset_audits)<16:self.reset_audits.append(dict(step=self.steps,ids=ids.tolist(),execution_episode=terminal_episodes,next_observation_episode=list(self.native['episodes']),next_tick=np.asarray(self.native['tick']).tolist()))
        info=dict(episode_results=results,action_source_episode=source,execution_episode=terminal_episodes,next_observation_episode=list(self.native['episodes']),
            encoder_available=availability,physical_safety_abort=physical_abort,fallback_policy=FALLBACK_POLICY,
            task_success=post_success,ever_success=ever_success,final_success=post_success.copy(),
            terminal_observation=terminal[ids],transition_valid=torch.ones(self.num_envs,dtype=torch.bool,device=self.device))
        self.timings['step_total_s']=time.perf_counter()-started
        return self.observations.clone(),torch.as_tensor(reward,device=self.device),torch.as_tensor(done.astype(np.int64),device=self.device),info


def geometric_reference(hand,pose,points,goal):
    """Current-CAD reference with proximal geometry optimization; no future data."""
    world=np.asarray(points)@pose[:3,:3].T+pose[:3,3]
    target=np.asarray(hand,dtype=np.float64).copy()
    center=world.mean(0)
    target+=np.clip(center-target[1:6].mean(0),-.04,.04)
    tips=target[6:].copy();initial=tips.copy()
    for _ in range(12):
        distance=np.linalg.norm(tips[:,None]-world[None],axis=-1)
        nearest=world[distance.argmin(1)]
        tips+=.2*(nearest-tips)-.04*(tips-initial)
        tips=np.clip(tips,initial-.025,initial+.025)
    target[6:]=tips
    movement=np.clip(goal[:3,3]-pose[:3,3],-.025,.025)
    fraction=np.arange(1,25,dtype=np.float64)/24
    return (fraction[:,None,None]*(target-hand)[None]+fraction[:,None,None]*movement[None,None]).astype(np.float32)


def history_features(objects,hands):
    """Shared causal H4: repeat earliest at startup, relative translation in 5cm units."""
    objects=np.asarray(objects,dtype=np.float64);hands=np.asarray(hands,dtype=np.float64)
    if not 1<=len(objects)==len(hands)<=4:raise ValueError('1–4 matching causal frames required')
    objects=np.concatenate((np.repeat(objects[:1],4-len(objects),axis=0),objects))
    hands=np.concatenate((np.repeat(hands[:1],4-len(hands),axis=0),hands))
    position=(objects[:,:3]-objects[-1:,:3])/.05
    quaternion=objects[:,3:7].copy()
    quaternion*=np.where((quaternion*quaternion[-1:]).sum(-1,keepdims=True)<0,-1.,1.)
    delta=(hands-hands[-1:])/.05
    return np.concatenate((position,quaternion,delta.reshape(4,33)),-1).reshape(160).astype(np.float32)

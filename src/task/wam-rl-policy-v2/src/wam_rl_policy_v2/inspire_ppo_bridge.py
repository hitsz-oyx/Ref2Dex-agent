"""Shadow official PPO environment adapter for pure measured Inspire physics."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .inspire_contract import expand_action,ACTIVE
from .inspire_shadow import CONTRACT,CURRENT_DIM,INPUT_DIM,SCHEMA,expanded_model,current_state,BatchedInspireInputs


def shadow_path_reward(objects,hands,actions,initial_z,decisions,previous_progress):
    """Shadow goal_cond=False approach terms; ordered object-path task adapter.

    The static-target distance is replaced by local SE3 surface tracking error,
    terminal bonus requires ordered completion. Progress bonus is monotonic;
    no hand-reference imitation, pose retargeting loss or DExplore reward.
    """
    wrist=np.linalg.norm(objects[:,:3]-hands[:,0],axis=-1).clip(max=.5)
    fingers=np.linalg.norm(objects[:,None,:3]-hands[:,[2,4,6,8,10]],axis=-1).sum(-1).clip(max=3.)
    coupled=(fingers<=.6)&(wrist<=.12)
    error=np.array([d['reference_error_m'] for d in decisions])
    progress=np.array([d['confirmed_progress'] for d in decisions])
    complete=np.array([d['complete'] for d in decisions])
    lift=objects[:,2]-initial_z
    # Translate the Shadow world-height thresholds to each episode's tabletop.
    hand_up=np.where(coupled&(lift>=.003),.1+.1*actions[:,2],0.)
    hand_up=np.where(coupled&(lift>=.17),.2,hand_up)
    goal=np.where(coupled,.9-2*error,0.)
    bonus=np.where(coupled&complete,1/(1+10*error),0.)
    advance=2*np.maximum(progress-previous_progress,0.)
    reward=-.5*fingers-wrist+goal+hand_up+bonus+advance
    return reward.astype(np.float32),progress


class RemoteInspireEnv:
    def __init__(self,connection,hello,cache,runtime,*,reset_probe=False):
        from gym import spaces
        self.connection=connection;self.runtime=runtime;self.cache=Path(cache);self.native=hello['state'];self.audit=hello['audit']
        self.num_envs=len(self.native['q']);self.num_environments=self.num_envs;self.num_observations=INPUT_DIM
        self.num_actions=12;self.num_states=0;self.rl_device=str(runtime.device)
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(INPUT_DIM,),dtype=np.float32)
        self.state_space=spaces.Box(-np.inf,np.inf,shape=(0,),dtype=np.float32)
        self.action_space=spaces.Box(-1.,1.,shape=(12,),dtype=np.float32)
        self.task=SimpleNamespace(cfg={'seed':246},max_episode_length=100000,random_time=False)
        meta=json.loads((self.cache/'manifest.json').read_text())
        self.references={};self.objects_by_sequence={}
        for row in meta['sequences']:
            if row['split']!='train':continue
            with np.load(Path(row['source'])/'trajectory.npz') as z:self.references[row['sequence']]=z['reference_obj'].copy()
            self.objects_by_sequence[row['sequence']]=row['object']
        references=[self.references[s] for s in self.native['sequences']]
        self.inputs=BatchedInspireInputs(runtime,references)
        self.previous=np.zeros((self.num_envs,12),np.float32);self.initial_z=np.zeros(self.num_envs)
        self.progress=np.zeros(self.num_envs);self.longest=np.zeros(self.num_envs,int);self.streak=np.zeros(self.num_envs,int)
        self.dropped=np.zeros(self.num_envs,bool);self.ever_complete=np.zeros(self.num_envs,bool)
        self.observations=np.zeros((self.num_envs,INPUT_DIM),np.float32);self.decisions=[None]*self.num_envs
        self.workspace=np.zeros(self.num_envs,bool);self.steps=0;self.reset_count=0
        self.reset_probe=reset_probe;self.reset_audits=[]
        self.completed_episodes=0;self.successes=0;self.ever_held=0;self.final_held=0;self.workspace_failures=0
        self._refresh(range(self.num_envs),reset=True)

    def rpc(self,op,**values):
        from trajectory_policy.wam_ipc import send,receive
        send(self.connection,dict(op=op,**values));return receive(self.connection)

    def _refresh(self,indices,reset=False):
        indices=list(indices);n=self.num_envs
        poses=np.repeat(np.eye(4)[None],n,0);objects=np.asarray(self.native['objects'],np.float32)
        poses[:,:3,:3]=Rotation.from_quat(objects[:,3:7]).as_matrix();poses[:,:3,3]=objects[:,:3]
        if reset:
            for i in indices:
                reference=self.references[self.native['sequences'][i]]
                self.inputs.reset(i,reference);self.previous[i]=0;self.initial_z[i]=objects[i,2]
                self.progress[i]=0;self.streak[i]=0;self.longest[i]=0;self.dropped[i]=False;self.ever_complete[i]=False
        goals=np.stack([self.references[s][-1,:3,3] for s in self.native['sequences']])
        current=current_state(self.native['q'],self.native['dq'],self.native['physical'],objects,goals,self.previous)
        result=self.inputs.observation(current,poses,np.asarray(self.native['hand']),np.asarray(self.native['tick']),indices=indices)
        self.observations[indices]=result['observation'][indices];self.workspace[indices]=result['workspace_invalid'][indices]
        for i in indices:self.decisions[i]=result['scheduler'][i]
        return objects

    def reset(self):
        self.native=self.rpc('reset',ids=list(range(self.num_envs)))['state'];self.reset_count+=self.num_envs
        self._refresh(range(self.num_envs),reset=True)
        return torch.tensor(self.observations,device=self.runtime.device)

    def get_state(self):return torch.zeros(self.num_envs,0,device=self.runtime.device)

    def step(self,actions,id=0):
        source_episodes=list(self.native['episodes']);raw=actions.detach().cpu().numpy().copy()
        canonical=np.clip(raw,-1,1).astype(np.float32)
        self.native=self.rpc('step',action=expand_action(canonical).tolist(),episodes=source_episodes)['state']
        if self.native['episodes']!=source_episodes:raise ValueError('native physics changed episode during policy step')
        self.previous[:]=canonical
        objects=self._refresh(range(self.num_envs));self.steps+=1
        # A subsequent partial reset may replace inputs.timings with a tiny
        # startup-only call. Report the actual full-environment policy tick.
        self.input_timings=dict(self.inputs.timings)
        reward,self.progress=shadow_path_reward(objects,np.asarray(self.native['hand']),canonical,self.initial_z,self.decisions,self.progress)
        complete=np.array([d['complete'] for d in self.decisions]);self.ever_complete|=complete
        timed_out=np.array([d['timed_out'] for d in self.decisions])
        horizon=np.array([len(self.references[s])+30 for s in self.native['sequences']])
        done=self.workspace|timed_out|(np.asarray(self.native['tick'])>=horizon)
        hold=(objects[:,2]-self.initial_z>=.1)&(np.linalg.norm(objects[:,:3]-np.stack([self.references[s][-1,:3,3] for s in self.native['sequences']]),axis=-1)<=.05)
        self.dropped|=(self.longest>=30)&~hold;self.streak=np.where(hold,self.streak+1,0);self.longest=np.maximum(self.longest,self.streak)
        # Engineering fixture: force only env0 and then env1 to terminate while
        # the other environment stays active, exercising actual indexed reset.
        if self.reset_probe:
            if self.steps==5:done[0]=True
            if self.steps==11:done[min(1,self.num_envs-1)]=True
        terminal=self.observations.copy();terminal_episodes=list(self.native['episodes']);ids=np.flatnonzero(done)
        episode_results={}
        for i in ids:
            episode_results[int(i)]=dict(episode_id=terminal_episodes[i],ordered_path_success=bool(self.ever_complete[i] and not self.workspace[i]),
                final_path_complete=bool(complete[i] and not self.workspace[i]),ever_held1s=bool(self.longest[i]>=30),final_held1s=bool(self.streak[i]>=30),
                drop_after_hold=bool(self.dropped[i]),workspace_failure=bool(self.workspace[i]),confirmed_progress=float(self.progress[i]),
                termination_reason='wam_workspace' if self.workspace[i] else 'scheduler_timeout' if timed_out[i] else 'horizon_or_engineering_reset')
            self.completed_episodes+=1;self.successes+=episode_results[int(i)]['ordered_path_success']
            self.ever_held+=int(self.longest[i]>=30);self.final_held+=int(self.streak[i]>=30);self.workspace_failures+=int(self.workspace[i])
        if len(ids):
            before=np.asarray(terminal_episodes);old_ticks=np.asarray(self.native['tick']);unaffected=np.setdiff1d(np.arange(self.num_envs),ids)
            self.native=self.rpc('reset',ids=ids.tolist())['state'];after=np.asarray(self.native['episodes']);new_ticks=np.asarray(self.native['tick'])
            if not (np.all(after[ids]==before[ids]+1) and np.all(after[unaffected]==before[unaffected]) and np.all(new_ticks[ids]==0)
                    and np.array_equal(old_ticks[unaffected],new_ticks[unaffected])):raise ValueError('indexed reset episode/clock corruption')
            self._refresh(ids,reset=True);self.reset_count+=len(ids)
            if len(self.reset_audits)<32:self.reset_audits.append(dict(step=self.steps,ids=ids.tolist(),old_episodes=before.tolist(),new_episodes=after.tolist(),new_ticks=new_ticks.tolist()))
        info=dict(episode_results=episode_results,action_source_episode=source_episodes,execution_episode=terminal_episodes,
            next_observation_episode=list(self.native['episodes']),terminal_observation=terminal[ids],transition_valid=torch.ones(self.num_envs,dtype=torch.bool,device=self.runtime.device))
        return torch.tensor(self.observations,device=self.runtime.device),torch.tensor(reward,device=self.runtime.device),torch.tensor(done,dtype=torch.long,device=self.runtime.device),info


def make_shadow_ppo(root,output,env,packet,seed):
    """The actual pinned Shadow PPO collector/storage/update; no replacement PPO."""
    import importlib.util
    import sys
    import yaml
    from .official_bridge import prepare_official
    official=prepare_official(root,output,masked_resets=False)
    sys.path.insert(0,str(official))
    spec=importlib.util.spec_from_file_location('inspire_official_ppo_process',official/'utils/process_sarl.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cfg=yaml.safe_load((official/'cfg/ppo/config.yaml').read_text());cfg['seed']=seed;env.task.cfg['seed']=seed
    cfg['policy'].update(backbone_type='pn',freeze_backbone=False,obs_dim=INPUT_DIM)
    torch.manual_seed(seed)
    actor=expanded_model(True);actor.load_state_dict(packet['model'],strict=True)
    if packet.get('schema')!=SCHEMA:raise ValueError('full-data Inspire scheduled BC required')
    ppo=module.process_ppo(SimpleNamespace(test=False,vision=False,model_dir=''),env,cfg,str(output/'models'))
    ppo.actor_critic.actor=actor.to(env.runtime.device)
    ppo.optimizer=torch.optim.Adam(ppo.actor_critic.parameters(),lr=cfg['learn']['optim_stepsize'])
    return ppo,cfg

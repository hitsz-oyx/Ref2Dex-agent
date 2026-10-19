"""Reuse pinned UniDexGrasp PPO; adapt only observations, BC mean and distribution.

Distribution formulas remain upstream, including squared scale_tril. Distributed
extensions average gradients/KL and normalize advantages over the global batch.
"""
import importlib.util,json,os,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from .ppo import reduce,synchronize_gradients,distributed_ready
from .batched_scheduled_inputs import BatchedScheduledInputs
from .goal_scheduler import OrderedGoalScheduler
from .scheduled_ppo_loop import parameter_sync_error
from .control_contracts import scheduled_conditioning_contract
from .checkpoint_retention import prune_checkpoints
from .episode_metrics import episode_outcome
from .execution_identity import execution_identity, validate_execution


def normalize_advantages(values, valid=None):
    selected=values if valid is None else values[valid]
    if not distributed_ready():
        result=(values-selected.mean())/(selected.std()+1e-8)
        return result if valid is None else torch.where(valid,result,0.)
    stats=torch.stack((selected.double().sum(),selected.double().square().sum(),values.new_tensor(selected.numel()).double()))
    dist.all_reduce(stats)
    mean=stats[0]/stats[2]
    std=((stats[1]-stats[0]*mean)/(stats[2]-1)).clamp_min(0).sqrt()
    result=((values.double()-mean)/(std+1e-8)).to(values.dtype)
    return result if valid is None else torch.where(valid,result,0.)


def valid_minibatches(storage, count):
    """Keep every valid transition and the same update count on all ranks."""
    ids=storage.transition_valid.flatten().nonzero().flatten().cpu().tolist()
    enough=reduce(torch.tensor(float(len(ids)>=count),device=storage.device),'sum')
    ranks=dist.get_world_size() if distributed_ready() else 1
    if enough.item()!=ranks:raise RuntimeError('Too few policy transitions for official minibatches')
    if storage.sampler=='random':
        ids=[ids[i] for i in torch.randperm(len(ids)).tolist()]
    return [part.tolist() for part in np.array_split(ids,count)]


def weighted_reduce(value, count):
    value=value.detach()
    if not distributed_ready():return value
    pair=torch.stack((value*count,value.new_tensor(count)))
    dist.all_reduce(pair)
    return pair[0]/pair[1]


def synchronize_sample_gradients(model,count):
    if distributed_ready():
        total=reduce(next(model.parameters()).new_tensor(float(count)),'sum')
        scale=count*dist.get_world_size()/total
        for parameter in model.parameters():
            if parameter.grad is not None:parameter.grad.mul_(scale)
    synchronize_gradients(model)


def prepare_official(root,output,masked_resets=False,correct_distribution_kl=False):
    """Same pinned source and dimension compatibility patch as official_unidex_ppo.py."""
    source=root/'third_party/UniDexGrasp2-official-ppo'
    target=output/'runtime-official';shutil.copytree(source,target)
    changes={}
    def edit(relative,replacements):
        p=target/relative;text=p.read_text();changes.setdefault(relative,[])
        for old,new,count in replacements:
            if text.count(old)!=count:raise ValueError('Pinned upstream patch point changed: '+old)
            text=text.replace(old,new);changes[relative].append(dict(old=old,new=new,count=count))
        p.write_text(text)
    module='dexgrasp/algorithms/rl/ppo/module.py'
    p=target/module;removed=[line for line in p.read_text().splitlines() if line.startswith('from algo.pn_utils.')]
    if len(removed)!=2:raise ValueError('Expected two unused vision imports')
    p.write_text('\n'.join(line for line in p.read_text().splitlines() if line not in removed)+'\n')
    edit(module,[('self.num_obs = 300',"self.num_obs = model_cfg.get('obs_dim', 300)",1),
        ('self.actor(observations[:,:300])','self.actor(observations)',3),
        ('if not self.use_pc:\n            observations = observations[:,:300]',
         'if not self.use_pc and self.num_obs == 300:\n            observations = observations[:,:300]',2)])
    edit('dexgrasp/algorithms/rl/ppo/ppo.py',[
        ('import torch\n','import torch\nfrom wam_rl_policy_v2.official_bridge import reduce, synchronize_gradients\n',1),
        ('kl_mean = torch.mean(kl)','kl_mean = reduce(torch.mean(kl))',1),
        ('loss.backward()','loss.backward()\n                synchronize_gradients(self.actor_critic)',1)])
    edit('dexgrasp/algorithms/rl/ppo/storage.py',[
        ('import torch\n','import torch\nfrom wam_rl_policy_v2.official_bridge import normalize_advantages\n',1),
        ('(self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)','normalize_advantages(self.advantages)',1)])
    if correct_distribution_kl:
        edit('dexgrasp/algorithms/rl/ppo/ppo.py',[
            ('import torch\n','import torch\nfrom wam_rl_policy_v2.distribution_kl import actual_policy_kl\n',1),
            ('kl = torch.sum(\n                        sigma_batch - old_sigma_batch + (torch.square(old_sigma_batch.exp()) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch.exp())) - 0.5, axis=-1)',
             'kl = actual_policy_kl(old_mu_batch, old_sigma_batch, mu_batch, sigma_batch)',1)])
    if masked_resets:
        edit('dexgrasp/algorithms/rl/ppo/ppo.py',[
            ('from wam_rl_policy_v2.official_bridge import reduce, synchronize_gradients',
             'from wam_rl_policy_v2.official_bridge import reduce, synchronize_gradients, weighted_reduce, synchronize_sample_gradients',1),
            ('actions_log_prob, mu, sigma)\n                    current_obs.copy_',
             'actions_log_prob, mu, sigma, infos["transition_valid"])\n                    current_obs.copy_',1),
            ('kl_mean = reduce(torch.mean(kl))','kl_mean = weighted_reduce(torch.mean(kl), len(indices))',1),
            ('synchronize_gradients(self.actor_critic)','synchronize_sample_gradients(self.actor_critic,len(indices))',1),
            ('new_ids = (dones > 0).nonzero(as_tuple=False)',
             'new_ids = ((dones > 0) & infos["transition_valid"]).nonzero(as_tuple=False)',1),
            ('cur_episode_length[:] += 1','cur_episode_length[:] += infos["transition_valid"]',1)])
        edit('dexgrasp/algorithms/rl/ppo/storage.py',[
            ('from wam_rl_policy_v2.official_bridge import normalize_advantages',
             'from wam_rl_policy_v2.official_bridge import normalize_advantages, valid_minibatches',1),
            ('self.dones = torch.zeros',
             'self.transition_valid = torch.ones(num_transitions_per_env,num_envs,1,device=self.device,dtype=torch.bool)\n        self.dones = torch.zeros',1),
            ('actions_log_prob, mu, sigma):','actions_log_prob, mu, sigma, transition_valid=None):',1),
            ('self.step += 1','self.transition_valid[self.step].copy_(transition_valid.view(-1,1))\n        self.step += 1',1),
            ('normalize_advantages(self.advantages)','normalize_advantages(self.advantages,self.transition_valid)',1),
            ('batch_size = self.num_envs * self.num_transitions_per_env',
             'return valid_minibatches(self,num_mini_batches)\n        batch_size = self.num_envs * self.num_transitions_per_env',1)])
    (output/'official-compatibility.json').write_text(json.dumps(dict(upstream_commit='6c9e710b04769bf5d9765e3233820101cb3d718f',
        previous_runner='src/task/wam-rl-policy/tools/run/official_unidex_ppo.py',removed_vision_imports=removed,
        changes=changes,ppo_run_changed=masked_resets,gae_changed=False,reward_changed=False,
        reset_initialization_excluded=masked_resets,
        distribution_changed=False,adaptive_kl_uses_actual_distribution=correct_distribution_kl,
        actor='self-trained scheduled BC mean with its frozen normalization; fully trainable',
        critic='official ELU architecture, fresh initialization',observation='565D current300 + world tau264 + valid1'),indent=2)+'\n')
    return target/'dexgrasp'


class ScheduledOfficialEnv:
    def __init__(self,base,snapshot,runtime,native_only=False,*,protocol='corrected',goal_type='position_only',termination_policy=None):
        from gym import spaces
        self.native_only=native_only
        if protocol not in ('corrected','legacy'):raise ValueError('Unknown reset protocol')
        self.protocol=protocol;self.goal_type=goal_type
        self.termination_policy=termination_policy or ('native' if native_only else 'wam_workspace')
        if self.termination_policy not in ('native','wam_workspace','common_workspace'):raise ValueError('Unknown termination policy')
        if not native_only and self.termination_policy=='native':raise ValueError('WAM requires explicit workspace protection')
        self.base=base;self.task=base.task;self.snapshot=snapshot;self.runtime=runtime
        for name in ('num_envs','num_environments','num_states','num_actions','state_space','action_space','rl_device'):
            setattr(self,name,getattr(base,name))
        self.num_observations=300 if native_only else 565
        self.observation_space=spaces.Box(low=-np.inf,high=np.inf,shape=(self.num_observations,),dtype=np.float32)
        self.ticks=np.zeros(self.num_envs,int);self.pending=np.zeros(self.num_envs,bool)
        self.inputs=None;self.reset_count=0;self.steps=0
        self.initial_z=np.zeros(self.num_envs);self.streak=np.zeros(self.num_envs,int);self.longest=np.zeros(self.num_envs,int)
        self.dropped_after_hold=np.zeros(self.num_envs,bool)
        self.completed_episodes=0;self.successful_episodes=0;self.held_episodes=0
        self.workspace_failures=0;self.workspace_invalid=np.zeros(self.num_envs,bool)
        self.episode_ids=np.zeros(self.num_envs,int)
        self.last_episode_results={};self.initialization_steps=0
        self.final_successful_episodes=self.final_held_episodes=self.drop_episodes=0

    def augment(self,native,reset_ids):
        state=self.snapshot()
        if self.inputs is None and not self.native_only:
            schedulers=[]
            for i in range(self.num_envs):schedulers.append(self.scheduler(state,i))
            self.inputs=BatchedScheduledInputs(self.runtime,schedulers)
        for i in reset_ids:
            if not self.native_only:
                self.inputs.histories[i].clear();self.inputs.schedulers[i]=self.scheduler(state,i)
            self.ticks[i]=0
            self.initial_z[i]=state['pose'][i,2,3];self.streak[i]=0;self.longest[i]=0
            self.dropped_after_hold[i]=False
        lift=state['pose'][:,2,3]-self.initial_z
        near=np.linalg.norm(state['pose'][:,:3,3]-state['goal'],axis=-1)<=.05
        self.final_near=near
        holding=near & (lift>=.1)
        self.dropped_after_hold |= (self.longest>=60) & ~holding
        self.streak=np.where(holding,self.streak+1,0);self.longest=np.maximum(self.longest,self.streak)
        if self.native_only:
            self.workspace_invalid[:]=False
            if self.termination_policy=='common_workspace':self.workspace_invalid=self.common_workspace_invalid(state)
            self.ticks+=1
            return native.detach()
        current=native.detach() if self.inputs.pipeline in ('tau','inputs','all') else native.detach().cpu().numpy()
        result=self.inputs.observation(current,state['pose'],state['hand'],self.ticks/60)
        self.workspace_invalid=result['workspace_invalid']
        if self.termination_policy=='common_workspace':self.workspace_invalid |= self.common_workspace_invalid(state)
        self.ticks+=1
        return torch.as_tensor(result['observation'],device=self.runtime.device)

    def scheduler(self,state,i):
        target=state['pose'][i].copy();target[:3,3]=state['goal'][i]
        return OrderedGoalScheduler([state['pose'][i],target],[0,2],self.runtime.geometry[i][2],interaction_points=self.runtime.geometry[i][0],goal_type=self.goal_type)

    def common_workspace_invalid(self,state):
        samples=[]
        for i,geometry in enumerate(self.runtime.geometry):
            hand=(state['hand'][i]-state['pose'][i,:3,3]) @ state['pose'][i,:3,:3]
            samples.append(dict(xyz=geometry[0],point_valid=np.ones(len(geometry[0]),bool),
                features=np.zeros((len(geometry[0]),18),np.float32),start_hand=hand,
                start_hand_valid=np.ones(11,bool)))
        return ~self.runtime.workspace_valid(samples)

    def reset(self):
        native=self.base.reset();self.pending[:]=False
        return self.augment(native,range(self.num_envs))

    def step(self,actions,id=0):
        # Preserve the sampled canonical actions in upstream storage despite
        # the native task's in-place wrist transformation.
        initializing=self.pending.copy()
        canonical=actions.clone()
        if self.protocol=='corrected':canonical[torch.as_tensor(initializing,device=actions.device)]=0
        source_episodes=self.episode_ids.copy()
        native,reward,done,info=self.base.step(canonical,id)
        reward=reward.clone();done=done.clone();info=dict(info)
        reset_ids=np.flatnonzero(self.pending)
        result=self.augment(native,reset_ids)
        # Out-of-domain WAM input is an explicit terminal failure, never a
        # clipped point or a fabricated trajectory used for another action.
        invalid=torch.as_tensor(self.workspace_invalid,device=done.device)
        done=done.clone();done[invalid]=1
        self.task.reset_buf[invalid]=1
        self.workspace_failures+=int(self.workspace_invalid.sum())
        self.reset_count+=len(reset_ids);self.steps+=1
        self.episode_ids[reset_ids]+=1
        if self.protocol=='corrected':self.initialization_steps+=len(reset_ids)
        self.pending=done.detach().cpu().numpy().astype(bool).copy()
        self.completed_episodes+=int(self.pending.sum())
        self.successful_episodes+=int((self.task.successes.detach().cpu().numpy()[self.pending]>0).sum())
        self.held_episodes+=int((self.longest[self.pending]>=60).sum())
        self.last_episode_results={int(i):dict(episode_id=int(self.episode_ids[i]),
            **episode_outcome(self.task.successes[i].item()>0,self.longest[i],self.streak[i],self.workspace_invalid[i],self.final_near[i],self.dropped_after_hold[i]))
            for i in np.flatnonzero(self.pending)}
        for record in self.last_episode_results.values():
            self.final_successful_episodes+=int(record['final_task_success'])
            self.final_held_episodes+=int(record['final_stable_success'])
            self.drop_episodes+=int(record['drop_after_hold'])
        valid=np.ones(self.num_envs,bool) if self.protocol=='legacy' else ~initializing
        if self.protocol=='corrected':
            mask=torch.as_tensor(initializing,device=done.device)
            reward[mask]=0;done[mask]=1  # cut GAE across excluded reset initialization
        info.update(transition_valid=torch.as_tensor(valid,device=done.device),
            action_source_episode=source_episodes,execution_episode=self.episode_ids.copy(),
            reset_initialization=initializing,episode_results=self.last_episode_results)
        return result,reward,done,info

    def get_state(self):return self.base.get_state()


def run(task,snapshot,sim_step,actor,runtime,args):
    import yaml
    from types import SimpleNamespace
    from tasks.hand_base.vec_task import VecTaskPython
    root=Path(os.environ.get('REF2DEX_RUN_ROOT',Path(__file__).resolve().parents[5]))
    native_only=getattr(args,'native_evaluation',False)
    protocol=getattr(args,'execution_protocol',None) or ('legacy' if native_only else 'corrected')
    official=prepare_official(root,args.output,masked_resets=protocol=='corrected');sys.path.insert(0,str(official))
    spec=importlib.util.spec_from_file_location('scheduled_official_process',official/'utils/process_sarl.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cfg=yaml.safe_load((official/'cfg/ppo/config.yaml').read_text())
    cfg['policy'].update(backbone_type='pn',freeze_backbone=False,obs_dim=300 if native_only else 565)
    cfg['seed']=args.seed;task.cfg['seed']=args.seed
    torch.manual_seed(args.seed)
    env=ScheduledOfficialEnv(VecTaskPython(task,'cuda:0',cfg['clip_observations'],cfg['clip_actions']),snapshot,runtime,native_only=native_only,
        protocol=protocol,goal_type=args.goal_type,termination_policy=args.termination_policy)
    identity=execution_identity(runtime,protocol=protocol,goal_type=args.goal_type,termination_policy=env.termination_policy)
    if distributed_ready():
        geometries=[None]*dist.get_world_size()
        dist.all_gather_object(geometries,identity['geometry'])
        merged={}
        for entries in geometries:
            for key,value in entries.items():
                if key in merged and merged[key]!=value:raise ValueError('Rank geometry conflict')
                merged[key]=value
        identity['geometry']=merged
    (args.output/'execution-identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    ppo=module.process_ppo(SimpleNamespace(test=False,vision=False,model_dir=''),env,cfg,str(args.output/'models'))
    # Replace only the policy mean network; upstream act/evaluate/inference and
    # Gaussian convention remain identical in all three call paths.
    if not native_only:ppo.actor_critic.actor=actor.actor
    ppo.optimizer=torch.optim.Adam(ppo.actor_critic.parameters(),lr=cfg['learn']['optim_stepsize'])
    completed=0
    if args.rl_checkpoint:
        packet=torch.load(args.rl_checkpoint,map_location=runtime.device,weights_only=False)
        if native_only:
            weights=packet.get('state_dict',packet)
            ppo.actor_critic.load_state_dict(weights,strict=True)
            from .official_evaluation import evaluate
            return evaluate(env,ppo.actor_critic,args,args.native_checkpoint_iteration)
        if packet.get('schema')!='ref2dex.scheduled-official-ppo.v1' or packet['conditioning']!=scheduled_conditioning_contract():raise ValueError('Official scheduled checkpoint required')
        migration=validate_execution(packet.get('execution_identity'),identity,evaluating=getattr(args,'evaluate',False),
            migration_reason=getattr(args,'checkpoint_migration_reason',None))
        (args.output/'checkpoint-migration.json').write_text(json.dumps(migration,indent=2)+'\n')
        ppo.actor_critic.load_state_dict(packet['model']);ppo.optimizer.load_state_dict(packet['optimizer'])
        completed=packet['iteration'];ppo.step_size=packet['step_size']
    if getattr(args,'evaluate',False):
        from .official_evaluation import evaluate
        return evaluate(env,ppo.actor_critic,args,completed)
    if distributed_ready():
        for parameter in ppo.actor_critic.parameters():dist.broadcast(parameter.data,src=0)
    (args.output/'training-config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    started=time.monotonic();state=dict(iteration=completed,optimizer_steps=0)
    original_step=ppo.optimizer.step
    def step(*a,**kw):
        result=original_step(*a,**kw);state['optimizer_steps']+=1;return result
    ppo.optimizer.step=step
    original_update=ppo.update
    def update():
        if (args.output.parent/'STOP').exists():raise InterruptedError('STOP requested')
        # Upstream collector and evaluator must agree on the action mean.
        with torch.no_grad():
            x=ppo.storage.observations[0];z=ppo.storage.actions[0]
            _,_,_,mu,_=ppo.actor_critic.evaluate(x,None,z)
            error=float((mu-ppo.actor_critic.act_inference(x)).abs().max())
            if error>1e-6:raise ValueError('Official action path mismatch')
            state['mean_path_error']=error
        result=original_update()
        if not np.isfinite(result).all():raise FloatingPointError('Nonfinite official PPO loss')
        return result
    ppo.update=update
    def log(locs):
        iteration=locs['it']+1
        metrics=reduce(torch.tensor([float(locs['mean_reward']),float(locs['mean_value_loss']),float(locs['mean_surrogate_loss'])],device=runtime.device))
        state.update(iteration=completed+iteration,rank=args.rank,world_size=args.world_size,batch_pipeline=env.inputs.pipeline,deep_opt=os.environ.get('REF2DEX_DEEP_OPT','none'),
            env_steps=iteration*args.envs*cfg['learn']['nsteps']*args.world_size,
            elapsed_s=time.monotonic()-started,collection_s=locs['collection_time'],learning_s=locs['learn_time'],lr=ppo.step_size,reward_mean=float(metrics[0]),
            value_loss=float(metrics[1]),surrogate_loss=float(metrics[2]),reset_count=env.reset_count,
            workspace_failures=env.workspace_failures,
            reset_initialization_steps=env.initialization_steps,reset_protocol=protocol,
            effective_std=float(ppo.actor_critic.log_std.exp().square().mean()),
            peak_mib=torch.cuda.max_memory_allocated()/2**20)
        counts=reduce(torch.tensor([env.completed_episodes,env.successful_episodes,env.held_episodes,
            env.final_successful_episodes,env.final_held_episodes,env.drop_episodes],device=runtime.device,dtype=torch.float64),'sum')
        state.update(completed_episodes=int(counts[0]),native_success_episodes=int(counts[1]),held60_lift10cm_episodes=int(counts[2]))
        state.update(ever_held60_episodes=int(counts[2]),final_task_success_episodes=int(counts[3]),
            final_stable_success_episodes=int(counts[4]),drop_after_hold_episodes=int(counts[5]))
        state['eta_s']=state['elapsed_s']/iteration*(args.ppo_updates-iteration)
        if iteration==1 or iteration%args.save_every==0 or iteration==args.ppo_updates:
            state['model_sync_iteration']=completed+iteration
            state['model_sync_error']=parameter_sync_error(ppo.actor_critic)
            if state['model_sync_error']!=0:raise RuntimeError('Rank parameters diverged')
        with (args.output/'curve.jsonl').open('a') as f:f.write(json.dumps(state)+'\n')
        print(json.dumps(state),flush=True)
        if state['iteration'] % 500 == 0:save(None)
        if timeline is not None:timeline.step()
    ppo.log=log
    def save(path):
        if args.rank==0:
            iteration=state['iteration'];dest=args.output/('official-iteration-%d.pt'%iteration)
            if dest.exists():return
            partial=dest.with_suffix('.pt.partial')
            torch.save(dict(schema='ref2dex.scheduled-official-ppo.v1',model=ppo.actor_critic.state_dict(),
                optimizer=ppo.optimizer.state_dict(),iteration=iteration,step_size=ppo.step_size,
                config=cfg,conditioning=scheduled_conditioning_contract(),execution_identity=identity,bc_source=str(args.policy),
                resume_semantics='model/optimizer; fresh native episodes'),partial)
            partial.replace(dest)
            removed=prune_checkpoints(args.output,final_iteration=completed+args.ppo_updates)
            if removed:
                with (args.output/'checkpoint-retention.jsonl').open('a') as f:
                    f.write(json.dumps(dict(saved=dest.name,removed=removed))+'\n')
    ppo.save=save
    timeline=None
    if os.environ.get('REF2DEX_PROFILE_CUDA')=='1':
        def export_trace(prof):
            events=[dict(name=e.key,count=e.count,self_cpu_us=e.self_cpu_time_total,
                         self_device_us=getattr(e,'self_device_time_total',0)) for e in prof.key_averages()]
            (args.output/'cuda-profile.json').write_text(json.dumps(events,indent=2)+'\n')
            prof.export_chrome_trace(str(args.output/'cuda-trace.json.gz'))
        timeline=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=2,warmup=1,active=1,repeat=1),on_trace_ready=export_trace)
        timeline.start()
    profiler=None
    if os.environ.get('REF2DEX_PROFILE_PPO')=='1':
        import cProfile
        profiler=cProfile.Profile();profiler.enable()
    try:
        ppo.run(num_learning_iterations=args.ppo_updates,log_interval=args.save_every)
        if state['optimizer_steps']!=args.ppo_updates*20:raise ValueError('Official optimizer count changed')
        (args.output/'learning-audit.json').write_text(json.dumps(dict(status='PASS',**state),indent=2)+'\n')
        return [state]
    finally:
        if timeline is not None:timeline.stop()
        if profiler is not None:
            profiler.disable();profiler.dump_stats(str(args.output/'collection-profile.pstats'))
        ppo.writer.close()

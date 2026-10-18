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


def normalize_advantages(values):
    if not distributed_ready():return (values-values.mean())/(values.std()+1e-8)
    stats=torch.stack((values.double().sum(),values.double().square().sum(),values.new_tensor(values.numel()).double()))
    dist.all_reduce(stats)
    mean=stats[0]/stats[2]
    std=((stats[1]-stats[0]*mean)/(stats[2]-1)).clamp_min(0).sqrt()
    return ((values.double()-mean)/(std+1e-8)).to(values.dtype)


def prepare_official(root,output):
    """Same pinned source and dimension compatibility patch as official_unidex_ppo.py."""
    source=root/'third_party/UniDexGrasp2-official-ppo'
    target=output/'runtime-official';shutil.copytree(source,target)
    changes={}
    def edit(relative,replacements):
        p=target/relative;text=p.read_text();changes[relative]=[]
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
    (output/'official-compatibility.json').write_text(json.dumps(dict(upstream_commit='6c9e710b04769bf5d9765e3233820101cb3d718f',
        previous_runner='src/task/wam-rl-policy/tools/run/official_unidex_ppo.py',removed_vision_imports=removed,
        changes=changes,ppo_run_changed=False,gae_changed=False,reward_changed=False,
        distribution_changed=False,actor='self-trained scheduled BC mean with its frozen normalization; fully trainable',
        critic='official ELU architecture, fresh initialization',observation='565D current300 + world tau264 + valid1'),indent=2)+'\n')
    return target/'dexgrasp'


class ScheduledOfficialEnv:
    def __init__(self,base,snapshot,runtime):
        from gym import spaces
        self.base=base;self.task=base.task;self.snapshot=snapshot;self.runtime=runtime
        for name in ('num_envs','num_environments','num_states','num_actions','state_space','action_space','rl_device'):
            setattr(self,name,getattr(base,name))
        self.num_observations=565
        self.observation_space=spaces.Box(low=-np.inf,high=np.inf,shape=(565,),dtype=np.float32)
        self.ticks=np.zeros(self.num_envs,int);self.pending=np.zeros(self.num_envs,bool)
        self.inputs=None;self.reset_count=0;self.steps=0
        self.initial_z=np.zeros(self.num_envs);self.streak=np.zeros(self.num_envs,int);self.longest=np.zeros(self.num_envs,int)
        self.completed_episodes=0;self.successful_episodes=0;self.held_episodes=0
        self.workspace_failures=0;self.workspace_invalid=np.zeros(self.num_envs,bool)

    def augment(self,native,reset_ids):
        state=self.snapshot()
        if self.inputs is None:
            schedulers=[]
            for i in range(self.num_envs):schedulers.append(self.scheduler(state,i))
            self.inputs=BatchedScheduledInputs(self.runtime,schedulers)
        for i in reset_ids:
            self.inputs.histories[i].clear();self.inputs.schedulers[i]=self.scheduler(state,i);self.ticks[i]=0
            self.initial_z[i]=state['pose'][i,2,3];self.streak[i]=0;self.longest[i]=0
        lift=state['pose'][:,2,3]-self.initial_z
        near=np.linalg.norm(state['pose'][:,:3,3]-state['goal'],axis=-1)<=.05
        self.streak=np.where(near & (lift>=.1),self.streak+1,0);self.longest=np.maximum(self.longest,self.streak)
        current=native.detach() if self.inputs.pipeline in ('tau','inputs','all') else native.detach().cpu().numpy()
        result=self.inputs.observation(current,state['pose'],state['hand'],self.ticks/60)
        self.workspace_invalid=result['workspace_invalid']
        self.ticks+=1
        return torch.as_tensor(result['observation'],device=self.runtime.device)

    def scheduler(self,state,i):
        target=state['pose'][i].copy();target[:3,3]=state['goal'][i]
        return OrderedGoalScheduler([state['pose'][i],target],[0,2],self.runtime.geometry[i][2],interaction_points=self.runtime.geometry[i][0])

    def reset(self):
        native=self.base.reset();self.pending[:]=False
        return self.augment(native,range(self.num_envs))

    def step(self,actions,id=0):
        # Preserve the sampled canonical actions in upstream storage despite
        # the native task's in-place wrist transformation.
        native,reward,done,info=self.base.step(actions.clone(),id)
        reset_ids=np.flatnonzero(self.pending)
        result=self.augment(native,reset_ids)
        # Out-of-domain WAM input is an explicit terminal failure, never a
        # clipped point or a fabricated trajectory used for another action.
        invalid=torch.as_tensor(self.workspace_invalid,device=done.device)
        done=done.clone();done[invalid]=1
        self.task.reset_buf[invalid]=1
        self.workspace_failures+=int(self.workspace_invalid.sum())
        self.reset_count+=len(reset_ids);self.steps+=1
        self.pending=done.detach().cpu().numpy().astype(bool).copy()
        self.completed_episodes+=int(self.pending.sum())
        self.successful_episodes+=int((self.task.successes.detach().cpu().numpy()[self.pending]>0).sum())
        self.held_episodes+=int((self.longest[self.pending]>=60).sum())
        return result,reward,done,info

    def get_state(self):return self.base.get_state()


def run(task,snapshot,sim_step,actor,runtime,args):
    import yaml
    from types import SimpleNamespace
    from tasks.hand_base.vec_task import VecTaskPython
    root=Path(os.environ.get('REF2DEX_RUN_ROOT',Path(__file__).resolve().parents[5]))
    official=prepare_official(root,args.output);sys.path.insert(0,str(official))
    spec=importlib.util.spec_from_file_location('scheduled_official_process',official/'utils/process_sarl.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cfg=yaml.safe_load((official/'cfg/ppo/config.yaml').read_text())
    cfg['policy'].update(backbone_type='pn',freeze_backbone=False,obs_dim=565)
    cfg['seed']=args.seed;task.cfg['seed']=args.seed
    torch.manual_seed(args.seed)
    env=ScheduledOfficialEnv(VecTaskPython(task,'cuda:0',cfg['clip_observations'],cfg['clip_actions']),snapshot,runtime)
    ppo=module.process_ppo(SimpleNamespace(test=False,vision=False,model_dir=''),env,cfg,str(args.output/'models'))
    # Replace only the policy mean network; upstream act/evaluate/inference and
    # Gaussian convention remain identical in all three call paths.
    ppo.actor_critic.actor=actor.actor
    ppo.optimizer=torch.optim.Adam(ppo.actor_critic.parameters(),lr=cfg['learn']['optim_stepsize'])
    completed=0
    if args.rl_checkpoint:
        packet=torch.load(args.rl_checkpoint,map_location=runtime.device,weights_only=False)
        if packet.get('schema')!='ref2dex.scheduled-official-ppo.v1' or packet['conditioning']!=scheduled_conditioning_contract():raise ValueError('Official scheduled checkpoint required')
        ppo.actor_critic.load_state_dict(packet['model']);ppo.optimizer.load_state_dict(packet['optimizer'])
        completed=packet['iteration'];ppo.step_size=packet['step_size']
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
            effective_std=float(ppo.actor_critic.log_std.exp().square().mean()),
            peak_mib=torch.cuda.max_memory_allocated()/2**20)
        counts=reduce(torch.tensor([env.completed_episodes,env.successful_episodes,env.held_episodes],device=runtime.device,dtype=torch.float64),'sum')
        state.update(completed_episodes=int(counts[0]),native_success_episodes=int(counts[1]),held60_lift10cm_episodes=int(counts[2]))
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
                config=cfg,conditioning=scheduled_conditioning_contract(),bc_source=str(args.policy),
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

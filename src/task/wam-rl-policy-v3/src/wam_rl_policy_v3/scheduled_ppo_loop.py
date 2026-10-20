"""Bounded synchronous PPO pilot with real final-state bootstrap.

Each rollout is an explicit time truncation before native reset; histories are
cleared at each new rollout. This pilot does not claim long-horizon holding.
"""
import json,random,time,shutil
import numpy as np
import torch
from .batched_scheduled_inputs import BatchedScheduledInputs
from .goal_scheduler import OrderedGoalScheduler
from .ppo import Critic,reduce,distributed_ready
import torch.distributed as distributed
from .recipe_ppo import update_minibatches
from .control_contracts import scheduled_conditioning_contract


@torch.no_grad()
def parameter_sync_error(module):
    flat=torch.cat([p.detach().flatten() for p in module.parameters()])
    reference=flat.clone()
    if distributed_ready():distributed.broadcast(reference,src=0)
    return float(reduce((flat-reference).abs().max(),'max'))


def run(task,snapshot,sim_step,actor,runtime,args):
    device=runtime.device;n=args.envs;started=time.monotonic()
    torch.manual_seed(args.seed)
    critic=Critic(obs_dim=565,value_scale=100.).to(device)
    actor_opt=torch.optim.Adam(actor.parameters(),lr=1e-5)
    critic_opt=torch.optim.Adam(critic.parameters(),lr=1e-4)
    generator=torch.Generator(device=device).manual_seed(args.seed)
    offset=0
    if args.rl_checkpoint:
        packet=torch.load(args.rl_checkpoint,map_location=device,weights_only=False)
        if packet.get('schema')!='ref2dex.scheduled-shadow-ppo.v1' or packet.get('conditioning')!=scheduled_conditioning_contract():
            raise ValueError('Scheduled PPO checkpoint contract mismatch')
        actor.load_state_dict(packet['actor']);critic.load_state_dict(packet['critic'])
        actor_opt.load_state_dict(packet['actor_optimizer']);critic_opt.load_state_dict(packet['critic_optimizer'])
        offset=int(packet['update'])
    if distributed_ready():
        for module in (actor,critic):
            for parameter in module.parameters():distributed.broadcast(parameter.data,src=0)
    reports=[]
    for update in range(args.ppo_updates+2):
        stop=(args.output/'STOP').exists() or (args.output.parent/'STOP').exists()
        if bool(reduce(torch.tensor(float(stop),device=device),'max')):raise InterruptedError('STOP requested')
        if shutil.disk_usage(args.output).free<20*2**30:raise RuntimeError('20GiB disk floor')
        evaluation=update in (0,args.ppo_updates+1)
        seed=args.seed+args.rank*10000+(0 if evaluation else offset+update)
        for _ in range(2):
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            task.reset_buf[:]=1;task.step(torch.zeros((n,24),device=device),-1)
        if task.reset_buf.any():raise ValueError('reset pending at rollout start')
        first=snapshot();schedulers=[]
        for i in range(n):
            target=first['pose'][i].copy();target[:3,3]=first['goal'][i]
            schedulers.append(OrderedGoalScheduler([first['pose'][i],target],[0,2],runtime.geometry[i][2],interaction_points=runtime.geometry[i][0]))
        inputs=BatchedScheduledInputs(runtime,schedulers)
        rows=[];distances=[];heights=[]
        state=first
        def encode(state,tick):
            packed=inputs.observation(state['current'],state['pose'],state['hand'],np.full(n,tick/60))
            return torch.as_tensor(packed['observation'],device=device)
        for tick in range(args.ticks):
            with torch.no_grad():
                x=encode(state,tick);dist=actor.distribution(x)
                raw=dist.loc if evaluation else dist.sample()
                row=dict(observation=x.clone(),latent=raw.clone(),logp=dist.log_prob(raw).sum(-1),
                    mean=dist.loc.clone(),std=dist.scale.clone(),value=critic(x))
                canonical=raw.clamp(-1,1).clone()
                sim_step(canonical.clone())
                row.update(reward=task.rew_buf.clone(),done=task.reset_buf.bool().clone())
            if task.reset_buf.any():
                raise RuntimeError('Unexpected native terminal inside bounded rollout; refusing cross-episode history')
            rows.append(row);state=snapshot()
            distances.append(np.linalg.norm(state['pose'][:,:3,3]-first['goal'],axis=-1))
            heights.append(state['pose'][:,2,3]-first['pose'][:,2,3])
        with torch.no_grad():bootstrap=critic(encode(state,args.ticks)).detach()
        batch={k:torch.stack([r[k] for r in rows]) for k in rows[0]};batch['bootstrap']=bootstrap
        d=np.stack(distances);near=d<.05;longest=np.zeros(n,int);streak=np.zeros(n,int)
        for step in near:
            streak=np.where(step,streak+1,0);longest=np.maximum(longest,streak)
        report=dict(update=offset+update,rank=args.rank,world_size=args.world_size,evaluation=evaluation,seed=seed,transitions=n*args.ticks,
            final_distance_mean_m=float(d[-1].mean()),final_within_5cm=float(near[-1].mean()),
            longest_within_5cm_frames=longest.tolist(),max_lift_m=np.max(heights,axis=0).tolist(),
            reward_mean=float(batch['reward'].mean()),elapsed_s=time.monotonic()-started,
            peak_mib=torch.cuda.max_memory_allocated()/2**20)
        aggregate=reduce(torch.tensor([report['final_distance_mean_m'],report['final_within_5cm'],report['reward_mean']],device=device))
        report['global_metrics']=dict(zip(('final_distance_mean_m','final_within_5cm','reward_mean'),aggregate.cpu().tolist()))
        report['global_transitions']=n*args.ticks*args.world_size
        if not evaluation:
            report['ppo']=update_minibatches(actor,critic,batch,actor_opt,critic_opt,bc_weight=0.,
                target_kl=.01,epochs=2,minibatches=4,shuffle_generator=generator)
            if report['ppo']['actor_steps']==0:raise RuntimeError('No accepted actor update')
        checkpoint_due=not evaluation and (update==1 or update==args.ppo_updates or update%args.save_every==0)
        if checkpoint_due:
            report['actor_sync_error']=parameter_sync_error(actor)
            report['critic_sync_error']=parameter_sync_error(critic)
            if report['actor_sync_error'] or report['critic_sync_error']:raise RuntimeError('Distributed model divergence')
        reports.append(report)
        with (args.output/'progress.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
        print(json.dumps(report),flush=True)
        if evaluation:
            np.savez_compressed(args.output/('baseline.npz' if update==0 else 'final-evaluation.npz'),
                distances=d,initial_observation=first['current'],max_lift=np.max(heights,axis=0))
        # Save unique endpoints, never overwrite BC or an earlier checkpoint.
        if checkpoint_due and args.rank==0:
            torch.save(dict(schema='ref2dex.scheduled-shadow-ppo.v1',actor=actor.state_dict(),critic=critic.state_dict(),
                actor_optimizer=actor_opt.state_dict(),critic_optimizer=critic_opt.state_dict(),
                update=offset+update,world_size=args.world_size,resume_semantics='model and optimizer; fresh measured episodes',bc_source=str(args.policy),conditioning=scheduled_conditioning_contract()),
                args.output/('ppo-update-%d.pt'%(offset+update)))
    return reports

"""Native Shadow BC/PPO loops called after the pinned simulator is initialized."""
import json
from pathlib import Path
import random
import shutil
import subprocess
import time
import numpy as np
import torch
from .ppo import SCHEMA,Actor,Critic,observation,apply_residual,update
from shadow_finetune.execution import Controller
from shadow_finetune.metrics import native_report


def write(path,value):
    path=Path(path);part=path.with_suffix(path.suffix+'.part')
    part.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');part.replace(path)


def run(task,snapshot,env_objects,teacher,args):
    out=args.output;started=time.monotonic()
    rng=(torch.get_rng_state(),torch.cuda.get_rng_state(),np.random.get_state(),random.getstate())
    bc=Controller(args.checkpoint,env_objects,teacher,args.cache)
    bc.model.requires_grad_(False)
    torch.manual_seed(231)
    actor=Actor().cuda();critic=Critic().cuda()
    if args.rl_checkpoint:
        saved=torch.load(args.rl_checkpoint,map_location='cpu',weights_only=False)
        if saved.get('schema')!=SCHEMA or saved['base_checkpoint']!=str(args.checkpoint.resolve()):
            raise ValueError('Matching frozen Shadow BC reference required')
        actor.load_state_dict(saved['actor'],strict=True)
    torch.set_rng_state(rng[0]);torch.cuda.set_rng_state(rng[1]);np.random.set_state(rng[2]);random.setstate(rng[3])
    before={k:v.clone() for k,v in bc.model.state_dict().items()}
    actor_optimizer=torch.optim.Adam(actor.parameters(),lr=1e-4)
    critic_optimizer=torch.optim.Adam(critic.parameters(),lr=3e-4)
    generator=torch.Generator(device='cuda');generator.manual_seed(232)
    training=args.mode=='train'
    if training and any(row['split']!='train' for row in env_objects):raise ValueError('DEV objects cannot enter PPO')
    manifest=dict(schema=SCHEMA,status='RUNNING',mode=args.mode,pid=__import__('os').getpid(),
        experiment_id='P-20261016-shadow-bc-ppo',run_id=out.name,git_commit=args.code_commit,
        base_checkpoint=str(args.checkpoint.resolve()),rl_checkpoint=str(args.rl_checkpoint) if args.rl_checkpoint else None,
        frozen_wam=True,frozen_bc=True,trainable='new327D actor/critic only; no WAM token adapter in Stage0',
        alpha=args.alpha,exploration='tanh latent Normal; initial log_std=-2.5; independent CUDA RNG232',
        policy_seed=231,reset_seed=args.seed,envs=args.envs,updates=args.updates,
        reward='unchanged official goal_cond=False native task reward',gamma=.99,gae_lambda=.95,
        bc_weight=.1,actor_lr=1e-4,critic_lr=3e-4,ppo_epochs=4,target_joint_kl=.02,
        max_seconds=args.seconds,artifact_budget_bytes=2*2**30,
        controls_per_episode=int(task.max_episode_length)-1,objects=env_objects,
        selection='final update, no DEV checkpoint selection',
        evaluation_metric='sticky official5cm; terminal5cm;60 consecutive60Hz states within5cm and lift>=10cm (position proxy)')
    write(out/'manifest.json',manifest)
    curves=[]
    for iteration in range(1,(args.updates if training else 1)+1):
        task.reset_buf[:]=1
        task.step(torch.zeros((args.envs,24),device='cuda'),-1)
        frames=[snapshot()];commands=[];bases=[];latents=[];intendeds=[];applied=[];forces=[];torques=[];targets=[]
        rewards=[];storage={k:[] for k in ('observation','latent','mean','std','logp','value','reward','done')}
        for tick in range(1,int(task.max_episode_length)):
            if time.monotonic()-started>args.seconds:raise TimeoutError('Shadow PPO wall budget')
            if tick>1 and task.reset_buf.any():raise ValueError('Unexpected interior reset; reject unspliced episode')
            snap=frames[-1]
            with torch.no_grad():
                base=bc.act(snap,task.obs_buf,tick)
                if bc.invalid:raise ValueError('BC history outside admitted workspace: '+str(sorted(bc.invalid)))
                relative_goal=torch.as_tensor(snap['goal']-snap['object'][:,:3],device='cuda')
                obs=observation(task.obs_buf,relative_goal,base)
                dist=actor.distribution(obs)
                latent=(dist.loc+dist.scale*torch.randn(dist.loc.shape,device='cuda',generator=generator)) if training else dist.loc
                if args.mode=='bc':latent=torch.zeros_like(latent)
                action,intended,delta=apply_residual(base,latent,args.alpha)
                if args.mode in ('bc','zero') and not torch.equal(action,base):raise ValueError('Zero residual changed canonical BC')
                value=critic(obs);logp=dist.log_prob(latent).sum(-1)
            commands.append(action.cpu().numpy().copy());bases.append(base.cpu().numpy().copy())
            latents.append(latent.cpu().numpy().copy());intendeds.append(intended.cpu().numpy().copy())
            task.pre_physics_step(action.clone())
            # These tensors are read before post's observation code mutates action views.
            applied.append(task.actions.cpu().numpy().copy());forces.append(task.apply_forces.cpu().numpy().copy())
            torques.append(task.apply_torque.cpu().numpy().copy());targets.append(task.cur_targets.cpu().numpy().copy())
            for _ in range(task.control_freq_inv):
                task.gym.simulate(task.sim);task.gym.fetch_results(task.sim,True)
            task.post_physics_step()
            reward=task.rew_buf.detach().clone();done=task.reset_buf.bool().clone()
            if not torch.isfinite(reward).all():raise FloatingPointError('Nonfinite native reward')
            rewards.append(reward.cpu().numpy())
            for key,val in dict(observation=obs,latent=latent,mean=dist.loc,std=dist.scale,
                                logp=logp,value=value,reward=reward,done=done).items():storage[key].append(val.detach())
            frames.append(snapshot())
            if tick==1 or tick%50==0:
                write(out/'progress.json',dict(update=iteration,tick=tick,elapsed_s=time.monotonic()-started,
                    completed_env_steps=((iteration-1)*(int(task.max_episode_length)-1)+tick)*args.envs,
                    gpu=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()))
        if not task.reset_buf.bool().all():raise ValueError('Full fixed horizon must terminate all environments')
        data={key:np.stack([frame[key] for frame in frames]) for key in frames[0]}
        data.update(action=np.stack(commands),base=np.stack(bases),latent=np.stack(latents),intended=np.stack(intendeds),
            applied=np.stack(applied),force=np.stack(forces),torque=np.stack(torques),target=np.stack(targets),reward=np.stack(rewards),
            time=np.arange(1,len(frames)+1)/60.)
        if any(not np.isfinite(v).all() for v in data.values()):raise FloatingPointError('Nonfinite physics trace')
        report=native_report(data,env_objects)
        stats=dict(update=iteration,env_steps=iteration*(int(task.max_episode_length)-1)*args.envs,
            elapsed_s=time.monotonic()-started,native=report,
            mean_episode_return=float(data['reward'].sum(0).mean()),
            applied_residual_rms=float(np.sqrt(np.mean((data['action']-data['base'])**2))),
            clip_fraction=float(np.mean(np.abs(data['intended']-data['action'])>1e-6)))
        if training:
            batch={k:torch.stack(v) for k,v in storage.items()};batch['bootstrap']=torch.zeros(args.envs,device='cuda')
            stats['ppo']=update(actor,critic,batch,actor_optimizer,critic_optimizer)
        curves.append(stats);write(out/'curve.json',curves)
        np.savez_compressed(out/'last-episode.npz',**data)
        if iteration==1:np.savez_compressed(out/'first-episode.npz',**data)
        if training:
            checkpoint=dict(schema=SCHEMA,update=iteration,actor=actor.state_dict(),critic=critic.state_dict(),
                actor_optimizer=actor_optimizer.state_dict(),critic_optimizer=critic_optimizer.state_dict(),
                base_checkpoint=str(args.checkpoint.resolve()),config=manifest,torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(),exploration_rng=generator.get_state(),
                numpy_rng=np.random.get_state(),python_rng=random.getstate(),
                resume_scope='episode boundary; simulator must reset (not exact physics continuation)')
            partial=out/'latest.pt.part';torch.save(checkpoint,partial);partial.replace(out/'latest.pt')
        write(out/'progress.json',dict(stats,eta_s=(time.monotonic()-started)/iteration*(args.updates-iteration) if training else 0))
        print(json.dumps(stats),flush=True)
        if shutil.disk_usage(out).free<20*2**30:raise RuntimeError('Disk reserve')
    unchanged=all(torch.equal(v,bc.model.state_dict()[k]) for k,v in before.items())
    if not unchanged:raise ValueError('Frozen BC parameters/buffers changed')
    if any(p.grad is not None for p in bc.model.parameters()):raise ValueError('Gradient reached frozen BC')
    write(out/'freeze-audit.json',dict(status='PASS',exact_state_dict_unchanged=unchanged,gradients_absent=True,
        fields=len(before),optimizer_contains_only_new_actor_and_critic=True))
    manifest.update(status='COMPLETED',elapsed_s=time.monotonic()-started)
    write(out/'manifest.json',manifest);write(out/'result.json',dict(status='COMPLETED',final=curves[-1],updates=len(curves)))

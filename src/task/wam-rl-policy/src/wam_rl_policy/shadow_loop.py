"""Native Shadow BC/PPO loops called after the pinned simulator is initialized."""
import json
from pathlib import Path
import random
import shutil
import subprocess
import time
from datetime import timedelta
import numpy as np
import torch
from .ppo import SCHEMA,Actor,Critic,observation,apply_policy_action,update_rollout
from shadow_finetune.execution import Controller
from shadow_finetune.metrics import native_report
from .bc_mlp import MLPController
from .fast_bc import FastController
from .rewards import position_hold_bonus
from .wam_features import FeatureController


def write(path,value):
    path=Path(path);part=path.with_suffix(path.suffix+'.part')
    part.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');part.replace(path)


def run(task,snapshot,env_objects,teacher,args):
    # Existing bounded launchers may be initializing assets while optional diagnostics are added.
    for name,default in [('initial_log_std',-2.5),('synchronize_warm_step',False),('capture_warm_state',False),('retain_reset_indices',False),('single_reset_target_setter',False),('force_zero_residual',False),('outside_support_zero',False),('observation_kind','native'),('rollout_steps',199),('network_kind','small'),('action_kind','residual'),('reward_kind','native'),('ppo_epochs',4),('ppo_minibatches',1),('wam_feature_checkpoint',None)]:
        if not hasattr(args,name):setattr(args,name,default)
    out=args.output;started=time.monotonic()
    if args.world_size>1:
        torch.distributed.init_process_group('nccl',init_method=args.rendezvous,rank=args.rank,world_size=args.world_size,
            timeout=timedelta(minutes=10))
    rng=(torch.get_rng_state(),torch.cuda.get_rng_state(),np.random.get_state(),random.getstate())
    bc_type=(FastController if args.experiment_id=='P-20261016-shadow-scale-continuous' else Controller) if args.bc_kind=='wam' else MLPController
    bc=bc_type(args.checkpoint,env_objects,teacher,args.cache)
    bc.allow_invalid_zero=args.outside_support_zero
    bc.model.requires_grad_(False)
    saved=None;offset=0
    if args.rl_checkpoint:
        saved=torch.load(args.rl_checkpoint,map_location='cpu',weights_only=False)
        if saved.get('schema')!=SCHEMA or saved['base_checkpoint']!=str(args.checkpoint.resolve()):
            raise ValueError('Matching frozen Shadow BC reference required')
        saved_kind=saved.get('config',{}).get('observation_kind','native')
        if args.mode=='train' and saved_kind!=args.observation_kind:raise ValueError('Resume must retain observation representation')
        args.observation_kind=saved_kind
        saved_steps=saved.get('config',{}).get('rollout_steps',199)
        if args.mode=='train' and saved_steps!=args.rollout_steps:raise ValueError('Resume must retain PPO rollout length')
        args.rollout_steps=saved_steps
        for name,default in [('network_kind','small'),('action_kind','residual'),('reward_kind','native'),('ppo_epochs',4),('ppo_minibatches',1)]:
            value=saved.get('config',{}).get(name,default)
            if args.mode=='train' and getattr(args,name)!=value:raise ValueError('Resume must retain '+name)
            setattr(args,name,value)
        feature_path=saved.get('config',{}).get('wam_feature_checkpoint')
        if args.mode=='train' and feature_path!=(str(args.wam_feature_checkpoint) if args.wam_feature_checkpoint else None):raise ValueError('Resume must retain independent WAM feature source')
        args.wam_feature_checkpoint=Path(feature_path) if feature_path else None
        args.initial_log_std=saved.get('config',{}).get('initial_log_std',args.initial_log_std)
        if saved.get('config',{}).get('outside_support_policy')=='persistent zero BC reference until episode reset':
            args.outside_support_zero=True;bc.allow_invalid_zero=True
    use_wam=args.observation_kind.startswith('wam-')
    if args.observation_kind in ('normalized','bc-hidden') and args.bc_kind!='mlp':raise ValueError('Current BC features require the MLP reference')
    if args.wam_feature_checkpoint and not use_wam:raise ValueError('Independent WAM source requires an explicit z condition')
    feature_bc=FeatureController(args.wam_feature_checkpoint,env_objects,args.cache) if args.wam_feature_checkpoint else bc
    if use_wam and not isinstance(feature_bc,FastController):raise ValueError('Direct WAM summaries require a WAM feature source')
    feature_bc.expose_wam_features=use_wam
    feature_bc.allow_invalid_zero=args.outside_support_zero
    obs_dim=2631 if use_wam else (327 if args.observation_kind=='native' else 539)
    permutation=list(range(len(env_objects)))
    if args.observation_kind=='wam-shuffled':
        order=sorted(permutation,key=lambda i:env_objects[i]['object_code'])
        counts={row['object_code']:sum(other['object_code']==row['object_code'] for other in env_objects) for row in env_objects}
        shift=max(counts.values());donors=order[shift:]+order[:shift]
        for index,donor in zip(order,donors):permutation[index]=donor
        if any(env_objects[i]['object_code']==env_objects[j]['object_code'] for i,j in enumerate(permutation)):raise ValueError('Shuffle must use a different object')
    donor_indices=torch.as_tensor(permutation,device='cuda')
    torch.manual_seed(args.policy_seed)
    actor=Actor(args.initial_log_std,obs_dim=obs_dim,network_kind=args.network_kind,action_kind=args.action_kind).cuda()
    critic=Critic(obs_dim=obs_dim,network_kind=args.network_kind).cuda()
    if saved:actor.load_state_dict(saved['actor'],strict=True)
    torch.set_rng_state(rng[0]);torch.cuda.set_rng_state(rng[1]);np.random.set_state(rng[2]);random.setstate(rng[3])
    before={k:v.clone() for k,v in bc.model.state_dict().items()}
    before_features={k:v.clone() for k,v in feature_bc.model.state_dict().items()} if feature_bc is not bc else None
    actor_optimizer=torch.optim.Adam(actor.parameters(),lr=1e-4)
    critic_optimizer=torch.optim.Adam(critic.parameters(),lr=3e-4)
    generator=torch.Generator(device='cuda');generator.manual_seed(args.exploration_seed+args.rank)
    optimizer_generator=torch.Generator(device='cuda');optimizer_generator.manual_seed(args.exploration_seed+1000000+args.rank)
    training=args.mode=='train'
    if saved and training:
        critic.load_state_dict(saved['critic'],strict=True)
        actor_optimizer.load_state_dict(saved['actor_optimizer']);critic_optimizer.load_state_dict(saved['critic_optimizer'])
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng']);generator.set_state(saved['exploration_rng'])
        if 'optimizer_shuffle_rng' in saved:optimizer_generator.set_state(saved['optimizer_shuffle_rng'])
        np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng']);offset=int(saved['update'])
        if offset>=args.updates:raise ValueError('Resume target update must exceed checkpoint update')
    if training and any(row['split']!='train' for row in env_objects):raise ValueError('DEV objects cannot enter PPO')
    manifest=dict(schema=SCHEMA,status='RUNNING',mode=args.mode,pid=__import__('os').getpid(),
        experiment_id=args.experiment_id,run_id=out.name,git_commit=args.code_commit,bc_kind=args.bc_kind,
        base_checkpoint=str(args.checkpoint.resolve()),rl_checkpoint=str(args.rl_checkpoint) if args.rl_checkpoint else None,
        uses_wam=args.bc_kind=='wam' or use_wam,frozen_wam=True if args.bc_kind=='wam' or use_wam else None,frozen_bc=True,
        wam_feature_checkpoint=str(args.wam_feature_checkpoint) if args.wam_feature_checkpoint else None,
        wam_feature_schema=getattr(feature_bc,'schema',None),wam_feature_step=getattr(feature_bc,'source_step',None),
        trainable='new%dD actor/critic only; frozen BC features; no WAM token adapter'%obs_dim,
        observation_kind=args.observation_kind,observation_dim=obs_dim,
        network_kind=args.network_kind,action_kind=args.action_kind,
        wam_z='frozen masked mean per scene/hand/known-goal type; 3x768; original current300 retained' if use_wam else None,
        wam_donor_indices=permutation if use_wam else None,
        direct_wam_condition=args.observation_kind if use_wam else None,
        representation=('native current300 + per-type LayerNorm WAM2304 + explicit goal3/reference24' if use_wam else
            ('native clamp current300' if obs_dim==327 else '512D current-only frozen prefix, per-state LayerNorm; explicit goal3/reference24 unchanged')),
        alpha=args.alpha if args.action_kind=='residual' else None,
        exploration=('tanh latent Normal residual' if args.action_kind=='residual' else 'direct canonical Normal around BC+learned mean; native clip only')+'; independent per-rank CUDA RNG',initial_log_std=args.initial_log_std,
        policy_seed=args.policy_seed,exploration_seed=args.exploration_seed+args.rank,reset_seed=args.seed,envs=args.envs,updates=args.updates,
        optimizer_shuffle_seed=args.exploration_seed+1000000+args.rank,optimizer_shuffle_rng='independent saved generator; never consumes simulator/global RNG',
        rank=args.rank,world_size=args.world_size,global_envs=args.envs*args.world_size,resume_update=offset,
        synchronize_warm_step=args.synchronize_warm_step,capture_warm_state=args.capture_warm_state,
        retain_reset_indices=args.retain_reset_indices,single_reset_target_setter=args.single_reset_target_setter,
        force_zero_residual=args.force_zero_residual,
        outside_support_policy='persistent zero BC reference until episode reset' if args.outside_support_zero else 'reject run',
        reward_kind=args.reward_kind,reward='official goal_cond=False native reward'+(' + 1/tick low-speed position hold: goal<=5cm,lift>=10cm,speed<=0.1m/s' if args.reward_kind=='hold' else ''),gamma=.99,gae_lambda=.95,
        bc_weight=.1,actor_lr=1e-4,critic_lr=3e-4,ppo_epochs=args.ppo_epochs,ppo_minibatches=args.ppo_minibatches,target_joint_kl=.02,
        rollout_steps=args.rollout_steps,interior_bootstrap='actual next-state critic before optimizer update; no simulator reset',
        max_seconds=args.seconds,artifact_budget_bytes=2*2**30,
        controls_per_episode=int(task.max_episode_length)-1,objects=env_objects,
        selection='final update, no DEV checkpoint selection',
        evaluation_metric='sticky official5cm; terminal5cm;60 consecutive60Hz states within5cm and lift>=10cm (position proxy)')
    write(out/'manifest.json',manifest)
    curves=[]
    for iteration in range(offset+1,(args.updates if training else 1)+1):
        if args.retain_reset_indices:
            torch.cuda.synchronize();task._retained_gpu_indices=[]
        task.reset_buf[:]=1
        if args.synchronize_warm_step:
            task.id=-1
            if task.dr_randomizations:raise ValueError('Synchronized warm probe requires native no-domain-randomization config')
            task.pre_physics_step(torch.zeros((args.envs,24),device='cuda'))
            torch.cuda.synchronize()
            if args.capture_warm_state and iteration==offset+1:
                np.savez(out/'warm-before.npz',root_state=task.root_state_tensor.cpu().numpy(),dof=task.dof_state.cpu().numpy(),
                    prev_targets=task.prev_targets.cpu().numpy(),cur_targets=task.cur_targets.cpu().numpy(),
                    z_theta=task.z_theta.cpu().numpy(),force=task.apply_forces.cpu().numpy(),torque=task.apply_torque.cpu().numpy())
            for _ in range(task.control_freq_inv):
                task.gym.simulate(task.sim);task.gym.fetch_results(task.sim,True)
            torch.cuda.synchronize();task.post_physics_step()
        else:task.step(torch.zeros((args.envs,24),device='cuda'),-1)
        frames=[snapshot()];commands=[];bases=[];latents=[];intendeds=[];applied=[];forces=[];torques=[];targets=[]
        rewards=[];native_rewards=[];hold_bonuses=[];storage={k:[] for k in ('observation','latent','mean','std','logp','value','reward','done')}
        initial_z=torch.as_tensor(frames[0]['object'][:,2],device='cuda')
        execution_error=dict(canonical_to_world=0.,force=0.,torque=0.,finger_target=0.);fallback_env_ticks=0;feature_fallback_ticks=0;ppo_segments=[]
        for tick in range(1,int(task.max_episode_length)):
            if time.monotonic()-started>args.seconds:raise TimeoutError('Shadow PPO wall budget')
            if tick>1 and task.reset_buf.any():raise ValueError('Unexpected interior reset; reject unspliced episode')
            snap=frames[-1]
            with torch.no_grad():
                base=bc.act(snap,task.obs_buf,tick)
                if use_wam and feature_bc is not bc:
                    feature_bc.act(snap,task.obs_buf,tick);feature_fallback_ticks+=len(feature_bc.invalid)
                if bc.invalid:
                    if not args.outside_support_zero:raise ValueError('BC history outside admitted workspace: '+str(sorted(bc.invalid)))
                    fallback_env_ticks+=len(bc.invalid)
                    if not (out/'first-bc-support-fallback.npz').exists():
                        np.savez_compressed(out/'first-bc-support-fallback.npz',**snap)
                        write(out/'first-bc-support-fallback.json',dict(update=iteration,tick=tick,envs=sorted(bc.invalid),
                            objects=[env_objects[i] for i in sorted(bc.invalid)]))
                relative_goal=torch.as_tensor(snap['goal']-snap['object'][:,:3],device='cuda')
                features=None
                if args.observation_kind=='bc-hidden':features=bc.last_features
                elif args.observation_kind=='normalized':
                    normalized=(task.obs_buf.clamp(-5,5)-bc.model.x_mean)/bc.model.x_scale
                    features=torch.nn.functional.pad(normalized,(0,212))
                wam_features=None
                if use_wam:
                    wam_features=feature_bc.last_wam_features
                    if args.observation_kind=='wam-none':wam_features=torch.zeros_like(wam_features)
                    elif args.observation_kind=='wam-shuffled':wam_features=wam_features[donor_indices]
                obs=observation(task.obs_buf,relative_goal,base,features,wam_features)
            if training and len(storage['reward'])>=args.rollout_steps:
                ppo_segments.append(update_rollout(actor,critic,storage,actor_optimizer,critic_optimizer,obs,epochs=args.ppo_epochs,minibatches=args.ppo_minibatches,shuffle_generator=optimizer_generator))
            with torch.no_grad():
                dist=actor.distribution(obs)
                latent=(dist.loc+dist.scale*torch.randn(dist.loc.shape,device='cuda',generator=generator)) if training else dist.loc
                if args.mode=='bc' or args.force_zero_residual:latent=base if args.action_kind=='reference-gaussian' else torch.zeros_like(latent)
                action,intended,delta=apply_policy_action(base,latent,args.action_kind)
                if (args.mode in ('bc','zero') or args.force_zero_residual) and not torch.equal(action,base):raise ValueError('Zero residual changed canonical BC')
                value=critic(obs);logp=dist.log_prob(latent).sum(-1)
            commands.append(action.cpu().numpy().copy());bases.append(base.cpu().numpy().copy())
            latents.append(latent.cpu().numpy().copy());intendeds.append(intended.cpu().numpy().copy())
            # Independent analytic yaw rotation and native finger target contract.
            theta=task.z_theta if task.repose_z else torch.zeros_like(task.z_theta)
            c,s=theta.cos(),theta.sin()
            expected=action.clone()
            for lo in (0,3):
                x=action[:,lo:lo+3]
                expected[:,lo:lo+3]=torch.stack((c*x[:,0]-s*x[:,1],s*x[:,0]+c*x[:,1],x[:,2]),-1)
            ids=task.actuated_dof_indices
            lower,upper=task.shadow_hand_dof_lower_limits[ids],task.shadow_hand_dof_upper_limits[ids]
            target=.5*(action[:,6:]+1)*(upper-lower)+lower
            target=task.act_moving_average*target+(1-task.act_moving_average)*task.prev_targets[:,ids]
            target=torch.maximum(torch.minimum(target,upper),lower)
            task.pre_physics_step(action.clone())
            expected_force=expected[:,:3]*task.dt*task.transition_scale*100000
            expected_torque=expected[:,3:6]*task.dt*task.orientation_scale*1000
            checks=dict(canonical_to_world=(task.actions,expected),force=(task.apply_forces[:,1],expected_force),
                torque=(task.apply_torque[:,1],expected_torque),finger_target=(task.cur_targets[:,ids],target))
            for key,(actual,wanted) in checks.items():
                execution_error[key]=max(execution_error[key],float((actual-wanted).abs().max()))
                torch.testing.assert_close(actual,wanted,rtol=2e-6,atol=1e-4 if key=='force' else 1e-6)
            # These tensors are read before post's observation code mutates action views.
            applied.append(task.actions.cpu().numpy().copy());forces.append(task.apply_forces.cpu().numpy().copy())
            torques.append(task.apply_torque.cpu().numpy().copy());targets.append(task.cur_targets.cpu().numpy().copy())
            for _ in range(task.control_freq_inv):
                task.gym.simulate(task.sim);task.gym.fetch_results(task.sim,True)
            task.post_physics_step()
            native_reward=task.rew_buf.detach().clone();done=task.reset_buf.bool().clone()
            hold_bonus=position_hold_bonus(task.object_pos,task.goal_pos,task.object_linvel,initial_z)
            reward=native_reward+hold_bonus if args.reward_kind=='hold' else native_reward
            if not torch.isfinite(reward).all():raise FloatingPointError('Nonfinite native reward')
            rewards.append(reward.cpu().numpy())
            native_rewards.append(native_reward.cpu().numpy());hold_bonuses.append(hold_bonus.cpu().numpy())
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
            native_reward=np.stack(native_rewards),position_hold_bonus=np.stack(hold_bonuses),
            time=np.arange(1,len(frames)+1)/60.)
        if any(not np.isfinite(v).all() for v in data.values()):raise FloatingPointError('Nonfinite physics trace')
        report=native_report(data,env_objects)
        stats=dict(update=iteration,env_steps=iteration*(int(task.max_episode_length)-1)*args.envs,
            global_env_steps=iteration*(int(task.max_episode_length)-1)*args.envs*args.world_size,
            elapsed_s=time.monotonic()-started,native=report,
            mean_episode_return=float(data['reward'].sum(0).mean()),
            mean_native_episode_return=float(data['native_reward'].sum(0).mean()),mean_position_hold_bonus=float(data['position_hold_bonus'].sum(0).mean()),
            native_execution_error=execution_error,
            bc_support_fallback_envs=sorted(bc.invalid),bc_support_fallback_env_ticks=fallback_env_ticks,
            wam_feature_fallback_envs=sorted(feature_bc.invalid) if use_wam else [],wam_feature_fallback_env_ticks=feature_fallback_ticks,
            applied_residual_rms=float(np.sqrt(np.mean((data['action']-data['base'])**2))),
            clip_fraction=float(np.mean(np.abs(data['intended']-data['action'])>1e-6)))
        if training:
            ppo_segments.append(update_rollout(actor,critic,storage,actor_optimizer,critic_optimizer,epochs=args.ppo_epochs,minibatches=args.ppo_minibatches,shuffle_generator=optimizer_generator))
            stats['ppo']=ppo_segments[-1]
            stats['ppo_update_batches']=len(ppo_segments)
            stats['actor_optimizer_steps_this_episode']=sum(segment['actor_steps'] for segment in ppo_segments)
            if len(ppo_segments)>1:stats['ppo_segments']=ppo_segments
        curves.append(stats);write(out/'curve.json',curves)
        if not training or iteration%args.trace_every==0 or iteration==args.updates:np.savez_compressed(out/'last-episode.npz',**data)
        if iteration==1:np.savez_compressed(out/'first-episode.npz',**data)
        if training:
            checkpoint=dict(schema=SCHEMA,update=iteration,actor=actor.state_dict(),critic=critic.state_dict(),
                actor_optimizer=actor_optimizer.state_dict(),critic_optimizer=critic_optimizer.state_dict(),
                base_checkpoint=str(args.checkpoint.resolve()),config=manifest,torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(),exploration_rng=generator.get_state(),
                optimizer_shuffle_rng=optimizer_generator.get_state(),
                numpy_rng=np.random.get_state(),python_rng=random.getstate(),
                resume_scope='episode boundary; simulator must reset (not exact physics continuation)')
            partial=out/'latest.pt.part';torch.save(checkpoint,partial);partial.replace(out/'latest.pt')
            if iteration in (50,100,200):torch.save(checkpoint,out/('update%d.pt'%iteration))
        write(out/'progress.json',dict(stats,eta_s=(time.monotonic()-started)/(iteration-offset)*(args.updates-iteration) if training else 0))
        print(json.dumps(stats),flush=True)
        if shutil.disk_usage(out).free<20*2**30:raise RuntimeError('Disk reserve')
    unchanged=all(torch.equal(v,bc.model.state_dict()[k]) for k,v in before.items())
    if not unchanged:raise ValueError('Frozen BC parameters/buffers changed')
    if any(p.grad is not None for p in bc.model.parameters()):raise ValueError('Gradient reached frozen BC')
    features_unchanged=True
    if before_features is not None:
        features_unchanged=all(torch.equal(value,feature_bc.model.state_dict()[key]) for key,value in before_features.items())
        if not features_unchanged or any(p.grad is not None for p in feature_bc.model.parameters()):raise ValueError('Independent frozen WAM changed')
    write(out/'freeze-audit.json',dict(status='PASS',exact_state_dict_unchanged=unchanged,gradients_absent=True,
        fields=len(before),optimizer_contains_only_new_actor_and_critic=True))
    if before_features is not None:write(out/'feature-freeze-audit.json',dict(status='PASS',exact_state_dict_unchanged=features_unchanged,
        gradients_absent=True,fields=len(before_features),checkpoint=str(args.wam_feature_checkpoint)))
    manifest.update(status='COMPLETED',elapsed_s=time.monotonic()-started)
    write(out/'manifest.json',manifest);write(out/'result.json',dict(status='COMPLETED',final=curves[-1],updates=len(curves)))
    if args.world_size>1:torch.distributed.destroy_process_group()

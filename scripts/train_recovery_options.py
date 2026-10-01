#!/usr/bin/env python3
"""Native categorical PPO; frozen expert commands and physical Cm forecasts."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from collect_contact_consequences import build_player
from run_paired_evaluator_resolution import sha


def recovery_player(original,args,torch,gymtorch):
    from src.task.CmResidual.recovery_option_policy import FrozenRecoveryPhysics,RecoveryOptionPolicy,policy_inputs,ppo_loss,supported_height_reward
    from src.task.CmResidual.physical_value_live import contacts
    from src.task.CmResidual.physical_value_contract import HoldTracker,private_initialization
    from src.task.CmResidual.paired_evaluation import fingerprint
    parent=build_player(original,args,torch,gymtorch)

    class RecoveryPlayer(parent):
        @torch.no_grad()
        def run(self):
            begin=time.monotonic();torch.set_num_threads(2)
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
            task=self.env.task;n=task.num_envs
            task._hybrid_init_prob=0.;task._enable_early_termination=False;task._adaptive_kappa_enabled=False
            if n!=96 or self.is_rnn or abs(task.dt-1/30)>1e-8:raise ValueError('native96env/stateless/30Hz required')
            physics=FrozenRecoveryPhysics(args.trajectory,self.device)
            learnable_guide=getattr(args,'learnable_guide',False);physical_reward=getattr(args,'physical_reward',False)
            with private_initialization(9381):policy=RecoveryOptionPolicy(learnable_guide=learnable_guide).to(self.device)
            initial=fingerprint(policy.state_dict())
            if args.policy:
                saved=torch.load(args.policy,map_location=self.device,weights_only=False)
                if saved['cm_on']!=args.cm_on or saved['trajectory_sha256']!=sha(args.trajectory):raise ValueError('policy mode/input drift')
                if saved.get('learnable_guide',False)!=learnable_guide:raise ValueError('guide implementation drift')
                policy.load_state_dict(saved['policy'])
            optimizer=torch.optim.Adam(policy.parameters(),lr=3e-4)
            generator=torch.Generator(device=self.device).manual_seed(args.assignment_seed)
            frozen=fingerprint([m.state_dict() for m in physics.models]+[dict(model=m.state_dict(),rms=r.state_dict()) for m,r in self.frozen_experts])
            reports=[];all_buffers=[];updates=0;latencies=[];cold_states=[];episode_traces=[]
            ids=torch.arange(n,device=self.device);empty=ids[:0]
            for rollout in range(args.rollouts):
                observation=self.env_reset(ids)
                if self.get_batch_size(observation['obs'],1)!=n:raise ValueError('native batch')
                motion=task.data_id.clone();start=task.start_times.clone()
                rest=task.hoi_refs[task.data_id,task.ref_index,0,108].clone()
                cold_states.append(dict(root=task._root_states.cpu().clone(),dof=task._dof_state.cpu().clone(),
                                        rigid_body=task._rigid_body_state.cpu().clone(),observation=observation['obs'].cpu().clone(),
                                        target=task._target_states.cpu().clone(),
                                        contact_forces=task._contact_forces.cpu().clone(),object_forces=task._tar_contact_forces.cpu().clone(),
                                        motion=motion.cpu(),start=start.cpu(),rest=rest.cpu(),
                                        cpu_rng=torch.get_rng_state().clone(),option_rng=generator.get_state().cpu(),cuda_rng=torch.cuda.get_rng_state().cpu() if torch.cuda.is_available() else None))
                tracker=HoldTracker(n,self.device);tracker.reset(ids,rest)
                history=torch.zeros(n,10,69,device=self.device);previous=torch.zeros(n,18,device=self.device)
                contact_run=torch.zeros(n,dtype=torch.long,device=self.device);ended=torch.zeros(n,dtype=torch.bool,device=self.device)
                elapsed=torch.full_like(contact_run,-1);cooldown=torch.zeros_like(contact_run);cached=torch.zeros(n,18,device=self.device)
                horizon=int(task.max_episode_length.max())+5
                reward=torch.zeros(horizon,n,device=self.device);native_rewards=torch.zeros_like(reward);episode_steps=torch.zeros_like(contact_run)
                object_trace=torch.zeros(horizon,n,13,device=self.device);contact_trace=torch.zeros(horizon,n,2,dtype=torch.bool,device=self.device)
                action_trace=torch.zeros(horizon,n,18,device=self.device);done_trace=torch.zeros(horizon,n,dtype=torch.bool,device=self.device)
                metrics={k:torch.zeros(n,dtype=torch.bool,device=self.device) for k in ['stable_success','ever_stable','drop_after_stable','five_step_hold','acquired_lift','post_lift_release']}
                acquired=torch.zeros_like(ended);released=torch.zeros_like(ended);lost=torch.zeros_like(contact_run)
                buffer=[];reset_ids=empty;infer_ms=[]
                for tick in range(horizon):
                    if time.monotonic()-begin>args.wall_seconds:raise TimeoutError('bounded native learning')
                    observation=self.env_reset(reset_ids)
                    state=torch.cat((task._dof_pos.clone(),task._dof_vel.clone(),task._target_states.clone()),-1)
                    contact=contacts(task)
                    history=torch.cat((history[:,1:],torch.cat((state,contact,previous),-1)[:,None]),1)
                    contact_run=torch.where(contact.bool().all(-1),contact_run+1,0)
                    candidate=self.candidates(observation);base=candidate[:,4]
                    eligible=(~ended)&(elapsed<0)&(cooldown<=0)&(contact_run>=3)&(tick>=10)&(task.max_episode_length[task.data_id]-task.progress_buf>11)
                    rows=eligible.nonzero().flatten()
                    if len(rows):
                        torch.cuda.synchronize();t=time.monotonic()
                        physical=physics.inputs(history[rows],candidate[rows],rest[rows],motion[rows],start[rows],torch.full_like(start[rows],tick))
                        x=policy_inputs(physical,args.cm_on,learnable_guide=learnable_guide);distribution,value=policy(**x)
                        chosen=distribution.probs.argmax(-1) if args.evaluate else torch.multinomial(distribution.probs,1,generator=generator).squeeze(-1)
                        torch.cuda.synchronize();infer_ms.append(1000*(time.monotonic()-t))
                        command=candidate[rows,chosen].clone();cached[rows]=command;elapsed[rows]=0
                        entry=dict(inputs={k:v.clone() for k,v in x.items()},selected=chosen.clone(),old_logprob=distribution.log_prob(chosen).clone(),
                                   old_value=value.clone(),env=rows.clone(),tick=tick,candidate=command,executed=torch.zeros(len(rows),2,18,device=self.device))
                        entry['reference_option']=physical['recommended'].clone() if args.cm_on else torch.full_like(chosen,4)
                        entry['candidate_bank']=candidate[rows].clone()
                        buffer.append(entry)
                    live=(elapsed>=0)&~ended;override=live&(elapsed<2)
                    action=base.clone();action[override]=cached[override];applied=action.clone()
                    for entry in buffer[-2:]:
                        offset=tick-entry['tick']
                        if 0<=offset<2:entry['executed'][:,offset]=applied[entry['env']]
                    _,native_reward,done,_=self.env_step(self.env,action)
                    if not torch.isfinite(native_reward).all():raise ValueError('nonfinite native reward')
                    native_rewards[tick]=native_reward.reshape(-1)*~ended
                    reward[tick]=native_rewards[tick]*.01
                    episode_steps+=~ended
                    pair=contacts(task).bool().all(-1);z=task._target_states[:,2]
                    if not torch.isfinite(task._target_states[~ended]).all():raise ValueError('nonfinite active physical trajectory')
                    object_trace[tick]=task._target_states.clone();contact_trace[tick]=contacts(task).bool();action_trace[tick]=applied
                    if physical_reward:reward[tick]=supported_height_reward(z,rest,pair)*(~ended)*.01
                    tracker.step(z,pair)
                    acquired|=(~ended)&(z-rest>=.03)&pair
                    lost=torch.where(acquired&~pair,lost+1,0)
                    released|=(~ended)&acquired&((z-rest<.02)|(lost>=6))
                    done=done.bool().reshape(-1);new=done&~ended
                    done_trace[tick]=done
                    metrics['ever_stable'][new]=tracker.stable[new]
                    metrics['drop_after_stable'][new]=tracker.drop_after_success[new]
                    metrics['stable_success'][new]=tracker.stable[new]&~tracker.drop_after_success[new]
                    metrics['five_step_hold'][new]=tracker.max_run[new]>=5/30-1e-6
                    metrics['acquired_lift'][new]=acquired[new];metrics['post_lift_release'][new]=released[new]
                    elapsed[live]+=1;complete=live&(elapsed==10);elapsed[complete]=-1;cooldown[complete]=6
                    cooldown=(cooldown-1).clamp_min(0);ended|=done;reset_ids=done.nonzero().flatten();previous=applied
                    if tick%200==0:print(json.dumps(dict(rollout=rollout,tick=tick,episodes_complete=int(ended.sum()),decisions=sum(len(b['selected']) for b in buffer))),flush=True)
                    if ended.all():break
                if not ended.all():raise ValueError('incomplete episodes; no trimming')
                if not buffer:raise ValueError('no contact decisions')
                if not all(torch.equal(b['executed'],b['candidate'][:,None].expand(-1,2,-1)) for b in buffer):raise ValueError('selected option was not executed')
                running=torch.zeros(n,device=self.device);returns=torch.zeros_like(reward)
                for tick in reversed(range(horizon)):running=reward[tick]+.99*running;returns[tick]=running
                x={k:torch.cat([b['inputs'][k] for b in buffer]) for k in buffer[0]['inputs']}
                selected=torch.cat([b['selected'] for b in buffer]);old=torch.cat([b['old_logprob'] for b in buffer]);old_value=torch.cat([b['old_value'] for b in buffer])
                target=torch.cat([returns[b['tick'],b['env']] for b in buffer]);advantage=target-old_value
                advantage=(advantage-advantage.mean())/advantage.std(unbiased=False).clamp_min(1e-6)
                before_update=fingerprint(policy.state_dict());losses=[];first_ratio_error=0.
                if not args.evaluate:
                    with torch.enable_grad():
                        for epoch in range(args.epochs):
                            permutation=torch.randperm(len(selected),device=self.device,generator=generator)
                            for offset in range(0,len(selected),128):
                                r=permutation[offset:offset+128];distribution,value=policy(**{k:v[r] for k,v in x.items()})
                                loss,stats=ppo_loss(distribution,value,selected[r],old[r],advantage[r],target[r])
                                if updates==0:first_ratio_error=float((stats['ratio']-1).abs().max())
                                if not torch.isfinite(loss):raise ValueError('nonfinite PPO loss')
                                optimizer.zero_grad();loss.backward()
                                if not all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None):raise ValueError('nonfinite gradient')
                                norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),1)
                                if not torch.isfinite(norm):raise ValueError('nonfinite gradient norm')
                                optimizer.step();updates+=1;losses.append(float(loss))
                distribution,_=policy(**x)
                reference=torch.cat([b['reference_option'] for b in buffer])
                changed=(distribution.probs.argmax(-1)!=reference).float().mean()
                original_return=torch.zeros(n,device=self.device)
                for tick in reversed(range(horizon)):original_return=native_rewards[tick]+.99*original_return
                report=dict(rollout=rollout,episodes=n,decisions=len(selected),episode_steps=episode_steps.cpu().tolist(),
                            native_mean_discounted_return=float(original_return.mean()),training_mean_discounted_return=float(returns[0].mean()/ .01),
                            metrics={k:int(v.sum()) for k,v in metrics.items()},
                            selected_counts=torch.bincount(selected,minlength=6).tolist(),recommended_counts=torch.bincount(reference,minlength=6).tolist(),
                            argmax_after_update_differs_from_prior=float(changed),first_actual_option_ratio_error=first_ratio_error,
                            parameter_update=before_update!=fingerprint(policy.state_dict()),optimizer_updates=updates,loss_mean=sum(losses)/len(losses) if losses else None)
                reports.append(report);latencies.extend(infer_ms)
                episode_traces.append(dict(object_state=object_trace[:int(episode_steps.max())].cpu(),
                                           contact=contact_trace[:int(episode_steps.max())].cpu(),action=action_trace[:int(episode_steps.max())].cpu(),
                                           done=done_trace[:int(episode_steps.max())].cpu(),episode_steps=episode_steps.cpu(),
                                           native_reward=native_rewards[:int(episode_steps.max())].cpu(),training_reward=reward[:int(episode_steps.max())].cpu()/.01))
                all_buffers.append(dict(inputs={k:v.cpu() for k,v in x.items()},selected=selected.cpu(),old_logprob=old.cpu(),returns=target.cpu(),
                                        reference_option=reference.cpu(),
                                        candidate_actions=torch.cat([b['candidate_bank'] for b in buffer]).cpu(),
                                        env=torch.cat([b['env'] for b in buffer]).cpu(),tick=torch.cat([torch.full_like(b['env'],b['tick']) for b in buffer]).cpu(),
                                        chosen_command=torch.cat([b['candidate'] for b in buffer]).cpu()))
                print(json.dumps(report),flush=True)
            if frozen!=fingerprint([m.state_dict() for m in physics.models]+[dict(model=m.state_dict(),rms=r.state_dict()) for m,r in self.frozen_experts]):raise ValueError('frozen model drift')
            torch.save(dict(schema='ref2dex.recovery_option_policy.v1',policy=policy.state_dict(),cm_on=args.cm_on,initial_fingerprint=initial,
                            trajectory_sha256=sha(args.trajectory),optimizer_updates=updates,learnable_guide=learnable_guide,
                            reward_mode='supported_height_fraction' if physical_reward else 'native_imitation'),args.output/'policy.pt')
            torch.save(all_buffers,args.output/'decisions.pt')
            torch.save(cold_states,args.output/'cold_states.pt')
            torch.save(episode_traces,args.output/'episode_traces.pt')
            result=dict(run_status='COMPLETED',cm_on=args.cm_on,evaluate=args.evaluate,smoke=args.smoke,rollouts=reports,optimizer_updates=updates,
                        frozen_cm_experts=True,policy_parameters_changed=initial!=fingerprint(policy.state_dict()),
                        native_env_steps=sum(sum(r['episode_steps']) for r in reports),batched_inference_ms=latencies,
                        policy_sha256=sha(args.output/'policy.pt'),decisions_sha256=sha(args.output/'decisions.pt'),elapsed_seconds=time.monotonic()-begin,
                        learnable_guide=learnable_guide,guide_weight=float(policy.guide_weight) if learnable_guide else None,
                        reward_mode='supported_height_fraction' if physical_reward else 'native_imitation',cold_states_sha256=sha(args.output/'cold_states.pt'),
                        episode_traces_sha256=sha(args.output/'episode_traces.pt'),
                        scientific_label='ENGINEERING_SMOKE' if args.smoke else 'AWAITING_MATCHED_ANALYSIS')
            (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['rollouts','batched_inference_ms']}),flush=True)
    return RecoveryPlayer


def main():
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    p.add_argument('--output-dir',dest='output',type=Path,required=True);p.add_argument('--trajectory',type=Path,required=True)
    p.add_argument('--policy',type=Path);p.add_argument('--cm-on',action='store_true');p.add_argument('--evaluate',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--rollouts',type=int,default=4);p.add_argument('--epochs',type=int,default=20);p.add_argument('--assignment-seed',type=int,default=7381)
    p.add_argument('--wall-seconds',type=int,default=1200)
    p.add_argument('--learnable-guide',action='store_true');p.add_argument('--physical-reward',action='store_true')
    args,remaining=p.parse_known_args()
    if args.learnable_guide!=args.physical_reward:raise ValueError('trainable guide route requires fixed physical task reward')
    base=(ROOT/'src/task/CmResidual/research/contact_consequence/output').resolve()
    if base not in args.output.resolve().parents or args.output.is_symlink():raise ValueError('owned output required')
    if sha(args.trajectory)!='027202015c32ba783aa1bbef5a0a3c501643bfa904e0b460cdf1877193971355':raise ValueError('frozen physical artifact required')
    if args.evaluate and args.rollouts!=1:raise ValueError('evaluation first episode only')
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    manifest=dict(run_status='STARTED',pid=os.getpid(),command=sys.argv,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),cm_training=False,expert_training=False,option_policy_training=not args.evaluate)
    path=args.output/'run_manifest.json';path.write_text(json.dumps(manifest,indent=2)+'\n')
    try:
        sys.path.insert(0,str(ROOT/'third_party/DExplore/dexplore'))
        from isaacgym import gymtorch
        import torch
        import evaluate as original
        original.EvalPlayer=recovery_player(original,args,torch,gymtorch)
        sys.argv=[sys.argv[0],*remaining];original.main();manifest['run_status']='COMPLETED'
    except BaseException as error:manifest.update(run_status='FAILED',error=repr(error));raise
    finally:manifest['elapsed_seconds']=time.monotonic()-started;path.write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()

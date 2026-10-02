#!/usr/bin/env python3
"""Frozen HF15 controllers own full H10 windows; execute two then reobserve."""
import argparse,hashlib,json,os,sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from collect_contact_consequences import build_player
from run_paired_evaluator_resolution import sha
ASSETS=ROOT/'third_party/DExplore/dexplore/data/assets/mjcf'
POLICIES=['cm','state_policy','shuffled','always_base','always_fixed']


def closed_loop_player(original,args,torch,gymtorch):
    from src.task.CmResidual.executable_contact_options import hold_target,obj_vertices,TableClearance,INDEPENDENT,COUPLINGS
    from src.task.CmResidual.physical_value_live import contacts as legacy_contacts
    from src.task.CmResidual.weight_normalized_contact import weight_normalized_contacts
    from src.task.CmResidual.orientation_anchored_options import orientation_anchored_action
    from src.task.CmResidual.paired_evaluation import fingerprint
    from src.task.CmResidual.native_pd_selector import FrozenNativePDSelector,native_pd_targets
    from src.task.CmResidual.relative_task_value_selector import FrozenRelativeTaskValueSelector
    parent=build_player(original,args,torch,gymtorch)
    class ClosedLoopPlayer(parent):
        @torch.no_grad()
        def run(self):
            begin=time.monotonic();torch.set_num_threads(2)
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
            task=self.env.task;n=task.num_envs;w=args.windows_per_episode
            task._hybrid_init_prob=0.;task._enable_early_termination=False;task._adaptive_kappa_enabled=False
            if set(getattr(task,'object_name',['airplane']))!={'airplane'} or n!=96 or self.is_rnn or abs(task.dt-1/30)>1e-8:raise ValueError('native execution contract')
            selector=FrozenNativePDSelector(args.checkpoint,self.device)
            if selector.fixed!=7:raise ValueError('frozen fixed candidate must be7')
            relative_selector=(FrozenRelativeTaskValueSelector(args.relative_value_checkpoint,self.device)
                               if args.relative_value_checkpoint else None)
            active_policies=POLICIES+(['cm_calibrated'] if relative_selector else [])
            cm_before=sha(args.checkpoint)
            relative_before=sha(args.relative_value_checkpoint) if args.relative_value_checkpoint else None
            model_before=fingerprint({mode:[m.state_dict() for m in models] for mode,models in selector.models.items()})
            expert_before=fingerprint([dict(model=m.state_dict(),rms=r.state_dict()) for m,r in self.frozen_experts])
            geometry=TableClearance(obj_vertices(ASSETS/'objects/airplane/airplane.obj',self.device),obj_vertices(ASSETS/'objects/table/table.obj',self.device),getattr(task,'ball_size',1.))
            ids=torch.arange(n,device=self.device);obs=self.env_reset(ids)
            if self.get_batch_size(obs['obs'],1)!=n:raise ValueError('native batch')
            motion=task.data_id.clone();start=task.start_times.clone();rest=task.hoi_refs[task.data_id,task.ref_index,0,108].clone()
            held=torch.tensor([int(hashlib.sha256(f'9851/{int(i)}/{int(j)}'.encode()).hexdigest()[:8],16)%100>=70 for i,j in zip(motion,start)],device=self.device)
            mass=torch.tensor([task.gym.get_actor_rigid_body_properties(e,h)[0].mass for e,h in zip(task.envs,task._target_handles)],device=self.device)
            gravity=abs(task.sim_params.gravity.z);body_ids=task._contact_body_ids
            def contact_state():return weight_normalized_contacts(task._contact_forces[:,body_ids],task._tar_contact_forces,mass,gravity)
            history=torch.zeros(n,10,69,device=self.device);previous=torch.zeros(n,18,device=self.device)
            count=torch.zeros(n,dtype=torch.long,device=self.device);elapsed=torch.full_like(count,-1);owner=torch.full_like(count,3);program=torch.full_like(count,4)
            cooldown=torch.zeros_like(count);contact_run=torch.zeros_like(count);ended=torch.zeros(n,dtype=torch.bool,device=self.device)
            anchor=torch.zeros(n,18,device=self.device);fixed_anchor=anchor.clone()
            def zeros(*shape,dtype=torch.float32):return torch.zeros(n,w,*shape,dtype=dtype,device=self.device)
            trigger=torch.full((n,w),-1,dtype=torch.long,device=self.device);assignment=trigger.clone();steps=torch.zeros_like(trigger)
            pre=zeros(49);hist=zeros(10,69);tables=zeros(7);initial_clearance=zeros();future=zeros(10,49)
            force=zeros(10,2,dtype=torch.bool);actual=zeros(10,18);pd=zeros(10,18);feedback=zeros(10,18);base_pd=zeros(10,18);fixed_pd=zeros(10,18)
            anchors=zeros(10,18);fixed_anchors=zeros(10,18);programs=zeros(10,dtype=torch.long)
            raw_hand=zeros(10,len(body_ids),3);raw_object=zeros(10,3);ratios=zeros(10,2);old_force=zeros(10,2,dtype=torch.bool)
            pre_hand=zeros(len(body_ids),3);pre_object=zeros(3);native_obs=zeros(10,obs['obs'].shape[-1]);clearance=zeros(10);terminal=zeros(10,dtype=torch.bool)
            decisions=zeros(10,10,69);recommend=zeros(10,len(active_policies),dtype=torch.long);ood=zeros(10,3,dtype=torch.bool);candidate_pd=zeros(10,8,18)
            diagnostics={key:zeros(10,3,8) for key in ['score_mm','relative_std_mm','retention','release']}
            changed_base=zeros(10,dtype=torch.bool);changed_fixed=zeros(10,dtype=torch.bool)
            generator=torch.Generator(device=self.device).manual_seed(args.assignment_seed);reset=ids[:0];first_episode_frames=0
            for tick in range(args.max_steps):
                if time.monotonic()-begin>args.wall_seconds:raise TimeoutError('bounded closed-loop execution')
                obs=self.env_reset(reset)
                state=torch.cat((task._dof_pos.clone(),task._dof_vel.clone(),task._target_states.clone()),-1);contact=contact_state()[0].float()
                history=torch.cat((history[:,1:],torch.cat((state,contact,previous),-1)[:,None]),1)
                contact_run=torch.where(contact.bool().all(-1),contact_run+1,0)
                bank=self.candidates(obs);base=bank[:,4];mesh_clearance=geometry.clearance(task._target_states,task._table_states)
                lifted=(state[:,38]-rest>=.03)&(mesh_clearance>=.002)
                eligible=held&(~ended)&(elapsed<0)&(cooldown<=0)&(contact_run>=3)&lifted&(tick>=10)&(tick<=args.max_steps-10)&(task.max_episode_length[task.data_id]-task.progress_buf>11)&(count<w)
                rows=eligible.nonzero().flatten();slot=count[rows]
                if len(rows):
                    draw=torch.multinomial(torch.full((len(rows),len(active_policies)),1/len(active_policies),device=self.device),1,generator=generator).squeeze(-1)
                    owner[rows]=draw;assignment[rows,slot]=draw;trigger[rows,slot]=tick;elapsed[rows]=0
                    pre[rows,slot]=state[rows];hist[rows,slot]=history[rows];tables[rows,slot]=task._table_states[rows,:7];initial_clearance[rows,slot]=mesh_clearance[rows]
                    pre_hand[rows,slot]=task._contact_forces[rows][:,body_ids];pre_object[rows,slot]=task._tar_contact_forces[rows]
                live=(elapsed>=0)&~ended;rows=live.nonzero().flatten();slot=count[rows];offset=elapsed[rows]
                replan=live;rr=replan.nonzero().flatten();ss=count[rr];dd=elapsed[rr]
                if len(rr):
                    anchor[rr]=hold_target(task._dof_pos[rr],task._pd_action_offset,task._pd_action_scale);fixed_anchor[rr]=anchor[rr]
                    base_hold=orientation_anchored_action(bank[rr,4],anchor[rr],task._dof_pos[rr],task._pd_action_offset,task._pd_action_scale)
                    cup_hold=orientation_anchored_action(bank[rr,1],anchor[rr],task._dof_pos[rr],task._pd_action_offset,task._pd_action_scale)
                    raw_candidates=torch.cat((bank[rr],base_hold[:,None],cup_hold[:,None]),1)
                    motor_candidates=native_pd_targets(raw_candidates,task._dof_pos[rr,None].expand(-1,8,-1),task._pd_action_offset,task._pd_action_scale)
                    candidate_pd[rr,ss,dd]=motor_candidates
                    choices=[];cm_d=None
                    for mode_index,mode in enumerate(POLICIES[:3]):
                        choice,d=selector.choose(mode,history[rr],obs['obs'][rr],rest[rr],motor_candidates,task._contact_forces[rr][:,body_ids],task._tar_contact_forces[rr],mass[rr],gravity,mesh_clearance[rr]);choices.append(choice)
                        if mode=='cm':cm_d=d
                        if any(not torch.isfinite(d[key]).all() for key in diagnostics):raise ValueError('nonfinite frozen model prediction')
                        ood[rr,ss,dd,mode_index]=d['ood']
                        for key in diagnostics:diagnostics[key][rr,ss,dd,mode_index]=d[key]
                    if relative_selector:
                        # Build a causal ten-step diagnostic prefix for the
                        # calibrated head.  Unknown suffix entries repeat the
                        # current one-step Cm prediction and never use future
                        # simulator outcomes.
                        prefix={}
                        for key in diagnostics:
                            current=diagnostics[key][rr,ss,dd,0]
                            if relative_selector.mode == 'current':
                                prefix[key]=current
                            else:
                                value=diagnostics[key][rr,ss,:,0].clone()
                                future_mask=torch.arange(10,device=self.device)[None,:]>dd[:,None]
                                prefix[key]=torch.where(future_mask[:,:,None],current[:,None,:],value)
                        calibrated_choice,_=relative_selector.choose(
                            prefix,pre[rr,ss],rest[rr],initial_clearance[rr,ss],
                            hist[rr,ss][:,-1,49:51],
                            {key: cm_d[key] for key in ('risk','contact','ood','candidate_ood')})
                    if relative_selector:
                        choices=torch.stack(choices+[torch.full_like(rr,4),torch.full_like(rr,7),calibrated_choice],-1)
                    else:
                        choices=torch.stack(choices+[torch.full_like(rr,4),torch.full_like(rr,7)],-1)
                    recommend[rr,ss,dd]=choices;decisions[rr,ss,dd]=history[rr];program[rr]=choices[torch.arange(len(rr),device=self.device),owner[rr]]
                action=base.clone();experts=live&(program<6);action[experts]=bank[experts,program[experts]]
                holding=live&(program>=6);expert_index=torch.where(program==7,1,torch.where(program==6,4,program));selected_feedback=bank[ids,expert_index]
                if holding.any():action[holding]=orientation_anchored_action(selected_feedback[holding],anchor[holding],task._dof_pos[holding],task._pd_action_offset,task._pd_action_scale)
                counter=base.clone()
                if len(rows):counter[rows]=orientation_anchored_action(bank[rows,1],fixed_anchor[rows],task._dof_pos[rows],task._pd_action_offset,task._pd_action_scale)
                applied=action.clone();targets=task._action_to_pd_targets(applied.clone()).clone();base_targets=task._action_to_pd_targets(base.clone()).clone();fixed_targets=task._action_to_pd_targets(counter.clone()).clone()
                if holding.any() and not torch.allclose(targets[holding,3:6],anchor[holding,3:6],atol=2e-5,rtol=1e-5):raise ValueError('anchored rotation target not realizable')
                if len(rows) and not torch.allclose(targets[rows],candidate_pd[rows,slot,offset,program[rows]],atol=2e-5,rtol=1e-5):raise ValueError('scored native motor target differs from executed target')
                feedback[rows,slot,offset]=selected_feedback[rows];native_obs[rows,slot,offset]=obs['obs'][rows]
                actual[rows,slot,offset]=applied[rows];pd[rows,slot,offset]=targets[rows];base_pd[rows,slot,offset]=base_targets[rows];fixed_pd[rows,slot,offset]=fixed_targets[rows]
                anchors[rows,slot,offset]=anchor[rows];fixed_anchors[rows,slot,offset]=fixed_anchor[rows];programs[rows,slot,offset]=program[rows]
                changed_base[rows,slot,offset]=((targets[rows]-base_targets[rows])[:,list(INDEPENDENT)].abs()>2e-5).any(-1)
                changed_fixed[rows,slot,offset]=((targets[rows]-fixed_targets[rows])[:,list(INDEPENDENT)].abs()>2e-5).any(-1)
                first_episode_frames+=int((~ended).sum())
                _,_,done,_=self.env_step(self.env,action);done=done.bool().reshape(-1)
                future[rows,slot,offset]=torch.cat((task._dof_pos.clone(),task._dof_vel.clone(),task._target_states.clone()),-1)[rows]
                bits,norm=contact_state();force[rows,slot,offset]=bits[rows];ratios[rows,slot,offset]=norm[rows]
                raw_hand[rows,slot,offset]=task._contact_forces[rows][:,body_ids];raw_object[rows,slot,offset]=task._tar_contact_forces[rows]
                old_force[rows,slot,offset]=legacy_contacts(task)[rows].bool();clearance[rows,slot,offset]=geometry.clearance(task._target_states,task._table_states)[rows]
                terminal[rows,slot,offset]=done[rows];steps[rows,slot]+=1
                elapsed[live]+=1;complete=live&(elapsed==10);count[complete]+=1;elapsed[complete]=-1;cooldown[complete]=6;cooldown=(cooldown-1).clamp_min(0)
                ended|=done;reset=done.nonzero().flatten();previous=applied
                if tick%100==0:print(json.dumps(dict(tick=tick,windows=int((steps==10).sum()),held_initial_envs=int(held.sum()))),flush=True)
                if (ended|~held|(count>=w)).all():break
            valid=steps==10
            if ((steps>0)&(~valid|terminal.any(-1))).any():raise ValueError('incomplete/reset-contaminated window')
            if int(valid.sum())<(1 if args.engineering_smoke else 24):raise ValueError('insufficient collected support')
            # Replay the frozen feedback experts independently on actual saved observations.
            saved_obs=native_obs[valid].reshape(-1,native_obs.shape[-1]);saved_feedback=feedback[valid].reshape(-1,18);choice=programs[valid].reshape(-1)
            index=torch.where(choice==7,1,torch.where(choice==6,4,choice));replay_error=0.;counter_error=0.
            prior_position=torch.cat((pre[valid][:,None,:18],future[valid][:,:-1,:18]),1).reshape(-1,18)
            saved_fixed_anchor=fixed_anchors[valid].reshape(-1,18)
            saved_base_pd=base_pd[valid].reshape(-1,18);saved_fixed_pd=fixed_pd[valid].reshape(-1,18)
            def native_pd(raw,position):
                value=raw.clone();value[:,6:]=(1+value[:,6:])/2
                target=task._pd_action_offset+task._pd_action_scale*value;target[:,:6]+=position[:,:6]
                for dst,src,ratio in COUPLINGS:target[:,dst]=target[:,src]*ratio
                return target
            for begin_batch in range(0,len(saved_obs),96):
                batch=saved_obs[begin_batch:begin_batch+96];k=len(batch);padded=torch.cat((batch,batch[:1].expand(96-k,-1)),0) if k<96 else batch
                replay=self.candidates({'obs':padded})[:k];selected=replay[torch.arange(k,device=self.device),index[begin_batch:begin_batch+k]]
                replay_error=max(replay_error,float((selected-saved_feedback[begin_batch:begin_batch+k]).abs().max()))
                pos=prior_position[begin_batch:begin_batch+k];fix=orientation_anchored_action(replay[:,1],saved_fixed_anchor[begin_batch:begin_batch+k],pos,task._pd_action_offset,task._pd_action_scale)
                counter_error=max(counter_error,float((native_pd(replay[:,4],pos)-saved_base_pd[begin_batch:begin_batch+k]).abs().max()),float((native_pd(fix,pos)-saved_fixed_pd[begin_batch:begin_batch+k]).abs().max()))
            if replay_error>2e-5 or counter_error>2e-5 or not torch.equal(actual[valid][:,:,:3],feedback[valid][:,:,:3]) or not torch.equal(actual[valid][:,:,6:],feedback[valid][:,:,6:]):raise ValueError('expert feedback/XYZ/fingers replay mismatch')
            if expert_before!=fingerprint([dict(model=m.state_dict(),rms=r.state_dict()) for m,r in self.frozen_experts]) or model_before!=fingerprint({mode:[m.state_dict() for m in models] for mode,models in selector.models.items()}) or cm_before!=sha(args.checkpoint):raise ValueError('frozen model drift')
            env=ids[:,None].expand(-1,w)[valid];rest_rows=rest[:,None].expand(-1,w)[valid];state_rows=pre[valid];f=future[valid];pair=force[valid].all(-1);clr=clearance[valid]
            retention=(clr[:,-3:]>=.002).all(-1)&pair[:,-3:].all(-1)
            score=((f[:,-3:,38].amin(-1)-rest_rows).clamp_min(0)*retention-(state_rows[:,38]-rest_rows).clamp_min(0))*1000
            payload=dict(schema='ref2dex.native_pd_closed_loop.v1',seed=args.seed,assignment_seed=args.assignment_seed,episode_id=[f's{args.seed}/env{int(i)}/first' for i in env.cpu()],env_id=env.cpu(),motion_id=motion[:,None].expand(-1,w)[valid].cpu(),start_frame=start[:,None].expand(-1,w)[valid].cpu(),
                trigger=trigger[valid].cpu(),assignment=assignment[valid].cpu(),propensity=torch.full((int(valid.sum()),),1/len(active_policies)),allocation_probabilities=torch.full((int(valid.sum()),len(active_policies)),1/len(active_policies)),state=state_rows.cpu(),history=hist[valid].cpu(),table_pose=tables[valid].cpu(),initial_clearance=initial_clearance[valid].cpu(),rest_z=rest_rows.cpu(),future_state=f.cpu(),future_contact=force[valid].cpu(),future_clearance=clr.cpu(),future_done=terminal[valid].cpu(),
                actual_action=actual[valid].cpu(),actual_pd_targets=pd[valid].cpu(),feedback_action=feedback[valid].cpu(),native_observation=native_obs[valid].cpu(),program=programs[valid].cpu(),rotation_anchor=anchors[valid].cpu(),fixed_rotation_anchor=fixed_anchors[valid].cpu(),base_pd_targets=base_pd[valid].cpu(),fixed_pd_targets=fixed_pd[valid].cpu(),changed_base=changed_base[valid].cpu(),changed_fixed=changed_fixed[valid].cpu(),decision_history=decisions[valid].cpu(),candidate_pd_targets=candidate_pd[valid].cpu(),recommendations=recommend[valid].cpu(),ood=ood[valid].cpu(),diagnostics={key:v[valid].cpu() for key,v in diagnostics.items()},
                mass_kg=mass[env].cpu(),gravity_magnitude=gravity,contact_collection=int(task.sim_params.physx.contact_collection),substeps=task.sim_params.substeps,initial_hand_force=pre_hand[valid].cpu(),initial_object_force=pre_object[valid].cpu(),future_hand_force=raw_hand[valid].cpu(),future_object_force=raw_object[valid].cpu(),future_force_ratio=ratios[valid].cpu(),legacy_future_contact=old_force[valid].cpu(),outcome=dict(score_mm=score.cpu(),retained=retention.cpu(),lost_clearance=(clr<.002).any(-1).cpu(),joint_last3=pair[:,-3:].all(-1).cpu()),
                policy_names=active_policies,intervention='random controller owner for H10; ten observed-state decisions each execute1feedbackstep; clear prestates held initial groups only;6tickcooldown',assignment_after_observation=True,policy_propensity_definition=f'{len(active_policies)} distinct whole-window controller laws p=1/{len(active_policies)}; coincident recommendations never merge policies',frozen_experts=True,frozen_cm=True,cm_used=True,optimizer_used=False,first_episode_frames=first_episode_frames,sim_frames=n*(tick+1),checkpoint_sha256=cm_before,relative_value_checkpoint_sha256=relative_before,pd_offset=task._pd_action_offset.cpu(),pd_scale=task._pd_action_scale.cpu(),geometry_definition='full25002vertex source mesh support over upper table plane; approximation to nativeVHACD',contact_definition='netforce norms divided by actual object weight >.1 presence proxy; not identified contact pairs',countercommand_definition='same observed-state base and reanchored fixed7 native PD references; not counterfactual outcome trajectories')
            torch.save(payload,args.output/'records.pt');result=dict(run_status='COMPLETED',rows=len(score),episodes=len(set(payload['episode_id'])),policies=torch.bincount(payload['assignment'],minlength=len(active_policies)).tolist(),elapsed_seconds=time.monotonic()-begin,record_sha256=sha(args.output/'records.pt'),frozen_experts=True,frozen_cm=True,checkpoint_sha256=cm_before,relative_value_checkpoint_sha256=relative_before,complete_labels=True,held_initial_groups_only=True,saved_observation_expert_replay_max_error=replay_error,countercommand_replay_max_error=counter_error,first_episode_frames=first_episode_frames,sim_frames=payload['sim_frames'])
            (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    return ClosedLoopPlayer


def main():
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False);p.add_argument('--output-dir',dest='output',type=Path,required=True);p.add_argument('--native-pd-checkpoint',dest='checkpoint',type=Path,required=True);p.add_argument('--relative-value-checkpoint',type=Path)
    p.add_argument('--panel-seed',dest='seed',type=int,required=True);p.add_argument('--assignment-seed',type=int,required=True);p.add_argument('--engineering-smoke',action='store_true');p.add_argument('--windows-per-episode',type=int,default=8);p.add_argument('--max-steps',type=int,default=650);p.add_argument('--wall-seconds',type=int,default=240)
    args,remaining=p.parse_known_args();base=(ROOT/'src/task/CmResidual/research/contact_consequence/output').resolve()
    if base not in args.output.resolve().parents or args.output.is_symlink():raise ValueError('new owned output required')
    args.output.mkdir(parents=True,exist_ok=False);begin=time.monotonic();m=dict(run_status='STARTED',pid=os.getpid(),command=sys.argv,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),expert_training=False,cm_training=False)
    path=args.output/'run_manifest.json';path.write_text(json.dumps(m,indent=2)+'\n')
    try:
        sys.path.insert(0,str(ROOT/'third_party/DExplore/dexplore'));from isaacgym import gymtorch
        import torch
        import evaluate as original
        original.EvalPlayer=closed_loop_player(original,args,torch,gymtorch);sys.argv=[sys.argv[0],*remaining];original.main();m['run_status']='COMPLETED'
    except BaseException as e:m.update(run_status='FAILED',error=repr(e));raise
    finally:m['elapsed_seconds']=time.monotonic()-begin;path.write_text(json.dumps(m,indent=2)+'\n')


if __name__=='__main__':main()

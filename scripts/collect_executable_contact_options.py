#!/usr/bin/env python3
"""Actual randomized H10 feedback-expert/pose-hold plans; no old Cm calls."""
import argparse,json,os,sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from collect_contact_consequences import build_player
from run_paired_evaluator_resolution import sha
ASSETS=ROOT/'third_party/DExplore/dexplore/data/assets/mjcf'


def executable_player(original,args,torch,gymtorch):
    from src.task.CmResidual.executable_contact_options import hold_target,hold_action,obj_vertices,TableClearance
    from src.task.CmResidual.physical_value_live import contacts
    from src.task.CmResidual.paired_evaluation import fingerprint
    from src.task.CmResidual.contact_consequence import local_outcomes
    parent=build_player(original,args,torch,gymtorch)
    class ExecutablePlayer(parent):
        @torch.no_grad()
        def run(self):
            begin=time.monotonic();torch.set_num_threads(2)
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
            task=self.env.task;n=task.num_envs;w=2*args.windows_per_stratum
            task._hybrid_init_prob=0.;task._enable_early_termination=False;task._adaptive_kappa_enabled=False
            if set(getattr(task,'object_name',['airplane']))!={'airplane'}:raise ValueError('airplane geometry only')
            if n!=96 or self.is_rnn or abs(task.dt-1/30)>1e-8:raise ValueError('native execution contract')
            geometry=TableClearance(obj_vertices(ASSETS/'objects/airplane/airplane.obj',self.device),
                                    obj_vertices(ASSETS/'objects/table/table.obj',self.device),getattr(task,'ball_size',1.))
            ids=torch.arange(n,device=self.device);obs=self.env_reset(ids)
            if self.get_batch_size(obs['obs'],1)!=n:raise ValueError('native batch')
            motion=task.data_id.clone();start=task.start_times.clone();rest=task.hoi_refs[task.data_id,task.ref_index,0,108].clone()
            before=fingerprint([dict(model=m.state_dict(),rms=r.state_dict()) for m,r in self.frozen_experts])
            history=torch.zeros(n,10,69,device=self.device);previous=torch.zeros(n,18,device=self.device)
            count=torch.zeros(n,dtype=torch.long,device=self.device);strata_count=torch.zeros(n,2,dtype=torch.long,device=self.device)
            elapsed=torch.full_like(count,-1);arm=torch.full_like(count,4);cooldown=torch.zeros_like(count);contact_run=torch.zeros_like(count)
            ended=torch.zeros(n,dtype=torch.bool,device=self.device);anchor=torch.zeros(n,18,device=self.device)
            trigger=torch.full((n,w),-1,dtype=torch.long,device=self.device);assignment=trigger.clone();allocation=trigger.clone();steps=torch.zeros_like(trigger)
            pre=torch.zeros(n,w,49,device=self.device);hist=torch.zeros(n,w,10,69,device=self.device)
            candidates=torch.zeros(n,w,7,18,device=self.device);anchors=torch.zeros(n,w,18,device=self.device);tables=torch.zeros(n,w,7,device=self.device)
            initial_clearance=torch.zeros(n,w,device=self.device);future=torch.zeros(n,w,10,49,device=self.device)
            force=torch.zeros(n,w,10,2,dtype=torch.bool,device=self.device);actual=torch.zeros(n,w,10,18,device=self.device);pd=torch.zeros_like(actual)
            expected=torch.zeros_like(actual);clearance=torch.zeros(n,w,10,device=self.device);terminal=torch.zeros(n,w,10,dtype=torch.bool,device=self.device)
            generator=torch.Generator(device=self.device).manual_seed(args.assignment_seed);reset=ids[:0]
            for tick in range(args.max_steps):
                if time.monotonic()-begin>args.wall_seconds:raise TimeoutError('bounded executable collection')
                obs=self.env_reset(reset)
                state=torch.cat((task._dof_pos.clone(),task._dof_vel.clone(),task._target_states.clone()),-1);contact=contacts(task)
                history=torch.cat((history[:,1:],torch.cat((state,contact,previous),-1)[:,None]),1)
                contact_run=torch.where(contact.bool().all(-1),contact_run+1,0)
                bank=self.candidates(obs);base=bank[:,4]
                mesh_clearance=geometry.clearance(task._target_states,task._table_states)
                lifted=(state[:,38]-rest>=.03)&(mesh_clearance>=.002)
                stratum=lifted.long()
                eligible=(~ended)&(elapsed<0)&(cooldown<=0)&(contact_run>=3)&((ids>=64)|lifted)&(tick>=10)&(tick<=args.max_steps-10)&(task.max_episode_length[task.data_id]-task.progress_buf>11)&(strata_count[ids,stratum]<args.windows_per_stratum)
                rows=eligible.nonzero().flatten();slot=count[rows]
                if len(rows):
                    # Independent duplicate-base labels form a randomized null;
                    # they do not claim individual cold/hot replay pairs.
                    probability=torch.full((len(rows),8),.125,device=self.device)
                    rare=lifted[rows]
                    probability[rare]=torch.tensor([.04,.04,.04,.04,.20,.04,.40,.20],device=self.device)
                    draw=torch.multinomial(probability,1,generator=generator).squeeze(-1)
                    selected=torch.where(draw==7,4,draw);anchor[rows]=hold_target(task._dof_pos[rows],task._pd_action_offset,task._pd_action_scale)
                    holding=hold_action(anchor[rows],task._dof_pos[rows],task._pd_action_offset,task._pd_action_scale)
                    trigger[rows,slot]=tick;allocation[rows,slot]=draw;assignment[rows,slot]=selected;arm[rows]=selected;elapsed[rows]=0
                    pre[rows,slot]=state[rows];hist[rows,slot]=history[rows];candidates[rows,slot]=torch.cat((bank[rows],holding[:,None]),1)
                    anchors[rows,slot]=anchor[rows];tables[rows,slot]=task._table_states[rows,:7];initial_clearance[rows,slot]=mesh_clearance[rows]
                    strata_count[rows,stratum[rows]]+=1
                live=(elapsed>=0)&~ended;rows=live.nonzero().flatten();slot=count[rows];offset=elapsed[rows]
                action=base.clone()
                experts=live&(arm<6);action[experts]=bank[experts,arm[experts]]
                holding=live&(arm==6)
                if holding.any():action[holding]=hold_action(anchor[holding],task._dof_pos[holding],task._pd_action_offset,task._pd_action_scale)
                applied=action.clone();targets=task._action_to_pd_targets(applied.clone()).clone()
                if holding.any() and not torch.allclose(targets[holding],anchor[holding],atol=2e-5,rtol=1e-5):raise ValueError('fixed hold target not realizable under native limits')
                expected[rows,slot,offset]=action[rows];pd[rows,slot,offset]=targets[rows]
                _,_,done,_=self.env_step(self.env,action);done=done.bool().reshape(-1)
                future[rows,slot,offset]=torch.cat((task._dof_pos.clone(),task._dof_vel.clone(),task._target_states.clone()),-1)[rows]
                force[rows,slot,offset]=contacts(task)[rows].bool();actual[rows,slot,offset]=applied[rows]
                clearance[rows,slot,offset]=geometry.clearance(task._target_states,task._table_states)[rows];terminal[rows,slot,offset]=done[rows];steps[rows,slot]+=1
                elapsed[live]+=1;complete=live&(elapsed==10);count[complete]+=1;elapsed[complete]=-1;cooldown[complete]=6;cooldown=(cooldown-1).clamp_min(0)
                ended|=done;reset=done.nonzero().flatten();previous=applied
                if tick%100==0:print(json.dumps(dict(tick=tick,windows=int((steps==10).sum()),initial_clear_windows=int(strata_count[:,1].sum()))),flush=True)
                if (ended|(strata_count==args.windows_per_stratum).all(-1)).all():break
            valid=steps==10
            if ((steps>0)&(~valid|terminal.any(-1))).any():raise ValueError('incomplete/reset-contaminated window')
            if int(valid.sum())<96:raise ValueError('insufficient support')
            if not torch.equal(actual[valid],expected[valid]):raise ValueError('feedback plan execution mismatch')
            if before!=fingerprint([dict(model=m.state_dict(),rms=r.state_dict()) for m,r in self.frozen_experts]):raise ValueError('expert drift')
            env=ids[:,None].expand(-1,w)[valid];rest_rows=rest[:,None].expand(-1,w)[valid]
            state_rows=pre[valid];f=future[valid];pair=force[valid].all(-1);clr=clearance[valid];initial=initial_clearance[valid]
            signed=(f[:,-3:,38].amin(-1)-rest_rows).clamp_min(0)*pair[:,-3:].all(-1)-(state_rows[:,38]-rest_rows).clamp_min(0)
            outcome=local_outcomes(f,pair,state_rows[:,38],rest_rows)
            outcome.update(supported_change_mm=signed*1000,retained_mesh_clearance_mm=clr[:,-3:].amin(-1)*1000,
                           initially_clear=(initial>=.002)&(state_rows[:,38]-rest_rows>=.03),retained_clear=(clr[:,-3:]>=.002).all(-1)&pair[:,-3:].all(-1))
            selected=assignment[valid];propensity=torch.where(selected==4,.25,.125)
            rare=outcome['initially_clear'];propensity[rare]=torch.where((selected[rare]==4)|(selected[rare]==6),.40,.04)
            allocations=torch.full((len(selected),8),.125,device=self.device)
            allocations[rare]=torch.tensor([.04,.04,.04,.04,.20,.04,.40,.20],device=self.device)
            payload=dict(schema='ref2dex.executable_contact_options.v1',seed=args.seed,assignment_seed=args.assignment_seed,
                         episode_id=[f's{args.seed}/env{int(i)}/first' for i in env.cpu()],env_id=env.cpu(),motion_id=motion[:,None].expand(-1,w)[valid].cpu(),
                         start_frame=start[:,None].expand(-1,w)[valid].cpu(),trigger=trigger[valid].cpu(),state=state_rows.cpu(),history=hist[valid].cpu(),
                         candidate_actions=candidates[valid].cpu(),hold_target=anchors[valid].cpu(),table_pose=tables[valid].cpu(),initial_clearance=initial.cpu(),
                         assignment=selected.cpu(),allocation=allocation[valid].cpu(),propensity=propensity.cpu(),future_state=f.cpu(),future_contact=force[valid].cpu(),
                         future_clearance=clr.cpu(),actual_action=actual[valid].cpu(),actual_pd_targets=pd[valid].cpu(),future_done=terminal[valid].cpu(),
                         rest_z=rest_rows.cpu(),outcome={k:v.cpu() for k,v in outcome.items()},frozen_experts=True,cm_used=False,optimizer_used=False,
                         intervention='full10step feedback expert or anchored native-PD pose-hold;6tick cooldown',
                         option_names=['balanced','cup','duck','mixed12','base','train5','pose_hold'],allocation_probabilities=allocations.cpu(),
                         sampling_cohort=torch.where(env<64,0,1).cpu(),
                         base_propensity_definition='slots4/7 identical feedback base; actual probability.25general/.40initiallyclear; hold.125general/.40initiallyclear; other5experts.125general/.04clear',
                         geometry_definition='full25002vertex source collision mesh support over oriented thin-table upper plane; scale matches native ball_size; VHACD approximation and pairwise contacts unavailable',
                         assignment_after_observation=True)
            torch.save(payload,args.output/'records.pt');result=dict(run_status='COMPLETED',rows=len(selected),episodes=len(set(payload['episode_id'])),
                 arms=torch.bincount(selected,minlength=7).tolist(),initially_clear=int(outcome['initially_clear'].sum()),
                 elapsed_seconds=time.monotonic()-begin,record_sha256=sha(args.output/'records.pt'),frozen_experts=True,cm_used=False,
                 table_thickness_axis=geometry.axis,pose_hold_target_contract_passed=True,feedback_execution_contract_passed=True)
            (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    return ExecutablePlayer


def main():
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False);p.add_argument('--output-dir',dest='output',type=Path,required=True)
    p.add_argument('--panel-seed',dest='seed',type=int,required=True);p.add_argument('--assignment-seed',type=int,required=True)
    p.add_argument('--windows-per-stratum',type=int,default=8);p.add_argument('--max-steps',type=int,default=650);p.add_argument('--wall-seconds',type=int,default=240)
    args,remaining=p.parse_known_args();base=(ROOT/'src/task/CmResidual/research/contact_consequence/output').resolve()
    if base not in args.output.resolve().parents or args.output.is_symlink():raise ValueError('new owned output required')
    args.output.mkdir(parents=True,exist_ok=False);begin=time.monotonic();m=dict(run_status='STARTED',pid=os.getpid(),command=sys.argv,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),expert_training=False,cm_training=False)
    path=args.output/'run_manifest.json';path.write_text(json.dumps(m,indent=2)+'\n')
    try:
        sys.path.insert(0,str(ROOT/'third_party/DExplore/dexplore'));from isaacgym import gymtorch
        import torch
        import evaluate as original
        original.EvalPlayer=executable_player(original,args,torch,gymtorch);sys.argv=[sys.argv[0],*remaining];original.main();m['run_status']='COMPLETED'
    except BaseException as e:m.update(run_status='FAILED',error=repr(e));raise
    finally:m['elapsed_seconds']=time.monotonic()-begin;path.write_text(json.dumps(m,indent=2)+'\n')


if __name__=='__main__':main()

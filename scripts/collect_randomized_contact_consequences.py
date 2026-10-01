#!/usr/bin/env python3
"""Sequential randomized contact interventions with recorded propensity.

Unlike a solver-state branch, each label belongs to its actual observed
pre-action state. Group by first episode for all learning/evaluation splits.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha
from collect_contact_consequences import build_player


def randomized_player(original,args,torch,gymtorch):
    from src.task.CmResidual.contact_consequence import HISTORY,HORIZON,EXECUTION_STEPS,BASE_INDEX,local_outcomes
    from src.task.CmResidual.paired_evaluation import fingerprint
    from src.task.CmResidual.physical_value_live import contacts
    parent=build_player(original,args,torch,gymtorch)

    class RandomizedPlayer(parent):
        @torch.no_grad()
        def run(self):
            started=time.monotonic();torch.set_num_threads(2)
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
            task=self.env.task;task._hybrid_init_prob=0.0
            task._enable_early_termination=False;task._adaptive_kappa_enabled=False
            if self.is_rnn or task.num_envs!=96 or abs(task.dt-1/30)>1e-8:
                raise ValueError('frozen stateless96env/30Hz contract')
            ids=torch.arange(task.num_envs,device=self.device)
            obs=self.env_reset(ids)
            if self.get_batch_size(obs['obs'],1)!=task.num_envs:raise ValueError('native batch')
            n=task.num_envs;w=args.windows_per_episode
            history=torch.zeros(n,HISTORY,69,device=self.device)
            previous=torch.zeros(n,18,device=self.device)
            count=torch.zeros(n,dtype=torch.long,device=self.device)
            elapsed=torch.full((n,),-1,dtype=torch.long,device=self.device)
            current_arm=torch.full((n,),BASE_INDEX,dtype=torch.long,device=self.device)
            cached=torch.zeros(n,18,device=self.device)
            cooldown=torch.zeros(n,dtype=torch.long,device=self.device)
            contact_run=torch.zeros(n,dtype=torch.long,device=self.device)
            ended=torch.zeros(n,dtype=torch.bool,device=self.device)
            trigger=torch.full((n,w),-1,dtype=torch.long,device=self.device)
            assignment=torch.full_like(trigger,-1)
            trigger_state=torch.zeros(n,w,49,device=self.device)
            trigger_history=torch.zeros(n,w,HISTORY,69,device=self.device)
            proposals=torch.zeros(n,w,6,18,device=self.device)
            future=torch.zeros(n,w,HORIZON,49,device=self.device)
            future_contact=torch.zeros(n,w,HORIZON,2,dtype=torch.bool,device=self.device)
            actual=torch.zeros(n,w,HORIZON,18,device=self.device)
            terminal=torch.zeros(n,w,HORIZON,dtype=torch.bool,device=self.device)
            steps=torch.zeros(n,w,dtype=torch.long,device=self.device)
            motion=task.data_id.clone();start=task.start_times.clone()
            rest=task.hoi_refs[task.data_id,task.ref_index,0,108].clone()
            generator=torch.Generator(device=self.device).manual_seed(args.assignment_seed)
            before=fingerprint([dict(model=m.state_dict(),rms=r.state_dict()) for m,r in self.frozen_experts])
            selector=None;policy_trace={};latencies=[];selector_pool=getattr(args,'trajectory',None) is not None
            if getattr(args,'ranker',None):
                from src.task.CmResidual.contact_selector import FrozenContactSelector
                selector=FrozenContactSelector(args.ranker,self.device)
                selector_before=fingerprint([m.state_dict() for m in selector.models])
                for key in ['proposed_arm','treatment','propensity','predicted_gain_mm','lower_gain_mm','predicted_contact','predicted_drop']:
                    policy_trace[key]=torch.zeros(n,w,device=self.device)
            elif selector_pool:
                from src.task.CmResidual.trajectory_selector import FrozenTrajectorySelectors,recommendation_probability,effective_commands
                selector=FrozenTrajectorySelectors(args.trajectory,self.device)
                selector_before=fingerprint([m.state_dict() for m in selector.models])
                for key in ['proposed_arm','treatment','propensity','selected_policy','predicted_gain_mm','lower_gain_mm','predicted_contact','predicted_release']:
                    policy_trace[key]=torch.zeros(n,w,device=self.device)
                policy_trace['policy_proposals']=torch.zeros(n,w,5,device=self.device)
                candidate_pd=torch.zeros(n,w,6,18,device=self.device)
                actual_pd=torch.zeros(n,w,HORIZON,18,device=self.device)
            reset_ids=torch.empty(0,dtype=torch.long,device=self.device)
            for tick in range(args.max_steps):
                if time.monotonic()-started>args.wall_seconds:raise TimeoutError('randomized native budget')
                obs=self.env_reset(reset_ids)
                state=torch.cat((task._dof_pos.clone(),task._dof_vel.clone(),task._target_states.clone()),-1)
                contact=contacts(task)
                history=torch.cat((history[:,1:],torch.cat((state,contact,previous),-1)[:,None]),1)
                contact_run=torch.where(contact.bool().all(-1),contact_run+1,0)
                candidate=self.candidates(obs)
                base=candidate[:,BASE_INDEX]
                eligible=((elapsed<0)&(count<w)&~ended&(cooldown<=0)&(contact_run>=3)&
                          (tick>=HISTORY)&(tick<=args.max_steps-HORIZON)&
                          (task.max_episode_length[task.data_id]-task.progress_buf>HORIZON+1))
                rows=eligible.nonzero().flatten();slots=count[rows]
                if len(rows):
                    # This private draw is after observing the current state,
                    # independent of actor/FPS RNG and all future labels.
                    if selector is None:
                        arm=torch.randint(6,(len(rows),),device=self.device,generator=generator)
                    else:
                        torch.cuda.synchronize();begin=time.monotonic()
                        predicted=selector.predict(history[rows],candidate[rows],rest[rows],motion[rows],start[rows],
                                                   torch.full_like(start[rows],tick))
                        torch.cuda.synchronize();latencies.append(dict(batch=len(rows),milliseconds=1000*(time.monotonic()-begin)))
                        proposal=predicted['proposed_arm'];active=proposal!=BASE_INDEX
                        if selector_pool:
                            selected_policy=torch.randint(5,(len(rows),),device=self.device,generator=generator)
                            arm=predicted['policy_proposals'][torch.arange(len(rows),device=self.device),selected_policy]
                            probability=recommendation_probability(candidate[rows],predicted['policy_proposals'],arm)
                            treatment=(selected_policy==0)&active
                            policy_trace['selected_policy'][rows,slots]=selected_policy.float()
                        else:
                            treatment=(torch.rand(len(rows),device=self.device,generator=generator)<.5)&active
                            arm=torch.where(treatment,proposal,torch.full_like(proposal,BASE_INDEX))
                            probability=torch.where(active,.5,1.)
                        for key,value in predicted.items():policy_trace[key][rows,slots]=value.float()
                        policy_trace['treatment'][rows,slots]=treatment.float()
                        policy_trace['propensity'][rows,slots]=probability
                        if selector_pool:
                            # Exact native mapping includes wrist increments,
                            # finger target ranges and overwritten couplings.
                            candidate_pd[rows,slots]=torch.stack([task._action_to_pd_targets(candidate[:,a].clone()).clone() for a in range(6)],1)[rows]
                    current_arm[rows]=arm;cached[rows]=candidate[rows,arm]
                    trigger[rows,slots]=tick;assignment[rows,slots]=arm
                    trigger_state[rows,slots]=state[rows]
                    trigger_history[rows,slots]=history[rows]
                    proposals[rows,slots]=candidate[rows]
                    elapsed[rows]=0
                live=(elapsed>=0)&~ended
                rows=live.nonzero().flatten();slots=count[rows];offset=elapsed[rows]
                action=base.clone()
                override=live&(elapsed<EXECUTION_STEPS)
                action[override]=cached[override]
                applied=action.clone()
                if selector_pool:actual_pd[rows,slots,offset]=task._action_to_pd_targets(applied.clone())[rows].clone()
                _,_,done,_=self.env_step(self.env,action)
                done=done.bool().reshape(-1)
                after=torch.cat((task._dof_pos.clone(),task._dof_vel.clone(),task._target_states.clone()),-1)
                future[rows,slots,offset]=after[rows]
                future_contact[rows,slots,offset]=contacts(task)[rows].bool()
                actual[rows,slots,offset]=applied[rows]
                terminal[rows,slots,offset]=done[rows]
                steps[rows,slots]+=1
                elapsed[live]+=1
                complete=live&(elapsed==HORIZON)
                count[complete]+=1;elapsed[complete]=-1;cooldown[complete]=6
                cooldown=(cooldown-1).clamp_min(0)
                ended|=done
                reset_ids=done.nonzero().flatten();previous=applied
                if tick%100==0:print(json.dumps(dict(tick=tick,complete_windows=int((steps==HORIZON).sum()),first_episodes_ended=int(ended.sum()))),flush=True)
                if bool((ended|(count>=w)).all()):break
            valid=(steps==HORIZON)&~terminal.any(-1)&(assignment>=0)
            if int(valid.sum())<96:raise ValueError('insufficient complete randomized support')
            if ((steps>0)&~valid).any():raise ValueError('incomplete/terminal-contaminated window; do not discard')
            expected=proposals[valid][torch.arange(int(valid.sum()),device=self.device),assignment[valid]]
            if not torch.equal(actual[valid][:,:EXECUTION_STEPS],expected[:,None].expand(-1,EXECUTION_STEPS,-1)):
                raise ValueError('random candidate not executed')
            if selector_pool:
                expected_pd=candidate_pd[valid][torch.arange(int(valid.sum()),device=self.device),assignment[valid]]
                if not torch.equal(actual_pd[valid][:,0],expected_pd):raise ValueError('native PD target not equal to selected current command')
            outcome=local_outcomes(future[valid],future_contact[valid].all(-1),trigger_state[valid][:,38],rest[:,None].expand(-1,w)[valid])
            if before!=fingerprint([dict(model=m.state_dict(),rms=r.state_dict()) for m,r in self.frozen_experts]):
                raise ValueError('frozen expert updated')
            if selector is not None and selector_before!=fingerprint([m.state_dict() for m in selector.models]):
                raise ValueError('frozen selector updated')
            env_id=torch.arange(n,device=self.device)[:,None].expand(-1,w)[valid]
            payload=dict(schema='ref2dex.randomized_contact_consequence.v1',seed=args.seed,assignment_seed=args.assignment_seed,
                         episode_id=[f's{args.seed}/env{int(i)}/first' for i in env_id.cpu()],
                         env_id=env_id.cpu(),motion_id=motion[:,None].expand(-1,w)[valid].cpu(),
                         start_frame=start[:,None].expand(-1,w)[valid].cpu(),trigger=trigger[valid].cpu(),
                         state=trigger_state[valid].cpu(),history=trigger_history[valid].cpu(),
                         candidate_actions=proposals[valid].cpu(),assignment=assignment[valid].cpu(),
                         propensity=torch.full((int(valid.sum()),),1/6),
                         future_state=future[valid].cpu(),future_contact=future_contact[valid].cpu(),
                         actual_action=actual[valid].cpu(),future_done=terminal[valid].cpu(),
                         rest_z=rest[:,None].expand(-1,w)[valid].cpu(),outcome={k:v.cpu() for k,v in outcome.items()},
                         contact_definition='native hand/object force proxy; collision pair unavailable',
                         intervention='cached current candidate2steps + own base8steps;6step cooldown',
                         teacher_or_optimizer=False,assignment_after_observation=True)
            if selector is not None:
                payload['schema']='ref2dex.targeted_contact_consequence.v1'
                payload['policy_trace']={k:v[valid].cpu() for k,v in policy_trace.items()}
                payload['propensity']=payload['policy_trace']['propensity']
                payload['ranker_sha256']=sha(args.trajectory if selector_pool else args.ranker);payload['inference_latency']=latencies
                active=payload['policy_trace']['proposed_arm'].long()!=BASE_INDEX
                treated=payload['policy_trace']['treatment'].bool()
                if selector_pool:
                    payload['schema']='ref2dex.selector_pool_contact_consequence.v1'
                    payload['policy_names']=list(selector.policy_names);payload['policy_allocation_probability']=.2
                    payload.pop('teacher_or_optimizer')
                    payload.update(actor_training=False,selector_training=False,frozen_proposal_models=True,
                                   predicted_release_scope='all-window acquired or pre-existing lift release')
                    pool=payload['policy_trace']['policy_proposals'].long()
                    expected_arm=pool[torch.arange(len(pool)),payload['policy_trace']['selected_policy'].long()]
                    if not torch.equal(payload['propensity'],recommendation_probability(payload['candidate_actions'],pool,expected_arm)):
                        raise ValueError('duplicate recommendation propensity mismatch')
                    payload.update(candidate_pd_targets=candidate_pd[valid].cpu(),actual_pd_targets=actual_pd[valid].cpu(),
                                   effective_candidate_commands=effective_commands(payload['candidate_actions']),
                                   command_definition='Inspire12independent normalized channels;6overwritten channels cannot cause physical differences',
                                   actual_propensity_definition='uniform5policy allocation, merge equal12channel constant2step plans')
                else:expected_arm=torch.where(treated,payload['policy_trace']['proposed_arm'].long(),BASE_INDEX)
                if not torch.equal(expected_arm,payload['assignment']):raise ValueError('allocation/proposal mismatch')
            torch.save(payload,args.output/'records.pt')
            result=dict(run_status='COMPLETED',rows=int(valid.sum()),episodes=len(set(payload['episode_id'])),
                        arms=torch.bincount(payload['assignment'],minlength=6).tolist(),
                        drop_eligible=int(outcome['drop_eligible'].sum()),drops=int(outcome['drop'].sum()),
                        elapsed_seconds=time.monotonic()-started,sha256=sha(args.output/'records.pt'))
            if selector is not None:
                result.update(active_windows=int(active.sum()),treated_windows=int((active&treated).sum()),
                              control_windows=int((active&~treated).sum()),frozen_selector=True,
                              decision_windows_per_episode=torch.bincount(env_id,minlength=n).cpu().tolist())
                if selector_pool:
                    result.pop('treated_windows');result.pop('control_windows')
                    result.update(policy_assignment_counts=torch.bincount(payload['policy_trace']['selected_policy'].long(),minlength=5).tolist(),
                                  cm_nonbase_recommendation_executed=int((active&(payload['assignment']==pool[:,0])).sum()))
            (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    return RandomizedPlayer


def main():
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    p.add_argument('--output-dir',dest='output',type=Path,required=True)
    p.add_argument('--panel-seed',dest='seed',type=int,required=True)
    p.add_argument('--assignment-seed',type=int,required=True)
    p.add_argument('--windows-per-episode',type=int,default=8)
    p.add_argument('--max-steps',type=int,default=650)
    p.add_argument('--wall-seconds',type=int,default=240)
    p.add_argument('--ranker',type=Path)
    p.add_argument('--trajectory',type=Path)
    args,remaining=p.parse_known_args()
    if args.ranker and args.trajectory:raise ValueError('one frozen selector mode')
    root=(ROOT/'src/task/CmResidual/research/contact_consequence/output').resolve()
    if root not in args.output.resolve().parents:raise ValueError('outside owned output')
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=dict(run_status='STARTED',pid=os.getpid(),command=sys.argv,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  actor_training=False,cm_training=False)
    (args.output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    started=time.monotonic()
    try:
        sys.path.insert(0,str(ROOT/'third_party/DExplore/dexplore'))
        from isaacgym import gymtorch
        import torch
        import evaluate as original
        original.EvalPlayer=randomized_player(original,args,torch,gymtorch)
        sys.argv=[sys.argv[0],*remaining];original.main();manifest['run_status']='COMPLETED'
    except BaseException as error:
        manifest.update(run_status='FAILED',error=repr(error));raise
    finally:
        manifest['elapsed_seconds']=time.monotonic()-started
        (args.output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()

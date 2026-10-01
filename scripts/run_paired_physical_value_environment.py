#!/usr/bin/env python3
"""Fresh-simulator paired, actor-only inference with state and action replay."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def make_player(original, args, torch, gymtorch):
    from src.task.CmResidual.paired_evaluation import (
        SCHEMA, capture_initial, capture_rng, cpu_copy, fingerprint, restore_initial, restore_rng,
        physical_property_value,
    )
    from src.task.CmResidual.physical_value_contract import HoldTracker
    from src.task.CmResidual.physical_value_live import contacts, snapshot

    def physics_properties(task):
        result=[]
        for env in task.envs:
            actors=[]
            for actor in range(task.gym.get_actor_count(env)):
                actors.append(dict(name=task.gym.get_actor_name(env,actor),
                    dof=cpu_copy(task.gym.get_actor_dof_properties(env,actor)),
                    rigid_body=physical_property_value(task.gym.get_actor_rigid_body_properties(env,actor)),
                    rigid_shape=physical_property_value(task.gym.get_actor_rigid_shape_properties(env,actor))))
            result.append(actors)
        return result

    class PairedPlayer(original.EvalPlayer):
        @torch.no_grad()
        def run(self):
            started=time.monotonic()
            torch.set_num_threads(2)
            torch.backends.cudnn.benchmark=False
            torch.backends.cudnn.deterministic=True
            torch.backends.cudnn.allow_tf32=False
            torch.backends.cuda.matmul.allow_tf32=False
            task=self.env.task
            task._enable_early_termination=False
            task._adaptive_kappa_enabled=False
            task._hybrid_init_prob=1.0
            if abs(task.dt-1/30)>1e-8 or task.num_envs!=96:
                raise ValueError('frozen time/environment contract')
            ids=torch.arange(task.num_envs,device=self.device)
            observation=self.env_reset(ids)
            if self.get_batch_size(observation['obs'],1)!=task.num_envs:
                raise ValueError('batch mismatch')
            if self.is_rnn: self.init_rnn()
            properties=physics_properties(task)
            saved=None
            if args.initial:
                saved=torch.load(args.initial,map_location='cpu',weights_only=False)
            else:
                saved=capture_initial(task,self,observation,properties)
            # Match native setter order even in the first run: each fresh
            # process performs reset -> full-state restore -> verification.
            observation=restore_initial(task,self,saved,gymtorch.unwrap_tensor,properties)
            initial_hash=fingerprint(saved)
            torch.save(saved,args.run_dir/'initial_state.pt')
            trace_reference=torch.load(args.trace,map_location='cpu',weights_only=False) if args.trace else None
            if trace_reference is not None:
                if trace_reference['schema']!=SCHEMA or trace_reference['initial_state_fingerprint']!=initial_hash:
                    raise ValueError('trace/initial state mismatch')
            if args.replay_actions and trace_reference is None:
                raise ValueError('action replay needs trace')
            # The initial player state is reused across arms; subsequent RNN
            # states evolve under each actor's own observations.
            tracker=HoldTracker(task.num_envs,self.device)
            tracker.reset(ids,task._target_states[:,2])
            motion=task.data_id.clone(); start_frame=task.start_times.clone()
            if (start_frame!=0).any() or [(motion==i).sum().item() for i in range(3)]!=[32]*3:
                raise ValueError('balanced frame0 panel missing')
            finished=torch.zeros(task.num_envs,dtype=torch.bool,device=self.device)
            steps=torch.zeros(task.num_envs,dtype=torch.long,device=self.device)
            lift_sum=torch.zeros(task.num_envs,device=self.device)
            contact_sum=torch.zeros_like(lift_sum)
            first_success=torch.full_like(steps,-1)
            first_drop=torch.full_like(steps,-1)
            trace_parts={}; rng_trace=[]; rnn_trace=[]; episodes=[]
            done_indices=torch.empty(0,dtype=torch.long,device=self.device)
            max_shadow_error=0.; max_obs_error=0.; max_rnn_error=0.
            model_before=fingerprint(self.model.state_dict())
            rms_before=fingerprint(self.running_mean_std.state_dict()) if self.normalize_input else None
            for tick in range(700):
                if time.monotonic()-started>args.wall_seconds:
                    raise TimeoutError('paired native wall budget')
                if trace_reference is not None:
                    if tick>=len(trace_reference['rng']): raise ValueError('reference trace exhausted')
                    restore_rng(trace_reference['rng'][tick]['before_reset'])
                before_reset=capture_rng()
                observation=self.env_reset(done_indices)
                if self.is_rnn and len(done_indices):
                    for state in self.states: state[:,done_indices,:]=0
                if trace_reference is not None:
                    restore_rng(trace_reference['rng'][tick]['before_action'])
                before_action=capture_rng()
                rnn_trace.append(cpu_copy(self.states))
                active=~finished
                raw_obs=observation['obs'].detach().cpu().clone()
                proposed=self.get_action(observation,True).clamp(-1,1).clone()
                action=proposed.clone()
                if args.replay_actions:
                    expected=trace_reference['action'][tick].to(self.device)
                    max_shadow_error=max(max_shadow_error,float((proposed[active]-expected[active]).abs().max()))
                    max_obs_error=max(max_obs_error,float((raw_obs[active.cpu()]-trace_reference['observation'][tick][active.cpu()]).abs().max()))
                    if self.is_rnn:
                        for actual,reference in zip(rnn_trace[-1],trace_reference['rnn'][tick]):
                            max_rnn_error=max(max_rnn_error,float((actual-reference).abs().max()))
                    action=expected.clone()
                if not torch.isfinite(action).all(): raise FloatingPointError('nonfinite action')
                if trace_reference is not None:
                    restore_rng(trace_reference['rng'][tick]['before_physics'])
                before_physics=capture_rng()
                # pre_physics_step transforms finger actions in place on its
                # private clone; retain executed normalized action first.
                normalized_action=action.detach().cpu().clone()
                observation,reward,done,info=self.env_step(self.env,action)
                done=done.bool().reshape(-1)
                terminate=info['terminate'].bool().reshape(-1)
                tracker.step(task._target_states[:,2],contacts(task).bool().all(-1))
                newly_success=active & tracker.stable & (first_success<0)
                newly_drop=active & tracker.drop_after_success & (first_drop<0)
                first_success[newly_success]=tick+1
                first_drop[newly_drop]=tick+1
                lift_sum[active]+=(task._target_states[active,2]-tracker.initial_height[active]).clamp_min(0)
                contact_sum[active]+=contacts(task)[active].bool().all(-1).float()
                steps[active]+=1
                values=dict(action=normalized_action,proposed_action=proposed.cpu(),observation=raw_obs,
                    active=active.cpu(),done=done.cpu(),terminate=terminate.cpu(),
                    state_after=snapshot(task,tracker).cpu(),root_after=task._root_states.cpu().clone(),
                    dof_after=task._dof_state.cpu().clone(),
                    hand_forces_after=task._contact_forces.cpu().clone(),
                    object_forces_after=task._tar_contact_forces.cpu().clone())
                for name,value in values.items(): trace_parts.setdefault(name,[]).append(value)
                rng_trace.append(dict(before_reset=before_reset,before_action=before_action,before_physics=before_physics))
                for i in (done & active).nonzero().flatten().tolist():
                    episodes.append(dict(pair_id=f't{args.training_seed}/s{args.eval_seed}/env{i}',env_id=i,
                        motion_id=int(motion[i]),start_frame=int(start_frame[i]),steps=int(steps[i]),
                        stable_success=bool(tracker.stable[i]),drop_after_success=bool(tracker.drop_after_success[i]),
                        first_success_step=int(first_success[i]),first_drop_step=int(first_drop[i]),
                        followup_after_success_steps=int(steps[i]-first_success[i]) if first_success[i]>=0 else None,
                        max_hold_seconds=float(tracker.max_run[i]),mean_lift_meters=float(lift_sum[i]/steps[i]),
                        contact_fraction=float(contact_sum[i]/steps[i]),
                        initial_object_height=float(tracker.initial_height[i]),control_dt=float(task.dt),
                        terminate=bool(terminate[i])))
                finished|=done
                if bool(finished.all()): break
                done_indices=done.nonzero().flatten()
                if tick%200==0: print(json.dumps(dict(tick=tick,complete_episodes=len(episodes))),flush=True)
            else: raise ValueError('incomplete first episodes')
            if fingerprint(self.model.state_dict())!=model_before:
                raise ValueError('actor parameter/buffer update')
            if self.normalize_input and fingerprint(self.running_mean_std.state_dict())!=rms_before:
                raise ValueError('normalizer update')
            trace={key:torch.stack(value) for key,value in trace_parts.items()}
            trace.update(schema=SCHEMA,initial_state_fingerprint=initial_hash,rng=rng_trace,rnn=rnn_trace,
                         checkpoint_sha256=args.checkpoint_sha256,arm=args.arm)
            trace_path=args.run_dir/'trace.pt'
            torch.save(trace,trace_path)
            trace_valid=True
            if trace_reference is not None:
                if len(rng_trace)!=len(trace_reference['rng']): raise ValueError('trace tick count mismatch')
                if fingerprint(rng_trace)!=fingerprint(trace_reference['rng']): raise ValueError('RNG schedule drift')
                for key in ('active','done','terminate'):
                    if not torch.equal(trace[key],trace_reference[key]): raise ValueError('terminal/active schedule drift')
                if args.replay_actions and not torch.equal(trace['action'],trace_reference['action']):
                    raise ValueError('applied action trace drift')
            result=dict(schema=SCHEMA,run_status='COMPLETED',mode='evaluate',arm=args.arm,
                training_seed=args.training_seed,evaluation_seed=args.eval_seed,
                complete_episodes=len(episodes),per_episode=episodes,
                stable_success_count=sum(e['stable_success'] for e in episodes),
                drop_after_success_count=sum(e['drop_after_success'] for e in episodes),
                initial_state_fingerprint=initial_hash,initial_state_sha256=sha(args.run_dir/'initial_state.pt'),
                trace_sha256=sha(trace_path),trace_contract_valid=trace_valid,
                checkpoint_sha256=args.checkpoint_sha256,actor_and_rms_unchanged=True,
                is_rnn=self.is_rnn,shadow_action_max_abs_error=max_shadow_error,
                shadow_observation_max_abs_error=max_obs_error,shadow_rnn_max_abs_error=max_rnn_error,
                closed_loop_equivalent=(max_shadow_error<=1e-5 and max_rnn_error<=1e-5) if args.replay_actions else None,
                elapsed_seconds=time.monotonic()-started)
            (args.run_dir/'results.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps({k:result[k] for k in ('stable_success_count','drop_after_success_count','shadow_action_max_abs_error','elapsed_seconds')}),flush=True)
    return PairedPlayer


def main():
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--checkpoint-sha256',required=True)
    parser.add_argument('--arm',choices=('plain_off','direct_q','cm_value'),required=True)
    parser.add_argument('--training-seed',type=int,required=True)
    parser.add_argument('--eval-seed',type=int,required=True)
    parser.add_argument('--initial',type=Path)
    parser.add_argument('--trace',type=Path)
    parser.add_argument('--replay-actions',action='store_true')
    parser.add_argument('--wall-seconds',type=int,default=240)
    args,remaining=parser.parse_known_args()
    checkpoint=Path(remaining[remaining.index('--checkpoint')+1])
    if sha(checkpoint)!=args.checkpoint_sha256: raise ValueError('checkpoint drift')
    if args.run_dir.exists(): raise FileExistsError(args.run_dir)
    base=(ROOT/'src/task/CmResidual/research/physical_value/output').resolve()
    if base not in args.run_dir.resolve().parents: raise ValueError('output outside owned task root')
    args.run_dir.mkdir(parents=True)
    manifest=dict(run_status='STARTED',pid=os.getpid(),command=sys.argv,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        created_at=datetime.now(timezone.utc).isoformat(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        checkpoint_sha256=args.checkpoint_sha256,training_or_optimizer_update=False)
    started=time.monotonic()
    try:
        sys.path.insert(0,str(ROOT/'third_party/DExplore/dexplore'))
        from isaacgym import gymtorch  # mandatory before torch
        import torch
        import evaluate as original
        original.EvalPlayer=make_player(original,args,torch,gymtorch)
        sys.argv=[sys.argv[0],*remaining]
        original.main()
        if sha(checkpoint)!=args.checkpoint_sha256: raise ValueError('checkpoint mutated')
        manifest['run_status']='COMPLETED'
    except BaseException as error:
        manifest.update(run_status='FAILED',error=repr(error)); raise
    finally:
        manifest['elapsed_seconds']=time.monotonic()-started
        (args.run_dir/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__': main()

#!/usr/bin/env python3
"""Single prospective randomized batch; no individually cloned solver states."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_paired_physical_value_environment import sha


def make_player(original,args,torch):
    from src.task.CmResidual.paired_evaluation import capture_rng,restore_rng,fingerprint
    from src.task.CmResidual.physical_value_contract import HoldTracker
    from src.task.CmResidual.physical_value_live import snapshot,contacts
    from src.task.CmResidual.dexplore_cm_geometry import DExploreCmv2GeometryBridge

    class RandomizedPlayer(original.EvalPlayer):
        @torch.no_grad()
        def run(self):
            begin=time.monotonic();torch.set_num_threads(2)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            task=self.env.task
            task._enable_early_termination=False;task._adaptive_kappa_enabled=False;task._hybrid_init_prob=1.
            if task.num_envs!=768 or abs(task.dt-1/30)>1e-8:raise ValueError('frozen batch contract')
            ids=torch.arange(768,device=self.device);obs=self.env_reset(ids);self.get_batch_size(obs['obs'],1)
            if self.is_rnn:self.init_rnn()
            if (task.start_times!=0).any() or [int((task.data_id==i).sum()) for i in range(3)]!=[256]*3:
                raise ValueError('balanced frame0 motions required')
            if not torch.allclose(task._pd_action_scale[:3],torch.ones(3,device=self.device)):
                raise ValueError('wrist target scales changed')
            motion=task.data_id.cpu().clone();generator=torch.Generator(device='cpu').manual_seed(9900+args.eval_seed)
            assignment=torch.empty(768,dtype=torch.long);propensity=torch.empty(768,7)
            if args.assignment_mode == 'iid':
                from src.task.CmResidual.direct_randomized_response import iid_assignment
                assignment, propensity = iid_assignment(768, args.eval_seed)
            else:
                for m in range(3):
                    env=(motion==m).nonzero().flatten();labels=torch.arange(256)%7
                    labels=labels[torch.randperm(256,generator=generator)]
                    assignment[env]=labels;propensity[env]=torch.bincount(labels,minlength=7).float()/256
            model_hash=fingerprint(self.model.state_dict());rms_hash=fingerprint(self.running_mean_std.state_dict()) if self.normalize_input else None
            initial_rng=capture_rng()
            torch.save(dict(root=task._root_states.cpu().clone(),dof=task._dof_state.cpu().clone(),rng=initial_rng,
                            assignment=assignment,propensity=propensity,motion=motion),args.run_dir/'initial.pt')
            asset=ROOT/'third_party/DExplore/dexplore/data/assets'
            bridge=DExploreCmv2GeometryBridge(hand_urdf=asset/'inspire_hand_new/inspire_hand_right.urdf',
                object_urdf=asset/'mjcf/airplane.urdf',device=task._dof_state.device,seed=42)
            restore_rng(initial_rng)
            tracker=HoldTracker(768,self.device);tracker.reset(ids,task._target_states[:,2])
            trigger=torch.full((768,),-1,dtype=torch.long,device=self.device)
            before=torch.zeros(768,55,device=self.device);after=torch.zeros_like(before)
            original_action=torch.zeros(768,18,device=self.device);applied=torch.zeros_like(original_action)
            gaps=torch.full((768,),float('inf'),device=self.device);complete=torch.zeros(768,dtype=torch.bool,device=self.device)
            assignment_gpu=assignment.to(self.device);empty=torch.empty(0,dtype=torch.long,device=self.device)
            for tick in range(360):
                if time.monotonic()-begin>args.wall_seconds:raise TimeoutError('randomized native budget')
                obs=self.env_reset(empty);state=snapshot(task,tracker)
                chosen=torch.empty(0,dtype=torch.long,device=self.device)
                if tick>=10 and tick%2==0:
                    candidates=((trigger<0)&state[:,49:51].bool().all(-1)).nonzero().flatten()
                    accepted=[]
                    for group in candidates.split(64):
                        current=state[group]
                        geo=bridge.current(current[:,:18],current[:,36:49])
                        gap=torch.cdist(geo.hand_points,geo.object_points).amin((1,2))
                        near=gap<=.02;accepted.append(group[near]);gaps[group[near]]=gap[near]
                    if accepted:chosen=torch.cat(accepted)
                ref=self.get_action(obs,True).clamp(-1,1).clone();action=ref.clone()
                if len(chosen):
                    trigger[chosen]=tick;before[chosen]=state[chosen]
                    original_action[chosen]=ref[chosen]
                    for label in range(1,7):
                        selected=chosen[assignment_gpu[chosen]==label];axis=(label-1)//2;delta=.01 if label%2 else -.01
                        request=action[selected,axis]+delta
                        if (request.abs()>1).any():raise ValueError('randomized intervention clips')
                        action[selected,axis]=request
                    applied[chosen]=action[chosen]
                obs,_,done,_=self.env_step(self.env,action)
                tracker.step(task._target_states[:,2],contacts(task).bool().all(-1))
                if len(chosen):after[chosen]=snapshot(task,tracker)[chosen];complete[chosen]=True
                if done.any():
                    complete[done.bool().reshape(-1)&(trigger==tick)]=False
                    break
                if complete.all():break
                if tick%50==0:print(json.dumps(dict(tick=tick,complete=int(complete.sum()))),flush=True)
            if fingerprint(self.model.state_dict())!=model_hash:raise ValueError('actor changed')
            if self.normalize_input and fingerprint(self.running_mean_std.state_dict())!=rms_hash:raise ValueError('RMS changed')
            if not torch.isfinite(before).all() or not torch.isfinite(after).all():raise FloatingPointError('nonfinite state')
            data=dict(state=before.cpu(),next_state=after.cpu(),reference_action=original_action.cpu(),action=applied.cpu(),
                      arm=assignment,propensity=propensity,trigger=trigger.cpu(),complete=complete.cpu(),gap=gaps.cpu(),
                      motion=motion,environment=torch.arange(768),training_seed=args.training_seed,evaluation_seed=args.eval_seed,
                      control_dt=float(task.dt),schema='ref2dex.randomized_effect.v1')
            torch.save(data,args.run_dir/'windows.pt')
            result=dict(run_status='COMPLETED',complete_windows=int(complete.sum()),required_windows=768,
                        all_windows_complete=bool(complete.all()),actor_and_rms_unchanged=True,
                        max_gap_mm=float(gaps[complete].max()*1000) if complete.any() else None,
                        windows_sha256=sha(args.run_dir/'windows.pt'),wall_seconds=time.monotonic()-begin,
                        assignment_mode=args.assignment_mode,
                        assignment_counts={str(m):torch.bincount(assignment[motion==m],minlength=7).tolist() for m in range(3)})
            (args.run_dir/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    return RandomizedPlayer


def main():
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--checkpoint-sha256',required=True)
    parser.add_argument('--training-seed',type=int,required=True);parser.add_argument('--eval-seed',type=int,required=True)
    parser.add_argument('--wall-seconds',type=int,default=600)
    parser.add_argument('--assignment-mode',choices=('balanced','iid'),default='balanced')
    args,remaining=parser.parse_known_args()
    if args.run_dir.exists() or ROOT not in args.run_dir.resolve().parents:raise ValueError('unique isolated output required')
    checkpoint=Path(remaining[remaining.index('--checkpoint')+1]);value=sha(checkpoint)
    if value!=args.checkpoint_sha256:raise ValueError('checkpoint drift')
    args.run_dir.mkdir(parents=True);begin=time.monotonic()
    manifest=dict(run_status='RUNNING',pid=os.getpid(),command=sys.argv,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  checkpoint_sha256=value,no_training=True)
    try:
        sys.path.insert(0,str(ROOT/'third_party/DExplore/dexplore'))
        from isaacgym import gymtorch
        import torch
        import evaluate as original
        original.EvalPlayer=make_player(original,args,torch);sys.argv=[sys.argv[0],*remaining];original.main()
        if sha(checkpoint)!=value:raise ValueError('checkpoint modified')
        manifest['run_status']='COMPLETED'
    except BaseException as error:
        manifest.update(run_status='FAILED',error=repr(error));raise
    finally:
        manifest['wall_seconds']=time.monotonic()-begin
        (args.run_dir/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()

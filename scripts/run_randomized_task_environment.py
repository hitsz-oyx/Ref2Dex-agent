#!/usr/bin/env python3
"""Randomized bounded corrections with complete first-episode task outcomes."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import sha


def make_player(original,args,torch):
    from src.task.CmResidual.paired_evaluation import capture_rng,restore_rng,fingerprint
    from src.task.CmResidual.physical_value_contract import HoldTracker
    from src.task.CmResidual.physical_value_live import snapshot,contacts
    from src.task.CmResidual.dexplore_cm_geometry import DExploreCmv2GeometryBridge
    from src.task.CmResidual.randomized_task_selection import FrozenFactualScores,candidate_validity,choose_actions

    class TaskPlayer(original.EvalPlayer):
        @torch.no_grad()
        def run(self):
            begin=time.monotonic();torch.set_num_threads(2)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            task=self.env.task;task._enable_early_termination=False;task._adaptive_kappa_enabled=False;task._hybrid_init_prob=1.
            if task.num_envs!=768 or abs(task.dt-1/30)>1e-8:raise ValueError('frozen batch contract')
            device=task._dof_state.device;ids=torch.arange(768,device=device)
            obs=self.env_reset(ids);self.get_batch_size(obs['obs'],1)
            if self.is_rnn:self.init_rnn()
            if (task.start_times!=0).any() or [int((task.data_id==i).sum()) for i in range(3)]!=[256]*3:raise ValueError('motion/frame0 contract')
            if not torch.allclose(task._pd_action_scale[:3],torch.ones(3,device=device)):raise ValueError('wrist scale drift')
            generator=torch.Generator().manual_seed(11000+args.eval_seed)
            assignment=torch.randint(4,(768,),generator=generator);policy=assignment.to(device)
            random_generators=[torch.Generator().manual_seed(21000+1000*args.eval_seed+i) for i in range(768)]
            motion=task.data_id.cpu().clone();rng=capture_rng()
            torch.save(dict(root=task._root_states.cpu().clone(),dof=task._dof_state.cpu().clone(),rng=rng,
                assignment=assignment,motion=motion),args.run_dir/'initial.pt')
            actor_hash=fingerprint(self.model.state_dict());rms_hash=fingerprint(self.running_mean_std.state_dict()) if self.normalize_input else None
            asset=ROOT/'third_party/DExplore/dexplore/data/assets'
            bridge=DExploreCmv2GeometryBridge(hand_urdf=asset/'inspire_hand_new/inspire_hand_right.urdf',
                object_urdf=asset/'mjcf/airplane.urdf',device=device,seed=42)
            controls=torch.load(args.controls,map_location='cpu',weights_only=False)
            checkpoints=[torch.load(Path(controls['models'])/f'factual_s{s}.pt',map_location='cpu',weights_only=False) for s in (611,612,613)]
            controller=FrozenFactualScores(checkpoints,device);global_scores=controls['global_vertical_scores'].to(device)
            restore_rng(rng)
            tracker=HoldTracker(768,device);tracker.reset(ids,task._target_states[:,2])
            initial_height=tracker.initial_height.cpu().clone();finished=torch.zeros(768,dtype=torch.bool,device=device)
            count=torch.zeros(768,dtype=torch.long,device=device);nonzero=torch.zeros_like(count)
            last=torch.full_like(count,-100);steps=torch.zeros_like(count);lift_sum=torch.zeros(768,device=device)
            done_ids=torch.empty(0,dtype=torch.long,device=device);episodes=[];trace={key:[] for key in ('height','contact','active','done')}
            decisions={key:[] for key in ('tick','environment','state','reference_action','prediction','valid','selected','gap')}
            inference_seconds=0.
            for tick in range(700):
                if time.monotonic()-begin>args.wall_seconds:raise TimeoutError('native task budget')
                obs=self.env_reset(done_ids)
                if len(done_ids):
                    tracker.reset(done_ids,task._target_states[done_ids,2])
                    if self.is_rnn:
                        for rnn in self.states:rnn[:,done_ids,:]=0
                active=~finished;before=snapshot(task,tracker);ref=self.get_action(obs,True).clamp(-1,1).clone();action=ref.clone()
                if tick>=10 and tick%2==0:
                    candidates=(active&(count<10)&(tick-last>=6)&before[:,49:51].bool().all(-1)).nonzero().flatten()
                    chosen=[];distances=[]
                    for group in candidates.split(64):
                        geo=bridge.current(before[group,:18],before[group,36:49]);gap=torch.cdist(geo.hand_points,geo.object_points).amin((1,2))
                        near=gap<=.02;chosen.append(group[near]);distances.append(gap[near])
                    if chosen:
                        group=torch.cat(chosen);gap=torch.cat(distances)
                        if len(group):
                            start=time.monotonic();predictions=controller(before[group],ref[group],bridge);torch.cuda.synchronize()
                            inference_seconds+=time.monotonic()-start;valid=candidate_validity(ref[group])
                            random_choice=[]
                            for env_id,allowed in zip(group.cpu().tolist(),valid.cpu()):
                                choices=allowed.nonzero().flatten();index=int(torch.randint(len(choices),(1,),generator=random_generators[env_id]))
                                random_choice.append(int(choices[index]))
                            selected=choose_actions(policy[group],predictions[:,:,2],global_scores,valid,torch.tensor(random_choice,device=device))
                            for arm in range(1,7):
                                changed=group[selected==arm];action[changed,(arm-1)//2]+=.01 if arm%2 else -.01
                            count[group]+=1;nonzero[group]+=(selected!=0).long();last[group]=tick
                            values=dict(tick=torch.full((len(group),),tick,dtype=torch.long),environment=group.cpu(),state=before[group].cpu(),
                                reference_action=ref[group].cpu(),prediction=predictions.cpu(),valid=valid.cpu(),selected=selected.cpu(),gap=gap.cpu())
                            for key,value in values.items():decisions[key].append(value)
                if not torch.isfinite(action).all() or (action.abs()>1).any():raise ValueError('invalid action')
                obs,_,done,info=self.env_step(self.env,action);done=done.bool().reshape(-1)
                pair=contacts(task);tracker.step(task._target_states[:,2],pair.bool().all(-1))
                height=task._target_states[:,2];steps[active]+=1;lift_sum[active]+=(height[active]-tracker.initial_height[active]).clamp_min(0)
                for key,value in dict(height=height,contact=pair,active=active,done=done).items():trace[key].append(value.cpu().clone())
                newly=done&active
                for env_id in newly.nonzero().flatten().tolist():
                    stable=bool(tracker.stable[env_id]);drop=bool(tracker.drop_after_success[env_id])
                    episodes.append(dict(environment=env_id,motion=int(motion[env_id]),policy=int(assignment[env_id]),steps=int(steps[env_id]),
                        stable_success=stable,drop_after_success=drop,retained_success=stable and not drop,
                        max_hold_seconds=float(tracker.max_run[env_id]),mean_lift_meters=float(lift_sum[env_id]/steps[env_id]),
                        decision_count=int(count[env_id]),nonzero_corrections=int(nonzero[env_id]),terminate=bool(info['terminate'].reshape(-1)[env_id])))
                finished|=done;done_ids=done.nonzero().flatten()
                if finished.all():break
                if tick%100==0:print(json.dumps(dict(tick=tick,complete=int(finished.sum()),decisions=int(count.sum()))),flush=True)
            else:raise ValueError('incomplete first episodes')
            if fingerprint(self.model.state_dict())!=actor_hash or (self.normalize_input and fingerprint(self.running_mean_std.state_dict())!=rms_hash):raise ValueError('actor/RMS changed')
            torch.save(dict(**{k:torch.stack(v) for k,v in trace.items()},initial_height=initial_height,
                assignment=assignment,motion=motion,episodes=episodes,control_dt=float(task.dt)),args.run_dir/'episodes.pt')
            widths={'state':55,'reference_action':18,'prediction':(7,3),'valid':7}
            values={}
            for key,parts in decisions.items():
                values[key]=torch.cat(parts) if parts else torch.empty((0,*((widths[key],) if isinstance(widths.get(key),int) else widths.get(key,()))))
            torch.save(values,args.run_dir/'decisions.pt')
            result=dict(run_status='COMPLETED',complete_episodes=len(episodes),required_episodes=768,all_episodes_complete=bool(finished.all()),
                assignment_counts={str(m):torch.bincount(assignment[motion==m],minlength=4).tolist() for m in range(3)},
                actor_and_rms_unchanged=True,wall_seconds=time.monotonic()-begin,inference_seconds=inference_seconds,
                episodes_sha256=sha(args.run_dir/'episodes.pt'),decisions_sha256=sha(args.run_dir/'decisions.pt'))
            (args.run_dir/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    return TaskPlayer


def main():
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--checkpoint-sha256',required=True)
    parser.add_argument('--training-seed',type=int,required=True);parser.add_argument('--eval-seed',type=int,required=True)
    parser.add_argument('--controls',type=Path,required=True);parser.add_argument('--wall-seconds',type=int,default=600)
    args,remaining=parser.parse_known_args()
    if args.run_dir.exists() or ROOT not in args.run_dir.resolve().parents:raise ValueError('unique isolated output required')
    checkpoint=Path(remaining[remaining.index('--checkpoint')+1])
    if sha(checkpoint)!=args.checkpoint_sha256:raise ValueError('actor drift')
    args.run_dir.mkdir(parents=True);begin=time.monotonic();manifest=dict(run_status='RUNNING',pid=os.getpid(),command=sys.argv)
    try:
        sys.path.insert(0,str(ROOT/'third_party/DExplore/dexplore'))
        from isaacgym import gymtorch
        import torch
        import evaluate as original
        original.EvalPlayer=make_player(original,args,torch);sys.argv=[sys.argv[0],*remaining];original.main()
        if sha(checkpoint)!=args.checkpoint_sha256:raise ValueError('actor modified')
        manifest['run_status']='COMPLETED'
    except BaseException as error:manifest.update(run_status='FAILED',error=repr(error));raise
    finally:
        manifest['wall_seconds']=time.monotonic()-begin
        (args.run_dir/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()

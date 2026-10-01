#!/usr/bin/env python3
"""Cold-simulator compact reference or fixed geometry-scheduled pulse replay."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_paired_physical_value_environment import sha


def make_player(original,args,torch,gymtorch):
    from src.task.CmResidual.paired_evaluation import (
        capture_initial,capture_rng,cpu_copy,fingerprint,restore_initial,restore_rng,physical_property_value,
    )
    from src.task.CmResidual.physical_value_contract import HoldTracker
    from src.task.CmResidual.physical_value_live import contacts,snapshot
    from src.task.CmResidual.causal_acquisition import axis_pulse

    class GeometryPlayer(original.EvalPlayer):
        @torch.no_grad()
        def run(self):
            begin=time.monotonic();torch.set_num_threads(2)
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
            task=self.env.task
            task._enable_early_termination=False;task._adaptive_kappa_enabled=False;task._hybrid_init_prob=1.
            if task.num_envs!=96 or abs(task.dt-1/30)>1e-8:
                raise ValueError('frozen native panel mismatch')
            if not torch.allclose(task._pd_action_scale[:3],torch.ones(3,device=self.device)):
                raise ValueError('native wrist target scales changed')
            ids=torch.arange(96,device=self.device);obs=self.env_reset(ids)
            self.get_batch_size(obs['obs'],1)
            if self.is_rnn:self.init_rnn()
            properties=[]
            for env in task.envs:
                actors=[]
                for actor in range(task.gym.get_actor_count(env)):
                    actors.append(dict(name=task.gym.get_actor_name(env,actor),
                        dof=cpu_copy(task.gym.get_actor_dof_properties(env,actor)),
                        rigid_body=physical_property_value(task.gym.get_actor_rigid_body_properties(env,actor)),
                        rigid_shape=physical_property_value(task.gym.get_actor_rigid_shape_properties(env,actor))))
                properties.append(actors)
            if args.mode=='reference':
                initial=capture_initial(task,self,obs,properties)
                trace=None;triggers=None;length=360
            else:
                initial=torch.load(args.initial,map_location='cpu',weights_only=False)
                trace=torch.load(args.trace,map_location='cpu',weights_only=False)
                schedule=torch.load(args.schedule,map_location='cpu',weights_only=False)
                triggers=schedule['triggers']
                length=int(triggers.max())+5
                if length<=0 or length>len(trace['action']):raise ValueError('invalid response window')
            obs=restore_initial(task,self,initial,gymtorch.unwrap_tensor,properties)
            initial_hash=fingerprint(initial)
            if trace is not None and trace['initial_state_fingerprint']!=initial_hash:
                raise ValueError('initial mismatch')
            if args.mode=='reference':torch.save(initial,args.run_dir/'initial_state.pt')
            if [int((task.data_id==i).sum()) for i in range(3)]!=[32,32,32] or (task.start_times!=0).any():
                raise ValueError('frame0 balance changed')
            tracker=HoldTracker(96,self.device);tracker.reset(ids,task._target_states[:,2])
            model_hash=fingerprint(self.model.state_dict())
            rms_hash=fingerprint(self.running_mean_std.state_dict()) if self.normalize_input else None
            parts={k:[] for k in ('state_before','state_after','action','done')};rng=[]
            empty=torch.empty(0,dtype=torch.long,device=self.device)
            for tick in range(length):
                if time.monotonic()-begin>args.wall_seconds:raise TimeoutError('native causal budget')
                if trace is not None:restore_rng(trace['rng'][tick]['before_reset'])
                before_reset=capture_rng()
                obs=self.env_reset(empty)
                if trace is not None:restore_rng(trace['rng'][tick]['before_action'])
                before_action=capture_rng()
                if trace is None:action=self.get_action(obs,True).clamp(-1,1).clone()
                else:action=axis_pulse(trace['action'][tick].to(self.device),triggers,tick,args.axis,args.amplitude)
                parts['state_before'].append(snapshot(task,tracker).cpu())
                parts['action'].append(action.detach().cpu().clone())
                if trace is not None:restore_rng(trace['rng'][tick]['before_physics'])
                before_physics=capture_rng()
                obs,_,done,_=self.env_step(self.env,action)
                tracker.step(task._target_states[:,2],contacts(task).bool().all(-1))
                after=snapshot(task,tracker).cpu()
                if not torch.isfinite(after).all():raise FloatingPointError('nonfinite response')
                parts['state_after'].append(after);parts['done'].append(done.bool().cpu().reshape(-1))
                if trace is None:rng.append(dict(before_reset=before_reset,before_action=before_action,before_physics=before_physics))
                if done.any():
                    if trace is not None:raise ValueError('termination in intervention window')
                    break
                if tick%100==0:print(json.dumps(dict(tick=tick,mode=args.mode)),flush=True)
            if fingerprint(self.model.state_dict())!=model_hash:
                raise ValueError('actor updated')
            if self.normalize_input and fingerprint(self.running_mean_std.state_dict())!=rms_hash:
                raise ValueError('normalizer updated')
            data={k:torch.stack(values) for k,values in parts.items()}
            data.update(initial_state_fingerprint=initial_hash,motion_id=task.data_id.cpu().clone(),control_dt=float(task.dt))
            if trace is None:
                data['rng']=rng;filename='trace.pt'
            else:
                data.update(triggers=triggers,axis=args.axis,amplitude=args.amplitude);filename='response.pt'
            torch.save(data,args.run_dir/filename)
            result=dict(run_status='COMPLETED',mode=args.mode,ticks=len(data['action']),
                        initial_state_fingerprint=initial_hash,output_sha256=sha(args.run_dir/filename),
                        actor_and_rms_unchanged=True,wall_seconds=time.monotonic()-begin)
            (args.run_dir/'results.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(result),flush=True)
    return GeometryPlayer


def main():
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--mode',choices=('reference','replay'),required=True)
    parser.add_argument('--checkpoint-sha256',required=True)
    parser.add_argument('--initial',type=Path);parser.add_argument('--trace',type=Path)
    parser.add_argument('--schedule',type=Path)
    parser.add_argument('--axis',type=int,default=0);parser.add_argument('--amplitude',type=float,default=0.)
    parser.add_argument('--wall-seconds',type=int,default=150)
    args,remaining=parser.parse_known_args()
    if args.run_dir.exists() or ROOT not in args.run_dir.resolve().parents:
        raise ValueError('unique output inside independent worktree required')
    checkpoint=Path(remaining[remaining.index('--checkpoint')+1])
    if sha(checkpoint)!=args.checkpoint_sha256:raise ValueError('checkpoint drift')
    inputs={str(p.resolve()):sha(p) for p in (checkpoint,args.initial,args.trace,args.schedule) if p is not None}
    args.run_dir.mkdir(parents=True);begin=time.monotonic()
    manifest=dict(run_status='RUNNING',pid=os.getpid(),mode=args.mode,command=sys.argv,input_sha256=inputs,
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),no_training=True)
    try:
        sys.path.insert(0,str(ROOT/'third_party/DExplore/dexplore'))
        from isaacgym import gymtorch
        import torch
        import evaluate as original
        original.EvalPlayer=make_player(original,args,torch,gymtorch)
        sys.argv=[sys.argv[0],*remaining];original.main()
        if any(sha(Path(p))!=h for p,h in inputs.items()):raise ValueError('input drift')
        manifest.update(run_status='COMPLETED',inputs_unchanged=True)
    except BaseException as error:
        manifest.update(run_status='FAILED',error=repr(error));raise
    finally:
        manifest['wall_seconds']=time.monotonic()-begin
        (args.run_dir/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()

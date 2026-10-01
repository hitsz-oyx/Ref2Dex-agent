#!/usr/bin/env python3
"""Native bounded finger-preload diagnostic with no learned policy calls."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import types
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import sha


def player_class(original,args,torch):
    from src.task.CmResidual.paired_evaluation import fingerprint
    from src.task.CmResidual.physical_value_live import contacts
    from src.task.CmResidual.static_hold_feasibility import mesh_vertices,static_pd_action
    from src.task.CmResidual.tabletop_clearance import clearance,tabletop_geometry
    from src.task.CmResidual.finger_preload import preload_target,assign_doses,DOSES

    class StaticPlayer(original.EvalPlayer):
        def get_action(self,*args,**kwargs):
            raise RuntimeError('learned policy invocation forbidden in static diagnostic')

        @torch.no_grad()
        def run(self):
            begin=time.monotonic();torch.set_num_threads(2)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            task=self.env.task;device=task._dof_pos.device
            if task.num_envs!=96 or abs(task.dt-1/30)>1e-8 or task.ball_size!=1:
                raise ValueError('frozen native shape/scale/dt')
            task._enable_early_termination=False;task._adaptive_kappa_enabled=False;task._hybrid_init_prob=1.
            generation=json.loads(args.references_manifest.read_text());by_name={r['name']:r for r in generation['references']}
            anchors=[]
            for motion,name in enumerate(task.motion_file):
                record=by_name[Path(name).name]
                if Path(name).resolve()!=Path(record['generated']).parent.resolve():raise ValueError('actual reference path drift')
                anchor=record['plateau_reference_frames_inclusive'][0];anchors.append(anchor)
                for key in ('obj_pos_vel','obj_rot_vel','robot_dof_pos_vel'):
                    if task.hoi_data_dict[motion][key][anchor].any():raise ValueError('reference initial velocity drift')
            anchors=torch.tensor(anchors,device=device)
            original_reset=task._reset_ref_state_init
            def elevated_reset(native,ids):
                original_reset(ids);times=anchors[native.data_id[ids]]
                native.progress_buf[ids]=times;native.start_times[ids]=times
                ref=native.hoi_refs[native.data_id[ids],native.ref_index[ids],times]
                native._set_env_state(env_ids=ids,dof_pos=ref[:,119:137],dof_vel=ref[:,137:155])
            task._reset_ref_state_init=types.MethodType(elevated_reset,task)
            ids=torch.arange(96,device=device);obs=self.env_reset(ids);self.get_batch_size(obs['obs'],1)
            if [int((task.data_id==m).sum()) for m in range(3)]!=[32]*3:raise ValueError('motion coverage')
            if task._dof_vel.any() or task._target_states[:,7:13].any():raise ValueError('ACTUAL initial velocity nonzero')
            base=task._dof_pos.clone();assignment=assign_doses(task.data_id,args.eval_seed).to(device)
            dose=torch.tensor(DOSES,device=device)[assignment]
            goal=preload_target(base,dose,task.dof_limits_lower,task.dof_limits_upper)
            initial_height=task._target_states[:,2].clone()
            rest=task.hoi_refs[task.data_id,task.ref_index,0,108].clone();motion=task.data_id.clone()
            model_hash=fingerprint(self.model.state_dict());rms_hash=fingerprint(self.running_mean_std.state_dict())
            assets=ROOT/'third_party/DExplore/dexplore/data/assets/mjcf'
            vertices=mesh_vertices(assets/'objects/airplane/airplane.obj').to(device)
            table_vertices=mesh_vertices(assets/'objects/table/table.obj').to(device)
            table=task._table_states.clone();normal,top,axis=tabletop_geometry(table_vertices,table)
            if axis!=1:raise ValueError('native tabletop thin axis changed')
            trace={key:[] for key in ('object_root','native_q','contact','hand_force','object_force','clearance','progress','action','target')}
            torch.save(dict(base_q=base.cpu(),goal_q=goal.cpu(),dose_assignment=assignment.cpu(),dose_rad=dose.cpu(),
                native_lower=task.dof_limits_lower.cpu(),native_upper=task.dof_limits_upper.cpu(),object_root=task._target_states.cpu().clone(),table_root=table.cpu(),
                initial_height=initial_height.cpu(),rest_height=rest.cpu(),motion=motion.cpu(),anchors=anchors.cpu(),
                object_vertex_count=len(vertices),table_thin_axis=axis,table_normal=normal.cpu(),table_top=top.cpu()),args.run_dir/'initial.pt')
            run=torch.zeros(96,dtype=torch.long,device=device);longest=run.clone();maximum_error=0.
            for tick in range(90):
                if time.monotonic()-begin>args.wall_seconds:raise TimeoutError('static diagnostic budget')
                action,error=static_pd_action(task,goal);maximum_error=max(maximum_error,error)
                _,_,done,_=self.env_step(self.env,action)
                if done.bool().any():raise ValueError('unexpected native terminal inside mechanical horizon')
                if not torch.equal(task.progress_buf,anchors[motion]+tick+1):raise ValueError('native mechanical progress drift')
                if not torch.equal(task._table_states,table):raise ValueError('static table moved')
                pair=contacts(task);root=task._target_states.clone()
                separation=clearance(root,table,vertices,table_vertices)
                valid=(root[:,2]-rest>=.03)&(root[:,2]-initial_height>=-.01)&pair.bool().all(-1)&(separation>=.02)
                run=torch.where(valid,run+1,torch.zeros_like(run));longest=torch.maximum(longest,run)
                values=dict(object_root=root,native_q=task._dof_pos,contact=pair,
                    hand_force=task._contact_forces[:,task._contact_body_ids],object_force=task._tar_contact_forces,
                    clearance=separation,progress=task.progress_buf,action=action,target=goal)
                for key,value in values.items():
                    if not torch.isfinite(value).all():raise ValueError('nonfinite static trace')
                    trace[key].append(value.cpu().clone())
            if fingerprint(self.model.state_dict())!=model_hash or fingerprint(self.running_mean_std.state_dict())!=rms_hash:
                raise ValueError('unused model/RMS changed')
            torch.save(dict(**{k:torch.stack(v) for k,v in trace.items()},max_hold_steps=longest.cpu()),args.run_dir/'trace.pt')
            result=dict(run_status='COMPLETED',mechanical_trajectories=96,physical_steps_each=90,no_policy_calls=True,
                actor_and_rms_unchanged=True,initial_velocities_zero=True,max_pd_goal_error=maximum_error,
                dose_counts=torch.bincount(assignment,minlength=4).cpu().tolist(),retained75_count=int((longest>=75).sum()),initial_sha256=sha(args.run_dir/'initial.pt'),
                trace_sha256=sha(args.run_dir/'trace.pt'),wall_seconds=time.monotonic()-begin)
            (args.run_dir/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    return StaticPlayer


def main():
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--checkpoint-sha256',required=True)
    parser.add_argument('--training-seed',type=int,required=True);parser.add_argument('--eval-seed',type=int,required=True)
    parser.add_argument('--references-manifest',type=Path,required=True);parser.add_argument('--wall-seconds',type=int,default=180)
    args,remaining=parser.parse_known_args()
    if args.run_dir.exists() or ROOT not in args.run_dir.resolve().parents:raise ValueError('unique isolated output')
    checkpoint=Path(remaining[remaining.index('--checkpoint')+1])
    if sha(checkpoint)!=args.checkpoint_sha256:raise ValueError('checkpoint drift')
    args.run_dir.mkdir(parents=True);begin=time.monotonic();manifest=dict(run_status='RUNNING',pid=os.getpid(),command=sys.argv)
    try:
        sys.path.insert(0,str(ROOT/'third_party/DExplore/dexplore'))
        from isaacgym import gymtorch
        import torch
        import evaluate as original
        original.EvalPlayer=player_class(original,args,torch);sys.argv=[sys.argv[0],*remaining];original.main()
        if sha(checkpoint)!=args.checkpoint_sha256:raise ValueError('checkpoint changed')
        manifest['run_status']='COMPLETED'
    except BaseException as error:manifest.update(run_status='FAILED',error=repr(error));raise
    finally:
        manifest['wall_seconds']=time.monotonic()-begin
        (args.run_dir/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()

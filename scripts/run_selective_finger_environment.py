#!/usr/bin/env python3
"""Native prospective independent effective finger responses."""
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
    from src.task.CmResidual.paired_evaluation import fingerprint,capture_rng
    from src.task.CmResidual.physical_value_live import contacts
    from src.task.CmResidual.static_hold_feasibility import mesh_vertices,static_pd_action
    from src.task.CmResidual.tabletop_clearance import clearance,tabletop_geometry
    from src.task.CmResidual.frame0_tracking import tracking_target
    from src.task.CmResidual.observation_hold_policy import hold_context,canonical_action
    from src.task.CmResidual.reference_target_policy import SCHEMA,target_from_residual
    from src.task.CmResidual.dexplore_bc_policy import DExploreBcPolicy,normalized_action
    from src.task.CmResidual.support_response import assignments,placement_offsets,support_features
    from src.task.CmResidual.selective_finger_response import PRIMITIVES,primitive_target,INDEPENDENT,validate_dof_names
    from src.task.CmResidual.native_reset_transaction import NativeResetQueue
    from isaacgym import gymapi,gymtorch

    class StaticPlayer(original.EvalPlayer):
        def get_action(self,*args,**kwargs):
            raise RuntimeError('learned policy invocation forbidden in static diagnostic')

        @torch.no_grad()
        def run(self):
            begin=time.monotonic();torch.set_num_threads(2)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            task=self.env.task;device=task._dof_pos.device
            if task.num_envs!=768 or abs(task.dt-1/30)>1e-8 or task.ball_size!=1:
                raise ValueError('frozen native shape/scale/dt')
            task._enable_early_termination=False;task._adaptive_kappa_enabled=False;task._hybrid_init_prob=1.
            generation=json.loads(args.references_manifest.read_text());by_name={r['name']:r for r in generation['references']}
            anchors=[];stops=[];lift_starts=[]
            for motion,name in enumerate(task.motion_file):
                record=by_name[Path(name).name]
                if Path(name).resolve()!=Path(record['generated']).parent.resolve():raise ValueError('actual reference path drift')
                anchor,stop=record['plateau_reference_frames_inclusive'];anchors.append(anchor);stops.append(stop);lift_starts.append(record['first_lift_interval'][0])
                for key in ('obj_pos_vel','obj_rot_vel','robot_dof_pos_vel'):
                    if task.hoi_data_dict[motion][key][anchor].any():raise ValueError('reference initial velocity drift')
            anchors=torch.tensor(anchors,device=device);stops=torch.tensor(stops,device=device);lift_starts=torch.tensor(lift_starts,device=device)
            native_gym=task.gym;reset_queue=NativeResetQueue(native_gym,gymtorch.wrap_tensor);task.gym=reset_queue
            ids=torch.arange(768,device=device);obs=self.env_reset(ids);self.get_batch_size(obs['obs'],1)
            if [int((task.data_id==m).sum()) for m in range(3)]!=[256]*3:raise ValueError('motion coverage')
            if task.progress_buf.any() or task.start_times.any():raise ValueError('native initial frame0 required')
            if task.ref_index.any():raise ValueError('single generated reference index0 required')
            base=task._dof_pos.clone()
            motion=task.data_id.clone();assignment=assignments(motion,args.eval_seed).to(device)
            offsets=placement_offsets(768,args.eval_seed).to(device);before_placement=task._target_states.clone()
            task._target_states[:,:2]+=offsets
            target_ids=task._tar_actor_ids[ids].to(torch.int32)
            task.gym.set_actor_root_state_tensor_indexed(task.sim,gymtorch.unwrap_tensor(task._root_states),gymtorch.unwrap_tensor(target_ids),len(target_ids))
            root_commit_ids,dof_commit_ids=reset_queue.commit(task.sim,task._root_states,task._dof_state,gymtorch.unwrap_tensor)
            task.gym=native_gym
            sdk_ids=torch.tensor([task.gym.get_actor_index(task.envs[i],task._target_handles[i],gymapi.DOMAIN_SIM) for i in range(768)],dtype=torch.int32,device=device)
            if not torch.equal(sdk_ids,target_ids):raise ValueError('SDK target actor index mismatch')
            expected=before_placement.clone();expected[:,:2]+=offsets
            if not torch.allclose(task._target_states,expected,atol=1e-7,rtol=0):raise ValueError('initial XY placement drift')
            class FrozenStateGym:
                def __init__(self,gym):self.original=gym
                def __getattr__(self,name):
                    if name in ('set_actor_root_state_tensor_indexed','set_actor_root_state_tensor','set_dof_state_tensor','set_dof_state_tensor_indexed','set_sim_rigid_body_states','apply_rigid_body_force_tensors','apply_rigid_body_force_at_pos_tensors'):
                        raise RuntimeError('post-initialization state/force write forbidden: '+name)
                    return getattr(self.original,name)
            task.gym=FrozenStateGym(task.gym)
            parameters=torch.tensor(PRIMITIVES,device=device)[assignment]
            decision_steps=lift_starts[motion]-8
            decision_features=torch.full((768,69),float('nan'),device=device)
            decision_context=torch.full((768,70),float('nan'),device=device)
            decision_recorded=torch.zeros(768,dtype=torch.bool,device=device)
            learned_model=mean=std=None;learned_fingerprint=stats_fingerprint=None
            if args.mode=='policy':
                if sha(args.policy_checkpoint)!=args.policy_sha256:raise ValueError('policy checkpoint drift')
                p=torch.load(args.policy_checkpoint,map_location='cpu',weights_only=False)
                if p['schema']!=SCHEMA or p['official_policy_checkpoint'] is not None or p['source_actor_weights_used'] or p['updates']!=2000:raise ValueError('scratch policy provenance/schema')
                learned_model=DExploreBcPolicy(70,18,tuple(p['hidden_dims'])).to(device)
                learned_model.load_state_dict(p['model']);learned_model.eval()
                mean=p['observation_mean'].to(device);std=p['observation_std'].to(device)
                learned_fingerprint=fingerprint(learned_model.state_dict());stats_fingerprint=fingerprint((mean,std))
                if learned_fingerprint!=p['final_model_fingerprint']:raise ValueError('policy model fingerprint mismatch')
            names=task.gym.get_actor_dof_names(task.envs[0],task.humanoid_handles[0]);validate_dof_names(names)
            body_properties=[]
            for i in range(768):
                props=task.gym.get_actor_rigid_body_properties(task.envs[i],task._target_handles[i])
                if len(props)!=1:raise ValueError('single rigid target body required')
                p=props[0]
                if p.flags & gymapi.RIGID_BODY_DISABLE_GRAVITY or p.mass<=0:raise ValueError('target gravity/mass invalid')
                body_properties.append(dict(mass=float(p.mass),flags=int(p.flags),
                    inertia=[[float(getattr(getattr(p.inertia,axis),coordinate)) for coordinate in ('x','y','z')] for axis in ('x','y','z')]))
            gravity=task.sim_params.gravity
            physical_metadata=dict(native_dof_names=names,independent_finger_coordinates=list(INDEPENDENT),object_body_properties=body_properties,gravity=[gravity.x,gravity.y,gravity.z],actors_per_env=task.get_num_actors_per_env())
            if physical_metadata['actors_per_env']!=3:raise ValueError('unexpected possible supporting actors')
            if abs(gravity.z+9.81)>1e-5 or gravity.x or gravity.y:raise ValueError('native gravity drift')
            (args.run_dir/'physical_metadata.json').write_text(json.dumps(physical_metadata,indent=2)+'\n')
            initial_height=task._target_states[:,2].clone()
            rest=task.hoi_refs[task.data_id,task.ref_index,0,108].clone();motion=task.data_id.clone()
            model_hash=fingerprint(self.model.state_dict());rms_hash=fingerprint(self.running_mean_std.state_dict())
            assets=ROOT/'third_party/DExplore/dexplore/data/assets/mjcf'
            vertices=mesh_vertices(assets/'objects/airplane/airplane.obj').to(device)
            table_vertices=mesh_vertices(assets/'objects/table/table.obj').to(device)
            table=task._table_states.clone();normal,top,axis=tabletop_geometry(table_vertices,table)
            if axis!=1:raise ValueError('native tabletop thin axis changed')
            trace={key:[] for key in ('object_root','native_q','native_dq','context','contact','hand_force','object_force','clearance','progress','action','target','model_residual','base_target','effective_finger_delta')}
            torch.save(dict(base_q=base.cpu(),initial_dof_vel=task._dof_vel.cpu().clone(),arm_assignment=assignment.cpu(),
                rng=capture_rng(),mode=args.mode,initial_contact=contacts(task).cpu(),before_placement=before_placement.cpu(),placement_offsets=offsets.cpu(),
                initial_state_semantics='committed native reset packet before firstsimulate',reset_root_actor_ids=root_commit_ids.cpu(),reset_dof_actor_ids=dof_commit_ids.cpu(),reset_events=reset_queue.events,refreshes_suppressed=reset_queue.refreshes_suppressed,
                placement_seed=args.eval_seed+11000,assignment_seed=args.eval_seed+12000,primitive_parameters=parameters.cpu(),decision_steps=decision_steps.cpu(),
                pd_offset=task._pd_action_offset.cpu(),pd_scale=task._pd_action_scale.cpu(),phase_stop=stops.cpu(),lift_start=lift_starts.cpu(),
                native_reference_q=task.hoi_refs[:,0,:int(stops.max())+1,119:137].cpu().clone(),
                native_lower=task.dof_limits_lower.cpu(),native_upper=task.dof_limits_upper.cpu(),object_root=task._target_states.cpu().clone(),table_root=table.cpu(),
                initial_height=initial_height.cpu(),rest_height=rest.cpu(),motion=motion.cpu(),anchors=anchors.cpu(),
                native_dof_names=names,independent_finger_coordinates=list(INDEPENDENT),object_vertex_count=len(vertices),table_thin_axis=axis,table_normal=normal.cpu(),table_top=top.cpu()),args.run_dir/'initial.pt')
            current_clearance=clearance(task._target_states,table,vertices,table_vertices)
            run=torch.zeros(768,dtype=torch.long,device=device);longest=run.clone();maximum_error=0.
            phase_steps=torch.zeros_like(run)
            for tick in range(202):
                if time.monotonic()-begin>args.wall_seconds:raise TimeoutError('static diagnostic budget')
                next_progress=torch.minimum(task.progress_buf+1,stops[motion])
                reference=task.hoi_refs[task.data_id,task.ref_index,next_progress,119:137]
                context=hold_context(task,stops)
                at_decision=task.progress_buf==decision_steps
                if at_decision.any():
                    if decision_recorded[at_decision].any():raise ValueError('duplicate decision')
                    decision_context[at_decision]=context[at_decision]
                    decision_features[at_decision]=support_features(context,initial_height,current_clearance)[at_decision]
                    decision_recorded|=at_decision
                model_residual=canonical_action(normalized_action(learned_model,context,mean,std))
                base_goal=target_from_residual(reference,model_residual,task.dof_limits_lower,task.dof_limits_upper)
                active=task.progress_buf>=decision_steps
                applied=parameters*active[:,None]
                goal=primitive_target(base_goal,applied,task.dof_limits_lower,task.dof_limits_upper)
                action,error=static_pd_action(task,goal);maximum_error=max(maximum_error,error)
                _,_,done,_=self.env_step(self.env,action)
                if done.bool().any():raise ValueError('unexpected native terminal inside mechanical horizon')
                if not torch.equal(task.progress_buf,torch.full_like(task.progress_buf,tick+1)):raise ValueError('native mechanical progress drift')
                if not torch.equal(task._table_states,table):raise ValueError('static table moved')
                pair=contacts(task);root=task._target_states.clone()
                separation=clearance(root,table,vertices,table_vertices)
                current_clearance=separation
                phase=(task.progress_buf>=anchors[motion])&(task.progress_buf<=stops[motion])
                phase_steps+=phase.long()
                valid=phase&(root[:,2]-initial_height>=.03)&pair.bool().all(-1)&(separation>=.02)
                run=torch.where(valid,run+1,torch.zeros_like(run));longest=torch.maximum(longest,run)
                values=dict(object_root=root,native_q=task._dof_pos,native_dq=task._dof_vel,context=context,contact=pair,
                    hand_force=task._contact_forces[:,task._contact_body_ids],object_force=task._tar_contact_forces,
                    clearance=separation,progress=task.progress_buf,action=action,target=goal,model_residual=model_residual,base_target=base_goal,effective_finger_delta=goal[:,INDEPENDENT]-base_goal[:,INDEPENDENT])
                for key,value in values.items():
                    if not torch.isfinite(value).all():raise ValueError('nonfinite static trace')
                    trace[key].append(value.cpu().clone())
            if (phase_steps!=90).any():raise ValueError('incomplete native holding phase')
            if fingerprint(self.model.state_dict())!=model_hash or fingerprint(self.running_mean_std.state_dict())!=rms_hash:
                raise ValueError('unused model/RMS changed')
            if not decision_recorded.all() or not torch.isfinite(decision_features).all():raise ValueError('complete decision rows')
            torch.save(dict(decision_features=decision_features.cpu(),decision_context=decision_context.cpu(),**{k:torch.stack(v) for k,v in trace.items()},max_hold_steps=longest.cpu()),args.run_dir/'trace.pt')
            if args.mode=='policy' and (fingerprint(learned_model.state_dict())!=learned_fingerprint or fingerprint((mean,std))!=stats_fingerprint):raise ValueError('policy/statistics changed')
            result=dict(mode=args.mode,run_status='COMPLETED' ,mechanical_trajectories=768,physical_steps_each=202,phase_steps_each=90,frame0_initialization=True,no_source_actor_calls=True,learned_policy_calls=202 if args.mode=='policy' else 0,
                actor_and_rms_unchanged=True,initial_velocities_preserved=True,max_pd_goal_error=maximum_error,
                initial_object_placement_only=True,no_object_writes_after_first_tick=True,assignment_seed=args.eval_seed+12000,placement_seed=args.eval_seed+11000,schema=SCHEMA,policy_model_fingerprint=learned_fingerprint,policy_statistics_fingerprint=stats_fingerprint,policy_checkpoint_sha256=args.policy_sha256,physical_metadata_sha256=sha(args.run_dir/'physical_metadata.json'),retained75_count=int((longest>=75).sum()),initial_sha256=sha(args.run_dir/'initial.pt'),
                trace_sha256=sha(args.run_dir/'trace.pt'),wall_seconds=time.monotonic()-begin)
            (args.run_dir/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    return StaticPlayer


def main():
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument('--mode',choices=['policy'],required=True);parser.add_argument('--policy-checkpoint',type=Path);parser.add_argument('--policy-sha256');
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--checkpoint-sha256',required=True)
    parser.add_argument('--training-seed',type=int,required=True);parser.add_argument('--eval-seed',type=int,required=True)
    parser.add_argument('--references-manifest',type=Path,required=True);parser.add_argument('--wall-seconds',type=int,default=180)
    args,remaining=parser.parse_known_args()
    if args.mode=='policy' and (not args.policy_checkpoint or not args.policy_sha256):raise ValueError('policy checkpoint/SHA required')
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

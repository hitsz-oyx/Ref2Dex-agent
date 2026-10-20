"""Full recorded episode, measured-history WAM24 / display8 rolling preview."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

TASK=Path(__file__).resolve().parents[2]
ROOT=Path(os.environ.get('REF2DEX_RUN_ROOT',TASK.parents[2])).resolve()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('audit-manifest','geometry-cache','wam','mano-root','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--sequence',default='s1_apple_lift')
    parser.add_argument('--gpu',type=int,default=7)
    parser.add_argument('--seconds',type=int,default=1200)
    parser.add_argument('--history-source',choices=('measured','predicted'),default='measured')
    args=parser.parse_args();out=args.output.resolve()
    if out.exists() or ROOT/'outputs/wam-rl-policy-v3' not in out.parents:
        raise ValueError('fresh task-local output required')
    if not 30<=args.seconds<=1800:raise ValueError('bounded preparation time required')
    if __import__('shutil').disk_usage(ROOT).free<20*1024**3:raise ValueError('20GiB disk floor')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',REF2DEX_DEEP_OPT='all')
    sys.path.insert(0,str(TASK/'src'))
    import numpy as np
    import torch
    import trimesh
    from scipy.spatial.transform import Rotation
    from wam_rl_policy_v3.teacher_dataset import TeacherRolloutCorpus
    from wam_rl_policy_v3.robot_adapter import InspireAdapter
    from wam_rl_policy_v3.wam_v4_runtime import V4Runtime,collate_v4,v4_online_samples
    from wam_rl_policy_v3.measured_execution import task_wam_goal
    from wam_rl_policy_v3.mano_preview import forecast_world,fit_mano,rolling_windows,forecast_object_poses
    from wam_rl_policy_v3.goal_scheduler import rigid_goal_delta
    autoregressive=args.history_source=='predicted'
    torch.set_num_threads(2);torch.manual_seed(246)
    started=time.monotonic()
    def budget():
        if time.monotonic()-started>args.seconds:raise TimeoutError('preparation deadline')
    corpus=TeacherRolloutCorpus(args.audit_manifest.resolve(),'train')
    index=next(i for i,r in enumerate(corpus.records) if r['sequence']==args.sequence)
    episode=corpus.episode(index);row=corpus.records[index];source=Path(row['source'])
    with np.load(source/'trajectory.npz') as z:
        poses=z['obj'].copy();hands=InspireAdapter.canonical_hand(z['hand']);q=z['q'].copy()
        ticks=z['frame_tick'].copy()
    n=len(poses);prefix=row['training_prefix_commands']
    if n>1200 or not np.array_equal(ticks,np.arange(n)):raise ValueError('bounded consecutive30Hz recording required')
    with np.load(corpus.root/row['sidecar']/'admission.npz') as z:table_pose=z['table_pose'].copy()
    out.mkdir(parents=True);mesh_dir=out/'mano-assets';mesh_dir.mkdir()
    goals=np.repeat(episode['local_goal_pose'][-1:,:,:],n,axis=0)
    goals[:len(episode['local_goal_pose'])]=episode['local_goal_pose']
    targets=np.zeros_like(hands);valid=np.zeros(n,bool)
    anchor_ids=np.full(n,-1,np.int32);horizons=np.zeros(n,np.int32)
    forecasts=[];windows=[];goal_errors=[];scene_forecasts=[];rigid_errors=[]
    input_hands=[];input_poses=[];input_goals=[];stop=None
    roll_hands=np.full_like(hands,np.nan);roll_poses=np.full_like(poses,np.nan)
    roll_hands[:4]=hands[:4];roll_poses[:4]=poses[:4]
    runtime=V4Runtime(args.wam.resolve(),[dict(object_code='dexplore/'+row['object'],scale=1.)],
        args.geometry_cache.resolve(),dense_manifest=args.geometry_cache/'surface-manifest.json',
        allow_geometry_replacement=True,device='cuda:0')
    step=runtime.step
    # Include the final observed endpoint of the admitted command prefix.
    for anchor,frames in rolling_windows(n if autoregressive else prefix+1):
        budget()
        if not autoregressive and not episode['active'][anchor]:continue
        hp=roll_hands if autoregressive else hands;op=roll_poses if autoregressive else poses
        history=hp[anchor-3:anchor+1];object_history=op[anchor-3:anchor+1]
        objects=np.c_[object_history[:,:3,3],Rotation.from_matrix(object_history[:,:3,:3]).as_quat()][None]
        if autoregressive:
            goal=rigid_goal_delta(op[anchor],goals[anchor],runtime.geometry[0][2]);error=0.
        else:goal,error=task_wam_goal(op[anchor],goals[anchor],episode['task'][anchor],runtime.geometry[0][2])
        sample=v4_online_samples(objects,history[None],runtime.geometry,goal_delta=goal[None])[0]
        if not runtime.workspace_valid([sample])[0]:
            if not autoregressive:raise ValueError('WAM workspace invalid at tick%d'%anchor)
            stop=dict(anchor=anchor,reason='predicted_state_outside_WAM_workspace');break
        batch={k:v.to(runtime.device) for k,v in collate_v4([sample]).items()}
        with torch.inference_mode(),torch.autocast('cuda',enabled=False):
            prediction=runtime.predict(batch) if autoregressive else runtime.predict_hand(batch)
            delta=prediction['trajectory'][0,:,0].float().cpu().numpy()
        if not np.isfinite(delta).all():stop=dict(anchor=anchor,reason='nonfinite_hand_prediction');break
        world=forecast_world(hp[anchor],delta,op[anchor])
        if autoregressive:
            scene=prediction['scene'][0,0].float().cpu().numpy()
            if not np.isfinite(scene).all():stop=dict(anchor=anchor,reason='nonfinite_object_prediction');break
            object_world,residual=forecast_object_poses(runtime.geometry[0][2],scene,op[anchor])
            roll_hands[frames]=world[1:len(frames)+1];roll_poses[frames]=object_world[:len(frames)]
            scene_forecasts.append(scene);rigid_errors.append(residual)
        targets[frames]=world[1:len(frames)+1];valid[frames]=True
        anchor_ids[frames]=anchor;horizons[frames]=frames-anchor
        forecasts.append(delta);goal_errors.append(error)
        input_hands.append(history.copy());input_poses.append(object_history.copy());input_goals.append(goal)
        windows.append(dict(anchor=int(anchor),history_ticks=list(range(anchor-3,anchor+1)),
            first_frame=int(frames[0]),last_frame=int(frames[-1]),consumed=len(frames)))
        if len(windows)%16==0:print(json.dumps(dict(stage='prediction',windows=len(windows),anchor=anchor)),flush=True)
    query=runtime.geometry[0][2].copy();peak=torch.cuda.max_memory_allocated()/2**20
    del runtime,batch,prediction;torch.cuda.empty_cache()
    ids=np.flatnonzero(valid)
    if not len(ids):raise ValueError('no valid predicted frames to render')
    fit_hand=np.zeros_like(hands);errors=np.full((n,11),np.nan,np.float32)
    # The scale uses an actual observed current hand, never a predicted future.
    scale=fit_mano(hands[3:4],args.mano_root,steps=50)['scale'];faces=None
    # Bounded batches share one current-only scale for the entire video.
    for lo in range(0,len(ids),25):
        budget();part=ids[lo:lo+25]
        fitted=fit_mano(targets[part],args.mano_root,steps=300,fixed_scale=scale)
        faces=fitted['faces'];fit_hand[part]=fitted['landmarks'];errors[part]=fitted['errors_m']
        for frame,vertices in zip(part,fitted['vertices']):
            trimesh.Trimesh(vertices=vertices,faces=faces,process=False).export(mesh_dir/('%04d.obj'%frame))
            (mesh_dir/('%04d.urdf'%frame)).write_text('<robot name="mano"><link name="surface"><visual><geometry><mesh filename="%04d.obj"/></geometry></visual></link></robot>'%frame)
        print(json.dumps(dict(stage='fitting',frames_done=min(lo+25,len(ids)),frames_total=len(ids),elapsed_s=time.monotonic()-started)),flush=True)
    raw_error=np.full(n,np.nan,np.float32)
    raw_error[valid]=np.linalg.norm(targets[valid]-hands[valid],axis=-1).mean(-1)*1000
    extra={}
    if autoregressive:
        last=int(ids[-1]);roll_poses[last+1:]=roll_poses[last]
        extra=dict(display_object_poses=roll_poses,scene_delta_anchor=np.asarray(scene_forecasts),
            rigid_residual_m=np.asarray(rigid_errors),flow_query=query)
    np.savez_compressed(out/'preview.npz',target_hand=targets,prediction_valid=valid,anchor_tick=anchor_ids,
        horizon_frame=horizons,wam_delta_anchor=np.asarray(forecasts),fit_hand=fit_hand,fit_faces=faces,
        fit_errors_m=errors,teacher_hand=hands,teacher_q=q,object_poses=poses,goal_poses=goals,
        goal_pose=goals[0],table_pose=table_pose,input_hand_history=np.asarray(input_hands),
        input_object_history=np.asarray(input_poses),input_goal_delta=np.asarray(input_goals),**extra)
    def nullable(values):return [float(v) if np.isfinite(v) else None for v in values]
    report=dict(status='COMPLETED',preview_mode='rolling_measured_history',work_version='wam-mano-rolling-v1',task='wam-rl-policy-v3',
        experiment_id='P-20261020-wam-mano-preview',run_id=out.name,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        python=sys.executable,torch=torch.__version__,gpu=args.gpu,sequence=args.sequence,object=row['object'],
        source=str(source),source_manifest=str(args.audit_manifest.resolve()),wam=str(args.wam.resolve()),checkpoint_step=step,
        mano_root=str(args.mano_root),frames=n,recording_hz=30,recording_duration_s=n/30,
        admitted_prefix_commands=prefix,forecast_horizon=24,replan_stride=8,windows=windows,
        history_source='recorded measured hand/object H4 at every anchor; no predicted-state feedback',
        goal_source='same causal TaskController as teacher corpus; update from every measured tick',
        tail_policy='full recording retained; warmup and frames beyond admission show teacher only, no fabricated forecast',
        physics='offline forecast comparison; no action execution or physical rollout',
        mano_fit=dict(scale=scale,fit_mean_mm=float(np.nanmean(errors)*1000),fit_max_mm=float(np.nanmax(errors)*1000),
            shape='one fixed current-only palm scale; neutral betas',fitting='independent bounded25-frame batches; no smoothing of raw WAM outputs'),
        per_frame_fit_mean_mm=nullable(errors.mean(-1)*1000),per_frame_prediction_error_mm=nullable(raw_error),
        max_task_wam_goal_error_m=max(goal_errors),predicted_frames=int(valid.sum()),
        peak_cuda_mib=max(peak,torch.cuda.max_memory_allocated()/2**20),elapsed_s=time.monotonic()-started,
        budget=dict(seconds=args.seconds,final_artifact_mib=150,temporary_artifact_mib=512,disk_floor_gib=20))
    if autoregressive:
        report.pop('max_task_wam_goal_error_m')
        report.update(preview_mode='rolling_predicted_history',work_version='wam-mano-autoregressive-v1',
            history_source='only recorded H4 ticks0..3 initializes; every subsequent H4 uses old predicted hand and rigid-projected object poses',
            goal_source='fixed replay of prior measured-controller spatial goal sequence for matched instructions; last goal held beyond tick544; not a virtual TaskController or causal execution test',
            tail_policy='forecast continues through full recording; after a declared numerical/workspace stop, teacher comparison continues and right object holds last predicted pose',
            rollout_stop=stop,object_prediction='32 same-ID scene displacements projected to proper SE3 by Kabsch; no scale or translation correction',
            rigid_projection_mean_mm=float(np.mean(rigid_errors)*1000),rigid_projection_max_mm=float(np.max(rigid_errors)*1000),
            future_measured_hand_object_inputs=0)
    (out/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='COMPLETED',frames=n,windows=len(windows),elapsed_s=report['elapsed_s'],peak_cuda_mib=report['peak_cuda_mib'])),flush=True)


if __name__=='__main__':main()

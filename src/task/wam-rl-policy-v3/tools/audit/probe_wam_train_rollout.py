"""Frozen WAM TRAIN GRAB windows: measured versus free history, bounded stride."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

TASK=Path(__file__).resolve().parents[2]
ROOT=TASK.parents[2]


def build_sample(template,poses,hands,goal_pose):
    """Both real hand slots, same canonical queries and training endpoint goal."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    from wam_rl_policy_v3.wam_v4_runtime import v4_online_samples
    surface=template['xyz'];normals=template['features'][:,:3]
    flow=template['start_object_points']
    geometry=(surface,normals,flow,template['interaction_points'][0],template['interaction_normals'][0])
    rotation=poses[-1,:3,:3];center=poses[-1,:3,3]
    relative=np.linalg.inv(poses[-1])@goal_pose
    goal=flow@relative[:3,:3].T+relative[:3,3]
    objects=np.c_[poses[:,:3,3],Rotation.from_matrix(poses[:,:3,:3]).as_quat()][None]
    observed=v4_online_samples(objects,hands[None,:,0],[geometry],goal_delta=(goal-flow)[None])[0]
    sample=dict(template)
    for k in ['xyz','features','start_hand','start_hand_history','start_hand_valid']:
        sample[k]=observed[k]
    local=(hands-center)@rotation
    present=template['start_hand_valid']
    sample['start_hand']=np.where(present[...,None],local[-1],0).astype(np.float32)
    history=np.concatenate(((local[-1]-local[-2])*30,(local[-1]-2*local[-2]+local[-3])*900,local[-1]-local[0]),axis=-1)
    sample['start_hand_history']=np.where(present[...,None],history,0).astype(np.float32)
    sample['start_hand_valid']=present.copy()
    sample['goal_object_points']=goal.astype(np.float32)
    return sample


def infer(args):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',REF2DEX_DEEP_OPT='all')
    sys.path.insert(0,str(TASK/'src'))
    sys.path.insert(0,str(TASK.parent/'ptv3-shared-wam-v4/src'))
    import numpy as np
    import torch
    from ptv3_shared_wam_v4 import paths
    from ptv3_shared_wam_v4.model import SharedWAMV4
    from pose_mesh_data.v4_data import V4Main3Windows,v4_collate
    from wam_rl_policy_v3.mano_preview import forecast_object_poses
    from wam_rl_policy_v3.deferred_checks import install_clean_hooks,checked_prediction
    from ptv3_shared_wam.model import clean
    out=args.output.resolve()
    if out.exists() or ROOT/'outputs/wam-rl-policy-v3' not in out.parents:raise ValueError('fresh task output required')
    if __import__('shutil').disk_usage(ROOT).free<20*1024**3:raise ValueError('20GiB floor')
    out.mkdir(parents=True);started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(246)
    meta=json.loads((args.main3/'processed/manifest.json').read_text())
    relocated=json.loads(json.dumps(meta))
    for source in relocated['sources']:
        for split,record in source['indices'].items():
            record['path']=str(args.main3.resolve()/'processed'/Path(record['path']).name)
    derived=out/'relocated-main3';(derived/'processed').mkdir(parents=True)
    (derived/'processed/manifest.json').write_text(json.dumps(relocated,indent=2)+'\n')
    ds=V4Main3Windows(derived,'train')
    records=json.loads((args.fixture/'metadata.json').read_text())['records']
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if checkpoint['step']!=80000:raise ValueError('same frozen step80000 required')
    if checkpoint['stats']['recipe']['main3_identity']!=meta['normalization_source_manifest_sha256']:
        raise ValueError('training source identity mismatch')
    config=checkpoint['config'];model=SharedWAMV4(checkpoint['stats'],config).cuda().eval().requires_grad_(False)
    model.load_state_dict(checkpoint['model'],strict=True);del checkpoint
    install_clean_hooks(clean)
    def forward(sample):
        if time.monotonic()-started>600:raise TimeoutError('ten-minute inference deadline')
        origin=np.asarray(config['voxel_origin_m']);xyz=np.r_[sample['xyz'],sample['start_hand'][sample['start_hand_valid']]]
        grid=np.floor((xyz-origin)/config['voxel_m'])
        if not np.isfinite(grid).all() or grid.min()<0 or grid.max()>65535:raise ValueError('input outside WAM workspace')
        batch={k:v.cuda() for k,v in v4_collate([sample]).items()}
        with torch.inference_mode(),torch.autocast('cuda',enabled=False),checked_prediction():
            result=model(batch)
        return result['trajectory'][0].float().cpu().numpy(),result['scene'][0,0].float().cpu().numpy()
    reports=[]
    with np.load(args.fixture/'baseline.npz') as fixture:
        for case,record in enumerate(records):
            template=ds[record['index']]
            fields=['xyz','features','start_hand','start_hand_history','trajectory_target','scene_target','goal_object_points']
            exported_diff={k:float(np.max(abs(template[k].astype(float)-fixture[f'{case}__base__{k}'].astype(float)))) for k in fields}
            if max(exported_diff.values())!=0:raise ValueError('known frozen TRAIN sample differs')
            source=ds.base.sources[1];local=record['index']-ds.base.offsets[1]
            seq_id,slot,anchor,_=map(int,source.rows[local]);sequence=source.sequence(source.sequences[seq_id])
            end=min(anchor+180,len(sequence['hand'])-25)
            if end-anchor<8:raise ValueError('at least one consumed chunk required')
            start=anchor-3;real_hand=np.asarray(sequence['hand'][start:end+25],np.float64).copy()
            real_pose=np.asarray(sequence['poses'][start:end+25,slot],np.float64).copy()
            # Endpoints through end+24 are instruction inputs, not hand-history feedback.
            length=end-start+1;roll_hand=np.full_like(real_hand[:length],np.nan);roll_pose=np.full_like(real_pose[:length],np.nan)
            roll_hand[:4]=real_hand[:4];roll_pose[:4]=real_pose[:4]
            measured_hand=np.full_like(roll_hand,np.nan);measured_pose=np.full_like(roll_pose,np.nan)
            measured_hand[:4]=real_hand[:4];measured_pose[:4]=real_pose[:4]
            measured_full,scene0=forward(template)
            reconstructed=build_sample(template,real_pose[:4],real_hand[:4],real_pose[27])
            reconstructed_full,_=forward(reconstructed)
            builder_field_diff={k:float(np.max(abs(template[k].astype(float)-reconstructed[k].astype(float)))) for k in ['xyz','features','start_hand','start_hand_history','goal_object_points']}
            builder_prediction_diff=float(np.max(abs(measured_full-reconstructed_full)))
            if builder_prediction_diff>1e-4:raise ValueError('online/training builder prediction differs by>0.1mm')
            first_truth=template['trajectory_target'];mask=template['trajectory_valid']
            cv=np.arange(1,25)[:,None,None,None]/30*template['start_hand_history'][None,...,:3]
            first_error=np.linalg.norm(measured_full-first_truth,axis=-1)*1000
            single=dict(hand_epe8_mm=float(first_error[:8][mask[:8]].mean()),hand_epe24_mm=float(first_error[mask].mean()),
                hand_epe1_mm=float(first_error[0][mask[0]].mean()),
                hand_endpoint24_mm=float(first_error[-1][mask[-1]].mean()),
                hold_epe8_mm=float(np.linalg.norm(first_truth[:8],axis=-1)[mask[:8]].mean()*1000),
                hold_epe24_mm=float(np.linalg.norm(first_truth,axis=-1)[mask].mean()*1000),
                constant_velocity_epe8_mm=float(np.linalg.norm(cv[:8]-first_truth[:8],axis=-1)[mask[:8]].mean()*1000),
                hand_epe8_by_side_mm=[float(first_error[:8,s][mask[:8,s]].mean()) for s in range(2)])
            rigid=[];windows=[];stopped=None;input_hand=[];input_pose=[];all_delta=[];all_scene=[]
            for t in range(3,length-1,args.stride):
                frame_ids=np.arange(t+1,min(t+1+args.stride,length));goal_pose=real_pose[t+24]
                observed=build_sample(template,real_pose[t-3:t+1],real_hand[t-3:t+1],goal_pose)
                hd,sd=forward(observed)
                measured_hand[frame_ids]=real_hand[t]+hd[:len(frame_ids)]@real_pose[t,:3,:3].T
                obj,residual=forecast_object_poses(template['start_object_points'],sd,real_pose[t]);measured_pose[frame_ids]=obj[:len(frame_ids)]
                try:
                    predicted=build_sample(template,roll_pose[t-3:t+1],roll_hand[t-3:t+1],goal_pose)
                    hd,sd=forward(predicted)
                    if not np.isfinite(hd).all() or not np.isfinite(sd).all():raise ValueError('nonfinite prediction')
                    obj,residual=forecast_object_poses(template['start_object_points'],sd,roll_pose[t])
                except ValueError as exc:
                    stopped=dict(relative_tick=t,reason=str(exc));break
                input_hand.append(roll_hand[t-3:t+1].copy());input_pose.append(roll_pose[t-3:t+1].copy())
                all_delta.append(hd);all_scene.append(sd)
                roll_hand[frame_ids]=roll_hand[t]+hd[:len(frame_ids)]@roll_pose[t,:3,:3].T
                roll_pose[frame_ids]=obj[:len(frame_ids)];rigid.extend(residual[:len(frame_ids)].tolist())
                windows.append(dict(anchor=t,first=int(frame_ids[0]),last=int(frame_ids[-1])))
            valid=np.isfinite(roll_hand).all((1,2,3));valid[:4]=False
            # This fixed GRAB panel retains both hand slots; do not score absent anatomy.
            present=template['start_hand_valid']
            def errors(hand,poses):
                error=np.linalg.norm(hand-real_hand[:length],axis=-1)*1000
                per_frame=np.where(present,error,0).sum((1,2))/present.sum()
                object_error=np.linalg.norm(poses[:,:3,3]-real_pose[:length,:3,3],axis=-1)*1000
                return per_frame,object_error
            me,mo=errors(measured_hand,measured_pose);ae,ao=errors(roll_hand,roll_pose)
            last=int(np.flatnonzero(valid)[-1]);bound=np.array([w['first'] for w in windows[1:]],int)
            boundary=float(np.linalg.norm(roll_hand[bound]-roll_hand[bound-1],axis=-1).mean()*1000) if len(bound) else None
            report=dict(sequence=record['sequence'],global_train_index=record['index'],source_id=1,source_name='GRAB',
                original_train_tick=anchor,source_sequence_id=seq_id,frames=length,duration_s=length/30,
                consumed_frames=int(valid.sum()),single_forecast=single,measured_history_mean_epe_mm=float(np.mean(me[valid])),
                free_history_mean_epe_mm=float(np.mean(ae[valid])),free_history_last_epe_mm=float(ae[last]),
                measured_object_translation_mm=float(np.mean(mo[valid])),free_object_translation_mm=float(np.mean(ao[valid])),
                free_hand_by_side_mm=[float(np.linalg.norm(roll_hand[valid,s]-real_hand[:length][valid,s],axis=-1).mean()*1000) for s in range(2)],
                free_boundary_step_mm=boundary,object_rigid_residual_mean_mm=float(np.mean(rigid)*1000),stop=stopped,
                fixture_max_field_diff=exported_diff,builder_max_field_diff=builder_field_diff,builder_prediction_max_diff_mm=builder_prediction_diff*1000)
            data=dict(real_hand=real_hand[:length],real_object=real_pose[:length],measured_hand=measured_hand,
                measured_object=measured_pose,free_hand=roll_hand,free_object=roll_pose,valid=valid,hand_present=present,
                measured_error_mm=me,free_error_mm=ae,measured_object_error_mm=mo,free_object_error_mm=ao,
                flow_query=template['start_object_points'],surface=template['xyz'],
                first_prediction=measured_full,first_target=first_truth,input_hand_history=np.asarray(input_hand),
                input_object_history=np.asarray(input_pose),free_delta=np.asarray(all_delta),free_scene_delta=np.asarray(all_scene))
            np.savez_compressed(out/('case%d.npz'%case),**data)
            report['windows']=windows;reports.append(report)
            print(json.dumps(dict(stage='case',**{k:report[k] for k in ['sequence','single_forecast','measured_history_mean_epe_mm','free_history_mean_epe_mm','free_history_last_epe_mm','duration_s','stop']})),flush=True)
    result=dict(status='COMPLETED',work_version='wam-train-rollout-v2',experiment_id='P-20261020-wam-mano-preview',
        run_id=out.name,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        forecast_horizon=24,replan_stride=args.stride,seed=246,
        checkpoint=str(args.checkpoint.resolve()),checkpoint_step=80000,python=sys.executable,torch=torch.__version__,gpu=args.gpu,
        fixture=str(args.fixture.resolve()),main3=str(args.main3.resolve()),reports=reports,
        goal='true object pose t+24, matching WAM training; no future hand inputs',
        scope='four previously frozen admitted WAM TRAIN GRAB anchors; continuations in the same TRAIN sequences; later individual windows not rechecked against the remote v4 filter',
        limitations=['TRAIN diagnostic only, not generalization or all-nine-source audit','Both hands; differs from one-handed Inspire','Fixed true endpoint goals are provided instructions','No physics or BC/PPO execution'],
        elapsed_s=time.monotonic()-started,peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20)
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='COMPLETED',elapsed_s=result['elapsed_s'],peak_cuda_mib=result['peak_cuda_mib'])),flush=True)


def visualize(args):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    out=args.output.resolve();result=json.loads((out/'result.json').read_text());stride=result.get('replan_stride',8)
    baseline=None;comparison=[]
    if args.baseline is not None:
        baseline=json.loads((args.baseline/'result.json').read_text())
        for key in ('checkpoint','fixture','main3'):
            if result[key]!=baseline[key]:raise ValueError('comparison input mismatch: '+key)
        if baseline.get('replan_stride',8)!=8 or stride!=1:raise ValueError('comparison requires stride1 against stride8')
        if len(baseline['reports'])!=len(result['reports']):raise ValueError('comparison case count mismatch')
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for i,(ax,report) in enumerate(zip(axes.flat,result['reports'])):
        with np.load(out/('case%d.npz'%i)) as d:
            t=np.arange(len(d['valid']))/30;v=d['valid']
            ax.plot(t[v],d['measured_error_mm'][v],label='Measured H4 / show%d'%stride,color='#377eb8')
            ax.plot(t[v],d['free_error_mm'][v],label='Predicted H4 / show%d'%stride,color='#e67e22')
            if baseline:
                br=baseline['reports'][i]
                if (br['sequence'],br['original_train_tick'])!=(report['sequence'],report['original_train_tick']):
                    raise ValueError('comparison sequence/anchor mismatch')
                with np.load(args.baseline/('case%d.npz'%i)) as b:
                    for key in ('real_hand','real_object','flow_query','hand_present'):
                        np.testing.assert_array_equal(d[key],b[key])
                    first_diff_mm=float(np.max(np.abs(d['first_prediction']-b['first_prediction']))*1000)
                    if first_diff_mm>.1:raise ValueError('initial predictions differ by more than0.1mm')
                    common=v&b['valid'];ids=np.flatnonzero(common)
                    if not len(ids):raise ValueError('no common consumed frames')
                    ax.plot(t[common],b['free_error_mm'][common],label='Predicted H4 / show8',color='#984ea3')
                    ax.plot(t[common],b['measured_error_mm'][common],label='Measured H4 / show8',color='#4daf4a',ls='--')
                    row=dict(sequence=report['sequence'],common_consumed_frames=len(ids),
                        measured_stride8_mean_mm=float(b['measured_error_mm'][common].mean()),
                        measured_stride1_mean_mm=float(d['measured_error_mm'][common].mean()),
                        free_stride8_mean_mm=float(b['free_error_mm'][common].mean()),
                        free_stride1_mean_mm=float(d['free_error_mm'][common].mean()),
                        free_stride8_final_mm=float(b['free_error_mm'][ids[-1]]),
                        free_stride1_final_mm=float(d['free_error_mm'][ids[-1]]),
                        free_stride8_object_mean_mm=float(b['free_object_error_mm'][common].mean()),
                        free_stride1_object_mean_mm=float(d['free_object_error_mm'][common].mean()),
                        first_prediction_max_difference_mm=first_diff_mm)
                    comparison.append(row)
        ax.set(title=report['sequence'],xlabel='Seconds from H4 start',ylabel='Mean22-point error (mm)');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Frozen WAM-v4 step80000 / actual TRAIN GRAB / true object t+24 goal')
    fig.savefig(out/'train-rollout-errors.png',dpi=160);plt.close(fig)
    if baseline:
        (out/'stride-comparison.json').write_text(json.dumps(dict(status='PASS',baseline=str(args.baseline.resolve()),
            visualization_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            changed='consume/replan stride8 to stride1; same goal function evaluated at new anchors',reports=comparison),indent=2)+'\n')
        print(json.dumps(dict(stage='comparison',reports=comparison)),flush=True)
    # A raw-keypoint video avoids MANO reconstruction as a confound in this diagnosis.
    bones=np.array([(0,i) for i in range(1,6)]+[(i,i+5) for i in range(1,6)])
    for i,report in enumerate(result['reports']):
        with np.load(out/('case%d.npz'%i)) as archive:d={k:archive[k].copy() for k in archive.files}
        n=len(d['valid']);fig=plt.figure(figsize=(12,4),dpi=100)
        axes=[fig.add_subplot(1,3,k+1,projection='3d') for k in range(3)]
        series=[(d['real_hand'],d['real_object'],'Recorded TRAIN'),(d['measured_hand'],d['measured_object'],'Measured H4 / show%d'%stride),(d['free_hand'],d['free_object'],'Predicted H4 / show%d'%stride)]
        if baseline:
            with np.load(args.baseline/('case%d.npz'%i)) as b:
                series[1]=(b['free_hand'].copy(),b['free_object'].copy(),'Predicted H4 / show8')
        finite=np.concatenate([h[np.isfinite(h).all((1,2,3))].reshape(-1,3) for h,_,_ in series])
        points=np.r_[finite,d['real_object'][:,:3,3]];center=(points.min(0)+points.max(0))/2
        radius=max(.12,float((points.max(0)-points.min(0)).max()/2)*1.12)
        handles=[]
        for ax,(_,_,name) in zip(axes,series):
            ax.set_title(name);ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]-radius,center[1]+radius);ax.set_zlim(center[2]-radius,center[2]+radius)
            ax.set_box_aspect((1,1,1));ax.view_init(25,-65)
            ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');ax.set_zlabel('z (m)')
            lines=[ax.plot([],[],[],color=color,lw=1.8)[0] for color in ['#087f8c','#b05aa6'] for _ in bones]
            obj=ax.scatter([],[],[],s=5,c='#e99326');handles.append((lines,obj))
        title=fig.suptitle('');fig.tight_layout(rect=(0,0,1,.88))
        path=out/('train-case%d.mp4'%i)
        if path.exists():raise ValueError('refuse overwrite '+str(path))
        writer=FFMpegWriter(fps=15,codec='libx264',extra_args=['-threads','2','-crf','20','-pix_fmt','yuv420p'])
        with writer.saving(fig,str(path),100):
            for frame in range(n):
                for (hand,pose,_),(lines,obj) in zip(series,handles):
                    for side in range(2):
                        for j,(a,b) in enumerate(bones):
                            p=hand[frame,side,[a,b]]
                            lines[side*len(bones)+j].set_data_3d(p[:,0],p[:,1],p[:,2])
                    p=d['flow_query']@pose[frame,:3,:3].T+pose[frame,:3,3]
                    obj._offsets3d=(p[:,0],p[:,1],p[:,2])
                title.set_text('%s | t=%.2fs | WAM24/show%d | playback2x slower\nRaw points / predicted object / no physical simulation'%(report['sequence'],frame/30,stride))
                writer.grab_frame()
        plt.close(fig);print(json.dumps(dict(stage='video',case=i,file=str(path))),flush=True)
    (out/'visualization.json').write_text(json.dumps(dict(status='COMPLETED',videos=['train-case%d.mp4'%i for i in range(4)],plot='train-rollout-errors.png'),indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('infer','visualize'),required=True)
    parser.add_argument('--main3',type=Path);parser.add_argument('--fixture',type=Path);parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--gpu',type=int,default=7)
    parser.add_argument('--baseline',type=Path,help='Optional stride8 baseline for matched stride1 plots/video.')
    parser.add_argument('--stride',type=int,choices=range(1,25),default=8,
                        help='Consume this many forecast frames before predicting again; horizon remains24.')
    args=parser.parse_args()
    if args.stage=='infer':infer(args)
    else:visualize(args)


if __name__=='__main__':main()

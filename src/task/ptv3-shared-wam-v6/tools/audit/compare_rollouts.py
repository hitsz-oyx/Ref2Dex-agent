"""Matched old-v4 / v6-parent / geometry-v6 predicted-history diagnostics."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[2]
CODE_ROOT = TASK.parents[2]
ROOT = Path(os.environ.get("REF2DEX_RUN_ROOT", CODE_ROOT))
sys.path.insert(0, str(TASK/'src'))
from ptv3_shared_wam_v6 import paths
sys.path.insert(0, str(CODE_ROOT/'src/task/wam-rl-policy-v3/src'))
import numpy as np
import torch
from ptv3_shared_wam_v5.data import history, transform
from ptv3_shared_wam_v5.contracts import SPARSE11_IN_21
from ptv3_shared_wam_v6.data import collate
from ptv3_shared_wam_v6.model import SharedWAMV6
from ptv3_shared_wam_v6.geometry_consistency import skin_attachment, skin_future
from wam_rl_policy_v3.mano_preview import forecast_object_poses


def read(path): return json.loads(Path(path).read_text())
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def online(geo, poses, joints, dense, joint_valid, dense_valid, parts, goal, event=None):
    """Only H4 observed/generated geometry and instruction inputs; no future labels."""
    anchor = np.linalg.inv(poses[-1]); local = joints@anchor[:3,:3].T+anchor[:3,3]
    skin = dense@anchor[:3,:3].T+anchor[:3,3]
    cloud = transform(np.broadcast_to(geo['surface'], (4, len(geo['surface']), 3)), anchor@poses)
    features = np.zeros((len(geo['surface']),18),np.float32)
    features[:,:3]=geo['normals'];features[:,3:12]=history(cloud);features[:,12]=1
    features[:,13]=np.linalg.norm(features[:,9:12],axis=-1)<.002
    flow=geo['query'];relative=anchor@goal;endpoint=flow@relative[:3,:3].T+relative[:3,3]
    sample=dict(xyz=geo['surface'].astype(np.float32),features=features,point_valid=np.ones(len(cloud[0]),bool),
        start_hand=np.where(joint_valid[...,None],local[-1],0).astype(np.float32),
        start_hand_history=np.where(joint_valid[...,None],history(local),0).astype(np.float32),start_hand_valid=joint_valid,
        dense_hand=np.where(dense_valid[...,None],skin[-1],0).astype(np.float32),
        dense_history=np.where(dense_valid[...,None],history(skin),0).astype(np.float32),dense_hand_valid=dense_valid,
        dense_hand_part=parts.astype(np.int64),scene_points=flow[None].astype(np.float32),object_valid=np.ones(1,bool),
        start_object_points=flow.astype(np.float32),goal_object_points=endpoint.astype(np.float32),
        goal_object_delta=(endpoint-flow).astype(np.float32),object_goal_valid=np.ones(32,bool),
        interaction_points=geo['surface'][None].astype(np.float32),interaction_normals=geo['normals'][None].astype(np.float32),
        interaction_query_valid=np.ones((1,len(geo['surface'])),bool),
        event_type=np.int64(0),event_delta_steps=np.float32(0),event_valid=np.bool_(False),event_hz=np.int64(30))
    if event:
        sample.update({k:np.asarray(v,dtype=sample[k].dtype) for k,v in event.items()})
    return sample


def extend_skin(joints, dense, valid, dense_valid, future):
    with torch.no_grad():
        attach=skin_attachment(torch.as_tensor(joints[None],dtype=torch.float32),
            torch.as_tensor(dense[None],dtype=torch.float32),torch.as_tensor(valid[None]),torch.as_tensor(dense_valid[None]))
        return skin_future(torch.as_tensor(future[None],dtype=torch.float32),attach)[0].numpy()


def score(real, pred, valid, present):
    error=np.linalg.norm(real-pred,axis=-1)*1000
    per_frame=np.where(present,error,0).sum((1,2))/present.sum()
    ids=np.flatnonzero(valid)
    return per_frame,dict(frames=int(len(ids)),mean_mm=float(per_frame[ids].mean()) if len(ids) else None,
                         final_mm=float(per_frame[ids[-1]]) if len(ids) else None)


def run(args):
    started=time.monotonic();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=args.continue_existing)
    if __import__('shutil').disk_usage(out).free<25*2**30:raise ValueError('25GiB preflight floor')
    torch.set_num_threads(2);torch.manual_seed(246)
    torch.backends.cuda.matmul.allow_tf32=False
    dataset_root=ROOT/'outputs/ptv3-shared-wam-v6/local-data-20261021-r2/data'
    gm=read(dataset_root/'geometry/manifest.json');geometry={x['key']:x for x in gm['objects']}
    oldroot=ROOT/'outputs/wam-rl-policy-v3/wam-train-rollout-20261020-r1'
    old=read(oldroot/'result.json');models=[('parent4371',args.parent),('geometry4500',args.geometry)]
    report=dict(status='RUNNING',experiment_id='P-20261021-wam-v6-geometry-ab',run_id=out.name,
        git_commit=args.git_commit,work_version='v6.matched-rollout.v1',gpu=7,python=sys.executable,
        torch=torch.__version__,forecast_horizon=24,stride=8,seed=246,models={},reports=[],
        limitations=['four frozen TRAIN anchors, no generalization claim','v4/v6 architectures and observed geometry differ',
        'common11-point score; v6-native joint21/observed skin on GRAB',
        'unknown event on GRAB matches instructions available to v4; scheduler events additionally tested on apple',
        'apple uses measured sparse11 with remaining joints/skin explicitly invalid, no fabricated FK',
        'future hand never enters history; predicted skin is a transported proxy',
        'offline motion-only, no physical execution or contact forces'])
    if args.continue_existing:
        report=read(out/'result.json')
        if report['work_version']!='v6.matched-rollout.v1':raise ValueError('different evaluation protocol')
        report.setdefault('continuation_commits',[]).append(args.git_commit)
        report['status']='RUNNING'
        for label,checkpoint in models:
            if label in report['models'] and report['models'][label]['checkpoint']!=str(checkpoint.resolve()):raise ValueError('continuation checkpoint changed')
    write(out/'result.json',report)
    for label,checkpoint in models:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False);config=saved['config']
        model=SharedWAMV6(read(args.stats),config).cuda().eval().requires_grad_(False)
        model.load_state_dict(saved['model'],strict=True)
        report['models'][label]=dict(checkpoint=str(checkpoint.resolve()),step=saved['step'],config=config)
        del saved
        def forward(sample):
            if time.monotonic()-started>1800:raise TimeoutError('30min inference budget')
            if __import__('shutil').disk_usage(out).free<20*2**30:raise ValueError('disk floor')
            xyz=np.r_[sample['xyz'],sample['start_hand'][sample['start_hand_valid']],sample['dense_hand'][sample['dense_hand_valid']]]
            grid=np.floor((xyz-np.asarray(config['voxel_origin_m']))/config['voxel_m'])
            if not np.isfinite(grid).all() or grid.min()<0 or grid.max()>65535:raise ValueError('generated state outside WAM workspace')
            batch={k:v.cuda() for k,v in collate([sample]).items()}
            with torch.inference_mode(),torch.autocast('cuda',enabled=False):p=model(batch)
            h=p['trajectory'][0].cpu().numpy();s=p['scene'][0,0].cpu().numpy()
            if not np.isfinite(h).all() or not np.isfinite(s).all():raise ValueError('nonfinite prediction')
            return h,s
        for case,record in enumerate(old['reports']):
            if any(r['model']==label and r['kind']=='grab' and r['case']==case and r.get('event_mode','unknown')==args.grab_events for r in report['reports']):continue
            folder=dataset_root/'hands/sequences'/('s1-%05d'%record['source_sequence_id'])
            meta=read(folder/'meta.json');assert meta['source_sequence']==record['sequence']
            seq={p.stem:np.load(p,mmap_mode='r') for p in folder.glob('*.npy')}
            with np.load(oldroot/('case%d.npz'%case)) as z:base={k:z[k].copy() for k in z.files}
            start=record['original_train_tick']-3;n=len(base['valid']);slice_=slice(start,start+n+24)
            joints=np.asarray(seq['joints'][slice_],np.float64).copy();poses=np.asarray(seq['object_poses'][slice_,0],np.float64).copy()
            ids=np.asarray(SPARSE11_IN_21);np.testing.assert_allclose(joints[:n,:,ids],base['real_hand'],atol=3e-7,rtol=0)
            np.testing.assert_allclose(poses[:n],base['real_object'],atol=3e-7,rtol=0)
            geo_row=geometry['s1:'+meta['objects'][0]]
            with np.load(dataset_root/'geometry'/geo_row['path']) as z:geo={k:z[k].copy() for k in z.files}
            np.testing.assert_allclose(geo['query'],base['flow_query'],atol=1e-7,rtol=0)
            skin=np.asarray(seq['surface_offsets'][start:start+4],np.float32)+joints[:4,:,None,0,:]
            jv=np.asarray(seq['joint_valid'][start:start+4]).all(0);dv=np.broadcast_to(np.asarray(seq['surface_valid'][start:start+4]).all(0)[:,None],(2,778)).copy()
            parts=np.asarray(seq['surface_parts']);hand=np.full_like(joints[:n],np.nan);pose=np.full_like(poses[:n],np.nan)
            planned=None
            if args.grab_events=='planned':
                recipe=read(ROOT/'outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/manifest.json')
                events=read(recipe['event_manifest']);event_row=next(x for x in events['sequences'] if x['key']==folder.name)
                planned={k:np.load(Path(event_row['path'])/(k+'.npy'),mmap_mode='r') for k in ('event_type','event_delta_steps','event_valid')}
            dense=np.full((n,2,778,3),np.nan);hand[:4]=joints[:4];pose[:4]=poses[:4];dense[:4]=skin
            valid=np.zeros(n,bool);stop=None;inputs=[];outputs=[];scene_outputs=[];input_pose=[]
            # Independent roundtrip against the training reader at the initial anchor.
            from ptv3_shared_wam_v5.data import sample_window
            cf=dataset_root/'contacts'/folder.name
            contact={p.stem:np.load(p,mmap_mode='r') for p in cf.glob('*.npy')}
            reference=sample_window(seq,geo,contact,record['original_train_tick'],0,1,[0,0,0])
            initial=online(geo,poses[:4],joints[:4],skin,jv,dv,parts,poses[27])
            roundtrip={k:float(np.max(np.abs(initial[k].astype(float)-reference[k].astype(float)))) for k in initial if k in reference}
            if max(roundtrip.values())>.003:raise ValueError('initial training/online mismatch '+str(roundtrip))
            for t in range(3,n-1,8):
                try:
                    event=dict(event_hz=30,**{k:v[start+t,0] for k,v in planned.items()}) if planned else None
                    sample=online(geo,pose[t-3:t+1],hand[t-3:t+1],dense[t-3:t+1],jv,dv,parts,poses[t+24],event)
                    hd,sd=forward(sample);idx=np.arange(t+1,min(t+9,n))
                    future=hand[t]+hd@pose[t,:3,:3].T
                    future_skin=extend_skin(hand[t],dense[t],jv,dv,future[:len(idx)])
                    objects,_=forecast_object_poses(geo['query'],sd,pose[t])
                except (ValueError,RuntimeError) as e:stop=dict(tick=t,reason=str(e));break
                inputs.append(hand[t-3:t+1].copy());input_pose.append(pose[t-3:t+1].copy());outputs.append(hd);scene_outputs.append(sd)
                hand[idx]=future[:len(idx)];pose[idx]=objects[:len(idx)];dense[idx]=future_skin;valid[idx]=True
            common=valid&base['valid'];pred=hand[:,:,ids];error,metrics=score(base['real_hand'],pred,common,base['hand_present'])
            _,bm=score(base['real_hand'],base['free_hand'],common,base['hand_present'])
            objerr=np.linalg.norm(pose[:,:3,3]-base['real_object'][:,:3,3],axis=-1)*1000
            row=dict(model=label,case=case,event_mode=args.grab_events,sequence=record['sequence'],kind='grab',**metrics,
                old_v4_common_mean_mm=bm['mean_mm'],improvement_percent=100*(1-metrics['mean_mm']/bm['mean_mm']) if metrics['mean_mm'] is not None else None,
                object_translation_mean_mm=float(objerr[common].mean()) if common.any() else None,stop=stop,roundtrip_max_difference=roundtrip)
            filename=label+('-planned' if args.grab_events=='planned' else '')+'-case%d.npz'%case
            if (out/filename).exists():raise ValueError('refuse prediction overwrite')
            np.savez_compressed(out/filename,real=base['real_hand'],hand=pred,pose=pose,
                real_pose=base['real_object'],valid=valid,common=common,present=base['hand_present'],error_mm=error,
                input_hand=np.asarray(inputs),input_pose=np.asarray(input_pose),delta=np.asarray(outputs),scene=np.asarray(scene_outputs),
                generated_joint21=hand,old_hand=base['free_hand'],old_pose=base['free_object'],surface=base['surface'])
            report['reports'].append(row);write(out/'result.json',report);print(json.dumps(row),flush=True)
        apple(args,out,label,forward,report)
        del model;torch.cuda.empty_cache()
    report.update(status='COMPLETED',elapsed_s=time.monotonic()-started,peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20)
    write(out/'result.json',report)


def apple(args,out,label,forward,report):
    from scipy.spatial.transform import Rotation
    from wam_rl_policy_v3.teacher_dataset import TeacherRolloutCorpus
    from wam_rl_policy_v3.task_spec import TaskSpec
    from wam_rl_policy_v3.task_runtime import TaskController
    root=ROOT/'outputs/wam-rl-policy-v3/wam-mano-autoregressive-20261020-r1'
    meta=read(root/'manifest.json');corpus=TeacherRolloutCorpus(Path(meta['source_manifest']),'train')
    index=next(i for i,r in enumerate(corpus.records) if r['sequence']==meta['sequence']);ep=corpus.episode(index)
    cache=ROOT/'outputs/wam-rl-policy-v2/inspire-shadow-parity-20261019-r1/cache'
    gm=read(cache/'surface-manifest.json');row=next(r for r in gm['objects'] if r['key']=='dexplore/apple@1')
    with np.load(cache/row['path']) as z:geo={k:z[k].copy() for k in z.files}
    with np.load(root/'preview.npz') as z:base={k:z[k].copy() for k in z.files}
    np.testing.assert_allclose(geo['query'],base['flow_query'],atol=1e-7,rtol=0)
    actual=base['teacher_hand'];real_pose=base['object_poses'];n=len(actual);ids=np.asarray(SPARSE11_IN_21)
    for mode in ('fixed','scheduler-unknown','scheduler-events'):
        if any(r['model']==label and r['kind']=='apple' and r['mode']==mode for r in report['reports']):continue
        ctl=TaskController(TaskSpec.from_packet(ep['task_spec']),ep['object_points'])
        hand=np.zeros((n,2,21,3));pose=np.full_like(real_pose,np.nan);dense=np.zeros((4,2,778,3),np.float32)
        for t in range(4):hand[t,0,ids]=actual[t]
        pose[:4]=real_pose[:4];jv=np.zeros((2,21),bool);jv[0,ids]=True;dv=np.zeros((2,778),bool);parts=np.zeros((2,778),np.int64)
        valid=np.zeros(n,bool);traces=[];stop=None
        def observe(t):
            velocity=np.r_[(pose[t,:3,3]-pose[t-1,:3,3])*30,Rotation.from_matrix(pose[t,:3,:3]@pose[t-1,:3,:3].T).as_rotvec()*30] if t else np.zeros(6)
            result=ctl.observe(pose[t],hand[t,0,ids],velocity,t/30,force_valid=False,supported=None)
            d=result['decision'];traces.append(dict(tick=t,phase=d.phase.name,terminated=d.terminated,reason=d.failure_reason,grasp=result['feedback'].grasp_established))
            return d
        for t in range(4):decision=observe(t)
        for t in range(3,n-1,8):
            if mode!='fixed' and decision.terminated:stop=dict(tick=t,reason=decision.failure_reason or 'task_complete');break
            goal=base['goal_poses'][t] if mode=='fixed' else decision.local_goal_pose
            event=decision.next_event.wam_condition() if mode=='scheduler-events' else None
            try:
                hd,sd=forward(online(geo,pose[t-3:t+1],hand[t-3:t+1],dense,jv,dv,parts,goal,event))
                future=hand[t]+hd@pose[t,:3,:3].T;objects,_=forecast_object_poses(geo['query'],sd,pose[t])
            except (ValueError,RuntimeError) as e:stop=dict(tick=t,reason=str(e));break
            for k,frame in enumerate(range(t+1,min(t+9,n))):
                hand[frame]=future[k];pose[frame]=objects[k];valid[frame]=True
                if mode!='fixed':decision=observe(frame)
                if mode!='fixed' and decision.terminated:stop=dict(tick=frame,reason=decision.failure_reason or 'task_complete');break
            if stop:break
        pred=hand[:,0,ids];present=np.ones((1,11),bool)
        # Scheduler baseline is its historical motion-only run, rather than fixed goals.
        if mode!='fixed':
            with np.load(ROOT/'outputs/wam-rl-policy-v3/wam-task-feedback-20261020-r1/rollout.npz') as z:
                old_hand=z['hand'].copy();old_pose=z['pose'].copy();old_valid=z['valid'].copy()
        else:old_hand=base['target_hand'];old_pose=base['display_object_poses'];old_valid=base['prediction_valid']
        common=valid&old_valid;err,metrics=score(actual[:,None],pred[:,None],common,present)
        _,bm=score(actual[:,None],old_hand[:,None],common,present)
        row=dict(model=label,kind='apple',mode=mode,**metrics,old_v4_common_mean_mm=bm['mean_mm'],
            improvement_percent=100*(1-metrics['mean_mm']/bm['mean_mm']) if metrics['mean_mm'] is not None else None,
            consumed_frames=int(valid.sum()),stop=stop,grasp_ticks=sum(x['grasp'] for x in traces))
        np.savez_compressed(out/(label+'-apple-'+mode+'.npz'),real=actual[:,None],hand=pred[:,None],pose=pose,real_pose=real_pose,
            old_hand=old_hand[:,None],old_pose=old_pose,valid=valid,common=common,present=present,error_mm=err,surface=geo['surface'])
        row['traces']=traces;report['reports'].append(row);write(out/'result.json',report)
        print(json.dumps({k:v for k,v in row.items() if k!='traces'}),flush=True)


def render(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    out=args.output;reports=read(out/'result.json');bones=[(0,i) for i in range(1,6)]+[(i,i+5) for i in range(1,6)]
    movie_names=[]
    keys=['planned-case%d'%i for i in range(4)] if args.render_planned_only else ['case%d'%i for i in range(4)]+['apple-fixed','apple-scheduler-unknown','apple-scheduler-events']
    for key in keys:
        arrays=[]
        for label in ('parent4371','geometry4500'):
            with np.load(out/(label+'-'+key+'.npz')) as z:arrays.append({k:z[k].copy() for k in z.files})
        a,b=arrays;series=[(a['real'],a['real_pose'],'Recorded'),(a['old_hand'],a['old_pose'],'v4 step80000 / predicted H4'),
            (a['hand'],a['pose'],'v6 parent4371 / predicted H4'),(b['hand'],b['pose'],'v6 geometry4500 / predicted H4')]
        points=np.concatenate([h[np.isfinite(h).all((1,2,3))].reshape(-1,3) for h,_,_ in series]+[a['real_pose'][:,:3,3]])
        points=points[np.isfinite(points).all(-1)];center=(points.min(0)+points.max(0))/2;radius=max(.15,float(np.ptp(points,axis=0).max())*.56)
        fig=plt.figure(figsize=(16,4),dpi=90);handles=[]
        for j,(h,p,title) in enumerate(series):
            ax=fig.add_subplot(1,4,j+1,projection='3d');ax.set_title(title,fontsize=9)
            ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius));ax.set_box_aspect((1,1,1));ax.view_init(25,-65)
            lines=[ax.plot([],[],[],color=color,lw=1.8)[0] for color in ('#087f8c','#b05aa6')[:h.shape[1]] for _ in bones]
            obj=ax.scatter([],[],[],s=4,c='#e99326');text=ax.text2D(.02,.02,'',transform=ax.transAxes,fontsize=8);handles.append((lines,obj,text))
        title=fig.suptitle('');fig.tight_layout(rect=(0,0,1,.9));name=key+'-comparison.mp4';path=out/name
        if path.exists():raise ValueError('refuse video overwrite')
        writer=FFMpegWriter(fps=15,codec='libx264',extra_args=['-threads','2','-crf','22','-pix_fmt','yuv420p'])
        with writer.saving(fig,str(path),90):
            for t in range(len(a['real'])):
                for j,((h,p,_),(lines,obj,text)) in enumerate(zip(series,handles)):
                    good=np.isfinite(p[t]).all() and (j<2 or (t<4 or arrays[j-2]['valid'][t]))
                    for side in range(h.shape[1]):
                        for k,(u,v) in enumerate(bones):
                            q=h[t,side,[u,v]] if good else np.full((2,3),np.nan);lines[side*len(bones)+k].set_data_3d(*q.T)
                    pts=a['surface'][::max(1,len(a['surface'])//128)]@p[t,:3,:3].T+p[t,:3,3] if good else np.full((1,3),np.nan)
                    obj._offsets3d=tuple(pts.T)
                    error=np.linalg.norm(h[t]-a['real'][t],axis=-1).mean()*1000 if good else None
                    text.set_text(('error %.1fmm'%error) if good else 'STOP / no prediction')
                title.set_text('%s | WAM24 / consume8 | t=%.2fs | playback 2x slower\nCommon11 points; no physical execution'% (key,t/30));writer.grab_frame()
        plt.close(fig);movie_names.append(name);print('rendered',name,flush=True)
    suffix='-planned' if args.render_planned_only else ''
    listing=out/('video-list'+suffix+'.txt');listing.write_text(''.join("file '%s'\n"%str((out/x).resolve()) for x in movie_names[:4]))
    combined='train-cases'+suffix+'-comparison.mp4'
    subprocess.run(['ffmpeg','-v','error','-n','-f','concat','-safe','0','-i',str(listing),'-c','copy',str(out/combined)],check=True)
    write(out/('visualization'+suffix+'.json'),dict(status='COMPLETED',videos=movie_names,combined=combined,fps=15,real_hz=30))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['infer','render']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--parent',type=Path);p.add_argument('--geometry',type=Path);p.add_argument('--stats',type=Path);p.add_argument('--git-commit')
    p.add_argument('--continue-existing',action='store_true')
    p.add_argument('--grab-events',choices=['unknown','planned'],default='unknown')
    p.add_argument('--render-planned-only',action='store_true')
    args=p.parse_args();run(args) if args.mode=='infer' else render(args)

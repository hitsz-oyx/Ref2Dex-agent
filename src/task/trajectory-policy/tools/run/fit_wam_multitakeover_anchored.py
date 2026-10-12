"""Fixed multi-takeover continuation of the GT-anchored learned WAM/R pair."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(TASK/'src'), str(TASK/'tools/audit'), str(TASK/'tools/run'),
    str(ROOT/'src/task/goal-point-wam/src'), str(ROOT/'src/task/cm-pointflow-effect-pretrain/src')]
from audit_wam_ref2_shadow import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=2)
    parser.add_argument('--engineering-smoke',action='store_true')
    parser.add_argument('--gt-anchored',action='store_true',help='R learns GT tau only; predicted action trains WAM only')
    parser.set_defaults(gt_anchored=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if (args.gpu != 2 or out.exists() or ROOT/'outputs/trajectory-policy' not in out.parents
            or not out.name.startswith('wam-ref2-joint-')):
        raise ValueError('fresh task-owned joint output and assigned GPU2 required')
    if subprocess.check_output(['nvidia-smi','-i','2','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
        raise RuntimeError('GPU2 occupied')
    scratch = ROOT/'tmp/trajectory-policy/wam-multitakeover-fit';scratch.mkdir(parents=True,exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES='2',TMPDIR=str(scratch),TRITON_CACHE_DIR=str(scratch/'triton'))
    import torch
    from trajectory_policy.wam_shadow import robot_goal_sample, WAM_TO_NATIVE
    from trajectory_policy.retargeter_data import features
    from trajectory_policy.pd_retargeter import load_model, encode_target, decode_target, command
    from trajectory_policy.wam_retargeter_admission import BASE_R_SHA
    from trajectory_policy.wam_joint import (JOINT_SCHEMA,PATCH_SCHEMA,POLICY_MODULES,TRAIN_MODULES,
        state_hash,buffer_hash,joint_losses,action_gradient_evidence,offline_screen,own_expert_admission,
        paired_manifest_admission,policy_patch_admission,load_policy_patch)
    from trajectory_policy.wam_multitakeover import validate_aggregation, balanced_draw
    from trajectory_policy.wam_anchored import anchored_losses, anchored_gradient_evidence, GRADIENT_OWNERSHIP
    from goal_point_wam.data import goal_collate
    from oakink_wm.pointworld_performance import install_fused_hilbert
    torch.set_num_threads(2);torch.manual_seed(295);torch.backends.cuda.matmul.allow_tf32=True
    rng = np.random.default_rng(296);started=time.monotonic()
    owner = ROOT/'outputs/trajectory-policy'
    parent = owner/'wam-ref2-joint-anchored-20261012-r1'
    pm,pr = [json.loads((parent/k).read_text()) for k in ('manifest.json','result.json')]
    paired_manifest_admission(pm,pr,pm,sha(parent/'head.pt'),sha(parent/'tau.pt'))
    source = owner/'competent-pd-teacher-20261011-r2'
    sm,sr,sa=[json.loads((source/k).read_text()) for k in ('manifest.json','result.json','audit.json')]
    if (sm.get('status')!='COMPLETED' or sm.get('training_allowed') is not True
            or sr.get('teacher_readiness') is not True or sa.get('pass') is not True
            or sm.get('dt')!=1/30 or sha(source/'trajectory.npz')!=sm.get('trajectory_sha256')):
        raise ValueError('qualified nominal teacher required')
    with np.load(source/'trajectory.npz',allow_pickle=False) as f:
        data={'nominal':{k:f[k][:,:28].copy() for k in ('q','dq','hand','obj','velocity','pd_targets')}}
    live = owner/'wam-ref2-execution-startup-anchor-20261012-r1/native'
    lm,la,lr=[json.loads((live/k).read_text()) for k in ('manifest.json','root_recovery_audit.json','result.json')]
    group=json.loads((live.parent/'group_status.json').read_text())
    if lm.get('result_sha256')!=sha(live/'result.json'):raise ValueError('native result binding differs')
    with np.load(live/'trajectory.npz',allow_pickle=False) as f:data['recovery']={k:f[k].copy() for k in f.files}
    ref_path=owner/'competent-reference-20261011-r1/reference.npz'
    canonical_path=ROOT/'outputs/consequence-evaluator/old-utility-pw-20261009-r2/canonical.npz'
    with np.load(ref_path,allow_pickle=False) as f:reference=f['obj'].copy()
    with np.load(canonical_path,allow_pickle=False) as f:canonical={k:f[k].copy() for k in f.files}
    admission=own_expert_admission(lm,la,lr,data['recovery'],sha(live/'trajectory.npz'),group,reference)
    base=ROOT/'outputs/goal-point-wam/goal-memory-ddp-20261012-r3/train/final.pt'
    stats=ROOT/'outputs/cm-pointflow-effect-pretrain/pointworld-main3-sharedstats-20261009/norm_stats.json'
    original=owner/'wam-ref2-shadow-20261012-r2/code/model.py'
    r_path=parent/'tau.pt'
    qualified_r_path=owner/'competent-pd-fit-20261011-r1/tau.pt'
    aggregation=owner/'wam-ref2-multitakeover-20261012-r1'
    am=json.loads((aggregation/'manifest.json').read_text())
    aa=json.loads((aggregation/'root_aggregation_audit.json').read_text())
    with np.load(aggregation/'aggregation.npz',allow_pickle=False) as f:aggregate={k:f[k] for k in f.files}
    if sha(aggregation/'aggregation.npz')!=am['aggregation_sha256']:raise ValueError('aggregation data differs')
    multi_admission=validate_aggregation(am,aa,aggregate,(sha(parent/'head.pt'),sha(r_path)))
    for source_id,source_admission in enumerate(am['admissions']):
        native=Path(source_admission['native'])
        with np.load(native/'trajectory.npz',allow_pickle=False) as f:data['multi%d'%source_id]={k:f[k].copy() for k in f.files}
    consumed=[Path(__file__),TASK/'src/trajectory_policy/wam_joint.py',
        TASK/'src/trajectory_policy/wam_shadow.py',TASK/'src/trajectory_policy/pd_retargeter.py',
        TASK/'src/trajectory_policy/retargeter_data.py',TASK/'src/trajectory_policy/wam_retargeter_admission.py',
        TASK/'tools/audit/audit_wam_ref2_shadow.py',ROOT/'src/task/goal-point-wam/src/goal_point_wam/data.py',
        ref_path,canonical_path,base,stats,original,r_path,live.parent/'group_status.json']
    if args.gt_anchored:consumed.append(TASK/'src/trajectory_policy/wam_anchored.py')
    consumed += [parent/k for k in ('head.pt','tau.pt','manifest.json','result.json')]
    consumed += [aggregation/k for k in ('aggregation.npz','manifest.json','root_aggregation_audit.json')]
    consumed += [Path(k).resolve() for k in am['input_sha256']]
    consumed += [TASK/'src/trajectory_policy/wam_multitakeover.py', qualified_r_path]
    consumed += [source/k for k in ('trajectory.npz','manifest.json','result.json','audit.json')]
    consumed += [live/k for k in ('trajectory.npz','manifest.json','result.json','root_recovery_audit.json')]
    support=ROOT/'src/task/cm-pointflow-effect-pretrain/src/oakink_wm'
    consumed += [support/k for k in ('model.py','data.py','pointworld.py','pointworld_performance.py')]
    consumed += list((ROOT/'third_party/PointWorld/ptv3').rglob('*.py'))
    consumed += [ROOT/'third_party/PointWorld/ptv3/ptv3_arch.yaml']
    evidence={str(p):sha(p) for p in consumed}
    for p in (base,stats,original,qualified_r_path,ref_path,canonical_path,source/'trajectory.npz'):
        if pm['input_sha256'].get(str(p))!=sha(p):raise ValueError('original startup actual input differs: '+str(p))
    packet=torch.load(base,map_location='cpu',weights_only=False)
    if sha(original)!=packet['identity']['model_sha256'] or sha(stats)!=packet['identity']['stats_sha256']:
        raise ValueError('original archived WAM and stats required')
    spec=importlib.util.spec_from_file_location('original_wam_joint',original)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);install_fused_hilbert()
    model=module.GoalPointWAM(json.loads(stats.read_text()),packet['config']).cuda().eval()
    model.load_state_dict(packet['model'],strict=True)
    head=torch.load(parent/'head.pt',map_location='cpu',weights_only=False)
    if head.get('base_checkpoint_sha256')!=sha(base):raise ValueError('warm WAM base differs')
    load_policy_patch(model,head,policy_patch_admission(head,pm,pr))
    base_r_packet=torch.load(r_path,map_location='cpu',weights_only=False)
    r=load_model(base_r_packet,'cuda')
    for net in (model,r):
        for p in net.parameters():p.requires_grad_(False)
    modules={name:getattr(model,name) for name in POLICY_MODULES};modules['R']=r.net
    for net in modules.values():
        for p in net.parameters():p.requires_grad_(True)
    frozen={name:getattr(model,name) for name in ('action','world')}
    initial_hash={k:state_hash(v.state_dict()) for k,v in modules.items()}
    frozen_hash={k:state_hash(v.state_dict()) for k,v in frozen.items()}
    norms={'WAM':buffer_hash(model),'R':buffer_hash(r)}
    del packet,head
    nominal=[('nominal',t,e) for t in range(519) for e in range(16)]
    if args.engineering_smoke:nominal=[('nominal',t,e) for t in (0,16) for e in range(16)]
    panel=[0,1,2,4,8,16,32,64,96,128,192,256,320,384,448,518]
    parts=dict(nominal_report=[('nominal',t,e) for t in panel for e in range(24,28)],
        recovery_training=[('recovery',t,3) for t in range(8,105)],
        pure_PD_training=[('recovery',t,2) for t in range(53)],
        pure_PD_diagnostic=[('recovery',t,2) for t in range(56,105) if not data['recovery']['expert_clipped'][t,0]],
        startup_all=[('nominal',t,e) for t in range(16) for e in range(24,28)])
    rows=list(nominal);groups={}
    aggregate_rows={}
    for name,part in parts.items():groups[name]=np.arange(len(rows),len(rows)+len(part));rows+=part
    for t in range(3):groups['startup_t%d'%t]=groups['startup_all'][4*t:4*t+4]
    for sid in range(3):
        for valid,label in ((True,'future'),(False,'current_PD')):
            part=np.flatnonzero((aggregate['source_id']==sid)&(aggregate['trajectory_valid']==valid))
            ids=np.arange(len(rows),len(rows)+len(part))
            groups['takeover%d_%s_TRAIN'%((8,16,32)[sid],label)]=ids
            for index,ai in zip(ids,part):
                rows.append(('multi%d'%sid,int(aggregate['tick'][ai]),int(aggregate['role'][ai])))
                aggregate_rows[int(index)]=int(ai)
    expert_pools=[groups['takeover%d_future_TRAIN'%t] for t in (8,16,32)]
    pure_pools=[groups['takeover%d_current_PD_TRAIN'%t] for t in (8,16,32)]
    cache={};order=torch.as_tensor(WAM_TO_NATIVE,device='cuda')
    offset,scale=[torch.as_tensor(lm['native_controller'][k],device='cuda') for k in ('offset','scale')]
    if any(not np.array_equal(np.asarray(lm['native_controller'][k]),np.asarray(sm['native_controller'][k])) for k in ('offset','scale')):
        raise ValueError('nominal and live controller differ')
    def get(ids):
        samples=[];labels=[]
        for index in ids:
            index=int(index)
            if index not in cache:
                kind,tick,env=rows[index];d=data[kind];past=np.maximum(tick+np.arange(-3,1),0)
                sample=robot_goal_sample(d['obj'][past,env],d['hand'][past,env],reference[min(tick+24,542)],canonical,sample_id=index)
                valid=kind=='nominal' or env==3
                future=d['hand'][tick+np.arange(1,25),env] if valid else np.broadcast_to(d['hand'][tick,env],(24,11,3))
                state,tau=features(*(d[k][tick,env] for k in ('q','dq','hand','obj','velocity')),future)
                q=d['q'][tick,env];pd=d['pd_targets'][tick,env] if kind=='nominal' else d['expert_PD'][tick,env-2]
                if index in aggregate_rows:
                    ai=aggregate_rows[index]
                    state,tau,q,pd=tuple(aggregate[k][ai] for k in ('state87','tau24x33','current_q','expert_PD'))
                    valid=bool(aggregate['trajectory_valid'][ai])
                cache[index]=(sample,dict(state=state,tau=tau,pd_label=encode_target(pd,q),q=q,tau_label_valid=np.asarray(valid)))
            sample,label=cache[index];samples.append(sample);labels.append(label)
        batch={k:v.cuda() for k,v in goal_collate(samples).items()}
        batch.update({k:torch.as_tensor(np.stack([v[k] for v in labels]),device='cuda') for k in labels[0]})
        return batch
    def predict(b):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            scene=model.scene(b)
            goal,mask=model.goal(b['start_object_points'],b['goal_object_points'],b['object_goal_valid'])
            tau=model.trajectory(scene['scene_tokens'],scene['scene_mask'],goal,mask,model.observed_hand_condition(b)).float()
        return tau[:,:,0].index_select(2,order).reshape(-1,24,33)
    out.mkdir(parents=True);admission.update(multi_admission,source_aggregation=str(aggregation))
    write(out/'actor_admission.json',admission)
    evidence[str(out/'actor_admission.json')]=sha(out/'actor_admission.json')
    manifest=dict(schema=JOINT_SCHEMA,status='RUNNING',task='goal-point-wam',
        experiment_id='P-20261012-goal-point-wam-ref23-multitakeover-fit',run_id=out.name,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        seed=295,draw_seed=296,physical_gpu=2,input_sha256=evidence,training_allowed=False,
        base_checkpoint_sha256=sha(base),base_R_sha256=BASE_R_SHA,warm_head_sha256=sha(parent/'head.pt'),
        warm_head_dir=str(parent),train_modules=list(TRAIN_MODULES),policy_patch_schema=PATCH_SCHEMA,
        maximum_updates=1 if args.engineering_smoke else 1000,batch=64,
        learning_rates=dict(scene=1e-5,goal=1e-4,trajectory=1e-4,hand_query=1e-4,R=1e-4),
        loss='valid true tau MSE/.05 + all-row actual encoded PD normalized L1; equal weights',
        nominal_windows=len(nominal),recovery_windows=259,pure_current_PD_windows=384,
        nominal_sampling='16 startup +16 later',recovery_sampling='16, source-balanced5/5/6',pure_current_sampling='16, source-balanced5/5/6',
        cache_scope='raw samples/labels RAM only; FULL scene and goal forward per update',
        model_mode='eval with trainable policy/R parameters; all statistics buffers frozen',
        fixed_final_only=True,new_stage_artifact_limit_bytes=500_000_000,ref2_artifact_limit_bytes=850_000_000 if args.gt_anchored else 620_000_000)
    if args.gt_anchored:
        manifest.update(gradient_ownership=GRADIENT_OWNERSHIP,execution_horizon=24,
            loss='valid true tau MSE/.05 + all-row predicted PD L1 to WAM only + valid GT-tau PD L1 to R only; equal weights')
    manifest.update(warm_R_sha256=sha(r_path),warm_pair_directory=str(parent),aggregation_directory=str(aggregation),
        aggregation_sha256=am['aggregation_sha256'],all_aggregation_is_TRAIN=True)
    write(out/'manifest.json',manifest)
    def guard(phase,count,total):
        elapsed=time.monotonic()-started
        if elapsed>1800:raise TimeoutError('joint stage1800s deadline')
        pids=set(map(int,subprocess.check_output(['nvidia-smi','-i','2','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()))
        if pids-{os.getpid()}:raise RuntimeError('foreign GPU2 process')
        paths=list(owner.glob('wam-ref2-*'))+list((ROOT/'tmp').glob('wam-ref2-*'))+list((ROOT/'tmp/trajectory-policy').glob('wam*'))
        size=sum(int(subprocess.check_output(['du','-sb',str(p)],text=True).split()[0]) for p in paths if p.is_dir())
        own=sum(int(subprocess.check_output(['du','-sb',str(p)],text=True).split()[0]) for p in (out,scratch))
        if size>=manifest['ref2_artifact_limit_bytes'] or own>=500_000_000:raise RuntimeError('joint artifact boundary')
        print(json.dumps(dict(phase=phase,count=count,total=total,elapsed_s=elapsed,
            eta_s=(time.monotonic()-train_started)/max(count,1)*(total-count) if phase=='train' else None,
            artifact_bytes=size,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)),flush=True)
    def evaluate(ids):
        arrays={k:[] for k in ('tau','tau_true','tau_label_valid','pd','pd_true','GT_tau_R_PD','clipped','GT_tau_R_clipped')}
        with torch.no_grad():
            for begin in range(0,len(ids),16):
                b=get(ids[begin:begin+16]);tau=predict(b)
                pd=decode_target(r(b['state'],tau),b['q']);target=decode_target(b['pd_label'],b['q'])
                _,_,clipped=command(pd,b['q'],offset,scale)
                gt=decode_target(r(b['state'],b['tau']),b['q'])
                _,_,gt_clipped=command(gt,b['q'],offset,scale)
                values=dict(tau=tau,tau_true=b['tau'],tau_label_valid=b['tau_label_valid'],pd=pd,pd_true=target,
                    GT_tau_R_PD=gt,clipped=clipped,GT_tau_R_clipped=gt_clipped)
                for k,v in values.items():arrays[k].append(v.cpu().numpy())
        arrays={k:np.concatenate(v) for k,v in arrays.items()};delta=arrays['pd']-arrays['pd_true']
        def pdstats(d,c):return dict(xyz_p95_mm=float(np.percentile(np.linalg.norm(d[:,:3],axis=-1),95)*1000),
            rotation_p95_rad=float(np.percentile(np.abs(d[:,3:6]).max(-1),95)),
            finger_p95_rad=float(np.percentile(np.abs(d[:,[6,8,10,12,14,15]]).max(-1),95)),clipping_rate=float(c.mean()))
        stats=dict(count=len(ids),**pdstats(delta,arrays['clipped']));valid=arrays['tau_label_valid']
        if valid.any():
            e=np.linalg.norm((arrays['tau'][valid]-arrays['tau_true'][valid]).reshape(-1,24,11,3),axis=-1)*1000
            stats.update(h1_epe_mm=float(e[:,0].mean()),first8_epe_mm=float(e[:,:8].mean()),
                all24_epe_mm=float(e.mean()),wrist_h1_epe_mm=float(e[:,0,0].mean()))
            stats.update({'GT_tau_R_'+k:v for k,v in pdstats((arrays['GT_tau_R_PD']-arrays['pd_true'])[valid],arrays['GT_tau_R_clipped'][valid]).items()})
        arrays['row_ids']=np.asarray(ids)
        return stats,arrays
    initial={};raw={}
    for name,ids in groups.items():
        initial[name],values=evaluate(ids);raw.update({'initial_'+name+'_'+k:v for k,v in values.items()})
    guard('initial_report',1,1)
    params=[p for net in modules.values() for p in net.parameters()]
    optimizer=torch.optim.AdamW([dict(params=list(net.parameters()),lr=1e-5 if name=='scene' else 1e-4)
        for name,net in modules.items()],weight_decay=.01)
    nt=np.asarray([t for _,t,_ in nominal]);startup=np.flatnonzero(nt<16);later=np.flatnonzero(nt>=16)
    curve=[];draws=[];gradient_checks={};train_started=time.monotonic()
    for step in range(1,manifest['maximum_updates']+1):
        ids=np.r_[rng.choice(startup,16),rng.choice(later,16),
            balanced_draw(rng,expert_pools),balanced_draw(rng,pure_pools)]
        draws.append(ids);b=get(ids);optimizer.zero_grad(set_to_none=True)
        tau=predict(b)
        if args.gt_anchored:
            geometry,action,anchor=anchored_losses(r,b['state'],tau,b['tau'],b['tau_label_valid'],b['pd_label'])
            if step==1:gradient_checks=anchored_gradient_evidence(action,anchor,modules)
            loss=geometry+action+anchor
        else:
            pd=r(b['state'],tau)
            geometry,action=joint_losses(tau,b['tau'],b['tau_label_valid'],pd,b['pd_label'],r.target_scale)
            if step==1:gradient_checks=action_gradient_evidence(action,modules)
            loss=geometry+action
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite joint loss')
        loss.backward()
        if any(p.grad is not None for net in frozen.values() for p in net.parameters()):raise ValueError('frozen world/action gradient')
        norm=torch.nn.utils.clip_grad_norm_(params,1.)
        if not torch.isfinite(norm):raise FloatingPointError('nonfinite joint gradient')
        optimizer.step()
        if step==1 or step%25==0:
            curve.append(dict(step=step,tau_MSE=float(geometry),PD_L1=float(action),gradient=float(norm)))
            if args.gt_anchored:curve[-1]['R_GT_L1']=float(anchor)
            guard('train',step,manifest['maximum_updates'])
    final={}
    for name,ids in groups.items():
        final[name],values=evaluate(ids);raw.update({'final_'+name+'_'+k:v for k,v in values.items()})
    final_hash={k:state_hash(v.state_dict()) for k,v in modules.items()}
    frozen_final={k:state_hash(v.state_dict()) for k,v in frozen.items()}
    norms_final={'WAM':buffer_hash(model),'R':buffer_hash(r)}
    if frozen_final!=frozen_hash or norms_final!=norms:raise ValueError('frozen weights/normalization drift')
    if any(final_hash[k]==initial_hash[k] for k in TRAIN_MODULES):raise ValueError('one policy-chain module did not change')
    if any(sha(p)!=h for p,h in evidence.items()):raise ValueError('actual consumed input drift')
    if any(not np.isfinite(v).all() for v in raw.values()) or any(not torch.isfinite(p).all() for p in params):
        raise FloatingPointError('nonfinite joint final data/parameters')
    screens=offline_screen(final)
    result=dict(status='COMPLETED',initial=initial,final=final,offline_screen=screens,
        offline_screen_pass=all(screens.values()) and not args.engineering_smoke,curve=curve,
        action_gradient_checks=gradient_checks,train_initial_hash=initial_hash,train_final_hash=final_hash,
        frozen_initial_hash=frozen_hash,frozen_final_hash=frozen_final,normalization_buffers_unchanged=True,
        normalization_initial_hash=norms,normalization_final_hash=norms_final,base_R_sha256=BASE_R_SHA,
        updates=manifest['maximum_updates'],elapsed_s=time.monotonic()-started,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        conclusion='engineering PASS' if args.engineering_smoke else 'UNCLEAR; exact paired physics needed')
    if args.gt_anchored:result.update(gradient_checks)
    if not args.engineering_smoke:
        rp=dict(schema=base_r_packet['schema'],width=base_r_packet['width'],use_tau=True,
            model={k:v.detach().cpu() for k,v in r.state_dict().items()},base_R_sha256=BASE_R_SHA,
            joint_training_schema=JOINT_SCHEMA,run_id=out.name,input_sha256=evidence)
        if args.gt_anchored:rp.update(gradient_ownership=GRADIENT_OWNERSHIP,execution_horizon=24)
        torch.save(rp,out/'tau.pt');result['checkpoint_sha256']=sha(out/'tau.pt')
        saved={name:{k:v.detach().cpu() for k,v in getattr(model,name).state_dict().items()} for name in POLICY_MODULES}
        torch.save(dict(schema=JOINT_SCHEMA,head={k:saved[k] for k in ('trajectory','hand_query')},
            policy_patch={k:saved[k] for k in ('scene','goal')},policy_patch_schema=PATCH_SCHEMA,
            base_checkpoint_sha256=sha(base),joint_R_sha256=result['checkpoint_sha256'],input_sha256=evidence),out/'head.pt')
        result['head_sha256']=sha(out/'head.pt')
    np.savez_compressed(out/'predictions.npz',**raw,sampled_row_ids=np.asarray(draws),
        rows_kind=np.asarray([k for k,t,e in rows]),rows_tick=np.asarray([t for k,t,e in rows]),rows_role=np.asarray([e for k,t,e in rows]))
    write(out/'result.json',result)
    manifest.update(status='COMPLETED',result_sha256=sha(out/'result.json'),
        head_sha256=result.get('head_sha256'),checkpoint_sha256=result.get('checkpoint_sha256'),
        head_training_R_sha256=result.get('checkpoint_sha256'),predictions_sha256=sha(out/'predictions.npz'))
    write(out/'manifest.json',manifest);guard('completed',1,1)


if __name__=='__main__':
    main()

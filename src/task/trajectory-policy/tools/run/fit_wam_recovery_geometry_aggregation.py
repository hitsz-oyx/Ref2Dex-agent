"""Explicitly retire both inspected takeovers into geometry training.

Static sibling of fit_wam_recovery_geometry.py. Both own-expert sources are
TRAINING data; their error is not a validation/generalization result. Only a
fresh actor rollout can provide subsequent actual-state evidence. Original
source actor/mixed-future denial is preserved; new scoped admissions enumerate
only the role3 futures wholly caused by its actual unclipped expert controls.
"""
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
sys.path[:0] = [str(TASK / 'src'), str(TASK / 'tools/audit'), str(TASK / 'tools/run'),
    str(ROOT / 'src/task/goal-point-wam/src'), str(ROOT / 'src/task/cm-pointflow-effect-pretrain/src')]
from audit_wam_ref2_shadow import sha, write
from calibrate_wam_robot_startup import network_hash

PARENT_SHA = 'c03e75a9f55c881153c83161a06646ca7932ccd47f72b5038193aaa8544d7286'
SOURCE_SHA = {
    'train': 'b0edec52551bc9e7f528c80022f7aad946793d02e450fc234bac8eabde8c32ee',
    'cross': '27da16275bdbc2c482ac6bfa164bf206dba4781cebc0c6b07c9a01d180329102'}


def own_expert_admission(manifest, audit, result, data, trajectory_hash, split):
    """CPU contract: expert-only own futures, not source-wide actor permission."""
    if split not in SOURCE_SHA:
        raise ValueError('only preregistered train/cross sources supported')
    controls, takeover, stop = (542, 16, 96) if split == 'train' else (192, 32, 169)
    roles = ['teacher', 'gt_tau_R', 'wam_R', 'wam_command_replay_then_expert_tick%d' % takeover]
    if (manifest.get('status') != 'COMPLETED' or result.get('status') != 'COMPLETED'
            or manifest.get('controls') != controls or manifest.get('takeover_tick') != takeover
            or manifest.get('roles') != roles or manifest.get('training_allowed') is not True
            or manifest.get('expert_PD_training_allowed') is not True
            or manifest.get('mixed_policy_future_hand_training_allowed') is not False
            or manifest.get('actor_future_hand_training_allowed') not in (None, False)
            or audit.get('pass') is not True or audit.get('all_arrays_finite') is not True
            or audit.get('checks_max') != 0 or audit.get('check_count') != (9768 if split=='train' else 3468)
            or manifest.get('dt') != 1/30
            or audit.get('controls') != controls or audit.get('measured_states') != controls + 1
            or ('takeover_tick' in audit and audit['takeover_tick'] != takeover)
            or audit.get('native_data_sha256') != trajectory_hash
            or manifest.get('trajectory_sha256') != trajectory_hash
            or trajectory_hash != SOURCE_SHA[split]):
        raise ValueError('authenticated preregistered source and independent audit required')
    if split == 'train':
        if result.get('calibration_pass') is not True:
            raise ValueError('original teacher/GT calibration required')
    elif (result.get('partial_episode') is not True or result.get('calibration_pass') is not None
            or result.get('recovery_pass') is not None or result.get('local_grasp_calibration') is not True):
        raise ValueError('partial local calibration required, not placement success')
    expected = {'q': (controls+1,4,18), 'dq': (controls+1,4,18),
        'hand': (controls+1,4,11,3), 'obj': (controls+1,4,4,4), 'velocity': (controls+1,4,6),
        'applied': (controls,4,18), 'pd_targets': (controls,4,18),
        'expert_raw': (controls,2,18), 'expert_action': (controls,2,18),
        'expert_PD': (controls,2,18), 'expert_clipped': (controls,2), 'expert_applied': (controls,2),
        'learner_object_history': (controls,2,4,4,4), 'learner_hand_history': (controls,2,4,11,3),
        'learner_goal': (controls,2,4,4)}
    if any(k not in data or data[k].shape != shape or not np.isfinite(data[k]).all()
           for k, shape in expected.items()):
        raise ValueError('finite correctly shaped current/action/future traces required')
    ticks = np.arange(takeover, stop)
    causing = np.unique((ticks[:,None] + np.arange(24)).ravel())
    if (not data['expert_applied'][causing,1].all() or data['expert_clipped'][causing,1].any()
            or not np.array_equal(data['applied'][causing,3], data['expert_action'][causing,1])
            or not np.array_equal(data['expert_action'][causing,1], np.clip(data['expert_raw'][causing,1],-1,1))):
        raise ValueError('all 24 causing controls must be actual unclipped own expert')
    offset = np.asarray(manifest['native_controller']['offset'],np.float32)
    scale = np.asarray(manifest['native_controller']['scale'],np.float32)
    if offset.shape != (18,) or scale.shape != (18,) or not np.isfinite(offset).all() or not np.isfinite(scale).all() or (scale<=0).any():
        raise ValueError('finite positive native PD conversion required')
    action = data['expert_action'][causing,1].copy()
    action[:,6:] = (action[:,6:] + 1) / 2
    pd = offset + scale * action
    pd[:,:6] += data['q'][causing,3,:6]
    for distal,parent,ratio in ((7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8)):
        pd[:,distal] = pd[:,parent] * ratio
    if (not np.allclose(pd,data['expert_PD'][causing,1],atol=1e-6,rtol=0)
            or not np.array_equal(data['pd_targets'][causing,3], data['expert_PD'][causing,1])):
        raise ValueError('actual current-q expert PD inversion mismatch')
    for tick in ticks:
        history=np.maximum(tick+np.arange(-3,1),0)
        if (not np.array_equal(data['learner_hand_history'][tick,1],data['hand'][history,3])
                or not np.array_equal(data['learner_object_history'][tick,1],data['obj'][history,3])):
            raise ValueError('H4 must be measured own row including observed command prefix')
    return dict(schema='ref2dex.own-expert-actor-geometry-admission.v1',admitted=True,
        actor_expert_future_training_allowed=True,diagnostic_only=False,
        prior_cross_diagnostic_retired_to_training=split=='cross',training_source_split=split,
        source_actor_future_hand_training_allowed=manifest.get('actor_future_hand_training_allowed'),
        mixed_policy_future_hand_training_allowed=False,role=3,first_tick=takeover,last_tick=stop-1,
        horizon=24,windows=len(ticks),source_trajectory_sha256=trajectory_hash,
        future_causing_control_first=int(causing[0]),future_causing_control_last=int(causing[-1]),
        measured_future_state_last=int(causing[-1]+1),expert_clipping_count=0,
        actual_action_expert_max_error=0.,scope='own role3 actual expert-only futures; limited approach/grasp; no role2 or placement claim')


def balanced_recovery_draws(rng, old_ids, retired_ids):
    """Exactly32 own-expert windows from each source; neither is held out."""
    old_ids=np.asarray(old_ids);retired_ids=np.asarray(retired_ids)
    if not len(old_ids) or not len(retired_ids) or np.intersect1d(old_ids,retired_ids).size:
        raise ValueError('nonempty disjoint old and retired recovery train pools required')
    return np.r_[rng.choice(old_ids,32,replace=True),rng.choice(retired_ids,32,replace=True)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=2)
    parser.add_argument('--engineering-smoke',action='store_true')
    args=parser.parse_args()
    if args.gpu!=2:
        raise ValueError('this bounded geometry probe is assigned GPU2')
    started=time.monotonic();out=args.output.resolve()
    if out.exists() or ROOT/'outputs/trajectory-policy' not in out.parents:
        raise ValueError('fresh task-owned output required')
    if subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
        raise RuntimeError('GPU occupied')
    scratch=ROOT/'tmp/trajectory-policy/wam-robot-head';scratch.mkdir(parents=True,exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),TMPDIR=str(scratch),TRITON_CACHE_DIR=str(scratch/'triton'))
    import torch
    import torch.nn.functional as F
    from trajectory_policy.wam_shadow import robot_goal_sample, WAM_TO_NATIVE
    from trajectory_policy.retargeter_data import features
    from trajectory_policy.pd_retargeter import load_model,encode_target,decode_target,command
    from goal_point_wam.data import goal_collate
    from oakink_wm.pointworld_performance import install_fused_hilbert
    torch.set_num_threads(2);torch.manual_seed(295);torch.backends.cuda.matmul.allow_tf32=True
    rng=np.random.default_rng(296)
    previous=ROOT/'outputs/trajectory-policy/wam-ref2-head-recovery-geometry-20261012-r1'
    pm,pr=[json.loads((previous/k).read_text()) for k in ('manifest.json','result.json')]
    if pm['status']!='COMPLETED' or pr['status']!='COMPLETED' or sha(previous/'head.pt')!=PARENT_SHA or pr['head_sha256']!=PARENT_SHA:
        raise ValueError('fixed first geometry head required')
    if pm.get('schema')!='ref2dex.wam-recovery-geometry.v1' or pm.get('result_sha256')!=sha(previous/'result.json') or pr.get('updates')!=500:
        raise ValueError('completed authenticated fixed500 geometry parent required')
    evidence=dict(pm['input_sha256'])
    evidence.update({str(previous/k):sha(previous/k) for k in ('head.pt','manifest.json','result.json')})
    inputs={};admissions={}
    paths={'train':ROOT/'outputs/trajectory-policy/wam-ref2-execution-recovery-20261012-r2/native',
        'cross':ROOT/'outputs/trajectory-policy/wam-ref2-execution-aggcollect-20261012-r1/native'}
    fields=('q','dq','hand','obj','velocity','applied','pd_targets','expert_raw','expert_action',
        'expert_PD','expert_clipped','expert_applied','learner_object_history','learner_hand_history','learner_goal')
    for split,path in paths.items():
        m,a,res=[json.loads((path/k).read_text()) for k in ('manifest.json','root_recovery_audit.json','result.json')]
        with np.load(path/'trajectory.npz',allow_pickle=False) as f: inputs[split]={k:f[k].copy() for k in fields}
        admissions[split]=own_expert_admission(m,a,res,inputs[split],sha(path/'trajectory.npz'),split)
        for name in ('trajectory.npz','manifest.json','root_recovery_audit.json','result.json'):
            target=path/name;evidence[str(target)]=sha(target)
            admissions[split][name.replace('.json','').replace('.npz','')+'_path']=str(target)
            admissions[split][name.replace('.json','').replace('.npz','')+'_sha256']=sha(target)
    native=json.loads((paths['train']/'manifest.json').read_text())['native_controller']
    cross_native=json.loads((paths['cross']/'manifest.json').read_text())['native_controller']
    if native!=cross_native:raise ValueError('native conversion changed between sources')
    source=ROOT/'outputs/trajectory-policy/competent-pd-teacher-20261011-r2'
    sm,sr,sa=[json.loads((source/k).read_text()) for k in ('manifest.json','result.json','audit.json')]
    if (sm.get('status')!='COMPLETED' or sm.get('training_allowed') is not True or sr.get('teacher_readiness') is not True
            or sa.get('pass') is not True or sm.get('dt')!=1/30 or sha(source/'trajectory.npz')!=sm.get('trajectory_sha256')):
        raise ValueError('qualified actual nominal teacher required')
    with np.load(source/'trajectory.npz',allow_pickle=False) as f:
        inputs['nominal']={k:f[k][:,:28].copy() for k in ('q','dq','hand','obj','velocity','pd_targets')}
    for k in ('trajectory.npz','manifest.json','result.json','audit.json'):
        target=str(source/k);digest=sha(source/k)
        if target in evidence and evidence[target]!=digest:
            raise ValueError('nominal source differs from authenticated startup head')
        evidence[target]=digest
    ref_path=ROOT/'outputs/trajectory-policy/competent-reference-20261011-r1/reference.npz'
    canonical_path=ROOT/'outputs/consequence-evaluator/old-utility-pw-20261009-r2/canonical.npz'
    with np.load(ref_path,allow_pickle=False) as f:reference=f['obj'].copy()
    with np.load(canonical_path,allow_pickle=False) as f:canonical={k:f[k].copy() for k in f.files}
    for split,ticks in (('train',range(16,96)),('cross',range(32,169))):
        for tick in ticks:
            if not np.array_equal(inputs[split]['learner_goal'][tick,1],reference[min(tick+24,542)]):
                raise ValueError('decision-known planned goal differs from authenticated original reference')
    base=ROOT/'outputs/goal-point-wam/goal-memory-ddp-20261012-r3/train/final.pt'
    stats=ROOT/'outputs/cm-pointflow-effect-pretrain/pointworld-main3-sharedstats-20261009/norm_stats.json'
    original=ROOT/'outputs/trajectory-policy/wam-ref2-shadow-20261012-r2/code/model.py'
    r_path=ROOT/'outputs/trajectory-policy/competent-pd-fit-20261011-r1/tau.pt'
    for path in (Path(__file__),TASK/'tools/run/calibrate_wam_robot_startup.py',
            TASK/'src/trajectory_policy/wam_shadow.py',TASK/'src/trajectory_policy/retargeter_data.py',
            TASK/'src/trajectory_policy/pd_retargeter.py',ref_path,canonical_path,base,stats,original,r_path):
        evidence.setdefault(str(path),sha(path))
    if any(sha(p)!=h for p,h in evidence.items()):raise ValueError('warm source/input identity drift')
    packet=torch.load(base,map_location='cpu',weights_only=False)
    if sha(original)!=packet['identity']['model_sha256'] or sha(stats)!=packet['identity']['stats_sha256']:
        raise ValueError('original base model implementation/stats required')
    spec=importlib.util.spec_from_file_location('frozen_wam_recovery_geometry',original)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);install_fused_hilbert()
    model=module.GoalPointWAM(json.loads(stats.read_text()),packet['config']).cuda().eval()
    model.load_state_dict(packet['model'],strict=True)
    head=torch.load(previous/'head.pt',map_location='cpu',weights_only=False)
    if head['base_checkpoint_sha256']!=sha(base) or set(head['head'])!={'trajectory','hand_query'} or 'scene_patch' in head:
        raise ValueError('original head-only startup packet required')
    for name,value in head['head'].items():getattr(model,name).load_state_dict(value,strict=True)
    del head,packet
    r=load_model(torch.load(r_path,map_location='cpu',weights_only=False),'cuda')
    for net in (model,r):
        for p in net.parameters():p.requires_grad_(False)
    for net in (model.trajectory,model.hand_query):
        for p in net.parameters():p.requires_grad_(True)
    frozen={name:getattr(model,name) for name in ('scene','goal','action','world')};frozen['R']=r
    frozen_hashes={name:network_hash(net) for name,net in frozen.items()}
    if frozen_hashes!=pr['frozen_final_hash']:raise ValueError('frozen startup identities differ')
    offset,scale=[torch.as_tensor(native[k],device='cuda') for k in ('offset','scale')]
    nominal=[('nominal',t,e) for t in range(519) for e in range(16)]
    panel=[0,1,2,4,8,16,32,64,96,128,192,256,320,384,448,518]
    report=[('nominal',t,e) for t in panel for e in range(24,28)]
    train=[('train',t,3) for t in range(16,96)]
    cross=[('cross',t,3) for t in range(32,169)]
    pure=[('train',t,2) for t in range(32)]
    if args.engineering_smoke:nominal=[('nominal',t,e) for t in (0,16) for e in range(16)]
    rows=nominal+report+train+cross+pure
    groups={};begin=0
    for name,part in (('nominal_report',report),('recovery_training',train),('retired_cross_training',cross),('pure_PD_only',pure)):
        if begin==0:begin=len(nominal)
        groups[name]=torch.arange(begin,begin+len(part),device='cuda');begin+=len(part)
    order=torch.as_tensor(WAM_TO_NATIVE,device='cuda')
    def predict(b):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            tau=model.trajectory(b['scene'],b['scene_mask'],b['goal'],b['goal_mask'],model.observed_hand_condition(b)).float()
        return tau[:,:,0].index_select(2,order).reshape(-1,24,33)
    def guard(phase,count,total):
        elapsed=time.monotonic()-started
        if elapsed>600:raise TimeoutError('bounded geometry deadline')
        pids=set(map(int,subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()))
        if pids-{os.getpid()}:raise RuntimeError('foreign GPU process')
        paths=[p for p in (ROOT/'outputs/trajectory-policy').glob('wam-ref2-*') if p.is_dir()]
        paths += [scratch]+[p for p in (ROOT/'tmp').glob('wam-ref2-*') if p.is_dir()]
        size=sum(int(subprocess.check_output(['du','-sb',str(p)],text=True).split()[0]) for p in paths)
        if size>=100_000_000:raise RuntimeError('ref2 aggregate artifact limit')
        print(json.dumps(dict(phase=phase,count=count,total=total,elapsed_s=elapsed,eta_s=elapsed/max(count,1)*(total-count),artifact_bytes=size)),flush=True)
    out.mkdir(parents=True)
    for split,value in admissions.items():write(out/(split+'_actor_admission.json'),value)
    for split in admissions:
        admission_path=out/(split+'_actor_admission.json')
        evidence[str(admission_path)]=sha(admission_path)
    manifest=dict(schema='ref2dex.wam-recovery-geometry-aggregation.v1',status='RUNNING',task='goal-point-wam',
        experiment_id='P-20261012-goal-point-wam-ref2-recovery-geometry',run_id=out.name,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        seed=295,draw_seed=296,physical_gpu=args.gpu,input_sha256=evidence,training_allowed=False,
        maximum_updates=1 if args.engineering_smoke else 500,nominal_windows=len(nominal),
        recovery_training_windows=217,old_source_training_windows=80,retired_cross_training_windows=137,pure_PD_only_windows=32,
        previous_cross_diagnostic_retired_to_training=True,recovery_sampling='32 old t16 + 32 retired t32; replacement',
        parent_collection_screen_pass=pr['collection_screen_pass'],recovery_holdout_windows=0,
        warm_head_dir=str(previous),warm_head_sha256=PARENT_SHA,base_checkpoint_sha256=sha(base),
        head_training_R_sha256=sha(r_path),learning_rate=1e-4,batch=128,fixed_final_only=True,
        loss='nominal Huber tau/.05 + actual own expert recovery Huber tau/.05; NO R/PD gradient',
        trajectory_label_shape=[24,33],active_hand='right',absent_left_masked=True,
        actor_inputs='own measured H4/current points + decision-known planned object goal; no q/future/time',
        source_blanket_future_deny_preserved=True,cross_takeover_selection_allowed=False,
        recovery_groups_evidence_scope='TRAIN fit observations only; fresh new-actor rollout required',
        train_admission_path=str(out/'train_actor_admission.json'),train_admission_sha256=sha(out/'train_actor_admission.json'),
        cross_admission_path=str(out/'cross_actor_admission.json'),cross_admission_sha256=sha(out/'cross_actor_admission.json'))
    write(out/'manifest.json',manifest)
    cache={};equivalent=0.
    with torch.no_grad():
        for begin in range(0,len(rows),128):
            samples=[];states=[];taus=[];pds=[];qs=[];label_valid=[]
            for i,(kind,tick,env) in enumerate(rows[begin:begin+128]):
                d=inputs[kind];past=np.maximum(tick+np.arange(-3,1),0)
                samples.append(robot_goal_sample(d['obj'][past,env],d['hand'][past,env],reference[min(tick+24,542)],canonical,sample_id=begin+i))
                valid=not(kind=='train' and env==2)
                future=d['hand'][tick+np.arange(1,25),env] if valid else np.broadcast_to(d['hand'][tick,env],(24,11,3))
                state,tau=features(*(d[k][tick,env] for k in ('q','dq','hand','obj','velocity')),future)
                states.append(state);taus.append(tau);qs.append(d['q'][tick,env]);label_valid.append(valid)
                pd=d['pd_targets'][tick,env] if kind=='nominal' else d['expert_PD'][tick,env-2]
                pds.append(encode_target(pd,d['q'][tick,env]))
            batch={k:v.cuda() for k,v in goal_collate(samples).items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):
                scene=model.scene(batch);goal,mask=model.goal(batch['start_object_points'],batch['goal_object_points'],batch['object_goal_valid'])
            values=dict(scene=scene['scene_tokens'].to(torch.bfloat16),scene_mask=scene['scene_mask'],goal=goal.to(torch.bfloat16),goal_mask=mask,
                start_hand=batch['start_hand'],start_hand_history=batch['start_hand_history'],start_hand_valid=batch['start_hand_valid'],
                state=torch.as_tensor(np.stack(states),device='cuda'),tau=torch.as_tensor(np.stack(taus),device='cuda'),
                pd_label=torch.as_tensor(np.stack(pds),device='cuda'),q=torch.as_tensor(np.stack(qs),device='cuda'),
                tau_label_valid=torch.as_tensor(label_valid,device='cuda'))
            if begin==0:
                equivalent=float((predict(values)-predict(dict(values,scene=scene['scene_tokens'],goal=goal))).abs().max())
                if equivalent>1e-5:raise ValueError('cache/direct actor mismatch')
            for k,v in values.items():cache.setdefault(k,[]).append(v.detach())
            if begin==0 or begin//128%16==0:guard('cache',min(begin+128,len(rows)),len(rows))
    cache={k:torch.cat(v) for k,v in cache.items()};get=lambda ids:{k:v[ids] for k,v in cache.items()}
    def evaluate(ids):
        b=get(ids)
        with torch.no_grad():
            tau=predict(b);pd=decode_target(r(b['state'],tau),b['q']);target=decode_target(b['pd_label'],b['q'])
            _,action,clipped=command(pd,b['q'],offset,scale)
            gt_pd=decode_target(r(b['state'],b['tau']),b['q'])
        delta=(pd-target).cpu().numpy();valid=b['tau_label_valid'].cpu().numpy()
        epe=torch.linalg.vector_norm((tau-b['tau']).reshape(-1,24,11,3),dim=-1).cpu().numpy()*1000
        stats=dict(count=len(ids),xyz_p95_mm=float(np.percentile(np.linalg.norm(delta[:,:3],axis=-1),95)*1000),
            rotation_p95_rad=float(np.percentile(np.abs(delta[:,3:6]).max(-1),95)),
            finger_p95_rad=float(np.percentile(np.abs(delta[:,[6,8,10,12,14,15]]).max(-1),95)),clipping_rate=float(clipped.float().mean()))
        if valid.any():
            e=epe[valid];stats.update(h1_epe_mm=float(e[:,0].mean()),first8_epe_mm=float(e[:,:8].mean()),all24_epe_mm=float(e.mean()),
                wrist_h1_epe_mm=float(e[:,0,0].mean()),h1_p95_mm=float(np.percentile(e[:,0],95)),first8_p95_mm=float(np.percentile(e[:,:8],95)),
                all24_p95_mm=float(np.percentile(e,95)),GT_tau_R_xyz_p95_mm=float(np.percentile(np.linalg.norm((gt_pd-target).cpu().numpy()[valid,:3],axis=-1),95)*1000))
            if rows[int(ids[0])][0]=='cross':
                ticks=np.asarray([rows[int(i)][1] for i in ids]);stats['blocks']={}
                for name,lo,hi in (('early32_48',32,48),('early48_64',48,64),('later64_169',64,169)):
                    part=epe[(ticks>=lo)&(ticks<hi)];stats['blocks'][name]=dict(count=len(part),first8_epe_mm=float(part[:,:8].mean()),all24_epe_mm=float(part.mean()))
        return stats,dict(tau=tau.cpu().numpy(),tau_true=b['tau'].cpu().numpy(),tau_label_valid=valid,
            pd=pd.cpu().numpy(),pd_true=target.cpu().numpy(),GT_tau_R_PD=gt_pd.cpu().numpy(),action=action.cpu().numpy(),row_ids=ids.cpu().numpy())
    initial={};raw={}
    for name,ids in groups.items():
        initial[name],values=evaluate(ids);raw.update({'initial_'+name+'_'+k:v for k,v in values.items()})
    params=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
    nt=np.asarray([t for _,t,_ in nominal]);startup=np.flatnonzero(nt<16);later=np.flatnonzero(nt>=16)
    train_ids=groups['recovery_training'].cpu().numpy();retired_ids=groups['retired_cross_training'].cpu().numpy();curve=[];gradients={};recovery_gradient=0.
    for step in range(1,manifest['maximum_updates']+1):
        nr=np.r_[rng.choice(startup,16),rng.choice(later,48)];rr=balanced_recovery_draws(rng,train_ids,retired_ids)
        nb,rb=get(torch.as_tensor(nr,device='cuda')),get(torch.as_tensor(rr,device='cuda'))
        if not nb['tau_label_valid'].all() or not rb['tau_label_valid'].all():raise ValueError('only admitted tau-labelled rows train')
        optimizer.zero_grad(set_to_none=True)
        nominal_loss=F.huber_loss(predict(nb)/.05,nb['tau']/.05);recovery_loss=F.huber_loss(predict(rb)/.05,rb['tau']/.05)
        loss=nominal_loss+recovery_loss
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite geometry loss')
        if step==1:
            grads=torch.autograd.grad(recovery_loss,params,retain_graph=True,allow_unused=True)
            recovery_gradient=float(sum((g.abs().sum() for g in grads if g is not None),torch.tensor(0.,device='cuda')))
            if not np.isfinite(recovery_gradient) or recovery_gradient<=0:raise ValueError('recovery geometry gradient missing')
        loss.backward()
        if step==1:
            gradients={name:float(sum((p.grad.abs().sum() for p in net.parameters() if p.grad is not None),torch.tensor(0.,device='cuda')))
                for name,net in (('trajectory',model.trajectory),('hand_query',model.hand_query))}
            if not all(v>0 for v in gradients.values()):raise ValueError('head/hand-query gradient missing')
        if any(p.grad is not None for net in frozen.values() for p in net.parameters()):raise ValueError('frozen gradient')
        norm=torch.nn.utils.clip_grad_norm_(params,1.)
        if not torch.isfinite(norm):raise FloatingPointError('nonfinite gradient')
        optimizer.step()
        if step==1 or step%25==0:
            curve.append(dict(step=step,nominal_tau=float(nominal_loss),recovery_tau=float(recovery_loss),gradient=float(norm)))
            guard('train',step,manifest['maximum_updates'])
    final={}
    for name,ids in groups.items():
        final[name],values=evaluate(ids);raw.update({'final_'+name+'_'+k:v for k,v in values.items()})
    final_hashes={name:network_hash(net) for name,net in frozen.items()}
    if final_hashes!=frozen_hashes:raise ValueError('frozen weights changed')
    if any(sha(p)!=h for p,h in evidence.items()):raise ValueError('consumed input/source drift')
    all_arrays_finite=all(np.isfinite(value).all() for value in raw.values())
    if not all_arrays_finite:raise FloatingPointError('nonfinite final raw prediction/label/diagnostic')
    signal=dict(nominal_geometry_retained=final['nominal_report']['first8_epe_mm']<=21.522)
    collection=dict(geometry_screen_pass=all(signal.values()),nominal_PD_retained=final['nominal_report']['xyz_p95_mm']<=50.,
        nominal_clipping_lt1pct=final['nominal_report']['clipping_rate']<.01)
    result=dict(status='COMPLETED',initial=initial,final=final,adaptation_signal=signal,collection_signal=collection,
        prior_cross_diagnostic_retired_to_training=True,recovery_evidence_scope='TRAIN fit only; not generalization',
        parent_collection_screen_pass=pr['collection_screen_pass'],new_actor_rollout_required=True,
        all_arrays_finite=all_arrays_finite,
        collection_screen_pass=all(collection.values()) and not args.engineering_smoke,curve=curve,updates=manifest['maximum_updates'],
        frozen_initial_hash=frozen_hashes,frozen_final_hash=final_hashes,gradient_checks=gradients,
        recovery_tau_actor_gradient_sum=recovery_gradient,R_backward_used=False,cache_direct_max_error_m=equivalent,
        elapsed_s=time.monotonic()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        conclusion='engineering PASS' if args.engineering_smoke else ('PROMISING' if all(signal.values()) else 'UNCLEAR'))
    if not args.engineering_smoke:
        saved={name:{k:v.detach().cpu() for k,v in getattr(model,name).state_dict().items()} for name in ('trajectory','hand_query')}
        torch.save(dict(schema=manifest['schema'],head=saved,base_checkpoint_sha256=sha(base),input_sha256=evidence,
            head_training_R_sha256=sha(r_path),config=dict(seed=295,draw_seed=296,additional_updates=500)),out/'head.pt')
        if (out/'head.pt').stat().st_size>=10_000_000:raise ValueError('head too large')
        result['head_sha256']=sha(out/'head.pt')
    np.savez_compressed(out/'predictions.npz',**raw,rows_kind=np.asarray([k for k,t,e in rows]),rows_tick=np.asarray([t for k,t,e in rows]),rows_role=np.asarray([e for k,t,e in rows]))
    write(out/'result.json',result)
    manifest.update(status='COMPLETED',result_sha256=sha(out/'result.json'),predictions_sha256=sha(out/'predictions.npz'))
    write(out/'manifest.json',manifest);guard('completed',1,1)


if __name__=='__main__':
    main()

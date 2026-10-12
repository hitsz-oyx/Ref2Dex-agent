"""Startup-coverage continuation from the authenticated300-update robot head."""
import argparse
import hashlib
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
sys.path[:0] = [str(TASK / 'src'), str(ROOT / 'src/task/goal-point-wam/src'),
    str(ROOT / 'src/task/cm-pointflow-effect-pretrain/src'), str(TASK / 'tools/audit')]
from audit_wam_ref2_shadow import sha, write


def network_hash(network):
    digest = hashlib.sha256()
    for name, value in network.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def sample_training_rows(rng, train_ticks):
    """Exactly32 startup and96 later rows; only caller-qualified train rows."""
    startup=np.flatnonzero(train_ticks<16)
    later=np.flatnonzero(train_ticks>=16)
    if not len(startup) or not len(later):
        raise ValueError('both startup and later train partitions required')
    draws=np.r_[rng.choice(startup,32,replace=True),rng.choice(later,96,replace=True)]
    rng.shuffle(draws)
    return draws


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=2)
    p.add_argument('--engineering-smoke', action='store_true')
    args = p.parse_args()
    out = args.output.resolve()
    if out.exists() or ROOT / 'outputs/trajectory-policy' not in out.parents:
        raise ValueError('fresh task-owned output required')
    if subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True).strip():
        raise RuntimeError('GPU occupied')
    scratch = ROOT / 'tmp/trajectory-policy/wam-robot-head'
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), TMPDIR=str(scratch),
                      TRITON_CACHE_DIR=str(scratch / 'triton'))
    import torch
    import torch.nn.functional as F
    from trajectory_policy.wam_shadow import robot_goal_sample, WAM_TO_NATIVE
    from trajectory_policy.pd_retargeter import load_model, decode_target, command, encode_target
    from trajectory_policy.retargeter_data import features
    from goal_point_wam.data import goal_collate
    from oakink_wm.pointworld_performance import install_fused_hilbert
    torch.set_num_threads(2)
    torch.manual_seed(293)
    torch.backends.cuda.matmul.allow_tf32 = True
    rng = np.random.default_rng(294)
    started = time.monotonic()
    shadow = ROOT / 'outputs/trajectory-policy/wam-ref2-shadow-20261012-r2'
    shadow_manifest = json.loads((shadow / 'manifest.json').read_text())
    if shadow_manifest['status'] != 'COMPLETED':
        raise ValueError('completed shadow contract required')
    for path, digest in shadow_manifest['input_sha256'].items():
        if sha(path) != digest:
            raise ValueError('authenticated shadow input drift: ' + path)
    source = ROOT / 'outputs/trajectory-policy/competent-pd-teacher-20261011-r2'
    source_manifest=json.loads((source/'manifest.json').read_text())
    source_result=json.loads((source/'result.json').read_text())
    source_audit=json.loads((source/'audit.json').read_text())
    if (source_manifest.get('status')!='COMPLETED' or source_manifest.get('training_allowed') is not True
            or source_result.get('teacher_readiness') is not True or not source_audit.get('pass')
            or source_manifest.get('dt')!=1/30
            or sha(source/'trajectory.npz')!=source_manifest.get('trajectory_sha256')):
        raise ValueError('qualified, training-eligible actual teacher data required')
    with np.load(source / 'trajectory.npz', allow_pickle=False) as f:
        # Test env28:32 are never selected for input/label construction.
        d = {k: f[k][:, :28].copy() for k in ('q','dq','hand','obj','velocity','pd_targets')}
    with np.load(ROOT / 'outputs/trajectory-policy/competent-reference-20261011-r1/reference.npz', allow_pickle=False) as f:
        ref = f['obj'].copy()
    with np.load(ROOT / 'outputs/consequence-evaluator/old-utility-pw-20261009-r2/canonical.npz', allow_pickle=False) as f:
        canonical = {k: f[k].copy() for k in f.files}
    original = shadow / 'code/model.py'
    spec = importlib.util.spec_from_file_location('frozen_wam_robot_head', original)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    install_fused_hilbert()
    checkpoint = ROOT / 'outputs/goal-point-wam/goal-memory-ddp-20261012-r3/train/final.pt'
    stats = ROOT / 'outputs/cm-pointflow-effect-pretrain/pointworld-main3-sharedstats-20261009/norm_stats.json'
    packet = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if sha(original) != packet['identity']['model_sha256'] or sha(stats) != packet['identity']['stats_sha256']:
        raise ValueError('frozen base implementation/stats differ')
    model = module.GoalPointWAM(json.loads(stats.read_text()), packet['config']).cuda().eval()
    model.load_state_dict(packet['model'], strict=True)
    del packet
    previous=ROOT/'outputs/trajectory-policy/wam-ref2-head-calibration-20261012-r1'
    previous_manifest=json.loads((previous/'manifest.json').read_text())
    previous_result=json.loads((previous/'result.json').read_text())
    if (previous_manifest['status']!='COMPLETED' or previous_result['updates']!=300
            or sha(previous/'head.pt')!=previous_result['head_sha256']):
        raise ValueError('authenticated fixed300 initial robot head required')
    for path,digest in previous_manifest['input_sha256'].items():
        if sha(path)!=digest:
            raise ValueError('initial robot head evidence drift: '+path)
    initial_head=torch.load(previous/'head.pt',map_location='cpu',weights_only=False)
    if initial_head['base_checkpoint_sha256']!=sha(checkpoint) or set(initial_head['head'])!={'trajectory','hand_query'}:
        raise ValueError('initial head has another base or module set')
    for name,value in initial_head['head'].items():
        getattr(model,name).load_state_dict(value,strict=True)
    del initial_head
    r_path = ROOT / 'outputs/trajectory-policy/competent-pd-fit-20261011-r1/tau.pt'
    r = load_model(torch.load(r_path, map_location='cpu', weights_only=False), 'cuda')
    for network in (model, r):
        for parameter in network.parameters():
            parameter.requires_grad_(False)
    for network in (model.trajectory, model.hand_query):
        for parameter in network.parameters():
            parameter.requires_grad_(True)
    parameters = [value for value in model.parameters() if value.requires_grad]
    frozen_modules = {name:getattr(model,name) for name in ('scene','goal','action','world')}
    frozen_modules['R'] = r
    frozen_hash = {name:network_hash(network) for name,network in frozen_modules.items()}
    if frozen_hash!=previous_result['frozen_final_hash']:
        raise ValueError('initial frozen backbone/R differs from calibrated300 source')
    trainable_names = [name for name,value in model.named_parameters() if value.requires_grad]
    if not all(name.startswith(('trajectory.', 'hand_query.')) for name in trainable_names):
        raise ValueError('only trajectory/observed-hand adapter may train')
    native = json.loads((ROOT / 'outputs/trajectory-policy/competent-pd-execution-20261011-r1/manifest.json').read_text())['native_controller']
    offset = torch.tensor(native['offset'], device='cuda')
    scale = torch.tensor(native['scale'], device='cuda')
    horizon_ticks = np.asarray([0,1,2,4,8,16,32,64,96,128,192,256,320,384,448,518])
    if args.engineering_smoke:
        train_ticks = np.repeat([0,4,16],16)
        train_envs = np.tile(np.arange(16),3)
    else:
        train_ticks = np.repeat(np.arange(519),16)
        train_envs = np.tile(np.arange(16),519)
    val_ticks, val_envs = np.repeat(horizon_ticks,4), np.tile(np.arange(16,20),16)
    held_ticks, held_envs = np.repeat(horizon_ticks,4), np.tile(np.arange(24,28),16)
    startup_ticks,startup_envs=np.repeat(np.arange(16),4),np.tile(np.arange(16,20),16)
    ticks = np.r_[train_ticks,val_ticks,held_ticks,startup_ticks]
    envs = np.r_[train_envs,val_envs,held_envs,startup_envs]
    out.mkdir(parents=True)
    evidence = {**shadow_manifest['input_sha256'], str(Path(__file__)):sha(Path(__file__)),
                str(source/'result.json'):sha(source/'result.json'),
                **{str(previous/name):sha(previous/name) for name in ('head.pt','manifest.json','result.json')}}
    manifest = dict(status='RUNNING', schema='ref2dex.wam-robot-startup.v1',
        task='goal-point-wam', experiment_id='P-20261012-goal-point-wam-ref2-startup',
        engineering_only=args.engineering_smoke, training_allowed=False,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        physical_gpu=args.gpu, seed=293, draw_seed=294, base_checkpoint_sha256=sha(checkpoint),
        input_sha256=evidence, train_envs=list(range(16)), validation_envs=list(range(16,24)),
        validation_panel_envs=list(range(16,20)), report_envs=list(range(24,28)),
        untouched_test_envs=list(range(28,32)), train_windows=len(train_ticks),
        source_training_allowed=True, source_teacher_readiness=True,
        validation_windows=64, held_report_windows=64, startup_validation_windows=64,
        startup_sampling_fraction=.25,startup_sampling_ticks=[0,16],later_sampling_ticks=[16,519],
        continuation_head_sha256=sha(previous/'head.pt'), trainable_names=trainable_names,
        frozen_initial_hash=frozen_hash, batch=128, learning_rate=3e-4, weight_decay=.01,
        maximum_updates=1 if args.engineering_smoke else 300,
        loss='Huber tau/.05 + 1*Huber R(encoded PD)/R target_scale vs true encoded PD/R target_scale',
        token_cache='RAM/GPU only; no disk cache', head_only_checkpoint=True, deadline_seconds=600)
    write(out / 'manifest.json',manifest)
    cache_parts = {}
    cache_equivalence = 0.
    indices = torch.as_tensor(WAM_TO_NATIVE, device='cuda')

    def prediction(batch):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            result = model.trajectory(batch['scene'],batch['scene_mask'],batch['goal'],batch['goal_mask'],
                                      model.observed_hand_condition(batch)).float()
        return result[:, :, 0].index_select(2,indices).reshape(-1,24,33)

    def monitor(phase, count, total):
        elapsed = time.monotonic()-started
        if elapsed > 600:
            raise TimeoutError('10min startup/closure shared deadline')
        pids = subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-compute-apps=pid',
                                      '--format=csv,noheader,nounits'],text=True).split()
        if set(map(int,pids))-{os.getpid()}:
            raise RuntimeError('foreign GPU process appeared')
        owned=[path for path in (ROOT/'outputs/trajectory-policy').glob('wam-ref2-*') if path.is_dir()]
        owned += [scratch,*[path for path in (ROOT/'tmp').glob('wam-ref2-*') if path.is_dir()]]
        artifact_bytes=sum(int(subprocess.check_output(['du','-sb',str(path)],text=True).split()[0]) for path in owned)
        if artifact_bytes>=100_000_000:
            raise RuntimeError('ref2 combined artifact bound100MB reached')
        gpu = subprocess.check_output(['nvidia-smi','-i',str(args.gpu),
            '--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip()
        row=dict(phase=phase,count=count,total=total,elapsed_s=elapsed,
                 phase_eta_s=elapsed/max(count,1)*(total-count),gpu=gpu,ref2_artifact_bytes=artifact_bytes)
        with (out/'monitor.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)

    # Each row encodes only current/past geometry plus planned object endpoint.
    with torch.no_grad():
        for begin in range(0,len(ticks),128):
            samples=[]; states=[]; targets=[]; pd_labels=[]
            for row in range(begin,min(begin+128,len(ticks))):
                tick,env=int(ticks[row]),int(envs[row])
                past=np.maximum(np.arange(tick-3,tick+1),0)
                samples.append(robot_goal_sample(d['obj'][past,env],d['hand'][past,env],ref[tick+24],
                                                  canonical,sample_id=row))
                state,tau=features(*(d[k][tick,env] for k in ('q','dq','hand','obj','velocity')),
                                   d['hand'][tick+np.arange(1,25),env])
                states.append(state);targets.append(tau)
                pd_labels.append(encode_target(d['pd_targets'][tick,env],d['q'][tick,env]))
            batch={k:v.cuda() for k,v in goal_collate(samples).items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):
                scene=model.scene(batch)
                goal,goal_mask=model.goal(batch['start_object_points'],batch['goal_object_points'],batch['object_goal_valid'])
            values=dict(scene=scene['scene_tokens'].to(torch.bfloat16),scene_mask=scene['scene_mask'],
                goal=goal.to(torch.bfloat16),goal_mask=goal_mask,
                start_hand=batch['start_hand'],start_hand_history=batch['start_hand_history'],
                start_hand_valid=batch['start_hand_valid'], state=torch.as_tensor(np.stack(states),device='cuda'),
                tau=torch.as_tensor(np.stack(targets),device='cuda'),
                pd_label=torch.as_tensor(np.stack(pd_labels),device='cuda'),
                q=torch.as_tensor(d['q'][ticks[begin:begin+len(samples)],envs[begin:begin+len(samples)]],device='cuda'))
            if begin == 0:
                direct=dict(values,scene=scene['scene_tokens'],goal=goal)
                cache_equivalence=float((prediction(values)-prediction(direct)).abs().max())
                if cache_equivalence>1e-5:
                    raise ValueError('cached/direct actor equivalence gate failed')
            for key,value in values.items():
                cache_parts.setdefault(key,[]).append(value.detach())
            if begin==0 or (begin+128)%1024==0 or begin+len(samples)==len(ticks):
                monitor('cache',begin+len(samples),len(ticks))
    cache={key:torch.cat(value) for key,value in cache_parts.items()}
    del cache_parts
    train_count=len(train_ticks)
    val_rows=torch.arange(train_count,train_count+64,device='cuda')
    held_rows=torch.arange(train_count+64,train_count+128,device='cuda')
    startup_rows=torch.arange(train_count+128,train_count+192,device='cuda')
    get=lambda rows:{key:value[rows] for key,value in cache.items()}

    def evaluate(rows):
        batch=get(rows)
        with torch.no_grad():
            pred=prediction(batch)
            encoded=r(batch['state'],pred)
            pd=decode_target(encoded,batch['q'])
            pdgt=decode_target(r(batch['state'],batch['tau']),batch['q'])
            pdtrue=decode_target(batch['pd_label'],batch['q'])
            _,action,clipped=command(pd,batch['q'],offset,scale)
            epe=torch.linalg.vector_norm((pred-batch['tau']).reshape(-1,24,11,3),dim=-1).cpu().numpy()
            delta=(pd-pdgt).cpu().numpy()
            true=(pd-pdtrue).cpu().numpy()
            gttrue=(pdgt-pdtrue).cpu().numpy()
            stats=lambda x:dict(xyz_p95_mm=float(np.percentile(np.linalg.norm(x[:,:3],axis=-1),95)*1000),
                rotation_axis_p95_rad=float(np.percentile(np.abs(x[:,3:6]).max(-1),95)),
                finger_coordinate_p95_rad=float(np.percentile(np.abs(x[:,[6,8,10,12,14,15]]).max(-1),95)))
            row_ticks=ticks[rows.cpu().numpy()]
            strata={}
            for name,mask in (('startup_repeated_H4',row_ticks<3),('real_H4',row_ticks>=3),
                              ('later_than_tick8',row_ticks>8)):
                strata[name]=dict(windows=int(mask.sum()),first8_epe_mm=float(epe[mask,:8].mean()*1000),
                    h1_epe_mm=float(epe[mask,0].mean()*1000),wam_vs_gt_R=stats(delta[mask]))
            return (dict(first8_epe_mm=float(epe[:,:8].mean()*1000),full24_epe_mm=float(epe.mean()*1000),
                h1_epe_mm=float(epe[:,0].mean()*1000),wam_vs_gt_R=stats(delta),wam_vs_true_PD=stats(true),
                gt_R_vs_true_PD=stats(gttrue),clipping_rate=float(clipped.float().mean()),history_strata=strata,
                per_tick={str(t):dict(windows=int((row_ticks==t).sum()),
                    first8_epe_mm=float(epe[row_ticks==t,:8].mean()*1000),
                    wam_vs_gt_R=stats(delta[row_ticks==t]),wam_vs_true_PD=stats(true[row_ticks==t]))
                    for t in np.unique(row_ticks)}),
                dict(prediction=pred.cpu().numpy(),pd=pd.cpu().numpy(),pd_gt=pdgt.cpu().numpy(),
                     pd_true=pdtrue.cpu().numpy(),action=action.cpu().numpy(),
                     teacher_tau=batch['tau'].cpu().numpy(),ticks=row_ticks,envs=envs[rows.cpu().numpy()]))

    initial={}; initial_predictions={}
    for name,rows in (('validation',val_rows),('held_report',held_rows),('startup_validation',startup_rows)):
        initial[name],values=evaluate(rows)
        initial_predictions.update({'initial_'+name+'_'+k:v for k,v in values.items()})
    optimizer=torch.optim.AdamW(parameters,lr=3e-4,weight_decay=.01)
    curve=[]; gradient_checks={}; sampled_startup=sampled_rows=sampled_first16=0
    for step in range(1,manifest['maximum_updates']+1):
        draws=sample_training_rows(rng,train_ticks)
        rows=torch.as_tensor(draws,device='cuda')
        sampled_first16+=int((train_ticks[draws]<16).sum())
        sampled_rows+=len(rows)
        sampled_startup+=int((train_ticks[rows.cpu().numpy()]<3).sum())
        batch=get(rows)
        optimizer.zero_grad(set_to_none=True)
        pred=prediction(batch)
        tau_loss=F.huber_loss(pred/.05,batch['tau']/.05)
        encoded=r(batch['state'],pred)
        pd_loss=F.huber_loss(encoded/r.target_scale,batch['pd_label']/r.target_scale)
        loss=tau_loss+pd_loss
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite head calibration loss')
        loss.backward()
        if step==1:
            gradient_checks={name:float(sum((p.grad.abs().sum() for p in network.parameters() if p.grad is not None),
                                           torch.tensor(0.,device='cuda')))
                             for name,network in (('trajectory',model.trajectory),('hand_query',model.hand_query))}
            if not all(value>0 for value in gradient_checks.values()):
                raise ValueError('both trainable modules must receive gradients')
        if any(p.grad is not None for network in frozen_modules.values() for p in network.parameters()):
            raise ValueError('frozen parameter received a gradient')
        gradient=torch.nn.utils.clip_grad_norm_(parameters,1.)
        if not torch.isfinite(gradient):
            raise FloatingPointError('nonfinite trainable gradient')
        optimizer.step()
        if step==1 or step%25==0 or step==manifest['maximum_updates']:
            row=dict(step=step,tau_loss=float(tau_loss),pd_loss=float(pd_loss),gradient_norm=float(gradient))
            curve.append(row)
            monitor('train',step,manifest['maximum_updates'])
            print(json.dumps(row),flush=True)
    final={}; predictions={}
    for name,rows in (('validation',val_rows),('held_report',held_rows),('startup_validation',startup_rows)):
        final[name],values=evaluate(rows)
        predictions.update({name+'_'+k:v for k,v in values.items()})
    frozen_final={name:network_hash(network) for name,network in frozen_modules.items()}
    if frozen_hash!=frozen_final:
        raise ValueError('frozen scene/goal/world/action/R weights changed')
    if any(sha(path)!=digest for path,digest in evidence.items()):
        raise ValueError('source evidence drift during calibration')
    signal=dict(startup_pd_xyz_p95_le20mm=final['startup_validation']['wam_vs_gt_R']['xyz_p95_mm']<=20.,
        held_real_H4_first8_no_degradation_gt25pct=final['held_report']['history_strata']['real_H4']['first8_epe_mm']
            <=previous_result['final']['held_report']['history_strata']['real_H4']['first8_epe_mm']*1.25)
    pd=final['held_report']['wam_vs_gt_R']
    simulation_screen=dict(xyz=pd['xyz_p95_mm']<=10.,rotation=pd['rotation_axis_p95_rad']<=.05,
        fingers=pd['finger_coordinate_p95_rad']<=.05,clipping=final['held_report']['clipping_rate']<.01)
    result=dict(status='COMPLETED',engineering_only=args.engineering_smoke,updates=manifest['maximum_updates'],
        initial=initial,final=final,adaptation_signal=signal,simulation_screen=simulation_screen,
        cache_direct_max_error_m=cache_equivalence,gradient_checks=gradient_checks,
        frozen_initial_hash=frozen_hash,frozen_final_hash=frozen_final,curve=curve,
        elapsed_s=time.monotonic()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        sampled_train_rows=sampled_rows,sampled_startup_rows=sampled_startup,sampled_first16_rows=sampled_first16,
        conclusion='engineering PASS' if args.engineering_smoke else ('PROMISING' if all(signal.values()) else 'UNCLEAR'))
    if not args.engineering_smoke:
        head={name:{k:v.detach().cpu() for k,v in getattr(model,name).state_dict().items()}
              for name in ('trajectory','hand_query')}
        torch.save(dict(schema=manifest['schema'],head=head,base_checkpoint_sha256=sha(checkpoint),
                        input_sha256=evidence,config=dict(seed=293,draw_seed=294,additional_updates=300,total_updates=600,
                        continuation_head_sha256=sha(previous/'head.pt'))),out/'head.pt')
        if (out/'head.pt').stat().st_size>=10_000_000:
            raise ValueError('head checkpoint exceeds10MB')
        result['head_sha256']=sha(out/'head.pt')
    np.savez_compressed(out/'predictions.npz',**initial_predictions,**predictions)
    write(out/'result.json',result)
    manifest.update(status='COMPLETED',elapsed_s=result['elapsed_s'],result_sha256=sha(out/'result.json'),
                    predictions_sha256=sha(out/'predictions.npz'),frozen_final_hash=frozen_final)
    write(out/'manifest.json',manifest)
    monitor('completed',manifest['maximum_updates'],manifest['maximum_updates'])
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()

"""Robot-domain aggregation from audited autonomous/recovered states.

Pure-WAM prior holdout blocks are explicitly retired into training. Late
query blocks are diagnostics; only a new live rollout measures adaptation.
Actual mixed-policy hand futures are NEVER expert trajectory labels.
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


def admit_aggregation_source(manifest, audit, result, trajectory_hash):
    """This experiment admits the new 192/takeover32 collection only."""
    expected_roles=['teacher','gt_tau_R','wam_R','wam_command_replay_then_expert_tick32']
    local=result['outcomes'][3]
    valid=bool(manifest.get('status')=='COMPLETED'
        and manifest.get('controls')==192 and manifest.get('takeover_tick')==32
        and manifest.get('roles')==expected_roles and manifest.get('training_allowed') is True
        and manifest.get('expert_PD_training_allowed') is True
        and manifest.get('mixed_policy_future_hand_training_allowed') is False
        and audit.get('pass') is True and audit.get('controls')==192
        and audit.get('takeover_tick')==32 and audit.get('measured_states')==193
        and audit.get('native_data_sha256')==trajectory_hash==manifest.get('trajectory_sha256')
        and result.get('status')=='COMPLETED' and result.get('partial_episode') is True
        and result.get('calibration_pass') is None and result.get('recovery_pass') is None
        and result.get('local_grasp_calibration') is True
        and local['maximum_held_frames']>=45 and local['first_stable_tick'] is not None
        and local['first_stable_tick']<=160 and local['clipping_rate']<.01)
    if not valid:
        raise ValueError('audited 192/takeover32 locally recoverable source required')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=Path, required=True, help='completed recovery/native directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--head-dir',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=500)
    parser.add_argument('--seed',type=int,default=295)
    parser.add_argument('--draw-seed',type=int,default=296)
    parser.add_argument('--engineering-smoke', action='store_true')
    args = parser.parse_args()
    if not 1<=args.updates<=1500:
        raise ValueError('bounded fixed update count required')
    out, query_dir = args.output.resolve(), args.queries.resolve()
    if out.exists() or ROOT / 'outputs/trajectory-policy' not in out.parents:
        raise ValueError('fresh task output required')
    if ROOT / 'outputs/trajectory-policy' not in query_dir.parents:
        raise ValueError('task-owned live query source required')
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
    from trajectory_policy.pd_retargeter import load_model, encode_target, decode_target, command
    from trajectory_policy.retargeter_data import features
    from goal_point_wam.data import goal_collate
    from oakink_wm.pointworld_performance import install_fused_hilbert
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    rng = np.random.default_rng(args.draw_seed)
    started = time.monotonic()
    previous=args.head_dir.resolve()
    if ROOT/'outputs/trajectory-policy' not in previous.parents:
        raise ValueError('task-owned parent required')
    pm, pr = [json.loads((previous / name).read_text()) for name in ('manifest.json', 'result.json')]
    qm = json.loads((query_dir / 'manifest.json').read_text())
    qa = json.loads((query_dir / 'root_recovery_audit.json').read_text())
    qr = json.loads((query_dir / 'result.json').read_text())
    local_pass=admit_aggregation_source(qm,qa,qr,sha(query_dir/'trajectory.npz'))
    if pm['status'] != 'COMPLETED' or sha(previous / 'head.pt') != pr['head_sha256']:
        raise ValueError('authenticated fixed startup head required')
    evidence = dict(pm['input_sha256'])
    evidence.update({str(previous / k):sha(previous / k) for k in ('head.pt', 'manifest.json', 'result.json')})
    evidence.update({str(query_dir / k):sha(query_dir / k) for k in
        ('trajectory.npz', 'manifest.json', 'result.json', 'root_recovery_audit.json')})
    evidence.update({str(Path(__file__)):sha(Path(__file__)),
        str(TASK / 'tools/run/calibrate_wam_robot_startup.py'):sha(TASK / 'tools/run/calibrate_wam_robot_startup.py')})
    if sha(query_dir / 'trajectory.npz') != qm['trajectory_sha256']:
        raise ValueError('query trajectory identity differs')
    if any(sha(path) != digest for path, digest in evidence.items()):
        raise ValueError('authenticated warm-start/source drift')
    with np.load(query_dir / 'trajectory.npz', allow_pickle=False) as f:
        live = {k:f[k].copy() for k in ('q','dq','hand','obj','velocity','expert_PD','expert_clipped')}
    source = ROOT / 'outputs/trajectory-policy/competent-pd-teacher-20261011-r2'
    sm = json.loads((source / 'manifest.json').read_text())
    if (sm['status'] != 'COMPLETED' or not sm['training_allowed'] or
        sha(source / 'trajectory.npz') != sm['trajectory_sha256']):
        raise ValueError('qualified nominal source required')
    with np.load(source / 'trajectory.npz', allow_pickle=False) as f:
        data = {k:f[k][:,:28].copy() for k in ('q','dq','hand','obj','velocity','pd_targets')}
    ref_path = ROOT / 'outputs/trajectory-policy/competent-reference-20261011-r1/reference.npz'
    with np.load(ref_path, allow_pickle=False) as f:
        reference = f['obj'].copy()
    canonical_path = ROOT / 'outputs/consequence-evaluator/old-utility-pw-20261009-r2/canonical.npz'
    with np.load(canonical_path, allow_pickle=False) as f:
        canonical = {k:f[k].copy() for k in f.files}
    base = ROOT / 'outputs/goal-point-wam/goal-memory-ddp-20261012-r3/train/final.pt'
    stats = ROOT / 'outputs/cm-pointflow-effect-pretrain/pointworld-main3-sharedstats-20261009/norm_stats.json'
    original = ROOT / 'outputs/trajectory-policy/wam-ref2-shadow-20261012-r2/code/model.py'
    packet = torch.load(base, map_location='cpu', weights_only=False)
    if sha(original) != packet['identity']['model_sha256'] or sha(stats) != packet['identity']['stats_sha256']:
        raise ValueError('base implementation/stats drift')
    spec = importlib.util.spec_from_file_location('frozen_wam_visited_pd', original)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    install_fused_hilbert()
    model = module.GoalPointWAM(json.loads(stats.read_text()), packet['config']).cuda().eval()
    model.load_state_dict(packet['model'], strict=True)
    head = torch.load(previous / 'head.pt', map_location='cpu', weights_only=False)
    if head['base_checkpoint_sha256'] != sha(base) or set(head['head']) != {'trajectory','hand_query'}:
        raise ValueError('head base/module set differs')
    if 'scene_patch' in head:
        raise ValueError('this coverage probe requires the original frozen scene')
    for name, value in head['head'].items():
        getattr(model, name).load_state_dict(value, strict=True)
    del packet, head
    r_path = ROOT / 'outputs/trajectory-policy/competent-pd-fit-20261011-r1/tau.pt'
    r = load_model(torch.load(r_path, map_location='cpu', weights_only=False), 'cuda')
    for net in (model, r):
        for p in net.parameters():
            p.requires_grad_(False)
    for net in (model.trajectory, model.hand_query):
        for p in net.parameters():
            p.requires_grad_(True)
    frozen = {name:getattr(model, name) for name in ('scene','goal','action','world')}
    frozen['R'] = r
    frozen_hashes = {name:network_hash(net) for name,net in frozen.items()}
    if frozen_hashes != pr['frozen_final_hash']:
        raise ValueError('frozen module identities differ')
    native = qm['native_controller']
    offset, scale = [torch.as_tensor(native[k], device='cuda') for k in ('offset','scale')]
    # Fixed nominal pools:16 startup rows and48 later per batch, plus64 live PD.
    nominal = [(0,t,e) for t in range(519) for e in range(16)]
    panel = [0,1,2,4,8,16,32,64,96,128,192,256,320,384,448,518]
    report = [(0,t,e) for t in panel for e in range(24,28)]
    controls=qm['controls']; takeover=qm['takeover_tick']
    lt,lr=np.repeat(np.arange(controls),2),np.tile([2,3],controls)
    unclipped=~live['expert_clipped'].reshape(-1)
    train_mask=unclipped&(((lr==2)&(lt<63))|((lr==3)&(lt>=takeover)&(lt<160)))
    val_mask=unclipped&(((lr==2)&(lt>=73)&(lt<96))|((lr==3)&(lt>=160)&(lt<176)))
    query_rows=[(1,int(t),int(role)) for t,role in zip(lt,lr)]
    train_queries=[row for row,ok in zip(query_rows,train_mask) if ok]
    val_queries=[row for row,ok in zip(query_rows,val_mask) if ok]
    if not train_queries or not val_queries or set(train_queries)&set(val_queries):
        raise ValueError('nonempty disjoint aggregation/diagnostic pools required')
    if args.engineering_smoke:
        nominal = [(0,t,e) for t in (0,16) for e in range(16)]
        train_queries=[r for r in train_queries if r[2]==2][:4]+[r for r in train_queries if r[2]==3][:4]
        val_queries=val_queries[:8]
    rows = nominal + report + train_queries + val_queries
    cache = {}
    order = torch.as_tensor(WAM_TO_NATIVE, device='cuda')
    def predict(batch):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            tau = model.trajectory(batch['scene'],batch['scene_mask'],batch['goal'],batch['goal_mask'],
                                   model.observed_hand_condition(batch)).float()
        return tau[:,:,0].index_select(2,order).reshape(-1,24,33)
    def guard(phase, count, total):
        elapsed = time.monotonic()-started
        if elapsed > 600:
            raise TimeoutError('bounded correction deadline')
        pids = set(map(int,subprocess.check_output(['nvidia-smi','-i',str(args.gpu),
            '--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()))
        if pids-{os.getpid()}:
            raise RuntimeError('foreign GPU process')
        paths = [p for p in (ROOT/'outputs/trajectory-policy').glob('wam-ref2-*') if p.is_dir()]
        paths += [scratch]+[p for p in (ROOT/'tmp').glob('wam-ref2-*') if p.is_dir()]
        size = sum(int(subprocess.check_output(['du','-sb',str(p)],text=True).split()[0]) for p in paths)
        if size >= 100_000_000:
            raise RuntimeError('ref2 aggregate artifact limit')
        print(json.dumps(dict(phase=phase,count=count,total=total,elapsed_s=elapsed,
            eta_s=elapsed/max(count,1)*(total-count),artifact_bytes=size)),flush=True)
    out.mkdir(parents=True)
    manifest = dict(schema='ref2dex.wam-robot-aggregation.v1',status='RUNNING',task='goal-point-wam',
        experiment_id='P-20261012-goal-point-wam-ref2-robot-aggregation',run_id=out.name,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        seed=args.seed,draw_seed=args.draw_seed,physical_gpu=args.gpu,input_sha256=evidence,training_allowed=False,
        maximum_updates=1 if args.engineering_smoke else args.updates,nominal_windows=len(nominal),
        live_training_windows=len(train_queries),live_validation_windows=len(val_queries),
        split='prior approach/scene-thaw holdout retired; fixed late blocks diagnostic only; new actor rollout required',
        loss='nominal Huber tau/.05 + nominal Huber PD/scale + live Huber PD/scale; equal nominal/live batches',
        fixed_final_only=True,aggregation=True,full_recovery_pass=qr.get('recovery_pass'),
        previous_holdout_retired=True,source_local_grasp_pass=local_pass,learning_rate=1e-4,
        warm_head_dir=str(previous),live_sampling='32 pure-WAM + 32 recovered; replacement',
        base_checkpoint_sha256=sha(base),warm_head_sha256=sha(previous/'head.pt'))
    write(out/'manifest.json',manifest)
    equivalent = 0.
    with torch.no_grad():
        for begin in range(0,len(rows),128):
            samples, states, taus, labels, qs, labelled = [],[],[],[],[],[]
            for i,(kind,tick,env) in enumerate(rows[begin:begin+128]):
                d = data if kind == 0 else live
                past = np.maximum(tick+np.arange(-3,1),0)
                samples.append(robot_goal_sample(d['obj'][past,env],d['hand'][past,env],
                    reference[min(tick+24,542)],canonical,sample_id=begin+i))
                future = d['hand'][tick+np.arange(1,25),env] if kind == 0 else np.broadcast_to(d['hand'][tick,env],(24,11,3))
                state,tau = features(*(d[k][tick,env] for k in ('q','dq','hand','obj','velocity')),future)
                states.append(state); taus.append(tau); qs.append(d['q'][tick,env]); labelled.append(kind==0)
                label = d['pd_targets'][tick,env] if kind==0 else d['expert_PD'][tick,env-2]
                labels.append(encode_target(label,d['q'][tick,env]))
            batch = {k:v.cuda() for k,v in goal_collate(samples).items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):
                scene=model.scene(batch); goal, mask=model.goal(batch['start_object_points'],batch['goal_object_points'],batch['object_goal_valid'])
            values=dict(scene=scene['scene_tokens'].to(torch.bfloat16),scene_mask=scene['scene_mask'],
                goal=goal.to(torch.bfloat16),goal_mask=mask,start_hand=batch['start_hand'],
                start_hand_history=batch['start_hand_history'],start_hand_valid=batch['start_hand_valid'],
                state=torch.as_tensor(np.stack(states),device='cuda'),tau=torch.as_tensor(np.stack(taus),device='cuda'),
                pd_label=torch.as_tensor(np.stack(labels),device='cuda'),q=torch.as_tensor(np.stack(qs),device='cuda'),
                tau_label_valid=torch.as_tensor(labelled,device='cuda'))
            if begin==0:
                equivalent=float((predict(values)-predict(dict(values,scene=scene['scene_tokens'],goal=goal))).abs().max())
                if equivalent>1e-5:
                    raise ValueError('cache/direct actor mismatch')
            for k,v in values.items():
                cache.setdefault(k,[]).append(v.detach())
            if begin==0 or (begin//128)%16==0:
                guard('cache',min(begin+128,len(rows)),len(rows))
    cache={k:torch.cat(v) for k,v in cache.items()}
    get=lambda ids:{k:v[ids] for k,v in cache.items()}
    n=len(nominal); report_ids=torch.arange(n,n+len(report),device='cuda')
    qt=n+len(report); qv=qt+len(train_queries)
    query_train_ids=np.arange(qt,qv); query_val_ids=torch.arange(qv,len(rows),device='cuda')
    nt=np.asarray([t for _,t,_ in nominal]); startup=np.flatnonzero(nt<16); later=np.flatnonzero(nt>=16)
    def evaluate(ids):
        b=get(ids)
        with torch.no_grad():
            tau=predict(b); encoded=r(b['state'],tau); pd=decode_target(encoded,b['q'])
            target=decode_target(b['pd_label'],b['q']); _, action, clipped=command(pd,b['q'],offset,scale)
            delta=(pd-target).cpu().numpy(); valid=b['tau_label_valid'].cpu().numpy()
            epe=torch.linalg.vector_norm((tau-b['tau']).reshape(-1,24,11,3),dim=-1).cpu().numpy()
        stats=dict(xyz_p95_mm=float(np.percentile(np.linalg.norm(delta[:,:3],axis=-1),95)*1000),
            rotation_p95_rad=float(np.percentile(np.abs(delta[:,3:6]).max(-1),95)),
            finger_p95_rad=float(np.percentile(np.abs(delta[:,[6,8,10,12,14,15]]).max(-1),95)),
            clipping_rate=float(clipped.float().mean()))
        if valid.any():
            stats['first8_epe_mm']=float(epe[valid,:8].mean()*1000)
        else:
            selected=np.asarray(rows)[ids.cpu().numpy()]
            stats['by_role']={}
            clipped_np=clipped.cpu().numpy()
            for role,name in ((2,'pure_wam'),(3,'recovered')):
                mask=selected[:,2]==role
                part=delta[mask]
                stats['by_role'][name]=dict(count=int(mask.sum()),
                    xyz_p95_mm=float(np.percentile(np.linalg.norm(part[:,:3],axis=-1),95)*1000) if len(part) else None,
                    rotation_p95_rad=float(np.percentile(np.abs(part[:,3:6]).max(-1),95)) if len(part) else None,
                    finger_p95_rad=float(np.percentile(np.abs(part[:,[6,8,10,12,14,15]]).max(-1),95)) if len(part) else None,
                    clipping_rate=float(clipped_np[mask].mean()) if len(part) else None)
        return stats,dict(tau=tau.cpu().numpy(),pd=pd.cpu().numpy(),pd_true=target.cpu().numpy(),
            action=action.cpu().numpy(),tau_label_valid=valid,row_ids=ids.cpu().numpy())
    initial={}; raw={}
    query_training_ids=torch.as_tensor(query_train_ids,device='cuda')
    for name,ids in (('nominal_report',report_ids),('query_training',query_training_ids),('query_validation',query_val_ids)):
        initial[name],values=evaluate(ids); raw.update({'initial_'+name+'_'+k:v for k,v in values.items()})
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
    pure=np.asarray([i for i in query_train_ids if rows[i][2]==2])
    recovered=np.asarray([i for i in query_train_ids if rows[i][2]==3])
    if not len(pure) or not len(recovered):
        raise ValueError('both live distributions required')
    curve=[]; gradients={}
    for step in range(1,manifest['maximum_updates']+1):
        nr=np.r_[rng.choice(startup,16),rng.choice(later,48)]; qr=np.r_[rng.choice(pure,32),rng.choice(recovered,32)]
        nb,qb=get(torch.as_tensor(nr,device='cuda')),get(torch.as_tensor(qr,device='cuda'))
        if not nb['tau_label_valid'].all() or qb['tau_label_valid'].any():
            raise ValueError('live rows cannot have tau labels')
        optimizer.zero_grad(set_to_none=True)
        pred=predict(nb); nominal_tau=F.huber_loss(pred/.05,nb['tau']/.05)
        nominal_pd=F.huber_loss(r(nb['state'],pred)/r.target_scale,nb['pd_label']/r.target_scale)
        live_pd=F.huber_loss(r(qb['state'],predict(qb))/r.target_scale,qb['pd_label']/r.target_scale)
        loss=nominal_tau+nominal_pd+live_pd
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite correction loss')
        if step==1:
            live_grads=torch.autograd.grad(live_pd,params,retain_graph=True,allow_unused=True)
            live_gradient=float(sum((g.abs().sum() for g in live_grads if g is not None),torch.tensor(0.,device='cuda')))
            if not np.isfinite(live_gradient) or live_gradient<=0:
                raise ValueError('live expert PD loss does not reach actor through frozen R')
        loss.backward()
        if step==1:
            gradients={name:float(sum((p.grad.abs().sum() for p in net.parameters() if p.grad is not None),torch.tensor(0.,device='cuda')))
                for name,net in (('trajectory',model.trajectory),('hand_query',model.hand_query))}
            if not all(v>0 for v in gradients.values()):
                raise ValueError('head/hand-query gradient missing')
        if any(p.grad is not None for net in frozen.values() for p in net.parameters()):
            raise ValueError('frozen gradient')
        norm=torch.nn.utils.clip_grad_norm_(params,1.)
        if not torch.isfinite(norm):
            raise FloatingPointError('nonfinite gradient')
        optimizer.step()
        if step==1 or step%25==0:
            curve.append(dict(step=step,tau=float(nominal_tau),nominal_pd=float(nominal_pd),live_pd=float(live_pd),gradient=float(norm)))
            guard('train',step,manifest['maximum_updates'])
    final={}
    for name,ids in (('nominal_report',report_ids),('query_training',query_training_ids),('query_validation',query_val_ids)):
        final[name],values=evaluate(ids); raw.update({'final_'+name+'_'+k:v for k,v in values.items()})
    if frozen_hashes!={name:network_hash(net) for name,net in frozen.items()}:
        raise ValueError('frozen weights changed')
    if any(sha(path)!=digest for path,digest in evidence.items()):
        raise ValueError('fit input drift')
    pure_final=final['query_validation']['by_role']['pure_wam']
    pure_initial=initial['query_validation']['by_role']['pure_wam']
    signal=dict(nominal_PD_retained=final['nominal_report']['xyz_p95_mm']<=38.33471266552805,
        query_PD_halved=bool(pure_final['count']>=8 and pure_final['xyz_p95_mm']<=pure_initial['xyz_p95_mm']*.5),
        nominal_first8_no_degradation_gt25pct=final['nominal_report']['first8_epe_mm']<=21.521584130823612)
    result=dict(status='COMPLETED',initial=initial,final=final,adaptation_signal=signal,curve=curve,
        frozen_initial_hash=frozen_hashes,frozen_final_hash=frozen_hashes,gradient_checks=gradients,
        cache_direct_max_error_m=equivalent,live_PD_actor_gradient_sum=live_gradient,elapsed_s=time.monotonic()-started,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,updates=manifest['maximum_updates'],
        conclusion='engineering PASS' if args.engineering_smoke else ('PROMISING' if all(signal.values()) else 'UNCLEAR'))
    if not args.engineering_smoke:
        saved={name:{k:v.detach().cpu() for k,v in getattr(model,name).state_dict().items()} for name in ('trajectory','hand_query')}
        torch.save(dict(schema=manifest['schema'],head=saved,base_checkpoint_sha256=sha(base),
            input_sha256=evidence,config=dict(seed=args.seed,draw_seed=args.draw_seed,additional_updates=args.updates)),out/'head.pt')
        if (out/'head.pt').stat().st_size>=10_000_000:
            raise ValueError('head too large')
        result['head_sha256']=sha(out/'head.pt')
    np.savez_compressed(out/'predictions.npz',**raw,rows=np.asarray(rows))
    write(out/'result.json',result)
    manifest.update(status='COMPLETED',result_sha256=sha(out/'result.json'),predictions_sha256=sha(out/'predictions.npz'))
    write(out/'manifest.json',manifest)
    guard('completed',1,1)


if __name__=='__main__':
    main()

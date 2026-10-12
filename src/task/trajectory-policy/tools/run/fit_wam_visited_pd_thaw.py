"""Optional dec0 thaw: live expert PD plus nominal trajectory/PD replay.

Static derivative of fit_wam_visited_pd.py, which is unchanged. Only final
PTv3 decoder + trajectory/hand_query train. Raw-input RAM cache only, full
scene forward per microbatch; frozen R preserves gradients into tau.
World weights freeze but scene changes can change world outputs.
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


def query_partition(ticks, roles, approach_only=False):
    """Predeclared temporal blocks; no duplicated common prefix in both sets."""
    ticks, roles = np.asarray(ticks), np.asarray(roles)
    if ticks.shape != roles.shape:
        raise ValueError('matching tick and role arrays required')
    if approach_only:
        admitted=((roles==2)&(ticks<48))|((roles==3)&(ticks>=16)&(ticks<128))
        val=admitted&(((roles==2)&(ticks>=32))|((roles==3)&(ticks>=64)&(ticks<80)))
        return admitted&~val,val
    admitted = ((roles == 2) & (ticks < 64)) | ((roles == 3) & (ticks >= 16) & (ticks < 519))
    val = admitted & (((roles == 2) & (ticks >= 48)) | ((roles == 3) &
        (((ticks >= 64) & (ticks < 80)) | ((ticks >= 256) & (ticks < 272)) |
         ((ticks >= 448) & (ticks < 464)))))
    return admitted & ~val, val


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=Path, required=True, help='completed recovery/native directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--head-dir', type=Path, required=True, help='exact admitted approach-only final head')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--draw-seed', type=int, required=True)
    parser.add_argument('--experiment-id', required=True)
    parser.add_argument('--closure-reserve-seconds', type=float, default=150.)
    parser.add_argument('--engineering-smoke', action='store_true')
    args = parser.parse_args()
    args.approach_only = True
    if not 0 <= args.closure_reserve_seconds < 600:
        raise ValueError('closure reserve must fit shared 600-second budget')
    out, query_dir = args.output.resolve(), args.queries.resolve()
    if (out.exists() or ROOT / 'outputs/trajectory-policy' not in out.parents
            or not out.name.startswith('wam-ref2-')):
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
    from trajectory_policy.wam_scene_patch import (extract_scene_patch, scene_remainder_hash,
        SCENE_PATCH_SCHEMA, SCENE_PATCH_PREFIX)
    from goal_point_wam.data import goal_collate
    from oakink_wm.pointworld_performance import install_fused_hilbert
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    rng = np.random.default_rng(args.draw_seed)
    started = time.monotonic()
    previous = args.head_dir.resolve()
    if ROOT / 'outputs/trajectory-policy' not in previous.parents:
        raise ValueError('task-owned exact parent head required')
    pm, pr = [json.loads((previous / name).read_text()) for name in ('manifest.json', 'result.json')]
    qm = json.loads((query_dir / 'manifest.json').read_text())
    qa = json.loads((query_dir / 'root_recovery_audit.json').read_text())
    qr = json.loads((query_dir / 'result.json').read_text())
    local=qr['outcomes'][3]
    local_pass=bool(qr['calibration_pass'] and local['maximum_held_frames']>=45 and
        local['first_stable_tick'] is not None and local['first_stable_tick']<=128 and local['clipping_rate']<.01)
    admission=local_pass if args.approach_only else qr.get('recovery_pass')
    if (qm['status'] != 'COMPLETED' or qm.get('training_allowed') is not True or
        not qa.get('pass') or not admission):
        raise ValueError('independently audited expert recovery and live-PD admission required')
    if (pm['status'] != 'COMPLETED' or not pm.get('approach_only')
            or sha(previous / 'head.pt') != pr['head_sha256']):
        raise ValueError('authenticated fixed approach-only head required')
    evidence = dict(pm['input_sha256'])
    evidence.update({str(previous / k):sha(previous / k) for k in ('head.pt', 'manifest.json', 'result.json')})
    evidence.update({str(query_dir / k):sha(query_dir / k) for k in
        ('trajectory.npz', 'manifest.json', 'result.json', 'root_recovery_audit.json')})
    evidence.update({str(Path(__file__)):sha(Path(__file__)),
        str(TASK / 'tools/run/calibrate_wam_robot_startup.py'):sha(TASK / 'tools/run/calibrate_wam_robot_startup.py'),
        str(TASK / 'src/trajectory_policy/wam_scene_patch.py'):sha(TASK / 'src/trajectory_policy/wam_scene_patch.py')})
    if sha(query_dir / 'trajectory.npz') != qm['trajectory_sha256']:
        raise ValueError('query trajectory identity differs')
    if any(sha(path) != digest for path, digest in evidence.items()):
        raise ValueError('authenticated warm-start/source drift')
    with np.load(query_dir / 'trajectory.npz', allow_pickle=False) as f:
        live = {k:f[k].copy() for k in ('q','dq','hand','obj','velocity','expert_PD')}
    source = ROOT / 'outputs/trajectory-policy/competent-pd-teacher-20261011-r2'
    sm = json.loads((source / 'manifest.json').read_text())
    sr = json.loads((source / 'result.json').read_text())
    sa = json.loads((source / 'audit.json').read_text())
    if (sm['status'] != 'COMPLETED' or sm.get('training_allowed') is not True
            or sr.get('teacher_readiness') is not True or not sa.get('pass')
            or sm.get('dt') != 1/30 or
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
    if (head['base_checkpoint_sha256'] != sha(base) or set(head['head']) != {'trajectory','hand_query'}
            or 'scene_patch' in head):
        raise ValueError('head base/module set differs')
    for name, value in head['head'].items():
        getattr(model, name).load_state_dict(value, strict=True)
    del packet, head
    r_path = ROOT / 'outputs/trajectory-policy/competent-pd-fit-20261011-r1/tau.pt'
    r = load_model(torch.load(r_path, map_location='cpu', weights_only=False), 'cuda')
    for net in (model, r):
        for p in net.parameters():
            p.requires_grad_(False)
    dec0 = model.scene.backbone.dec.dec0
    for net in (model.trajectory, model.hand_query, dec0):
        for p in net.parameters():
            p.requires_grad_(True)
    frozen = {name:getattr(model, name) for name in ('goal','action','world')}
    frozen['R'] = r
    frozen_hashes = {name:network_hash(net) for name,net in frozen.items()}
    if dict(frozen_hashes, scene=network_hash(model.scene)) != pr['frozen_final_hash']:
        raise ValueError('frozen parent identities differ')
    scene_frozen_hash = scene_remainder_hash(model.scene)
    initial_scene_hash = network_hash(model.scene)
    trainable_names = [name for name,p in model.named_parameters() if p.requires_grad]
    if not trainable_names or not all(name.startswith(('trajectory.', 'hand_query.',
            'scene.' + SCENE_PATCH_PREFIX)) for name in trainable_names):
        raise ValueError('only dec0 and original head may train')
    native = qm['native_controller']
    offset, scale = [torch.as_tensor(native[k], device='cuda') for k in ('offset','scale')]
    # Fixed nominal pools:16 startup rows and48 later per batch, plus64 live PD.
    nominal = [(0,t,e) for t in range(519) for e in range(16)]
    panel = [0,1,2,4,8,16,32,64,96,128,192,256,320,384,448,518]
    report = [(0,t,e) for t in panel for e in range(24,28)]
    lt, lr = np.repeat(np.arange(519),2), np.tile([2,3],519)
    train_mask, val_mask = query_partition(lt, lr,args.approach_only)
    query_rows = [(1,int(t),int(role)) for t,role in zip(lt,lr)]
    train_queries = [row for row,ok in zip(query_rows,train_mask) if ok]
    val_queries = [row for row,ok in zip(query_rows,val_mask) if ok]
    if args.engineering_smoke:
        nominal = [(0,t,e) for t in (0,16) for e in range(16)]
        train_queries, val_queries = train_queries[:8], val_queries[:8]
    rows = nominal + report + train_queries + val_queries
    input_cache = {}  # Raw samples/labels only, never scene tokens or sparse Point objects.
    order = torch.as_tensor(WAM_TO_NATIVE, device='cuda')

    def get(ids):
        samples, labels = [], []
        for index in ids:
            index = int(index)
            if index not in input_cache:
                kind,tick,env = rows[index]
                d = data if kind == 0 else live
                past = np.maximum(tick+np.arange(-3,1),0)
                sample = robot_goal_sample(d['obj'][past,env],d['hand'][past,env],
                    reference[min(tick+24,542)],canonical,sample_id=index)
                # Current placeholder is for features only; never a live tau label.
                future = (d['hand'][tick+np.arange(1,25),env] if kind == 0 else
                    np.broadcast_to(d['hand'][tick,env],(24,11,3)))
                state,tau = features(*(d[k][tick,env] for k in ('q','dq','hand','obj','velocity')),future)
                q = d['q'][tick,env]
                pd = d['pd_targets'][tick,env] if kind == 0 else d['expert_PD'][tick,env-2]
                input_cache[index] = (sample,dict(state=state,tau=tau,pd_label=encode_target(pd,q),
                    q=q,tau_label_valid=np.asarray(kind == 0)))
            sample,label = input_cache[index]
            samples.append(sample); labels.append(label)
        batch = {k:v.cuda() for k,v in goal_collate(samples).items()}
        batch.update({k:torch.as_tensor(np.stack([v[k] for v in labels]),device='cuda') for k in labels[0]})
        return batch

    def predict(batch, return_scene=False):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            # Full ORIGINAL scene forward; no no_grad/detach around dec0.
            scene = model.scene(batch)
            with torch.no_grad():
                goal,mask = model.goal(batch['start_object_points'],batch['goal_object_points'],batch['object_goal_valid'])
            tau = model.trajectory(scene['scene_tokens'],scene['scene_mask'],goal,mask,
                                   model.observed_hand_condition(batch)).float()
        native_tau = tau[:,:,0].index_select(2,order).reshape(-1,24,33)
        return (native_tau,scene) if return_scene else native_tau

    def guard(phase, count, total):
        elapsed = time.monotonic()-started
        if elapsed > 600:
            if out.exists():
                write(out/'result.json',dict(status='ENGINEERING_BUDGET_STOP',phase=phase,
                    elapsed_s=elapsed,conclusion='engineering cost only; no scientific negative'))
                manifest.update(status='ENGINEERING_BUDGET_STOP')
                write(out/'manifest.json',manifest)
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
    manifest = dict(schema='ref2dex.wam-visited-pd-dec0.v1',status='RUNNING',task='goal-point-wam',
        experiment_id=args.experiment_id,run_id=out.name,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        seed=args.seed,draw_seed=args.draw_seed,physical_gpu=args.gpu,input_sha256=evidence,
        training_allowed=False,maximum_updates=1 if args.engineering_smoke else 100,
        nominal_windows=len(nominal),live_training_windows=len(train_queries),
        live_validation_windows=len(val_queries),microbatch=16,effective_batch=128,
        gradient_accumulation='4 nominal + 4 live microbatches; each loss divided by4',
        cache_scope='raw inputs/labels in RAM only; full original scene forward every microbatch',
        split='temporal blocks in one episode; correlated diagnostic, no independent generalization claim',
        loss='nominal Huber tau/.05 + nominal Huber PD/scale + live Huber PD/scale',
        learning_rates=dict(dec0=2e-5,head=1e-4),fixed_final_only=True,approach_only=True,
        full_recovery_pass=qr['recovery_pass'],base_checkpoint_sha256=sha(base),
        warm_head_sha256=sha(previous/'head.pt'),warm_head_dir=str(previous),
        trainable_parameters=trainable_names,scene_patch_schema=SCENE_PATCH_SCHEMA,
        scene_patch_prefix=SCENE_PATCH_PREFIX,model_mode='eval; selected weights have gradients',
        closure_reserve_seconds=args.closure_reserve_seconds,
        limitations=['frozen world weights do not protect world outputs when scene changes',
            'no human/video/world validation; nominal robot replay is a local screen'])
    write(out/'manifest.json',manifest)
    guard('admitted',1,1)
    n=len(nominal); report_ids=torch.arange(n,n+len(report),device='cuda')
    qt=n+len(report); qv=qt+len(train_queries)
    query_train_ids=np.arange(qt,qv); query_val_ids=np.arange(qv,len(rows))
    nt=np.asarray([t for _,t,_ in nominal]); startup=np.flatnonzero(nt<16); later=np.flatnonzero(nt>=16)
    def evaluate(ids):
        arrays = dict(tau=[],pd=[],pd_true=[],action=[],clipped=[],tau_label_valid=[])
        with torch.no_grad():
            for begin in range(0,len(ids),16):
                b=get(ids[begin:begin+16]); tau=predict(b)
                pd=decode_target(r(b['state'],tau),b['q']); target=decode_target(b['pd_label'],b['q'])
                _,action,clipped=command(pd,b['q'],offset,scale)
                for key,value in dict(tau=tau,pd=pd,pd_true=target,action=action,clipped=clipped,
                        tau_label_valid=b['tau_label_valid']).items():
                    arrays[key].append(value.cpu().numpy())
        arrays={k:np.concatenate(v) for k,v in arrays.items()}
        delta=arrays['pd']-arrays['pd_true']
        stats=dict(xyz_p95_mm=float(np.percentile(np.linalg.norm(delta[:,:3],axis=-1),95)*1000),
            rotation_p95_rad=float(np.percentile(np.abs(delta[:,3:6]).max(-1),95)),
            finger_p95_rad=float(np.percentile(np.abs(delta[:,[6,8,10,12,14,15]]).max(-1),95)),
            clipping_rate=float(np.mean(arrays['clipped'])))
        valid=arrays['tau_label_valid']
        if valid.any():
            targets=np.stack([input_cache[int(i)][1]['tau'] for i in ids])
            epe=np.linalg.norm((arrays['tau']-targets).reshape(-1,24,11,3),axis=-1)
            stats['first8_epe_mm']=float(epe[valid,:8].mean()*1000)
        arrays['row_ids']=np.asarray(ids)
        return stats,arrays

    def scene_diagnostic(ids):
        tokens,translations,rotations=[],[],[]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for begin in range(0,len(ids),16):
                b=get(ids[begin:begin+16]); _,scene=predict(b,return_scene=True)
                action,mask=model.action(b['action'].reshape(-1,24,2,11,9),b['action_valid'].reshape(-1,24,2,11))
                world=model.world(scene['scene_tokens'],scene['scene_mask'],scene['object_tokens'],
                    scene['object_mask'],action,mask)
                tokens.append(scene['scene_tokens'].float().cpu())
                translations.append(world['translation'].float().cpu().numpy())
                rotations.append(world['rotation'].float().cpu().numpy())
        return torch.cat(tokens),np.concatenate(translations),np.concatenate(rotations)

    initial={}; raw={}
    report_ids=np.asarray(report_ids.cpu())
    query_training_ids=query_train_ids
    initial_features,initial_world_t,initial_world_r=scene_diagnostic(report_ids[:16])
    for name,ids in (('nominal_report',report_ids),('query_training',query_training_ids),('query_validation',query_val_ids)):
        initial[name],values=evaluate(ids); raw.update({'initial_'+name+'_'+k:v for k,v in values.items()})
    params=[p for p in model.parameters() if p.requires_grad]
    head_params=[p for net in (model.trajectory,model.hand_query) for p in net.parameters()]
    optimizer=torch.optim.AdamW([dict(params=head_params,lr=1e-4),dict(params=list(dec0.parameters()),lr=2e-5)],weight_decay=.01)
    curve=[]; gradients={}; live_gradient={}; train_started=time.monotonic()
    for step in range(1,manifest['maximum_updates']+1):
        nr=np.r_[rng.choice(startup,16),rng.choice(later,48)]; qr=rng.choice(query_train_ids,64)
        optimizer.zero_grad(set_to_none=True); sums=np.zeros(3)
        for micro in range(4):
            nb,qb=get(nr[micro*16:(micro+1)*16]),get(qr[micro*16:(micro+1)*16])
            if not nb['tau_label_valid'].all() or qb['tau_label_valid'].any():
                raise ValueError('live rows cannot have tau labels')
            pred=predict(nb)
            nominal_tau=F.huber_loss(pred/.05,nb['tau']/.05)
            nominal_pd=F.huber_loss(r(nb['state'],pred)/r.target_scale,nb['pd_label']/r.target_scale)
            live_pd=F.huber_loss(r(qb['state'],predict(qb))/r.target_scale,qb['pd_label']/r.target_scale)
            loss=(nominal_tau+nominal_pd+live_pd)/4
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite thaw loss')
            if step==1 and micro==0:
                live_grads=torch.autograd.grad(live_pd,params,retain_graph=True,allow_unused=True)
                for name,net in (('dec0',dec0),('trajectory',model.trajectory),('hand_query',model.hand_query)):
                    net_ids={id(p) for p in net.parameters()}
                    live_gradient[name]=sum(float(g.abs().sum()) for p,g in zip(params,live_grads)
                        if id(p) in net_ids and g is not None)
                if not all(np.isfinite(v) and v>0 for v in live_gradient.values()):
                    raise ValueError('live expert PD must reach dec0/head through frozen R')
            loss.backward()
            sums+=np.asarray([float(nominal_tau),float(nominal_pd),float(live_pd)])/4
        if step==1:
            gradients={name:sum(float(p.grad.abs().sum()) for p in net.parameters() if p.grad is not None)
                for name,net in (('dec0',dec0),('trajectory',model.trajectory),('hand_query',model.hand_query))}
        if (any(p.grad is not None for net in frozen.values() for p in net.parameters()) or
            any(p.grad is not None for name,p in model.scene.named_parameters() if not name.startswith(SCENE_PATCH_PREFIX))):
            raise ValueError('frozen gradient')
        norm=torch.nn.utils.clip_grad_norm_(params,1.)
        if not torch.isfinite(norm):
            raise FloatingPointError('nonfinite gradient')
        optimizer.step()
        if step in (1,2) or step%10==0:
            curve.append(dict(step=step,tau=float(sums[0]),nominal_pd=float(sums[1]),live_pd=float(sums[2]),gradient=float(norm)))
            guard('train',step,manifest['maximum_updates'])
            projected=(time.monotonic()-started)+(time.monotonic()-train_started)/step*(manifest['maximum_updates']-step)
            if step>=2 and not args.engineering_smoke and projected+args.closure_reserve_seconds>600:
                write(out/'result.json',dict(status='ENGINEERING_BUDGET_STOP',updates=step,
                    projected_fit_elapsed_s=projected,conclusion='engineering cost only; no scientific negative'))
                manifest.update(status='ENGINEERING_BUDGET_STOP')
                write(out/'manifest.json',manifest)
                return
    final={}
    for name,ids in (('nominal_report',report_ids),('query_training',query_training_ids),('query_validation',query_val_ids)):
        final[name],values=evaluate(ids); raw.update({'final_'+name+'_'+k:v for k,v in values.items()})
    if (frozen_hashes!={name:network_hash(net) for name,net in frozen.items()}
            or scene_remainder_hash(model.scene)!=scene_frozen_hash):
        raise ValueError('frozen weights changed')
    if any(sha(path)!=digest for path,digest in evidence.items()):
        raise ValueError('fit input drift')
    baseline_path=ROOT/'outputs/trajectory-policy/wam-ref2-head-startup-20261012-r1/result.json'
    baseline=json.loads(baseline_path.read_text())['final']['held_report']
    if sha(baseline_path)!=evidence[str(baseline_path)]:
        raise ValueError('original startup retention baseline differs')
    retention_limits=dict(first8_epe_mm=baseline['first8_epe_mm']*1.25,
        nominal_PD_xyz_p95_mm=baseline['wam_vs_true_PD']['xyz_p95_mm']*1.5)
    signal=dict(query_PD_halved=final['query_validation']['xyz_p95_mm']<=initial['query_validation']['xyz_p95_mm']*.5,
        nominal_first8_no_degradation_gt25pct=final['nominal_report']['first8_epe_mm']<=retention_limits['first8_epe_mm'],
        nominal_PD_retained=final['nominal_report']['xyz_p95_mm']<=retention_limits['nominal_PD_xyz_p95_mm'])
    result=dict(status='COMPLETED',initial=initial,final=final,adaptation_signal=signal,curve=curve,
        frozen_initial_hash=frozen_hashes,frozen_final_hash=frozen_hashes,gradient_checks=gradients,
        live_PD_actor_gradient_sum=live_gradient,elapsed_s=time.monotonic()-started,
        frozen_scene_remainder_initial_hash=scene_frozen_hash,frozen_scene_remainder_final_hash=scene_remainder_hash(model.scene),
        scene_initial_hash=initial_scene_hash,scene_final_hash=network_hash(model.scene),
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,updates=manifest['maximum_updates'],
        retention_limits=retention_limits,
        conclusion='engineering PASS' if args.engineering_smoke else ('PROMISING' if all(signal.values()) else 'UNCLEAR'))
    final_features,final_world_t,final_world_r=scene_diagnostic(report_ids[:16])
    result['world_feature_drift']=dict(scope='16 nominal robot inputs, zero/invalid executed-action context',
        scene_token_rms=float((final_features-initial_features).square().mean().sqrt()),
        translation_max_abs=float(np.max(np.abs(final_world_t-initial_world_t))),
        rotation_matrix_max_abs=float(np.max(np.abs(final_world_r-initial_world_r))),
        validation='descriptive only; human/video/world quality not validated')
    raw.update(initial_world_translation=initial_world_t,final_world_translation=final_world_t,
        initial_world_rotation=initial_world_r,final_world_rotation=final_world_r)
    # Diagnostics and final serialization count toward the same wall-clock bound.
    guard('finalized',1,1)
    if any(not torch.isfinite(p).all() for p in params):
        raise FloatingPointError('nonfinite final trainable weights')
    if any(not np.isfinite(value).all() for value in raw.values()):
        raise FloatingPointError('nonfinite final prediction or diagnostic')
    if not args.engineering_smoke:
        saved={name:{k:v.detach().cpu() for k,v in getattr(model,name).state_dict().items()} for name in ('trajectory','hand_query')}
        torch.save(dict(schema=manifest['schema'],head=saved,base_checkpoint_sha256=sha(base),
            scene_patch=extract_scene_patch(model.scene),scene_patch_schema=SCENE_PATCH_SCHEMA,
            parent_head_sha256=sha(previous/'head.pt'),input_sha256=evidence,
            config=dict(seed=args.seed,draw_seed=args.draw_seed,additional_updates=100)),out/'head.pt')
        if (out/'head.pt').stat().st_size>=10_000_000:
            raise ValueError('head too large')
        result['head_sha256']=sha(out/'head.pt')
    np.savez_compressed(out/'predictions.npz',**raw,rows=np.asarray(rows))
    result['elapsed_s']=time.monotonic()-started
    write(out/'result.json',result)
    manifest.update(status='COMPLETED',result_sha256=sha(out/'result.json'),predictions_sha256=sha(out/'predictions.npz'))
    write(out/'manifest.json',manifest)
    guard('completed',1,1)


if __name__=='__main__':
    main()

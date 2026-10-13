"""Matched fixed-budget planned-wrist conditioning Probe; no physics promotion."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(TASK/'src'), str(ROOT/'src/task/trajectory-policy/src'),
               str(ROOT/'src/task/cm-pointflow-effect-pretrain/src'), str(TASK/'tools/run')]
from train_three_domain import atomic_json, gpu_snapshot
from goal_point_wam.control_context import ControlContextWAM
from goal_point_wam.data import GoalWindows
from oakink_wm.multisource import MixedWindows
from oakink_wm.pointworld_performance import install_fused_hilbert
from trajectory_policy.wam_control_data import ControlData, control_collate
from trajectory_policy.wam_joint import policy_patch_admission, load_policy_patch
from trajectory_policy.wam_effect_admission import effect_admission, chunk_admission
from trajectory_policy.chunk_retargeter import load_chunk
from trajectory_policy.chunk_objective import control_aware_loss, independent_PD_scale
from trajectory_policy.retargeter_data import ACTIVE
from trajectory_policy.wam_shadow import WAM_TO_NATIVE

from goal_point_wam.reference_wrist import wrist_context, ReferenceWristAdapter, wrist_actor
from trajectory_policy.wam_control_admission import control_admission

SCHEMA = 'ref2dex.reference-wrist-probe.v1'
FIELDS = ('start_hand', 'trajectory_target', 'trajectory_valid', 'action', 'action_valid',
          'points', 'effect', 'object_valid', 'radius', 'has_pd', 'native_state',
          'native_action', 'proprio', 'robot_goal', 'robot_mask', 'source_id', 'wrist_goal', 'wrist_shift16')


def concatenate(parts):
    if isinstance(parts[0], dict):
        return {k: concatenate([p[k] for p in parts]) for k in parts[0]}
    if isinstance(parts[0], tuple):
        return tuple(concatenate([p[i] for p in parts]) for i in range(len(parts[0])))
    if parts[0] is None:
        return None
    shape = [max(p.shape[i] for p in parts) for i in range(1, parts[0].ndim)]
    padded = []
    for p in parts:
        pad = [x for i in reversed(range(len(shape))) for x in (0, shape[i]-p.shape[i+1])]
        padded.append(F.pad(p, pad) if any(pad) else p)
    return torch.cat(padded)


def subset(value, ids):
    if isinstance(value, dict):
        return {k: subset(v, ids) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(subset(v, ids) for v in value)
    return None if value is None else value[ids]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=2)
    p.add_argument('--updates', type=int, default=1000)
    p.add_argument('--deadline-seconds', type=int, default=1200)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    out = a.output.resolve()
    if out.exists() or ROOT/'outputs/goal-point-wam' not in out.parents:
        raise ValueError('fresh owned output required')
    if not 1 <= a.updates <= 1000 or not 1 <= a.deadline_seconds <= 1200:
        raise ValueError('bounded ref2_8 update/time budget required')
    if subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True).strip():
        raise RuntimeError('GPU occupied')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    scratch = ROOT/'tmp'/out.name
    scratch.mkdir(parents=True, exist_ok=False)
    os.environ.update(TMPDIR=str(scratch), TRITON_CACHE_DIR=str(scratch/'triton'))
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.manual_seed(295)
    torch.backends.cuda.matmul.allow_tf32 = True
    install_fused_hilbert()
    started = time.monotonic()
    owner = ROOT/'outputs/trajectory-policy'
    base = ROOT/'outputs/goal-point-wam/goal-memory-ddp-20261012-r3/train/final.pt'
    c = ROOT/'outputs/goal-point-wam/wam-effect-ddp-20261013-r3/train/checkpoint_step50500.pt'
    warm = owner/'wam-ref2-joint-multitakeover-20261012-r1'
    fit = owner/'wam-v2-control-fit-20261012-r1'
    corpus = ROOT/'outputs/cm-pointflow-effect-pretrain/pointworld-main3-sharedstats-20261009/data-main-worktree'
    stats = json.loads((corpus.parent/'norm_stats.json').read_text())
    state = torch.load(base, map_location='cpu', weights_only=False)
    model = ControlContextWAM(stats, state['config']).cuda().eval()
    # New adapter is the only absent state; old base must load strictly.
    old = {k: v for k, v in model.state_dict().items() if k.startswith('control_adapter.')}
    model.load_state_dict(dict(state['model'], **old), strict=True)
    del state, old
    h = torch.load(warm/'head.pt', map_location='cpu', weights_only=False)
    load_policy_patch(model, h, policy_patch_admission(h,
        json.loads((warm/'manifest.json').read_text()), json.loads((warm/'result.json').read_text())))
    del h
    identity, decoder = effect_admission(c, warm, fit)
    model.trajectory.load_state_dict(decoder, strict=True)
    del decoder
    parent = ROOT/'outputs/goal-point-wam/wam-control-context-20261013-r3/stage2_step4500.pt'
    parent_identity, packet = control_admission(parent, identity)
    model.control_adapter.load_state_dict(packet['adapter'], strict=True)
    model.trajectory.load_state_dict(packet['trajectory'], strict=True)
    del packet
    adapter = ReferenceWristAdapter().cuda().eval()
    goal_source = ROOT/'outputs/goal-point-wam/planned-wrist-goal-20261013-r1'
    receipt = json.loads((goal_source/'result.json').read_text())
    if (receipt['status'] != 'COMPLETED' or not receipt['planned_reference_only']
            or receipt['offsets'] != [8,16,24] or max(receipt['checks'].values()) > 2e-5):
        raise ValueError('audited planned wrist export required')
    with np.load(goal_source/'sparse_reference_goals.npz', allow_pickle=False) as f:
        reference = {k:f[k].copy() for k in f.files}
    if (not reference['wrist_goal_valid'].all() or not (reference['active_hand']==0).all()
            or not np.array_equal(reference['offsets'], [8,16,24])):
        raise ValueError('this probe requires valid right-hand reference')
    chunk, ri, _ = chunk_admission(fit)
    r = load_chunk(torch.load(chunk, map_location='cpu', weights_only=False), 'cuda').requires_grad_(False)
    recovery = [owner/('wam-ref2-execution-C-t%d-20261013-r1' % t)/'native' for t in (32,40,48)]
    mixed, valbase = MixedWindows(corpus, 'train'), MixedWindows(corpus, 'val')
    train = ControlData(ROOT, GoalWindows(mixed, 'object_points', 32), recovery)
    val = ControlData(ROOT, GoalWindows(valbase, 'object_points', 32), recovery)
    pdscale = independent_PD_scale(torch.tensor(np.asarray(train.controller['scale'])[list(ACTIVE)],
                                                device='cuda', dtype=torch.float32))
    order = torch.tensor(WAM_TO_NATIVE, device='cuda')
    manifest = dict(schema=SCHEMA, status='RUNNING', task='goal-point-wam',
        experiment_id='P-20261013-goal-point-wam-reference-wrist-goal', run_id=out.name,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        parent_identity=parent_identity, retargeter_identity=ri, recovery_paths=list(map(str,recovery)),
        updates_per_arm=a.updates, mix=dict(nominal=30,recovery=50,human=20),
        lambda_a=1., lambda_E=.1, adapter_lr=1e-4, decoder_lr=1e-5, seed=295, draw_seed=296,
        deadline_seconds=a.deadline_seconds, training_allowed=not a.smoke,
        reference_source=str(goal_source),
        reference_admission='User permits reference demonstrations; geometric checks pass; admitted only as learned conditioning, never a robot command',
        context_contract='planned human wrist at +8/+16/+24 in actual current object frame; robot-only right query residual',
        selection='final fixed update only; A blank condition vs B planned, identical initialization/draws/capacity',
        primary='exact three C takeover queries TRAIN first8 mean B<=.9*A, each <=1.05*A+.1mm',
        limitations='same fixed reference/reset, development panels; only three actual C states, remaining recovery queries expert continuation')
    atomic_json(out/'manifest.json', manifest)
    # Fixed, bounded repeated-window Probe; no claim of new unique data.
    nominal_ticks = sorted(set(range(16)) | set(range(16,128,4)) | set(range(128,519,16)))
    nominal = [('nominal', t, e) for t in nominal_ticks for e in range(16)]
    rec = [(name,t,3) for name, ticks in train.valid_ticks.items() for t in ticks]
    human = []
    rng = np.random.default_rng(296)
    for sid in range(3):
        ids = rng.choice(np.arange(mixed.offsets[sid],mixed.offsets[sid+1]),64,replace=False)
        human.extend(('main',int(i),0) for i in ids)
    groups = dict(nominal=[('nominal',t,e) for t in nominal_ticks for e in range(24,28)],
        startup=[('nominal',t,e) for t in range(16) for e in range(24,28)])
    for name,takeover in train.takeovers.items():
        valid = train.valid_ticks[name]
        if not all(t in valid for t in range(takeover,takeover+16)):
            raise ValueError('contact16 panel has clipped/unapplied own expert futures')
        groups[name+'_contact_TRAIN'] = [(name,t,3) for t in range(takeover,takeover+16)]
    panel = np.load(base.parent/'validation_balanced.npy')
    for i, name in enumerate(('oakink2','grab','arctic')):
        ids = panel[(panel>=valbase.offsets[i]) & (panel<valbase.offsets[i+1])][:16]
        groups[name] = [('main',int(j),0) for j in ids]
    for name,takeover in train.takeovers.items():
        groups[name+'_exact_C_TRAIN'] = [(name,takeover,3)]
    if a.smoke:
        nominal, rec, human = nominal[:4], rec[:4], human[:4]
        groups = {k:v[:2] for k,v in groups.items()}
    atomic_json(out/'panel.json', groups)
    atomic_json(out/'training_keys.json', dict(nominal=nominal,recovery=rec,human=human))

    @torch.no_grad()
    def cache(data, keys):
        batches, contexts = [], []
        for begin in range(0,len(keys),16):
            rows = keys[begin:begin+16]
            b = {k:v.cuda() for k,v in control_collate([data[k] for k in rows]).items()}
            for field, shift in (('wrist_goal',0), ('wrist_shift16',16)):
                values=[]
                for kind,tick,role in rows:
                    if kind=='main':
                        values.append(np.zeros(27,np.float32))
                    else:
                        frame=min(int(tick)+shift, len(reference['planned_wrist_position'])-1)
                        values.append(wrist_context(data.data[kind]['obj'][tick,role],
                            reference['planned_wrist_position'][frame],reference['planned_wrist_rotation'][frame]))
                b[field]=torch.as_tensor(np.stack(values),device='cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                context = model.contexts(b)
                tau = model.actor(context,b)
                original = wrist_actor(model,adapter,context,b)
            if not torch.equal(tau, original):
                raise ValueError('zero-gate parent output equivalence failed')
            contexts.append(context)
            batches.append({k:b[k] for k in FIELDS})
            if time.monotonic()-started > a.deadline_seconds:
                raise TimeoutError('context preparation exceeded deadline')
        return concatenate(contexts), concatenate(batches)

    pools = {'nominal':cache(train,nominal), 'recovery':cache(train,rec), 'human':cache(train,human)}
    evaluations = {name:cache(val,keys) for name,keys in groups.items()}
    initial_decoder = {k:v.cpu().clone() for k,v in model.trajectory.state_dict().items()}
    initial_adapter = {k:v.cpu().clone() for k,v in adapter.state_dict().items()}
    frozen = {k:v.cpu().clone() for k,v in model.state_dict().items() if not k.startswith('trajectory.')}
    frozen_r = {k:v.cpu().clone() for k,v in r.state_dict().items()}
    model.requires_grad_(False)
    model.trajectory.requires_grad_(True)
    if model.trajectory.goal_query is not None and model.trajectory.goal_injection != 'query':
        model.trajectory.goal_query.requires_grad_(False)

    def freeze_check():
        maximum=max(float((model.state_dict()[k].cpu().float()-v.float()).abs().max()) for k,v in frozen.items())
        rd=max(float((r.state_dict()[k].cpu()-v).abs().max()) for k,v in frozen_r.items())
        if maximum or rd: raise ValueError('frozen module/buffer drift')
        return dict(frozen_max_difference=maximum,R_max_difference=rd)

    @torch.no_grad()
    def evaluate(label, mode):
        report, arrays = {}, {}
        for name,(context,batch) in evaluations.items():
            values={k:[] for k in ('h1_mm','first8_mm','all24_mm')}
            predictions=[]
            for begin in range(0,len(batch['proprio']),16):
                ids=slice(begin,begin+16)
                b,c=subset(batch,ids),subset(context,ids)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    tau=wrist_actor(model,adapter,c,b,mode)
                predictions.append(tau.cpu().numpy())
                error=torch.linalg.vector_norm(tau-b['trajectory_target'],dim=-1)*1000
                valid=b['trajectory_valid']
                for key,count in (('h1_mm',1),('first8_mm',8),('all24_mm',24)):
                    v=(error[:,:count]*valid[:,:count]).flatten(1).sum(1)/valid[:,:count].flatten(1).sum(1).clamp_min(1)
                    values[key].extend(v.cpu().tolist())
                if b['has_pd'].any():
                    action=r(b['native_state'],tau[:,:,0].index_select(2,order).reshape(-1,24,33))[:,0]
                    delta=(action.clamp(-1,1)-b['native_action'][:,0])*pdscale
                    values.setdefault('PD_xyz_mm',[]).extend((delta[:,:3].norm(dim=-1)*1000).cpu().tolist())
                    values.setdefault('clipping_rate',[]).extend((action.abs()>1+1e-6).any(-1).float().cpu().tolist())
            report[name]={k:float(np.mean(v)) for k,v in values.items() if k!='PD_xyz_mm'}
            if 'PD_xyz_mm' in values:report[name]['PD_xyz_mm_p95']=float(np.percentile(values['PD_xyz_mm'],95))
            arrays.update({name+'_'+k:np.asarray(v) for k,v in values.items()})
            arrays[name+'_tau']=np.concatenate(predictions)
            arrays[name+'_target']=batch['trajectory_target'].cpu().numpy()
            arrays[name+'_valid']=batch['trajectory_valid'].cpu().numpy()
        atomic_json(out/(label+'.json'),report)
        np.savez_compressed(out/(label+'.npz'),**arrays)
        return report

    initial=evaluate('initial','planned')
    atomic_json(out/'initial_contract.json',dict(exact_parent_output=True,**freeze_check()))
    print(json.dumps(dict(event='cache_complete',elapsed_s=time.monotonic()-started,
        pool_sizes={k:len(v[1]['proprio']) for k,v in pools.items()},gpu=gpu_snapshot(torch.device('cuda')))),flush=True)
    stop=[False]
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):
        signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    curve=[]; final={}; contracts={}; intervention={}; completed=True
    for arm,mode in (('A','blank'),('B','planned')):
        model.trajectory.load_state_dict(initial_decoder,strict=True)
        adapter.load_state_dict(initial_adapter,strict=True)
        torch.manual_seed(295)
        parameters=[{'params':[p for p in model.trajectory.parameters() if p.requires_grad],'lr':1e-5},
                    {'params':list(adapter.parameters()),'lr':1e-4}]
        optimizer=torch.optim.AdamW(parameters,weight_decay=.01)
        phase_start=time.monotonic();gradient_seen=False
        for step in range(1,a.updates+1):
            optimizer.zero_grad(set_to_none=True)
            draw=np.random.default_rng(np.random.SeedSequence([296,2,step]))
            total=0.
            for name,count in manifest['mix'].items():
                pc,pb=pools[name]
                ids=torch.tensor(draw.integers(0,len(pb['proprio']),count),device='cuda')
                b,c=subset(pb,ids),subset(pc,ids)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    tau=wrist_actor(model,adapter,c,b,mode)
                    effect=model.induced(tau,b,c[0])
                valid=b['trajectory_valid']
                te=((tau-b['trajectory_target'])/.05).square().mean(-1)
                tl=((te*valid).flatten(1).sum(1)/valid.flatten(1).sum(1).clamp_min(1)).mean()
                el,_=model.world_loss(effect,b)
                al=torch.zeros((),device='cuda')
                if name!='human':
                    native=tau[:,:,0].index_select(2,order).reshape(-1,24,33)
                    al,_=control_aware_loss(r(b['native_state'],native),b['native_action'],r.action_scale,pdscale)
                loss=(tl+al+.1*el)*count/100
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
                loss.backward();total+=float(loss.detach())
            trainable=[p for group in parameters for p in group['params']]
            norm=torch.nn.utils.clip_grad_norm_(trainable,1.)
            if not torch.isfinite(norm):raise FloatingPointError('nonfinite gradient')
            if any(p.grad is not None for p in list(model.parameters())+list(r.parameters()) if not p.requires_grad):
                raise ValueError('gradient reached frozen module')
            gradient_seen |= any(p.grad is not None and bool(p.grad.count_nonzero()) for p in adapter.net.parameters())
            optimizer.step()
            if step==1 or step%100==0:
                record=dict(arm=arm,step=step,loss=total,alpha=float(adapter.alpha),
                    elapsed_s=time.monotonic()-started,eta_arm_s=(time.monotonic()-phase_start)/step*(a.updates-step),
                    gpu=gpu_snapshot(torch.device('cuda')))
                atomic_json(out/'progress.json',record);print(json.dumps(record),flush=True)
            if stop[0] or time.monotonic()-started>a.deadline_seconds:
                completed=False;break
            if step%500==0 or step==a.updates:
                foreign=set(map(int,subprocess.check_output(['nvidia-smi','-i',str(a.gpu),
                    '--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()))-{os.getpid()}
                if foreign:raise RuntimeError('foreign GPU process')
                sizes=[int(subprocess.check_output(['du','-sb',str(p)],text=True).split()[0])
                       for p in (out,scratch,ROOT/'outputs',ROOT/'tmp')]
                if sum(sizes[:2])>1_000_000_000 or sum(sizes[2:])>400_000_000_000:raise RuntimeError('disk budget')
                last=evaluate(arm+'_step'+str(step),mode)
                curve.append(dict(arm=arm,step=step,metrics=last))
                atomic_json(out/'curve.json',curve)
        if not completed:break
        final[arm]=last
        contracts[arm]=dict(**freeze_check(),adapter_gradient_seen=gradient_seen,
            adapter_max_difference=max(float((adapter.state_dict()[k].cpu()-v).abs().max()) for k,v in initial_adapter.items()),
            decoder_max_difference=max(float((model.trajectory.state_dict()[k].cpu()-v).abs().max()) for k,v in initial_decoder.items()))
        torch.save(dict(schema=SCHEMA,arm=arm,adapter=adapter.state_dict(),trajectory=model.trajectory.state_dict(),
            manifest=manifest,contracts=contracts[arm],step=step),out/(arm+'_final.pt'))
        if arm=='B':
            intervention={mode:evaluate('B_'+mode,mode) for mode in ('blank','shift16')}
    checks={}
    if completed:
        aa,bb=final['A'],final['B']
        exact=[n for n in groups if 'exact_C' in n]
        contact=[n for n in groups if 'contact' in n]
        primary_a=float(np.mean([aa[n]['first8_mm'] for n in exact]))
        primary_b=float(np.mean([bb[n]['first8_mm'] for n in exact]))
        checks['exact_mean_10pct']=primary_b<=.9*primary_a
        for n in exact+contact:
            checks[n+'/first8_guard']=bb[n]['first8_mm']<=1.05*aa[n]['first8_mm']+.1
            checks[n+'/PD_guard']=bb[n]['PD_xyz_mm_p95']<=1.1*aa[n]['PD_xyz_mm_p95']+.1
        for n in ('nominal','startup'):
            checks[n+'/PD_guard']=bb[n]['PD_xyz_mm_p95']<=1.1*aa[n]['PD_xyz_mm_p95']+.1
            checks[n+'/clipping_guard']=bb[n]['clipping_rate']<=aa[n]['clipping_rate']+.01
        for n in ('oakink2','grab','arctic'):
            checks[n+'/human_guard']=bb[n]['all24_mm']<=1.05*aa[n]['all24_mm']+.1
        scope='PROMISING' if all(checks.values()) else 'UNPROMISING'
    else:
        primary_a=primary_b=None;scope='UNCLEAR'
    result=dict(status='COMPLETED' if completed else 'STOPPED',probe_status=scope,
        engineering_only=a.smoke,primary_A_mm=primary_a,primary_B_mm=primary_b,checks=checks,
        initial=initial,final=final,interventions=intervention,contracts=contracts,
        elapsed_s=time.monotonic()-started,scope='fixed-budget continuation; no generalization or physical grasp claim')
    atomic_json(out/'result.json',result)
    manifest['status']=result['status'];atomic_json(out/'manifest.json',manifest)
    print(json.dumps({k:v for k,v in result.items() if k not in ('initial','final','interventions')}),flush=True)


if __name__=='__main__':main()

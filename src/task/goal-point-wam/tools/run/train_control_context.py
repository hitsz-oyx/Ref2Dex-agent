"""Bounded ref2_8 robot adapter finetuning with frozen measured-state contexts."""
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

SCHEMA = 'ref2dex.robot-control-context.v1'
FIELDS = ('start_hand', 'trajectory_target', 'trajectory_valid', 'action', 'action_valid',
          'points', 'effect', 'object_valid', 'radius', 'has_pd', 'native_state',
          'native_action', 'proprio', 'robot_goal', 'robot_mask', 'source_id')


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
    p.add_argument('--stage1', type=int, default=2000)
    p.add_argument('--stage2', type=int, default=5000)
    p.add_argument('--deadline-seconds', type=int, default=7200)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--stage1-checkpoint', type=Path,
                   help='continue a frozen-decoder stage1 packet into bounded stage2')
    a = p.parse_args()
    out = a.output.resolve()
    if out.exists() or ROOT/'outputs/goal-point-wam' not in out.parents:
        raise ValueError('fresh owned output required')
    if not 1 <= a.stage1 <= 3000 or not 1 <= a.stage2 <= 20000 or not 1 <= a.deadline_seconds <= 7200:
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
        experiment_id='P-20261013-goal-point-wam-ref28-control-context', run_id=out.name,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        C_identity=identity, retargeter_identity=ri, recovery_paths=list(map(str, recovery)),
        stage1_updates=a.stage1, stage2_updates=a.stage2, stage2_mix=[20,30,50],
        stage1_nominal_recovery=[30,50], lambda_a=1., lambda_E=.1,
        adapter_lr=1e-4, decoder_lr=1e-5, normalization='existing train-only base stats; native context physical units',
        deadline_seconds=a.deadline_seconds, training_allowed=not a.smoke,
        frozen_context='bounded fixed training subset; cached once on GPU, not serialized',
        gate_protocol='contact first8 all nonregressing and mean ratio<=.9; each PD<=.9; startup PD<=max(relative limit,3mm); strict original gates retained',
        context_contract='q18,dq18,previous applied ACTIVE12; own-current-relative planned +8/+16/+24 poses')
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
    if a.smoke:
        nominal, rec, human = nominal[:4], rec[:4], human[:4]
        groups = {k:v[:2] for k,v in groups.items()}
    atomic_json(out/'panel.json', groups)
    atomic_json(out/'training_keys.json', dict(nominal=nominal,recovery=rec,human=human))

    @torch.no_grad()
    def cache(data, keys):
        batches, contexts = [], []
        for begin in range(0,len(keys),16):
            b = {k:v.cuda() for k,v in control_collate([data[k] for k in keys[begin:begin+16]]).items()}
            with torch.autocast('cuda', dtype=torch.bfloat16):
                context = model.contexts(b)
                tau = model.actor(context,b)
                original = model.trajectory(context[0]['scene_tokens'],context[0]['scene_mask'],
                    context[1],context[2],context[3])
            if not torch.equal(tau, original):
                raise ValueError('zero-gate C initial equivalence failed')
            contexts.append(context)
            batches.append({k:b[k] for k in FIELDS})
            if time.monotonic()-started > a.deadline_seconds:
                raise TimeoutError('context preparation exceeded deadline')
        return concatenate(contexts), concatenate(batches)

    pools = {'nominal':cache(train,nominal), 'recovery':cache(train,rec), 'human':cache(train,human)}
    evaluations = {name:cache(val,keys) for name,keys in groups.items()}
    freeze = {k:v.cpu().clone() for k,v in model.state_dict().items()
              if not k.startswith(('control_adapter.','trajectory.'))}
    old_decoder = {k:v.cpu().clone() for k,v in model.trajectory.state_dict().items()}
    old_r = {k:v.cpu().clone() for k,v in r.state_dict().items()}

    def frozen_check(stage):
        error = max(float((model.state_dict()[k].cpu().float()-v.float()).abs().max()) for k,v in freeze.items())
        rd = max(float((r.state_dict()[k].cpu()-v).abs().max()) for k,v in old_r.items())
        dd = max(float((model.trajectory.state_dict()[k].cpu().float()-v.float()).abs().max()) for k,v in old_decoder.items())
        if error != 0 or rd != 0 or (stage==1 and dd != 0):
            raise ValueError('frozen module/buffer drift')
        return dict(frozen_max_difference=error,R_max_difference=rd,decoder_difference=dd)

    @torch.no_grad()
    def evaluate(step, off=False):
        gates = (model.control_adapter.alpha_p.clone(),model.control_adapter.alpha_g.clone())
        if off:
            model.control_adapter.alpha_p.zero_();model.control_adapter.alpha_g.zero_()
        report, raw = {}, {}
        for name,(context,b) in evaluations.items():
            values = {k:[] for k in ('h1_mm','first8_mm','all24_mm')}
            predictions = []
            for start in range(0,len(b['proprio']),16):
                ids = slice(start,start+16)
                cb, cc = subset(b,ids), subset(context,ids)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    tau = model.actor(cc,cb)
                predictions.append(tau.cpu().numpy())
                error = torch.linalg.vector_norm(tau-cb['trajectory_target'],dim=-1)*1000
                valid = cb['trajectory_valid']
                for key,count in (('h1_mm',1),('first8_mm',8),('all24_mm',24)):
                    v = (error[:,:count]*valid[:,:count]).flatten(1).sum(1)/valid[:,:count].flatten(1).sum(1).clamp_min(1)
                    values[key].extend(v.cpu().tolist())
                if cb['has_pd'].any():
                    action = r(cb['native_state'],tau[:,:,0].index_select(2,order).reshape(-1,24,33))[:,0]
                    delta = (action.clamp(-1,1)-cb['native_action'][:,0])*pdscale
                    for k,v in dict(PD_xyz_mm=delta[:,:3].norm(dim=-1)*1000,
                        PD_rotation_rad=delta[:,3:6].norm(dim=-1),
                        PD_finger_rad=delta[:,6:].abs().mean(-1),clipping_rate=(action.abs()>1+1e-6).any(-1).float()).items():
                        values.setdefault(k,[]).extend(v.cpu().tolist())
            report[name] = {k:float(np.mean(v)) for k,v in values.items() if not k.startswith('PD_')}
            report[name].update({k+'_p95':float(np.percentile(v,95)) for k,v in values.items() if k.startswith('PD_')})
            raw.update({name+'_'+k:np.asarray(v) for k,v in values.items()})
            raw[name+'_tau'] = np.concatenate(predictions)
        if off:
            model.control_adapter.alpha_p.copy_(gates[0]);model.control_adapter.alpha_g.copy_(gates[1])
        suffix = '_adapter_off' if off else ''
        atomic_json(out/('evaluation_step%d%s.json'%(step,suffix)),report)
        np.savez_compressed(out/('panel_step%d%s.npz'%(step,suffix)),**raw)
        return report

    initial = evaluate(0)
    atomic_json(out/'initial_contract.json',dict(exact_C_output=True,human_mask_zero=True,**frozen_check(1)))
    curve = []
    stop = [False]
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):
        signal.signal(sig,lambda *_:stop.__setitem__(0,True))

    def contact_gate(now):
        checks = {}
        for name in (n for n in groups if 'contact' in n):
            checks[name+'/first8_nonregression'] = now[name]['first8_mm'] <= initial[name]['first8_mm']+.05
            checks[name+'/PD_xyz_mm_p95'] = now[name]['PD_xyz_mm_p95'] <= .9*initial[name]['PD_xyz_mm_p95']
        checks['contact/mean_first8_ratio'] = np.mean([now[n]['first8_mm']/initial[n]['first8_mm']
            for n in groups if 'contact' in n])<=.9
        for name in ('nominal','startup'):
            limit=1.1*initial[name]['PD_xyz_mm_p95']+.1
            if name=='startup':limit=max(limit,3.)
            checks[name+'/PD'] = now[name]['PD_xyz_mm_p95']<=limit
            checks[name+'/clipping'] = now[name]['clipping_rate']<=initial[name]['clipping_rate']+.01
        return checks

    def save(stage,step):
        path = out/('stage%d_step%d.pt'%(stage,step))
        torch.save(dict(schema=SCHEMA,adapter=model.control_adapter.state_dict(),trajectory=model.trajectory.state_dict(),
            manifest=manifest,stage=stage,step=step,initial=initial,curve=curve,
            optimizer=optimizer.state_dict(),contracts=frozen_check(stage)),path)
        return path

    stage1_pass = False
    best_score = float('inf')
    bad_streak = 0
    last, final, reason = initial, None, 'stage1_contact_gate_failed'
    phases=((1,a.stage1),(2,a.stage2))
    if a.stage1_checkpoint:
        parent_path=a.stage1_checkpoint.resolve()
        if ROOT/'outputs/goal-point-wam' not in parent_path.parents:raise ValueError('owned stage1 continuation required')
        parent=torch.load(parent_path,map_location='cpu',weights_only=False)
        if (parent.get('schema')!=SCHEMA or parent.get('stage')!=1
            or parent.get('manifest',{}).get('C_identity')!=identity
            or parent.get('contracts',{}).get('decoder_difference')!=0.
            or parent['contracts'].get('frozen_max_difference')!=0. or parent['contracts'].get('R_max_difference')!=0.):
            raise ValueError('stage1 frozen C/World/R continuation differs')
        model.control_adapter.load_state_dict(parent['adapter'],strict=True)
        model.trajectory.load_state_dict(parent['trajectory'],strict=True)
        frozen_check(1)
        last=evaluate(100000+parent['step'])
        checks=contact_gate(last)
        stage1_pass=all(checks.values())
        reason='stage2_complete' if stage1_pass else 'stage1_contact_gate_failed'
        manifest['stage1_parent']=str(parent_path)
        atomic_json(out/'manifest.json',manifest)
        atomic_json(out/'stage1_gate.json',dict(pass_=stage1_pass,checks=checks,
            original_strict_gate_pass=False,protocol_amended=True))
        phases=((2,a.stage2),)
        if not stage1_pass:raise ValueError('stage1 continuation fails revised contact gate')
    best_any_score=float('inf')
    best_any_path=None
    for stage,updates in phases:
        if stage==2 and not stage1_pass:
            break
        model.set_stage(stage)
        parameters = [{'params':list(model.control_adapter.parameters()),'lr':1e-4}]
        if stage==2:
            parameters.append({'params':[p for p in model.trajectory.parameters() if p.requires_grad],'lr':1e-5})
        optimizer = torch.optim.AdamW(parameters,weight_decay=.01)
        phase_start = time.monotonic()
        for step in range(1,updates+1):
            optimizer.zero_grad(set_to_none=True)
            counts = {'nominal':30,'recovery':50,'human':20} if stage==2 else {'nominal':30,'recovery':50}
            draw = np.random.default_rng(np.random.SeedSequence([296,stage,step]))
            loss_total = 0.
            for name,count in counts.items():
                pc,pb = pools[name]
                ids = torch.tensor(draw.integers(0,len(pb['proprio']),count),device='cuda')
                b,context = subset(pb,ids),subset(pc,ids)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    tau = model.actor(context,b)
                    effect = model.induced(tau,b,context[0])
                valid = b['trajectory_valid']
                te = ((tau-b['trajectory_target'])/.05).square().mean(-1)
                tl = ((te*valid).flatten(1).sum(1)/valid.flatten(1).sum(1).clamp_min(1)).mean()
                el,_ = model.world_loss(effect,b)
                al = torch.zeros((),device='cuda')
                if name!='human':
                    native = tau[:,:,0].index_select(2,order).reshape(-1,24,33)
                    al,_ = control_aware_loss(r(b['native_state'],native),b['native_action'],r.action_scale,pdscale)
                loss = (tl+al+.1*el)*count/sum(counts.values())
                if not torch.isfinite(loss):
                    raise FloatingPointError('control context loss nonfinite')
                loss.backward()
                loss_total += float(loss.detach())
            trainable = [p for p in model.parameters() if p.requires_grad]
            norm = torch.nn.utils.clip_grad_norm_(trainable,1.)
            if not torch.isfinite(norm):
                raise FloatingPointError('adapter gradient nonfinite')
            if any(p.grad is not None for p in list(model.parameters())+list(r.parameters()) if not p.requires_grad):
                raise ValueError('gradient reached frozen module')
            optimizer.step()
            stopping = stop[0] or time.monotonic()-started>=a.deadline_seconds
            if step==1 or step%100==0:
                record = dict(stage=stage,step=step,loss=loss_total,
                    alpha_p=float(model.control_adapter.alpha_p),alpha_g=float(model.control_adapter.alpha_g),
                    elapsed_s=time.monotonic()-started,eta_stage_s=(time.monotonic()-phase_start)/step*(updates-step),
                    gpu=gpu_snapshot(torch.device('cuda')))
                atomic_json(out/'progress.json',record)
                print(json.dumps(record),flush=True)
            if step%500==0 or step==updates or stopping:
                pids=set(map(int,subprocess.check_output(['nvidia-smi','-i',str(a.gpu),
                    '--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()))
                if pids-{os.getpid()}:
                    raise RuntimeError('foreign GPU process during owned training')
                sizes=subprocess.check_output(['du','-sb',str(out),str(scratch),
                    str(ROOT/'outputs'),str(ROOT/'tmp')],text=True).splitlines()
                sizes=[int(line.split()[0]) for line in sizes]
                if sum(sizes[:2])>=2_000_000_000 or sum(sizes[2:])>=400_000_000_000:
                    raise RuntimeError('adapter2GB / global400GB artifact cap')
                last = evaluate(stage*100000+step)
                checks = contact_gate(last)
                curve.append(dict(stage=stage,step=step,metrics=last,contact_gate=checks,**frozen_check(stage)))
                atomic_json(out/'curve.json',curve)
                candidate = save(stage,step)
                score = np.mean([last[n][k]/max(initial[n][k],1e-6)
                    for n in groups if 'contact' in n for k in ('first8_mm','PD_xyz_mm_p95')])
                preservation = all(v for k,v in checks.items() if k.startswith(('nominal/','startup/')))
                if stage==2:
                    preservation = preservation and all(last[n]['all24_mm']<=1.2*initial[n]['all24_mm']+.05
                        for n in ('oakink2','grab','arctic'))
                if score<best_any_score:
                    best_any_score,best_any_path=float(score),candidate
                if preservation and score < best_score:
                    best_score, final = float(score), candidate
                gross = any(last[n][k]>2.*initial[n][k]+.1
                    for n in groups if 'contact' in n for k in ('first8_mm','PD_xyz_mm_p95'))
                bad_streak = bad_streak+1 if gross else 0
                if bad_streak>=2:
                    stopping = True
                    reason = 'two_consecutive_gross_contact_regressions'
            if stopping:
                if reason != 'two_consecutive_gross_contact_regressions':
                    reason = 'deadline_or_user_stop'
                break
        if stopping:
            break
        if stage==1:
            stage1_pass = all(contact_gate(last).values())
            atomic_json(out/'stage1_gate.json',dict(pass_=stage1_pass,checks=contact_gate(last)))
            evaluate(100000+step,off=True)
            if stage1_pass:
                reason = 'stage2_complete'
        else:
            evaluate(200000+step,off=True)
    qualified_selection=final is not None
    if final is None:final=best_any_path
    result = dict(status='COMPLETED' if reason not in ('deadline_or_user_stop','two_consecutive_gross_contact_regressions') else 'STOPPED',reason=reason,
        stage1_pass=stage1_pass,final_checkpoint=str(final),last=last,initial=initial,
        curve=curve,best_contact_score=best_score if qualified_selection else best_any_score,
        selection_preservation_pass=qualified_selection,elapsed_s=time.monotonic()-started,
        scope='bounded repeated-window TRAIN contact Probe; no physical claim')
    atomic_json(out/'result.json',result)
    manifest['status'] = result['status'];atomic_json(out/'manifest.json',manifest)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()

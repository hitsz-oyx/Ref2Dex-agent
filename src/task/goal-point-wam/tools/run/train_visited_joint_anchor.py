"""Joint actor/R update on visited recoveries, with a GT-tau action anchor."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_control_context import (ROOT, FIELDS, SCHEMA, ControlContextWAM, GoalWindows, MixedWindows,
    control_collate, load_policy_patch, policy_patch_admission, effect_admission, chunk_admission,
    load_chunk, independent_PD_scale, control_aware_loss, ACTIVE, WAM_TO_NATIVE, concatenate, subset,
    atomic_json, gpu_snapshot, install_fused_hilbert)
from trajectory_policy.visited_control_data import VisitedControlData, collection_entries
from trajectory_policy.chunk_retargeter import SCHEMA as CHUNK_SCHEMA
from trajectory_policy.wam_retargeter_admission import sha

EXPERIMENT = 'P-20261013-goal-point-wam-visited-joint-anchor'
BUFFERS = ('state_mean', 'state_scale', 'tau_mean', 'tau_scale', 'action_mean', 'action_scale')


def grad_sum(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    total = 0.
    for grad in grads:
        if grad is None:
            continue
        if not torch.isfinite(grad).all():
            raise FloatingPointError('nonfinite joint gradient')
        total += float(grad.detach().abs().sum())
    return total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collections', nargs='+', type=Path, required=True)
    p.add_argument('--r-fit', type=Path, required=True)
    p.add_argument('--actor-parent', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--chunk-output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=2)
    p.add_argument('--updates', type=int, default=3000)
    a = p.parse_args()
    out = a.output.resolve()
    chunk_out = a.chunk_output.resolve()
    if (out.exists() or chunk_out.exists() or ROOT/'outputs/goal-point-wam' not in out.parents
            or ROOT/'outputs/trajectory-policy' not in chunk_out.parents or not 1 <= a.updates <= 3000):
        raise ValueError('fresh bounded actor and chunk outputs required')
    if subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-compute-apps=pid',
                               '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU occupied')
    scratch = ROOT/'tmp'/out.name
    scratch.mkdir(parents=True, exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), TMPDIR=str(scratch), CUBLAS_WORKSPACE_CONFIG=':4096:8')
    torch.set_num_threads(2)
    torch.manual_seed(295)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.use_deterministic_algorithms(True)
    install_fused_hilbert()
    started = time.monotonic()
    owner = ROOT/'outputs/trajectory-policy'
    base = ROOT/'outputs/goal-point-wam/goal-memory-ddp-20261012-r3/train/final.pt'
    warm = owner/'wam-ref2-joint-multitakeover-20261012-r1'
    c = ROOT/'outputs/goal-point-wam/wam-effect-ddp-20261013-r3/train/checkpoint_step50500.pt'
    parent_r = a.r_fit.resolve()
    parent_actor = a.actor_parent.resolve()
    parent_result = json.loads((parent_actor.parent/'result.json').read_text())
    if parent_result.get('offline_gate') is not True or parent_result.get('final_checkpoint') != str(parent_actor):
        raise ValueError('joint update starts from the admitted aligned actor')
    corpus = ROOT/'outputs/cm-pointflow-effect-pretrain/pointworld-main3-sharedstats-20261009/data-main-worktree'
    state = torch.load(base, map_location='cpu', weights_only=False)
    model = ControlContextWAM(json.loads((corpus.parent/'norm_stats.json').read_text()), state['config']).cuda().eval()
    adapter = {k: v for k, v in model.state_dict().items() if k.startswith('control_adapter.')}
    model.load_state_dict(dict(state['model'], **adapter), strict=True)
    del state, adapter
    head = torch.load(warm/'head.pt', map_location='cpu', weights_only=False)
    load_policy_patch(model, head, policy_patch_admission(
        head, json.loads((warm/'manifest.json').read_text()), json.loads((warm/'result.json').read_text())))
    del head
    identity, _ = effect_admission(c, warm, owner/'wam-v2-control-fit-20261012-r1')
    packet = torch.load(parent_actor, map_location='cpu', weights_only=False)
    if packet['manifest']['C_identity'] != identity:
        raise ValueError('parent actor C identity differs')
    model.trajectory.load_state_dict(packet['trajectory'])
    model.control_adapter.load_state_dict(packet['adapter'])
    del packet
    model.set_stage(2)
    chunk, ri, _ = chunk_admission(parent_r)
    if not json.loads((parent_r/'result.json').read_text())['offline_gate']:
        raise ValueError('parent R offline gate required')
    r = load_chunk(torch.load(chunk, map_location='cpu', weights_only=False), 'cuda')
    r.requires_grad_(True)
    r.eval()
    entries = collection_entries(a.collections)
    mixed = MixedWindows(corpus, 'train')
    valbase = MixedWindows(corpus, 'val')
    data = VisitedControlData(ROOT, GoalWindows(mixed, 'object_points', 32), entries)
    val = VisitedControlData(ROOT, GoalWindows(valbase, 'object_points', 32), entries)
    train_names = [n for n in data.takeovers if data.families[n] != 3]
    dev_names = [n for n in data.takeovers if data.families[n] == 3]
    if not train_names or not dev_names:
        raise ValueError('whole-prefix TRAIN/DEV required')
    ticks = sorted(set(range(16)) | set(range(16, 128, 4)) | set(range(128, 519, 16)))
    nominal = [('nominal', t, e) for t in ticks for e in range(16)]
    exact = [(n, data.takeovers[n], 3) for n in train_names]
    continuation = [(n, t, 3) for n in train_names for t in data.valid_ticks[n] if t > data.takeovers[n]]
    human = []
    rng = np.random.default_rng(296)
    for sid in range(3):
        human.extend(('main', int(i), 0) for i in rng.choice(
            np.arange(mixed.offsets[sid], mixed.offsets[sid+1]), 64, replace=False))
    keys = dict(nominal=nominal, exact=exact, continuation=continuation, human=human)
    groups = dict(nominal=[('nominal', t, e) for t in ticks for e in range(24, 28)],
                  startup=[('nominal', t, e) for t in range(16) for e in range(24, 28)])
    for split, names in [('train', train_names), ('dev', dev_names)]:
        groups['exact_'+split] = [(n, data.takeovers[n], 3) for n in names]
        groups['contact_'+split] = [(n, t, 3) for n in names for t in data.valid_ticks[n] if t < data.takeovers[n]+16]
    panel = np.load(base.parent/'validation_balanced.npy')
    for sid, name in enumerate(('oakink2', 'grab', 'arctic')):
        groups[name] = [('main', int(i), 0) for i in panel[
            (panel >= valbase.offsets[sid]) & (panel < valbase.offsets[sid+1])][:16]]
    out.mkdir()
    chunk_out.mkdir()
    manifest = dict(schema=SCHEMA, status='RUNNING', experiment_id=EXPERIMENT, run_id=out.name,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        C_identity=identity, parent_control=str(parent_actor), parent_R=str(chunk),
        retargeter_identity=ri, joint_anchor=True, training_allowed=True, alignment=False,
        collections=list(map(str, a.collections)), maximum_updates=a.updates, physical_gpu=a.gpu,
        loss='trajectory MSE/.05 + predicted-tau action L1 through R + detached GT-tau action L1; equal weights; no World loss',
        mix=dict(nominal=30, human=20, exact=25, continuation=25), excluded=data.excluded,
        chunk_output=str(chunk_out))
    atomic_json(out/'manifest.json', manifest)
    atomic_json(out/'training_keys.json', keys)
    atomic_json(out/'panel.json', groups)

    @torch.no_grad()
    def cache(dataset, items):
        batches = []
        contexts = []
        for start in range(0, len(items), 16):
            b = {k: v.cuda() for k, v in control_collate([dataset[k] for k in items[start:start+16]]).items()}
            with torch.autocast('cuda', dtype=torch.bfloat16):
                context = model.contexts(b)
            batches.append({k: b[k] for k in FIELDS})
            contexts.append(context)
        return concatenate(contexts), concatenate(batches)

    pools = {name: cache(data, ks) for name, ks in keys.items()}
    panels = {name: cache(val, ks) for name, ks in groups.items()}
    frozen = {k: v.cpu().clone() for k, v in model.state_dict().items()
              if not k.startswith(('trajectory.', 'control_adapter.'))}
    buffers = {k: getattr(r, k).detach().cpu().clone() for k in BUFFERS}
    scale = independent_PD_scale(torch.tensor(np.asarray(data.controller['scale'])[list(ACTIVE)],
                                              dtype=torch.float32, device='cuda'))
    order = torch.tensor(WAM_TO_NATIVE, device='cuda')

    def native_tau(tau):
        return tau[:, :, 0].index_select(2, order).reshape(-1, 24, 33)

    def contracts():
        frozen_diff = max(float((model.state_dict()[k].cpu().float()-v.float()).abs().max()) for k, v in frozen.items())
        buffer_diff = max(float((getattr(r, k).detach().cpu()-v).abs().max()) for k, v in buffers.items())
        if frozen_diff != 0 or buffer_diff != 0:
            raise ValueError('frozen module or R normalization drift')
        return dict(frozen_max_difference=frozen_diff, R_buffer_difference=buffer_diff)

    @torch.no_grad()
    def evaluate(step):
        reports = {}
        raw = {}
        for name, (context, batch) in panels.items():
            vals = {k: [] for k in ('first8_mm', 'all24_mm', 'PD_xyz_mm', 'GT_PD_xyz_mm', 'clipping', 'GT_clipping')}
            for start in range(0, len(batch['proprio']), 16):
                sl = slice(start, start+16)
                b, ctx = subset(batch, sl), subset(context, sl)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    tau = model.actor(ctx, b)
                error = (tau-b['trajectory_target']).norm(dim=-1)*1000
                valid = b['trajectory_valid']
                for key, count in [('first8_mm', 8), ('all24_mm', 24)]:
                    vals[key].extend((((error[:, :count]*valid[:, :count]).flatten(1).sum(1)
                        / valid[:, :count].flatten(1).sum(1).clamp_min(1)).cpu().tolist()))
                if b['has_pd'].any():
                    pred = r(b['native_state'], native_tau(tau))[:, 0]
                    gt = r(b['native_state'], native_tau(b['trajectory_target']))[:, 0]
                    delta = (pred.clamp(-1, 1)-b['native_action'][:, 0])*scale
                    gt_delta = (gt.clamp(-1, 1)-b['native_action'][:, 0])*scale
                    vals['PD_xyz_mm'].extend((delta[:, :3].norm(dim=-1)*1000).cpu().tolist())
                    vals['GT_PD_xyz_mm'].extend((gt_delta[:, :3].norm(dim=-1)*1000).cpu().tolist())
                    vals['clipping'].extend((pred.abs() > 1+1e-6).any(-1).float().cpu().tolist())
                    vals['GT_clipping'].extend((gt.abs() > 1+1e-6).any(-1).float().cpu().tolist())
            reports[name] = {k: float(np.mean(v)) for k, v in vals.items() if v}
            for key in ('PD_xyz_mm', 'GT_PD_xyz_mm'):
                if vals[key]:
                    reports[name][key.replace('_xyz_mm', '_p95_mm')] = float(np.percentile(vals[key], 95))
            raw.update({name+'_'+k: np.asarray(v) for k, v in vals.items() if v})
        atomic_json(out/f'evaluation_step{step}.json', reports)
        np.savez_compressed(out/f'panel_step{step}.npz', **raw)
        return reports

    def save_pair(step, report, checks, evidence):
        current = contracts()
        r_diff = max(float((p.detach().cpu()-q).abs().max()) for p, q in zip(
            r.parameters(), initial_parameters))
        current['R_max_difference'] = r_diff
        torch.save(dict(schema=CHUNK_SCHEMA, width=128, model={k: v.detach().cpu() for k, v in r.state_dict().items()},
                        native_controller=data.controller), chunk_out/'chunk.pt')
        digest = sha(chunk_out/'chunk.pt')
        manifest['training_R_sha256'] = digest
        manifest['training_R_checkpoint'] = str(chunk_out/'chunk.pt')
        torch.save(dict(schema=SCHEMA, adapter=model.control_adapter.state_dict(),
                        trajectory=model.trajectory.state_dict(), manifest=manifest, stage=2, step=step,
                        contracts=current, gradient_evidence=evidence), out/'selected.pt')
        return digest, current

    initial = evaluate(0)
    actor_params = [p for p in list(model.control_adapter.parameters())+list(model.trajectory.parameters()) if p.requires_grad]
    r_params = [p for p in r.parameters() if p.requires_grad]
    initial_parameters = [p.detach().cpu().clone() for p in r_params]
    optimizer = torch.optim.AdamW([
        {'params': model.control_adapter.parameters(), 'lr': 1e-4},
        {'params': [p for p in model.trajectory.parameters() if p.requires_grad], 'lr': 1e-5},
        {'params': r_params, 'lr': 1e-4}], weight_decay=.01)
    curve = []
    best = None
    bad = 0
    evidence = {}
    train_start = time.monotonic()
    try:
        for step in range(1, a.updates+1):
            optimizer.zero_grad(set_to_none=True)
            draw = np.random.default_rng(np.random.SeedSequence([296, step]))
            total = 0.
            for name, count in manifest['mix'].items():
                context, batch = pools[name]
                ids = torch.tensor(draw.integers(0, len(batch['proprio']), count), device='cuda')
                b, ctx = subset(batch, ids), subset(context, ids)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    tau = model.actor(ctx, b)
                error = ((tau-b['trajectory_target'])/.05).square().mean(-1)
                valid = b['trajectory_valid']
                tl = ((error*valid).flatten(1).sum(1)/valid.flatten(1).sum(1).clamp_min(1)).mean()
                al = anchor = torch.zeros((), device='cuda')
                if name != 'human':
                    if not bool(b['has_pd'].all()):
                        raise ValueError('robot pool lacks recorded actions')
                    predicted = r(b['native_state'], native_tau(tau))
                    al, _ = control_aware_loss(predicted, b['native_action'], r.action_scale, scale)
                    anchored = r(b['native_state'], native_tau(b['trajectory_target']).detach())
                    anchor, _ = control_aware_loss(anchored, b['native_action'], r.action_scale, scale)
                    if step == 1 and name == 'nominal' and not evidence:
                        evidence = dict(
                            predicted_actor=grad_sum(al, actor_params), predicted_R=grad_sum(al, r_params),
                            anchor_actor=grad_sum(anchor, actor_params), anchor_R=grad_sum(anchor, r_params))
                        if not (evidence['predicted_actor'] > 0 and evidence['predicted_R'] > 0
                                and evidence['anchor_actor'] == 0 and evidence['anchor_R'] > 0):
                            raise ValueError('joint gradient ownership failed: '+json.dumps(evidence))
                loss = (tl+al+anchor)*count/100
                if not torch.isfinite(loss):
                    raise FloatingPointError('joint loss')
                loss.backward()
                total += float(loss)
            trainable = actor_params+r_params
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError('joint gradient')
            optimizer.step()
            if step == 1 or step % 100 == 0:
                row = dict(step=step, loss=total, elapsed_s=time.monotonic()-started,
                           eta_s=(time.monotonic()-train_start)/step*(a.updates-step),
                           gpu=gpu_snapshot(torch.device('cuda')))
                atomic_json(out/'progress.json', row)
                print(json.dumps(row), flush=True)
            if step % 250 == 0 or step == a.updates:
                report = evaluate(step)
                current = contracts()
                checks = {}
                checks['exact_train/predicted_PD'] = report['exact_train']['PD_xyz_mm'] < initial['exact_train']['PD_xyz_mm']
                for name in ('nominal', 'startup', 'exact_train', 'exact_dev'):
                    checks[name+'/GT_PD'] = report[name]['GT_PD_p95_mm'] <= 1.1*initial[name]['GT_PD_p95_mm']+.1
                for name in ('nominal', 'startup'):
                    checks[name+'/PD'] = report[name]['PD_p95_mm'] <= 1.1*initial[name]['PD_p95_mm']+.1
                for name in ('exact_dev', 'contact_dev'):
                    checks[name+'/first8'] = report[name]['first8_mm'] <= 1.1*initial[name]['first8_mm']+.1
                    checks[name+'/PD_p95'] = report[name]['PD_p95_mm'] <= 1.1*initial[name]['PD_p95_mm']+.1
                for name in ('oakink2', 'grab', 'arctic'):
                    checks[name+'/trajectory'] = report[name]['all24_mm'] <= 1.1*initial[name]['all24_mm']+.1
                for name in ('exact_train', 'contact_train', 'exact_dev', 'contact_dev', 'nominal', 'startup'):
                    checks[name+'/clip'] = report[name]['clipping'] <= .01
                    checks[name+'/GT_clip'] = report[name]['GT_clipping'] <= .01
                score = (report['exact_train']['PD_xyz_mm']/max(initial['exact_train']['PD_xyz_mm'], .1)
                         + sum(report[n]['GT_PD_p95_mm']/max(initial[n]['GT_PD_p95_mm'], .1)
                               for n in ('nominal', 'startup')))/3
                curve.append(dict(step=step, reports=report, checks=checks, score=score, **current))
                atomic_json(out/'curve.json', curve)
                key = (not all(checks.values()), score)
                if best is None or key < best[0]:
                    digest, saved = save_pair(step, report, checks, evidence)
                    best = (key, step, report, checks, digest, saved)
                gross = any(report[n]['GT_PD_p95_mm'] > 2*initial[n]['GT_PD_p95_mm']+.1 for n in ('nominal', 'startup'))
                bad = bad+1 if gross else 0
                if bad >= 2:
                    raise RuntimeError('two gross GT-anchor regressions')
                pids = set(map(int, subprocess.check_output(
                    ['nvidia-smi', '-i', str(a.gpu), '--query-compute-apps=pid', '--format=csv,noheader'],
                    text=True).split()))
                if pids-{os.getpid()}:
                    raise RuntimeError('foreign GPU process')
            if time.monotonic()-started > 1500:
                raise TimeoutError('joint 1500s budget')
        r_diff = best[5]['R_max_difference']
        if r_diff == 0:
            raise ValueError('selected pair left R unchanged')
        passed = all(best[3].values())
        result = dict(status='COMPLETED', offline_gate=passed, selected_update=best[1],
                      final_checkpoint=str(out/'selected.pt'), checkpoint_sha256=sha(out/'selected.pt'),
                      training_R_sha256=best[4], initial=initial, reports=best[2], checks=best[3],
                      gradient_evidence=evidence, elapsed_s=time.monotonic()-started)
        atomic_json(out/'result.json', result)
        chunk_manifest = dict(status='COMPLETED', schema=CHUNK_SCHEMA, experiment_id=EXPERIMENT, run_id=chunk_out.name,
            git_commit=manifest['git_commit'], source=str(data.source), parent=str(chunk), parent_sha256=sha(chunk),
            source_sha256=sha(data.source/'trajectory.npz'), actor_checkpoint=str(out/'selected.pt'),
            collections=manifest['collections'], gpu=a.gpu, maximum_updates=a.updates,
            objective=manifest['loss'], training_allowed=False, joint_anchor=True)
        chunk_result = dict(status='COMPLETED', offline_gate=passed, selected_update=best[1],
                            checkpoint_sha256=best[4], reports=best[2], checks=best[3])
        atomic_json(chunk_out/'result.json', chunk_result)
        atomic_json(chunk_out/'manifest.json', chunk_manifest)
        manifest.update(status='COMPLETED', training_R_sha256=best[4])
        atomic_json(out/'manifest.json', manifest)
        print(json.dumps(dict(offline_gate=passed, selected_update=best[1], elapsed_s=result['elapsed_s'])), flush=True)
    except BaseException as error:
        manifest.update(status='FAILED', error=repr(error))
        atomic_json(out/'manifest.json', manifest)
        if chunk_out.exists():
            atomic_json(chunk_out/'manifest.json', dict(status='FAILED', error=repr(error), experiment_id=EXPERIMENT))
        raise


if __name__ == '__main__':
    main()

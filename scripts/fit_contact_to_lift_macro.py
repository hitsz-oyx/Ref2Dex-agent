#!/usr/bin/env python3
"""Matched GPU H10 feedback-law-conditioned physical information probe."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from run_paired_evaluator_resolution import sha, gpu_admission


def run(args):
    base = ROOT / 'src/task/CmResidual/research/contact_consequence/output'
    if args.output.resolve().parent != base.resolve() or args.output.exists() or args.output.is_symlink():
        raise ValueError('unique owned output required')
    qualification = json.loads(args.qualification.read_text())
    if not qualification['supervision_adequate']:
        raise ValueError('do not fit without frozen supervision qualification')
    extra = json.loads((args.additional / 'run_manifest.json').read_text())
    audit = json.loads(args.additional_audit.read_text())
    engineering = json.loads(args.engineering_audit.read_text())
    if extra['run_status'] != 'COMPLETED' or audit['run_status'] != 'COMPLETED' or engineering['run_status'] != 'COMPLETED':
        raise ValueError('terminal audited source required')
    prior_seconds = extra['cumulative_seconds'] + audit['elapsed_seconds'] + engineering['elapsed_seconds'] + qualification['elapsed_seconds']
    prior_bytes = extra['output_bytes'] + sum(p.stat().st_size for p in [args.additional_audit, args.engineering_audit, args.qualification])
    admission = gpu_admission(args.gpu); os.environ['CUDA_VISIBLE_DEVICES'] = admission['uuid']
    begin = time.monotonic()
    import torch
    from qualify_contact_to_lift_source import qualify
    from src.task.CmResidual.contact_to_lift_macro import ContactToLiftMacroModel, loss
    torch.set_num_threads(2); torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir()
    manifest = dict(experiment_id='P-20261002-contact-to-lift-macro', family='HF18', probe_index_in_family=1,
                    run_status='STARTED', pid=os.getpid(), command=sys.argv, gpu=admission,
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    prior_seconds=prior_seconds, prior_bytes=prior_bytes, wall_limit_seconds=3600,
                    output_limit_bytes=8 << 30, models=[], input_sha256={})
    def save(): (args.output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    save()
    try:
        raw, splits, counts, adequate, record_hashes = qualify(args.source, args.additional, args.additional_audit)
        if not adequate or counts != qualification['counts'] or record_hashes != qualification['input_sha256']:
            raise ValueError('qualified source drift')
        inputs = [Path(__file__), ROOT / 'scripts/qualify_contact_to_lift_source.py',
                  ROOT / 'src/task/CmResidual/contact_to_lift_macro.py',
                  ROOT / 'src/task/CmResidual/native_pd_selector.py',
                  ROOT / 'src/task/CmResidual/native_pd_consequence.py',
                  ROOT / 'src/task/CmResidual/native_pd_policy_controls.py',
                  ROOT / 'src/task/CmResidual/executable_contact_options.py',
                  ROOT / 'docs/decisions/D-20261002-contact-to-lift-macro.md',
                  ROOT / 'docs/experiments/probes/P-20261002-contact-to-lift-macro.md',
                  args.qualification, args.additional_audit, args.engineering_audit]
        for directory in [args.source, args.additional]:
            p = directory / 'run_manifest.json'; inputs.append(p)
            source = json.loads(p.read_text()); inputs += [Path(k) for k in source['input_sha256']]
        hashes = {str(p.resolve()): sha(p) for p in inputs}; hashes.update(record_hashes)
        manifest.update(run_status='RUNNING', input_sha256=hashes, counts=counts); save()
        data = {k: v.cuda() for k, v in raw.items()}; splits = {k: v.cuda() for k, v in splits.items()}
        normalization = {}
        for name in ['history', 'physical', 'native', 'goal', 'law']:
            x = data[name][splits['fit']]; dims = (0, 1) if name == 'history' else 0
            mean = x.mean(dims); std = x.std(dims, unbiased=False).clamp_min(.001)
            normalization[name+'_mean'] = mean; normalization[name+'_std'] = std
            data[name] = ((data[name]-mean)/std).clamp(-8, 8)
        fit = splits['fit'].nonzero().flatten(); models = {}; metrics = {}
        def check():
            if prior_seconds+time.monotonic()-begin > 3540: raise TimeoutError('whole slot budget')
            if prior_bytes+sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file()) > 8 << 30: raise ValueError('storage limit')
            if any(sha(Path(k)) != v for k, v in hashes.items()): raise ValueError('input drift')
        for mode in ['cm', 'state_only', 'shuffled']:
            trained, predictions = [], []
            for seed in [11601, 11602, 11603]:
                check(); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                model = ContactToLiftMacroModel(data['physical'].shape[-1], data['native'].shape[-1], mode).cuda()
                optimizer = torch.optim.Adam(model.parameters(), lr=.0003, weight_decay=.0001)
                generator = torch.Generator(device='cuda').manual_seed(seed+30000)
                goals = data['goal'].clone(); laws = data['law'].clone()
                if mode == 'shuffled':
                    shuffler = torch.Generator(device='cuda').manual_seed(seed+40000)
                    for condition in [False, True]:
                        rows = fit[data['early'][fit] == condition]
                        permuted = rows[torch.randperm(len(rows), device='cuda', generator=shuffler)]
                        goals[rows] = data['goal'][permuted]; laws[rows] = data['law'][permuted]
                model.train()
                for update in range(1000):
                    ids = fit[torch.randint(len(fit), (256,), device='cuda', generator=generator)]
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(data['history'][ids], data['physical'][ids], data['native'][ids], goals[ids], laws[ids], data['prior'][ids])
                    value = loss(prediction, data['target'][ids]); value.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
                    if not torch.isfinite(value): raise ValueError('nonfinite loss')
                model.eval().requires_grad_(False); outputs = []
                with torch.no_grad():
                    for offset in range(0, len(data['target']), 512):
                        s = slice(offset, offset+512)
                        outputs.append(model(data['history'][s], data['physical'][s], data['native'][s], data['goal'][s], data['law'][s], data['prior'][s]))
                predictions.append(torch.cat(outputs)); trained.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
                manifest['models'].append(dict(mode=mode, seed=seed, updates=1000, final_loss=float(value))); save()
                print(json.dumps(dict(mode=mode, seed=seed, status='COMPLETED')), flush=True)
                del model, optimizer
            mean = torch.stack(predictions).mean(0); report = {}
            for name, mask in [('held', splits['held']), ('held_early', splits['held'] & data['early']), ('cal_early', splits['cal'] & data['early'])]:
                target = data['target'][mask]; pred = mean[mask]; delta = pred-target
                report[name] = dict(rows=int(mask.sum()), height_rmse_mm=float(delta[:, :10].square().mean().sqrt()*10),
                                    clearance_mae_mm=float(delta[:, 10:20].abs().mean()*10),
                                    joint_brier=float((torch.sigmoid(pred[:, 20:30])-target[:, 20:30]).square().mean()),
                                    support_brier=float((torch.sigmoid(pred[:, 30])-target[:, 30]).square().mean()),
                                    loss_brier=float((torch.sigmoid(pred[:, 31])-target[:, 31]).square().mean()),
                                    new_lift_brier=float((torch.sigmoid(pred[:, 32])-target[:, 32]).square().mean()),
                                    persist_lift_brier=float((data['persist_lift'][mask]-target[:, 32]).square().mean()),
                                    joint_supported_height_mae_mm=float(delta[:, 33].abs().mean()*10))
            models[mode] = trained; metrics[mode] = report
        cm = metrics['cm']['held_early']; gates = dict(supervision=adequate, new_lift_no_worse_than_persist=cm['new_lift_brier'] <= cm['persist_lift_brier'])
        for control in ['state_only', 'shuffled']:
            other = metrics[control]['held_early']
            gates[control+'_height5pct'] = cm['height_rmse_mm'] <= .95*other['height_rmse_mm']
            gates[control+'_clearance5pct'] = cm['clearance_mae_mm'] <= .95*other['clearance_mae_mm']
            gates[control+'_new_lift'] = cm['new_lift_brier'] <= other['new_lift_brier']
            gates[control+'_joint_height'] = cm['joint_supported_height_mae_mm'] <= other['joint_supported_height_mae_mm']
        gates['passed'] = all(gates.values()); label = 'PROMISING' if gates['passed'] else 'UNPROMISING'
        checkpoint = dict(schema='ref2dex.contact_to_lift_macro.v1', physical_dim=data['physical'].shape[-1], native_dim=data['native'].shape[-1],
                          models=models, normalization={k: v.cpu() for k, v in normalization.items()}, input_sha256=hashes,
                          scope='H10 physical outcome conditional on fixed feedback law; pre-observation inputs, not unknown future motor commands')
        torch.save(checkpoint, args.output/'contact_to_lift_macro.pt'); check()
        result = dict(run_status='COMPLETED', label=label, metrics=metrics, gate=gates, counts=counts,
                      elapsed_seconds=time.monotonic()-begin, cumulative_seconds=prior_seconds+time.monotonic()-begin,
                      scoped_bytes=prior_bytes+sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file()),
                      checkpoint_sha256=sha(args.output/'contact_to_lift_macro.pt'), scope='factual H10 information; no ranking-regret/actual controller/RL/stable grasp claim')
        (args.output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
        manifest.update(run_status='COMPLETED', input_hashes_unchanged=True, result=result); print(json.dumps(result, indent=2))
    except BaseException as error:
        manifest.update(run_status='FAILED', error=repr(error)); raise
    finally:
        manifest['elapsed_seconds'] = time.monotonic()-begin; save()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'additional', 'additional-audit', 'engineering-audit', 'qualification', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0); run(p.parse_args())

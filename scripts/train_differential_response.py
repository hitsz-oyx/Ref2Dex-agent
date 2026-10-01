#!/usr/bin/env python3
"""Fixed endpoint predictive screen; never selects on held-out actor panels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from src.task.CmResidual.contact_response import sample_response
from scripts.run_contact_response_probe import ARMS, PANELS, admission, sha

METHODS = ('factual', 'differential', 'no_pulse')
SEEDS = (301, 302, 303)


def window_features(data, nominal_plan):
    triggers = data['triggers']
    env = torch.arange(len(triggers))
    indices = triggers[:, None] + torch.arange(-3, 1)[None]
    mask = indices >= 0
    history = data['state_before'][indices.clamp_min(0), env[:, None]] * mask[..., None]
    prefix = torch.cat((history.flatten(1), mask.float()), -1)
    plan_indices = triggers[:, None] + torch.arange(10)[None]
    plan = data['action'][plan_indices, env[:, None]].flatten(1)
    return torch.cat((prefix, plan), -1), torch.cat((prefix, nominal_plan), -1)


def prepare(run, actor):
    parts = {k: [] for k in ('x', 'x_no_pulse', 'y', 'motion')}
    inputs = {}
    for t, seed in PANELS:
        if t != actor:
            continue
        data = []
        for arm, _ in ARMS:
            path = run / f't{t}_s{seed}_{arm}/response.pt'
            result = json.loads((path.parent / 'results.json').read_text())
            if sha(path) != result['output_sha256']:
                raise ValueError('response hash drift')
            inputs[str(path)] = sha(path)
            data.append(torch.load(path, map_location='cpu', weights_only=False))
        triggers = data[0]['triggers']
        env = torch.arange(len(triggers))
        base_indices = triggers[:, None] + torch.arange(10)[None]
        nominal_plan = data[0]['action'][base_indices, env[:, None]].reshape(len(env), -1)
        features, blind, targets = [], [], []
        for d in data:
            if not torch.equal(d['triggers'], triggers):
                raise ValueError('trigger mismatch')
            full, no_pulse = window_features(d, nominal_plan)
            features.append(full)
            blind.append(no_pulse)
            targets.append(torch.cat([sample_response(d['state_before'], d['state_after'], triggers, h)[:, :3] for h in (5, 10)], -1))
        parts['x'].append(torch.stack(features, 1))
        parts['x_no_pulse'].append(torch.stack(blind, 1))
        parts['y'].append(torch.stack(targets, 1))
        parts['motion'].append(data[0]['motion_id'])
    return {k: torch.cat(v) for k, v in parts.items()}, inputs


def metrics(predicted, actual):
    result = {}
    for i, horizon in enumerate((5, 10)):
        part = slice(3 * i, 3 * (i + 1))
        error = predicted[:, part] - actual[:, part]
        result[str(horizon)] = dict(
            contrast_rmse_mm=float(error.square().sum(-1).mean().sqrt() * 1000),
            signed_z_mae_mm=float(error[:, 2].abs().mean() * 1000),
            z_sign_agreement=float((predicted[:, part][:, 2].sign() == actual[:, part][:, 2].sign()).float().mean()))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=4)
    args = parser.parse_args()
    if ROOT not in args.output.resolve().parents or args.output.exists():
        raise ValueError('unique output in isolated worktree required')
    physical = json.loads((args.input / 'analysis.json').read_text())
    if not all(physical['gates'][str(h)]['contrast_resolved'] for h in (5, 10)):
        raise ValueError('physical vector contrasts are unresolved')
    gpu = admission(args.gpu)
    # Admission UUID must match the CUDA-visible UUID selected by the launcher.
    import os
    if os.environ.get('CUDA_VISIBLE_DEVICES') != gpu['uuid']:
        raise ValueError('set CUDA_VISIBLE_DEVICES to the admitted GPU UUID')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('requires exactly one visible CUDA device')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True)
    started = time.monotonic()
    fit, fit_inputs = prepare(args.input, 286)
    test, test_inputs = prepare(args.input, 287)
    inputs = dict(fit_inputs, **test_inputs)
    inputs[str(Path(__file__).resolve())] = sha(__file__)
    design = ROOT / 'docs/experiments/probes/P-20261001-differential-response-learning.md'
    inputs[str(design)] = sha(design)
    manifest = dict(experiment_id='P-20261001-differential-response-learning', run_id=args.output.name,
                    run_status='RUNNING', pid=os.getpid(), command=sys.argv, gpu=gpu,
                    input_sha256=inputs, fixed_updates=1000, seeds=list(SEEDS),
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    fit_actor=286, test_actor=287, wall_limit_seconds=600, phases=[])
    manifest['physical_pilot_label'] = physical['label']
    manifest['decision'] = 'D-20261001-vector-response-learning; preserves failed monotone-lift gate'
    def save():
        manifest['wall_seconds'] = time.monotonic() - started
        (args.output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    save()
    try:
        mean = fit['x'].flatten(0, 1).mean(0)
        std = fit['x'].flatten(0, 1).std(0).clamp_min(.01)
        target_scale = fit['y'].flatten(0, 1).std(0).clamp_min(.001)
        delta = fit['y'][:, 2] - fit['y'][:, 3]
        delta_scale = delta.square().mean(0).sqrt().clamp_min(.001)
        fit = {k: v.cuda() for k, v in fit.items()}
        test = {k: v.cuda() for k, v in test.items()}
        mean, std, target_scale, delta_scale = (x.cuda() for x in (mean, std, target_scale, delta_scale))
        true_contrast = test['y'][:, 2] - test['y'][:, 3]
        constant = delta.mean(0).cuda().expand(len(true_contrast), -1)
        by_motion = torch.stack([delta[fit['motion'].cpu() == m].mean(0) for m in range(3)]).cuda()
        baseline = dict(global_fit_mean=metrics(constant, true_contrast),
                        motion_fit_mean=metrics(by_motion[test['motion']], true_contrast))
        results, predictions = [], {}
        for seed in SEEDS:
            for method in METHODS:
                if time.monotonic() - started > 570:
                    raise TimeoutError('learning screen wall budget')
                torch.manual_seed(seed)
                model = torch.nn.Sequential(torch.nn.Linear(len(mean), 128), torch.nn.SiLU(),
                                            torch.nn.Linear(128, 128), torch.nn.SiLU(),
                                            torch.nn.Linear(128, 6)).cuda()
                optimizer = torch.optim.Adam(model.parameters(), lr=.001)
                sampler = torch.Generator(device='cuda').manual_seed(seed + 10000)
                feature_key = 'x_no_pulse' if method == 'no_pulse' else 'x'
                x = (fit[feature_key] - mean) / std
                y = fit['y'] / target_scale
                losses = []
                for step in range(1000):
                    if step % 100 == 0 and time.monotonic() - started > 570:
                        raise TimeoutError('learning screen wall budget')
                    indices = torch.randint(len(x), (96,), device='cuda', generator=sampler)
                    output = model(x[indices])
                    loss = (output - y[indices]).square().mean()
                    if method == 'differential':
                        predicted_delta = (output[:, 2] - output[:, 3]) * target_scale
                        target_delta = fit['y'][indices, 2] - fit['y'][indices, 3]
                        loss = loss + ((predicted_delta - target_delta) / delta_scale).square().mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite training loss')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    if step in (0, 499, 999):
                        losses.append(dict(update=step + 1, loss=float(loss.detach())))
                model.eval()
                with torch.no_grad():
                    predicted = model((test[feature_key] - mean) / std) * target_scale
                    contrast = predicted[:, 2] - predicted[:, 3]
                    score = metrics(contrast, true_contrast)
                    factual_rmse_mm = float((predicted - test['y']).square().sum(-1).mean().sqrt() * 1000)
                key = f'{method}_s{seed}'
                predictions[key] = contrast.cpu()
                results.append(dict(method=method, seed=seed, contrast=score,
                                    factual_rmse_mm=factual_rmse_mm, fixed_updates=1000, losses=losses))
                torch.save(dict(model=model.cpu().state_dict(), mean=mean.cpu(), std=std.cpu(),
                                target_scale=target_scale.cpu(), method=method, seed=seed), args.output / (key + '.pt'))
                manifest['phases'].append(dict(method=method, seed=seed, run_status='COMPLETED'))
                save()
                print(json.dumps(results[-1]), flush=True)
                del model, optimizer
        averaged = {method: {str(h): {metric: sum(r['contrast'][str(h)][metric] for r in results if r['method'] == method) / 3
                                     for metric in ('contrast_rmse_mm', 'z_sign_agreement')}
                            for h in (5, 10)} for method in METHODS}
        gates = {}
        for h in ('5', '10'):
            diff = averaged['differential'][h]
            factual = averaged['factual'][h]['contrast_rmse_mm']
            constant_error = baseline['global_fit_mean'][h]['contrast_rmse_mm']
            gates[h] = dict(beats_factual10pct=diff['contrast_rmse_mm'] <= .9 * factual,
                            beats_constant10pct=diff['contrast_rmse_mm'] <= .9 * constant_error,
                            z_sign75pct=diff['z_sign_agreement'] >= .75)
        passed = all(v for gate in gates.values() for v in gate.values())
        report = dict(experiment_id=manifest['experiment_id'], run_status='COMPLETED',
                      label='PROMISING' if passed else 'UNPROMISING',
                      scope='predictive mechanism Probe; no policy utility or object generalization',
                      fit_windows=len(fit['x']), test_windows=len(test['x']), gates=gates,
                      results=results, averaged=averaged, baselines=baseline,
                      wall_seconds=time.monotonic() - started, input_sha256=inputs)
        torch.save(dict(predictions=predictions, actual=true_contrast.cpu(), motion=test['motion'].cpu()), args.output / 'predictions.pt')
        if any(sha(Path(p)) != h for p, h in inputs.items()):
            raise ValueError('input/source drift')
        (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
        manifest.update(run_status='COMPLETED', label=report['label'], inputs_unchanged=True)
        print(json.dumps(dict(label=report['label'], gates=gates, baselines=baseline)), flush=True)
    except BaseException as error:
        manifest.update(run_status='FAILED', error=repr(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()

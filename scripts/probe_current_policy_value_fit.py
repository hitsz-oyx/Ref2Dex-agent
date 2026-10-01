#!/usr/bin/env python3
"""Bounded GPU-only factual V adaptation; frozen labels, episode holdout."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scripts.audit_current_policy_value import (
    DiagnosticError, _groups, _metric, _phase_labels, _primary_labels,
    frozen_critic_lambda_targets, load_collections, sha256, validate_diagnostic_schema,
)
from src.task.CmResidual.physical_value_data import HISTORY
from src.task.CmResidual.physical_value_models import Features, OutcomeNetwork

SPLIT_SEED = 20261001
UPDATES = 1000
BATCH = 128


def episode_split(data, groups, split_seed=SPLIT_SEED, hold_per_motion=6):
    """Pair motion/env assignment across checkpoints; never inspect outcomes."""
    import hashlib
    panels = {}
    records = []
    for (checkpoint, episode), indices in groups:
        idx = torch.tensor(indices)
        motion = data['motion_id'][idx].unique()
        env = data['env_id'][idx].unique()
        if len(motion) != 1 or len(env) != 1:
            raise DiagnosticError('episode_motion_or_env_changes')
        key = (int(motion[0]), int(env[0]))
        if key in panels.setdefault(checkpoint, set()):
            raise DiagnosticError('duplicate_panel_assignment')
        panels[checkpoint].add(key)
        records.append(dict(checkpoint=checkpoint, episode_id=episode,
                            motion_id=key[0], env_id=key[1], rows=len(indices)))
    sets = list(panels.values())
    if not sets or any(s != sets[0] for s in sets):
        raise DiagnosticError('unpaired_checkpoint_panels')
    held = set()
    for motion in sorted({m for m, e in sets[0]}):
        keys = sorted(k for k in sets[0] if k[0] == motion)
        if len(keys) <= hold_per_motion:
            raise DiagnosticError('insufficient_fit_episodes')
        keys.sort(key=lambda k: hashlib.sha256(
            f'{split_seed}:{k[0]}:{k[1]}'.encode()).hexdigest())
        held.update(keys[:hold_per_motion])
    hold = torch.zeros(len(data['reward']), dtype=torch.bool)
    for record, (_, indices) in zip(records, groups):
        selected = (record['motion_id'], record['env_id']) in held
        hold[indices] = selected
        record['split'] = 'holdout' if selected else 'fit'
    return ~hold, hold, records


def history_rows(data, checkpoint, indices):
    offsets = torch.arange(HISTORY - 1, -1, -1, device=indices.device)
    history = indices[:, None] - offsets
    safe = history.clamp_min(0)
    mask = history >= 0
    for tensor in (data['episode_id'], checkpoint):
        mask &= tensor[safe] == tensor[indices, None]
    mask &= data['step'][safe] == data['step'][indices, None] - offsets
    return safe, mask[:, :, None].float()


def cached_history(cache, indices):
    safe, mask = history_rows(cache, cache['checkpoint_seed'], indices)
    return torch.cat((cache['state_features'][safe], cache['previous_action'][safe], mask), -1) * mask


def restore_value_adam(network, joint_saved):
    """Saved joint optimizer is ordered V followed by Q; retain V moments."""
    saved = copy.deepcopy(joint_saved)
    params = list(network.parameters())
    if len(saved['param_groups']) != 1:
        raise DiagnosticError('unexpected_optimizer_groups')
    group = saved['param_groups'][0]
    ids = group['params']
    if len(ids) != 2 * len(params) or len(set(ids)) != len(ids):
        raise DiagnosticError('unexpected_joint_optimizer_order')
    value_ids = ids[:len(params)]
    for pid, parameter in zip(value_ids, params):
        state = saved['state'].get(pid, {})
        if set(state) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise DiagnosticError('unexpected_adam_state')
        for name in ('exp_avg', 'exp_avg_sq'):
            if state[name].shape != parameter.shape or not torch.isfinite(state[name]).all():
                raise DiagnosticError('adam_parameter_shape_or_finite')
    group['params'] = value_ids
    saved['state'] = {pid: saved['state'][pid] for pid in value_ids}
    optimizer = torch.optim.Adam(params, lr=group['lr'])
    optimizer.load_state_dict(saved)
    return optimizer


@torch.no_grad()
def predict(network, cache, selected, check):
    values = []
    network.eval()
    for indices in selected.split(512):
        check()
        values.append(network(cached_history(cache, indices), cache['context'][indices]).flatten().cpu())
    result = torch.cat(values)
    if not torch.isfinite(result).all():
        raise DiagnosticError('nonfinite_prediction')
    return result


def fit_value(network, optimizer, cache, target, fit_indices, generator, scale,
              updates, check, on_update=None):
    network.train()
    for update in range(1, updates + 1):
        check()
        indices = fit_indices[torch.randint(len(fit_indices), (BATCH,),
                                            device=fit_indices.device, generator=generator)]
        prediction = network(cached_history(cache, indices), cache['context'][indices]).flatten()
        loss = ((prediction - target[indices]) / scale).square().mean()
        if not torch.isfinite(loss):
            raise DiagnosticError('nonfinite_loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(network.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
        if on_update and update % 250 == 0:
            on_update(update, float(loss.detach()), float(norm))


def metrics(data, prediction, target, mc, selected, fit, hold, phase, success):
    out = {}
    episode = data['episode_id']
    for name, split in (('fit', fit), ('holdout', hold)):
        active = split[selected]
        out[name] = {}
        strata = {'all': active, 'primary_success': active & success[selected],
                  'primary_failure': active & ~success[selected]}
        for i, label in enumerate(('pre_contact', 'contact', 'held_1s', 'drop_edge', 'other')):
            strata[label] = active & (phase[selected] == i)
        for label, mask in strata.items():
            idx = selected[mask]
            out[name][label] = {
                'gae': _metric(prediction[mask], target[idx], episode[idx].tolist()),
                'mc': _metric(prediction[mask], mc[idx], episode[idx].tolist()),
            }
    return out


def run(args):
    started = time.monotonic()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(run_status='STARTED', experiment_id='P-20261001-current-policy-v-fit',
                    run_id='r1', pid=os.getpid(), command=sys.argv,
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    gpu_visibility=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    updates_per_checkpoint=UPDATES, batch_size=BATCH, split_seed=SPLIT_SEED,
                    budget_seconds=1200, budget_bytes=1 << 30, cpu_threads=2)
    def write_manifest():
        (output / 'runtime_manifest.json').write_text(json.dumps(manifest, indent=2))
    def check():
        if time.monotonic() - started > 1170:
            raise DiagnosticError('walltime_budget')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) > (1 << 30):
            raise DiagnosticError('storage_budget')
    write_manifest()
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise DiagnosticError('requires_exactly_one_visible_gpu')
        device = torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        checkpoints = {286: args.checkpoint286, 287: args.checkpoint287}
        inputs = [Path(__file__), ROOT / 'scripts/audit_current_policy_value.py',
                  ROOT / 'src/task/CmResidual/physical_value_models.py',
                  ROOT / 'src/task/CmResidual/physical_value_data.py',
                  ROOT / 'src/task/CmResidual/physical_value_contract.py',
                  ROOT / 'src/task/CmResidual/dexplore_physical_value_agent.py',
                  ROOT / 'src/task/CmResidual/v118_planner.py',
                  args.pretrained, args.baseline_report, *checkpoints.values()]
        for collection in args.collection:
            inputs += [collection / 'results.json', collection / 'runtime_manifest.json']
            payload = json.loads((collection / 'results.json').read_text())
            inputs += [Path(s['path']) for s in payload['shards']]
        hashes = {str(p.resolve()): sha256(p) for p in inputs}
        manifest.update(inputs_sha256=hashes, gpu_name=torch.cuda.get_device_name(0), run_status='RUNNING')
        write_manifest()
        data = load_collections(args.collection)
        contract = validate_diagnostic_schema(data)
        cp = contract['checkpoint_id']
        groups = _groups(data, cp)
        if len(groups) != 192 or set(cp.tolist()) != {286, 287}:
            raise DiagnosticError('unexpected_frozen_panel')
        target, target_meta = frozen_critic_lambda_targets(data, groups, cp)
        fit, hold, split_records = episode_split(data, groups)
        for seed in checkpoints:
            panel = [r for r in split_records if r['checkpoint'] == seed]
            if len(panel) != 96 or sum(r['split'] == 'holdout' for r in panel) != 18:
                raise DiagnosticError('unexpected_split_counts')
            for motion in {r['motion_id'] for r in panel}:
                if sum(r['motion_id'] == motion for r in panel) != 32:
                    raise DiagnosticError('unexpected_motion_panel')
        package = torch.load(args.pretrained, map_location='cpu', weights_only=False)
        features = Features(**{k: v.to(device) for k, v in package['stats'].items()}, device=device).to(device)
        scale = float(package['return_scale'])
        if scale <= 0 or not torch.isfinite(torch.tensor(scale)):
            raise DiagnosticError('invalid_return_scale')
        cache = {key: data[key].to(device) for key in ('episode_id', 'step', 'previous_action')}
        cache['checkpoint_seed'] = cp.to(device)
        with torch.no_grad():
            cache['state_features'] = torch.cat([features.state(x.to(device)) for x in data['state'].split(2048)])
            cache['context'] = features.context(data['context'].to(device))
            # Include all episode starts and later steps in cache equivalence check.
            sampled = sorted(set([i for _, ix in groups for i in ix[:2]] + list(range(0, len(target), 257))))
            idx = torch.tensor(sampled, device=device)
            safe, mask = history_rows(cache, cache['checkpoint_seed'], idx)
            states = data['state'].to(device)[safe] * mask
            actions = cache['previous_action'][safe] * mask
            direct = features.history(states, actions, mask)
            max_error = float((direct - cached_history(cache, idx)).abs().max())
            if max_error > 1e-5:
                raise DiagnosticError('history_cache_equivalence')
        primary, primary_drop = _primary_labels(data, groups)
        success = torch.zeros(len(target), dtype=torch.bool)
        for key, indices in groups:
            success[indices] = primary[key]
        for record in split_records:
            key = (record['checkpoint'], record['episode_id'])
            record.update(primary_success=primary[key], drop_after_primary=primary_drop[key])
        (output / 'split.json').write_text(json.dumps(split_records, indent=2))
        phase = _phase_labels(data)
        target_gpu = target.to(device)
        baseline_reference = json.loads(args.baseline_report.read_text())['diagnostic']
        results = {}
        def snapshot(network, selected):
            prediction = predict(network, cache, selected.to(device), check)
            return prediction, metrics(data, prediction, target, data['return'], selected, fit, hold, phase, success)
        for seed, path in checkpoints.items():
            check()
            online = torch.load(path, map_location='cpu', weights_only=False)
            physical = online['physical_value']
            if physical['arm'] != 'cm_value':
                raise DiagnosticError('checkpoint_provenance')
            network = OutcomeNetwork(int(package['context_dim']), 0, 1, 9283).to(device)
            network.load_state_dict(physical['value'])
            optimizer = restore_value_adam(network, physical['value_optimizer'])
            before_steps = sorted({float(s['step']) for s in optimizer.state.values()})
            selected = (cp == seed).nonzero().flatten()
            prediction, baseline = snapshot(network, selected)
            actual = float((prediction - target[selected]).square().mean().sqrt())
            expected = baseline_reference['metrics']['saved_pv_v']['checkpoint'][str(seed)]['lambda']['rmse']
            if abs(actual - expected) > .001:
                raise DiagnosticError(f'baseline_drift:{seed}:{actual}:{expected}')
            ppo = metrics(data, data['value_at_state'][selected].flatten(), target, data['return'], selected,
                          fit, hold, phase, success)
            results[str(seed)] = dict(baseline=baseline, ppo_critic=ppo, stages={},
                                     optimizer_step_before=before_steps, reproduced_rmse=actual,
                                     source_actor_initial_model_sha256=physical['initial_model_sha256'])
            fit_indices = (fit & (cp == seed)).nonzero().flatten().to(device)
            generator = torch.Generator(device=device).manual_seed(SPLIT_SEED + seed)
            def on_update(update, loss, norm):
                prediction, stage = snapshot(network, selected)
                results[str(seed)]['stages'][str(update)] = stage
                record = dict(checkpoint=seed, update=update, loss=loss, gradient_norm=norm,
                              fit_rmse=stage['fit']['all']['gae']['rmse'],
                              holdout_rmse=stage['holdout']['all']['gae']['rmse'],
                              elapsed_seconds=time.monotonic() - started)
                with (output / 'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(record) + '\n')
                print(json.dumps(record), flush=True)
                network.train()
            fit_value(network, optimizer, cache, target_gpu, fit_indices, generator, scale, UPDATES, check, on_update)
            after_steps = sorted({float(s['step']) for s in optimizer.state.values()})
            if after_steps != [s + UPDATES for s in before_steps]:
                raise DiagnosticError('optimizer_step_mismatch')
            results[str(seed)]['optimizer_step_after'] = after_steps
            torch.save(dict(value=network.cpu().state_dict(), value_optimizer=optimizer.state_dict(),
                            origin_sha256=hashes[str(path.resolve())], updates=UPDATES), output / f'value_s{seed}.pt')
            del network, optimizer, online, physical
        if any(sha256(Path(p)) != digest for p, digest in hashes.items()):
            raise DiagnosticError('input_drift')
        reductions = {}
        for seed, result in results.items():
            reductions[seed] = {split: 1 - result['stages']['1000'][split]['all']['gae']['rmse'] /
                               result['baseline'][split]['all']['gae']['rmse'] for split in ('fit', 'holdout')}
        label = 'PROMISING' if all(v >= .2 for r in reductions.values() for v in r.values()) else 'UNPROMISING'
        report = dict(run_status='COMPLETED', label=label, rows=len(target), episodes=len(groups),
                      hypothesis='same V can adapt to current-policy frozen GAE with episode generalization',
                      target_metadata=target_meta, return_scale=scale, history_cache_max_abs_error=max_error,
                      rmse_reductions=reductions, checkpoints=results,
                      scope='offline factual V-only fit; no PPO/Cm/actor update or utility claim; held_1s is a diagnostic phase',
                      elapsed_seconds=time.monotonic() - started,
                      gpu_peak_memory_bytes=torch.cuda.max_memory_allocated())
        torch.save(dict(target=target, fit=fit, holdout=hold, checkpoint=cp,
                        episode_id=data['episode_id'], step=data['step']), output / 'frozen_targets.pt')
        (output / 'results.json').write_text(json.dumps(report, indent=2))
        manifest.update(run_status='COMPLETED', label=label, elapsed_seconds=time.monotonic()-started,
                        input_hashes_unchanged=True)
        write_manifest()
        check()
        print(json.dumps({k: report[k] for k in ('label', 'rmse_reductions', 'elapsed_seconds', 'gpu_peak_memory_bytes')}), flush=True)
    except BaseException as error:
        manifest.update(run_status='FAILED', error=repr(error), elapsed_seconds=time.monotonic()-started)
        write_manifest()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, action='append', required=True)
    parser.add_argument('--pretrained', type=Path, required=True)
    parser.add_argument('--checkpoint286', type=Path, required=True)
    parser.add_argument('--checkpoint287', type=Path, required=True)
    parser.add_argument('--baseline-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()

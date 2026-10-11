"""Recover historical Adam exactly, then compare one-update LR forks on saved data."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(TASK/'src'), str(ROOT/'src/task/consequence-evaluator/src')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists() or ROOT/'outputs/trajectory-policy' not in args.output.parents:
        raise ValueError('fresh task-owned output required')
    source = json.loads((args.training/'manifest.json').read_text())
    evaluation = json.loads((args.evaluation/'manifest.json').read_text())
    if source['status'] != 'COMPLETED' or source['engineering_smoke'] or source['updates'] != 24 or source['gae_lambda'] != 1. or source['matmul_precision'] != 'highest':
        raise ValueError('fixed full lambda1 training required')
    if evaluation['status'] != 'COMPLETED' or evaluation.get('action_mode') != 'paired_gaussian':
        raise ValueError('frozen measured paired-policy histories required')
    hashes = dict(source['input_sha256'])
    hashes.update(evaluation['input_sha256'])
    files = [args.training/name for name in ('manifest.json', 'high.npz', 'low.npz', 'monitor.jsonl', 'final.pt')]
    files += sorted((args.training/'actors').glob('u*.pt'))
    files += [args.evaluation/'manifest.json', args.evaluation/'plans.npz', Path(__file__)]
    for path, digest in hashes.items():
        if sha(path) != digest:
            raise ValueError('frozen input/source drift: '+path)
    hashes.update({str(path.resolve()): sha(path) for path in files})
    if sha(args.training/'final.pt') != source['final_checkpoint_sha256']:
        raise ValueError('final checkpoint drift')
    with np.load(args.training/'high.npz') as f:
        high = {key: f[key].copy() for key in ('history', 'c', 'mean', 'std', 'logp', 'value', 'reward', 'done', 'duration', 'bootstrap', 'low_index', 'advantage')}
    with np.load(args.training/'low.npz') as f:
        starts, held = f['episode_start'].copy(), f['held'].copy()
    if high['history'].shape != (384, 16, 328) or held.shape != (3062, 16):
        raise ValueError('fixed complete training trace required')
    selected = []
    for start in np.unique(starts):
        streak, longest = np.zeros(16, int), np.zeros(16, int)
        for flags in held[starts == start]:
            streak = np.where(flags, streak+1, 0)
            longest = np.maximum(longest, streak)
        query = np.flatnonzero(high['low_index'] == start)
        if len(query) != 1:
            raise ValueError('unique startup query required')
        selected.extend(dict(high_index=int(query[0]), row=int(row), maximum_held=int(longest[row])) for row in np.flatnonzero(longest >= 45))
    if len(selected) != 12:
        raise ValueError('predeclared all twelve long-held exploratory rows differ')
    with np.load(args.evaluation/'plans.npz') as f:
        rows = np.asarray(evaluation['roles']) == 'warm_start'
        diagnostic_h = f['history'][f['tick'] <= 40][:, rows].reshape(-1, 328).copy()
    gpu = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], text=True).strip()
    utilization, memory = map(int, gpu.split(','))
    if utilization > 10 or memory > 512:
        raise ValueError('GPU not idle: '+gpu)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONDONTWRITEBYTECODE='1')
    import torch
    from trajectory_policy.actor import load_actor
    from trajectory_policy.ppo import HistoryValue, update
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    args.output.mkdir(parents=True)
    run = dict(status='RUNNING', git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        input_sha256=hashes, physical_gpu=args.gpu, gpu_before=gpu,
        historical_adam='Reconstructed from empty u0 Adam by every actual saved update; bitwise actor/value checkpoint match required',
        fork='Each of all24 updates forks from actual before-state and recovered moments, resets actor LR to1e-6 for this update only; other contracts unchanged',
        selected=selected, claim='Matched one-update optimization mechanism only, no on-policy training or physical benefit')
    write(args.output/'manifest.json', run)
    started = time.monotonic()
    try:
        packet = torch.load(args.training/'actors/u0000.pt', map_location='cuda', weights_only=False)
        actor = load_actor(packet, 'cuda').train()
        value = HistoryValue(actor.history_mean, actor.history_scale).to('cuda')
        value.load_state_dict(packet['value_model'])
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-6)
        value_optimizer = torch.optim.Adam(value.parameters(), lr=3e-4)
        logs = [json.loads(line) for line in (args.training/'monitor.jsonl').read_text().splitlines()]
        probe_h = torch.as_tensor(diagnostic_h, device='cuda')
        records = []

        def equal_checkpoint(path):
            saved = torch.load(path, map_location='cuda', weights_only=False)
            for group, model in (('model', actor), ('value_model', value)):
                for key, actual in model.state_dict().items():
                    if not torch.equal(actual, saved[group][key]):
                        error = float((actual-saved[group][key]).abs().max())
                        raise ValueError('historical reconstruction differs: '+str(path)+' '+group+'/'+key+' max='+str(error))

        for u in range(24):
            equal_checkpoint(args.training/'actors'/('u%04d.pt' % u))
            batch = {key: torch.as_tensor(high[key][u*16:(u+1)*16], device='cuda') for key in
                ('history', 'c', 'mean', 'std', 'logp', 'value', 'reward', 'done', 'duration')}
            bootstrap = torch.as_tensor(high['bootstrap'][u*16], device='cuda')
            fork_actor, fork_value = copy.deepcopy(actor), copy.deepcopy(value)
            fork_optimizer = torch.optim.Adam(fork_actor.parameters(), lr=1e-6)
            fork_optimizer.load_state_dict(copy.deepcopy(actor_optimizer.state_dict()))
            fork_optimizer.param_groups[0]['lr'] = 1e-6
            fork_value_optimizer = torch.optim.Adam(fork_value.parameters(), lr=3e-4)
            fork_value_optimizer.load_state_dict(copy.deepcopy(value_optimizer.state_dict()))
            with torch.no_grad():
                before_probe = actor(probe_h).detach().clone()
                before_std = actor.log_std.detach().clone()
            historical = update(actor, value, batch, bootstrap, actor_optimizer, value_optimizer, gae_lambda=1.)
            with torch.no_grad():
                actor.log_std.clamp_(-4.605170186, -1.609437912)
            equal_checkpoint(args.training/'final.pt' if u == 23 else args.training/'actors'/('u%04d.pt' % (u+1)))
            if historical['actor_steps'] != logs[u]['actor_steps'] or historical['actor_lr'] != logs[u]['actor_lr']:
                raise ValueError('historical accepted update or LR differs')
            if not np.array_equal(historical['advantage'].cpu().numpy(), high['advantage'][u*16:(u+1)*16]):
                raise ValueError('historical advantage differs')
            forked = update(fork_actor, fork_value, batch, bootstrap, fork_optimizer, fork_value_optimizer, gae_lambda=1.)
            with torch.no_grad():
                fork_actor.log_std.clamp_(-4.605170186, -1.609437912)
            normalized = historical['advantage'].flatten()
            normalized = (normalized-normalized.mean())/normalized.std(unbiased=False).clamp_min(1e-6)
            h, c = batch['history'].flatten(0, 1), batch['c'].flatten(0, 1)
            old = torch.distributions.Normal(batch['mean'].flatten(0, 1), batch['std'].flatten(0, 1))
            arm_records = {}
            with torch.no_grad():
                for label, model, optimization in (('historical', actor, historical), ('reset_lr', fork_actor, forked)):
                    dist = model.distribution(h)
                    logp = dist.log_prob(c).sum(-1)
                    ratio = (logp-batch['logp'].flatten()).clamp(-20, 20).exp()
                    kl = torch.distributions.kl_divergence(old, dist).reshape(-1, 24, 12).mean(0)
                    xyz = (model(probe_h)-before_probe).reshape(-1, 24, 12)[:, :8, :3]
                    good = []
                    for row in selected:
                        if row['high_index']//16 == u:
                            index = row['high_index'] % 16 * 16 + row['row']
                            good.append(dict(row, normalized_advantage=float(normalized[index]),
                                logprob_change=float(logp[index]-batch['logp'].flatten()[index])))
                    arm_records[label] = dict(actor_steps=optimization['actor_steps'], actor_lr=optimization['actor_lr'],
                        joint_kl=float(kl.sum()), prefix_kl=float(kl[:8].sum()), suffix_kl=float(kl[8:].sum()),
                        xyz_kl=float(kl[:, :3].sum()), rotation_kl=float(kl[:, 3:6].sum()), fingers_kl=float(kl[:, 6:].sum()),
                        clipped_ratio_fraction=float(((ratio < .8) | (ratio > 1.2)).float().mean()),
                        surrogate=float(torch.minimum(ratio*normalized, ratio.clamp(.8, 1.2)*normalized).mean()),
                        precontact_xyz_mean_movement_mm=float(xyz.square().mean().sqrt()*10),
                        changed_std_coordinates=int((model.log_std != before_std).sum()),
                        maximum_log_std_change=float((model.log_std-before_std).abs().max()), long_held_startups=good)
            records.append(dict(update=u, **arm_records))
            if u == 0 or (u+1) % 8 == 0:
                processes = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True).strip()
                for process in processes.splitlines():
                    pid, used = map(int, process.split(','))
                    if pid != os.getpid() and used > 512:
                        raise RuntimeError('foreign GPU compute detected')
                records[-1]['monitor'] = dict(gpu=subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], text=True).strip(),
                    eta_s=(time.monotonic()-started)/(u+1)*(23-u))
            print(json.dumps(records[-1]), flush=True)
        if any(sha(path) != digest for path, digest in hashes.items()):
            raise ValueError('frozen replay input drift')
        aggregate = {}
        for label in ('historical', 'reset_lr'):
            arms = [r[label] for r in records]
            good = [row for arm in arms for row in arm['long_held_startups']]
            aggregate[label] = dict(accepted_epochs=sum(arm['actor_steps'] for arm in arms),
                median_joint_kl=float(np.median([arm['joint_kl'] for arm in arms])),
                median_actor_lr=float(np.median([arm['actor_lr'] for arm in arms])),
                median_precontact_xyz_movement_mm=float(np.median([arm['precontact_xyz_mean_movement_mm'] for arm in arms])),
                positive_startup_advantages=sum(row['normalized_advantage'] > 0 for row in good),
                startup_probability_increases=sum(row['logprob_change'] > 0 for row in good),
                mean_startup_logprob_change=float(np.mean([row['logprob_change'] for row in good])),
                median_surrogate=float(np.median([arm['surrogate'] for arm in arms])))
        promising_updates = sum(r['reset_lr']['precontact_xyz_mean_movement_mm'] >= 2*r['historical']['precontact_xyz_mean_movement_mm']
            and r['reset_lr']['joint_kl'] <= .02 and r['reset_lr']['surrogate'] > r['historical']['surrogate'] for r in records)
        signal_label = 'PROMISING' if promising_updates >= 12 else 'UNCLEAR'
        report = dict(status=signal_label, exact_historical_actor_value_matches=25,
            promising_updates=promising_updates, aggregate=aggregate, records=records,
            metric='XYZ per-coordinate RMS over first8 plan frames at all168 saved warm H, ticks0:8:40; raw coordinate mean change, no physical outcome',
            elapsed_s=time.monotonic()-started, claim=run['claim'])
        write(args.output/'result.json', report)
        run.update(status='COMPLETED', elapsed_s=report['elapsed_s'])
        write(args.output/'manifest.json', run)
        print(json.dumps(dict(status=signal_label, promising_updates=promising_updates, aggregate=aggregate), indent=2), flush=True)
    except BaseException as error:
        run.update(status='FAILED', elapsed_s=time.monotonic()-started, error=repr(error))
        write(args.output/'manifest.json', run)
        raise


if __name__ == '__main__':
    def deadline(signum, frame):
        raise TimeoutError('step replay exceeded120s')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(120)
    main()

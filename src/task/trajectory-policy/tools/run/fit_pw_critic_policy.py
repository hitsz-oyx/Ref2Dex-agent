"""Matched frozen-PW Q fits and bounded advantage-weighted actor improvement."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(TASK / 'src'), str(ROOT / 'src/task/consequence-evaluator/src')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--initial', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    a = p.parse_args()
    cache, initial, out = a.cache.resolve(), a.initial.resolve(), a.output.resolve()
    if out.exists() or ROOT / 'outputs/trajectory-policy' not in out.parents:
        raise ValueError('fresh task-owned fit required')
    if subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-compute-apps=pid',
                                '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU occupied')
    scratch = ROOT / 'tmp/trajectory-policy/pw-critic'
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), TMPDIR=str(scratch))
    import torch
    from trajectory_policy.reference_actor import SCHEMA as ACTOR_SCHEMA, load_reference_actor
    from trajectory_policy.pw_critic import SCHEMA, PhysicalQ, critic_input, advantage_weights
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(188)
    np.random.seed(188)
    cm = json.loads((cache / 'manifest.json').read_text())
    im = json.loads((initial.parent / 'manifest.json').read_text())
    if (cm['status'] != 'COMPLETED' or not cm['training_allowed'] or not cm['frozen_pw']
            or sha(cache / 'windows.npz') != cm['windows_sha256']
            or sha(initial) != '7c2d1fe98a27c7c3ba72ba44a217c631aea5e46b0c74384a0cc7c9d4f692677e'):
        raise ValueError('authenticated eligible cache and validated initialization required')
    with np.load(cache / 'windows.npz', allow_pickle=False) as f:
        d = {k: f[k].copy() for k in f.files}
    train, val = np.flatnonzero(d['train']), np.flatnonzero(~d['train'])
    target_mean, target_scale = float(d['returns'][train].mean()), max(float(d['returns'][train].std()), .001)
    y = torch.tensor((d['returns'] - target_mean) / target_scale, device='cuda')
    paths = [cache / 'manifest.json', cache / 'windows.npz', initial, initial.parent / 'manifest.json',
             Path(im['retargeter']), Path(im['reference']), Path(__file__), TASK / 'src/trajectory_policy/pw_critic.py',
             TASK / 'src/trajectory_policy/reference_actor.py', TASK / 'src/trajectory_policy/competent_actor.py']
    hashes = {str(path): sha(path) for path in paths}
    out.mkdir(parents=True)
    manifest = dict(schema=SCHEMA, status='RUNNING', cache=str(cache), initial=str(initial),
        source=im['source'], retargeter=im['retargeter'], retargeter_sha256=im['retargeter_sha256'],
        reference=im['reference'], seed=188, physical_gpu=a.gpu, input_sha256=hashes,
        normalization_frozen=True, frozen_pw=True, train_rows=len(train), validation_rows=len(val),
        q_updates=3000, actor_updates=1000, batch=256, q_lr=3e-4, actor_lr=1e-5,
        target_mean=target_mean, target_scale=target_scale, trust_rms_m=.00025,
        continuation_arms={'tau_only': 'PW-off', 'joint': 'PW-on'},
        training_objective='Q24 actual returns; clipped advantage-weighted proposal regression plus equal own-mean penalty',
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    write(out / 'manifest.json', manifest)
    start = time.monotonic()
    rng = np.random.default_rng(188)
    batches = rng.choice(train, (3000, 256))
    qs, qinputs, qmeaninputs, reports, evidence = {}, {}, {}, {}, {}
    high_tensor = torch.tensor(d['high'], device='cuda')
    tau_tensor = torch.tensor(d['tau'], device='cuda')
    initial_packet = torch.load(initial, map_location='cpu', weights_only=False)
    warm = load_reference_actor(initial_packet, 'cuda')
    with torch.no_grad():
        anchor = torch.cat([warm(x) for x in high_tensor.split(256)])
    # Producer uses8-row native GEMM; modern reference replay may differ below1e-6.
    if float((anchor - torch.tensor(d['mean'], device='cuda')).abs().max()) > 2e-5:
        raise ValueError('cache actor mean differs from frozen initial actor')
    train_tensor, val_tensor = torch.tensor(train, device='cuda'), torch.tensor(val, device='cuda')
    for name, use_pw in (('tau_only', False), ('joint', True)):
        xx = critic_input(d['high'], d['tau'], d['z'], use_pw)
        mean = xx[train].mean(0, dtype=np.float64).astype(np.float32)
        scale = np.maximum(xx[train].std(0, dtype=np.float64), .001).astype(np.float32)
        torch.manual_seed(188)
        q = PhysicalQ(mean, scale).cuda()
        optimizer = torch.optim.Adam(q.parameters(), lr=3e-4)
        x = torch.tensor(xx, device='cuda')
        xm = torch.tensor(critic_input(d['high'], d['mean'], d['z_mean'], use_pw), device='cuda')
        best, selected = float('inf'), None
        for update, indices in enumerate(batches, 1):
            ids = torch.tensor(indices, device='cuda')
            loss = (q(x[ids]) - y[ids]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if update % 250 == 0:
                with torch.no_grad():
                    prediction = q(x)
                    vmse = float((prediction[val_tensor] - y[val_tensor]).square().mean())
                    tmse = float((prediction[train_tensor] - y[train_tensor]).square().mean())
                if vmse < best:
                    best, selected = vmse, dict(model=copy.deepcopy(q.state_dict()), update=update)
                monitor = dict(stage='Q', arm=name, update=update, train_mse=tmse, validation_mse=vmse,
                    elapsed_s=time.monotonic() - start,
                    gpu=subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-gpu=utilization.gpu,memory.used',
                                                '--format=csv,noheader'], text=True).strip())
                with (out / 'monitor.jsonl').open('a') as stream:
                    stream.write(json.dumps(monitor) + '\n')
                if update % 1000 == 0:
                    print(json.dumps(monitor), flush=True)
                if time.monotonic() - start > 300:
                    raise TimeoutError('bounded learning exceeded300s')
        q.load_state_dict(selected['model'])
        q.eval()
        torch.save(dict(schema=SCHEMA, model={k: v.cpu() for k, v in q.state_dict().items()},
            selected_update=selected['update'], target_mean=target_mean, target_scale=target_scale,
            use_pw=use_pw), out / (name + '_q.pt'))
        with torch.no_grad():
            prediction, mean_prediction = q(x), q(xm)
        advantage = ((prediction - mean_prediction) * target_scale).cpu().numpy()
        weights, temperature = advantage_weights(advantage, d['train'])
        shuffle = np.arange(len(xx)).reshape(-1, 16)
        # Shuffle within phase and split, avoiding train-to-val feature mixing.
        for lo, hi in ((0, 12), (12, 16)):
            shuffle[:, lo:hi] = np.roll(shuffle[:, lo:hi], 1, axis=1)
        shuffled_z = d['z'][shuffle.ravel()]
        with torch.no_grad():
            shuffled = q(torch.tensor(critic_input(d['high'], d['tau'], shuffled_z, use_pw), device='cuda'))
        reports[name] = dict(selected_q_update=selected['update'], validation_normalized_mse=best,
            validation_constant_normalized_mse=float(y[val_tensor].square().mean()),
            validation_shuffled_z_mse=float((shuffled[val_tensor] - y[val_tensor]).square().mean()),
            validation_return_mse=best * target_scale**2,
            proposal_mean_Q_abs_delta=float(np.mean(np.abs(advantage))),
            train_weight_std=float(weights[train].std()), temperature=temperature)
        evidence[name + '_q_prediction'] = prediction.cpu().numpy() * target_scale + target_mean
        evidence[name + '_q_mean_prediction'] = mean_prediction.cpu().numpy() * target_scale + target_mean
        evidence[name + '_weight'] = weights
        evidence[name + '_q_shuffled_prediction'] = shuffled.cpu().numpy() * target_scale + target_mean
        qs[name], qinputs[name], qmeaninputs[name] = q, x, xm
    actor_batches = rng.choice(train, (1000, 256))
    result = dict(reports=reports, physical_screen_pass={}, checkpoint_sha256={})
    for name in ('tau_only', 'joint'):
        actor = load_reference_actor(initial_packet, 'cuda').train()
        optimizer = torch.optim.Adam(actor.parameters(), lr=1e-5)
        w = torch.tensor(evidence[name + '_weight'], device='cuda')
        first_gradient = None
        for update, indices in enumerate(actor_batches, 1):
            ids = torch.tensor(indices, device='cuda')
            prediction = actor(high_tensor[ids])
            error = ((prediction - tau_tensor[ids]) / actor.tau_scale).square().mean((1, 2))
            trust = ((prediction - anchor[ids]) / actor.tau_scale).square().mean()
            loss = (w[ids] * error).mean() + trust
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if first_gradient is None:
                first_gradient = sum(float(p.grad.abs().sum()) for p in actor.parameters() if p.grad is not None)
            optimizer.step()
        trained = copy.deepcopy(actor.state_dict())
        initial_state = warm.state_dict()
        factor = 1.
        for backtrack in range(24):
            actor.load_state_dict({k: initial_state[k] + factor * (trained[k] - initial_state[k]) for k in trained})
            with torch.no_grad():
                final_tau = torch.cat([actor(x) for x in high_tensor.split(256)])
                rms = float((final_tau - anchor).square().mean().sqrt())
            if rms <= .00025:
                break
            factor *= .5
        if rms > .00025 or not torch.isfinite(final_tau).all():
            raise ValueError('actor trust/nonfinite screen failed')
        changed = sum(float((v - initial_state[k]).abs().sum()) for k, v in actor.state_dict().items())
        for key in ('input_mean', 'input_scale', 'tau_mean', 'tau_scale'):
            if not torch.equal(actor.state_dict()[key], initial_state[key]):
                raise ValueError('actor normalization changed')
        torch.save(dict(schema=ACTOR_SCHEMA, width=actor.width, model={k: v.detach().cpu() for k, v in actor.state_dict().items()},
            selected_update=1000, backtrack_factor=factor, pw_critic=name), out / (name + '.pt'))
        result['physical_screen_pass'][name] = changed > 0 and first_gradient > 0
        result['checkpoint_sha256'][name] = sha(out / (name + '.pt'))
        reports[name].update(actor_gradient_l1=first_gradient, parameter_change_l1=changed,
            backtrack_factor=factor, trajectory_change_rms_m=rms,
            first8_change_rms_m=float((final_tau[:, :8] - anchor[:, :8]).square().mean().sqrt()),
            startup_change_rms_m=float((final_tau[torch.tensor(d['ticks'] == 0, device='cuda')] -
                anchor[torch.tensor(d['ticks'] == 0, device='cuda')]).square().mean().sqrt()))
        evidence[name + '_final_tau'] = final_tau.detach().cpu().numpy()
        evidence[name + '_trained_net_l1'] = np.array(sum(float((v - initial_state[k]).abs().sum()) for k, v in trained.items()))
        # Save unconstrained trained model for independently checking interpolation.
        torch.save({k: v.cpu() for k, v in trained.items()}, out / (name + '_before_backtrack.pt'))
    np.savez_compressed(out / 'learning.npz', **evidence)
    result['weight_arm_difference_rms'] = float(np.sqrt(np.mean((evidence['joint_weight'] - evidence['tau_only_weight'])**2)))
    result['actor_arm_difference_rms_m'] = float(np.sqrt(np.mean((evidence['joint_final_tau'] - evidence['tau_only_final_tau'])**2)))
    result['elapsed_s'] = time.monotonic() - start
    write(out / 'result.json', result)
    if any(sha(path) != digest for path, digest in hashes.items()):
        raise ValueError('learning inputs drifted')
    manifest.update(status='COMPLETED', elapsed_s=result['elapsed_s'], learning_sha256=sha(out / 'learning.npz'),
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
    write(out / 'manifest.json', manifest)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()

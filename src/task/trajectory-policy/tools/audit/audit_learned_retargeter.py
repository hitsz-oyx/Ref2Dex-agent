"""Independently reconstruct actual inverse labels, split, statistics and metrics."""
import argparse
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
ACTIVE = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15]
sys.path.insert(0, str(TASK/'src'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def reconstruct(data, starts):
    """Explicit source indexing and matrix multiplication; no fit data helpers."""
    indices = []
    for tick in range(len(data['q'])-24):
        if data['episode_start'][tick] not in starts:
            continue
        rows = np.arange(tick, tick+25)
        if not np.all(data['episode_start'][rows] == data['episode_start'][tick]):
            continue
        if not np.array_equal(data['episode_tick'][rows], data['episode_tick'][tick]+np.arange(25)):
            raise ValueError('non-contiguous actual episode ticks')
        indices.append(tick)
    indices = np.asarray(indices)
    q, dq, hand, obj, velocity = [data[key][indices] for key in ('q', 'dq', 'hand', 'obj', 'velocity')]
    rotation, center = obj[..., :3, :3], obj[..., :3, 3]
    local = (hand-center[..., None, :]) @ rotation
    state = np.concatenate((q, dq, local.reshape(len(indices), 16, 33),
        obj[..., :3, :].reshape(len(indices), 16, 12), velocity), axis=-1).astype(np.float32)
    future = np.stack([data['hand'][indices+k] for k in range(1, 25)], axis=2)
    tau = ((future-hand[:, :, None]) @ rotation[:, :, None]).reshape(len(indices), 16, 24, 33).astype(np.float32)
    action = np.stack([data['applied'][indices+k][..., ACTIVE] for k in range(8)], axis=2).astype(np.float32)
    return state, tau, action, indices


def report(prediction, truth, action_scale, controller, startup):
    normalized = np.abs(prediction-truth)/action_scale
    native_delta = prediction[:, 0].clip(-1, 1)-truth[:, 0]
    scale = np.asarray(controller['scale'])[ACTIVE].copy()
    scale[6:] *= .5
    pd_error = np.abs(native_delta*scale)
    values = dict(normalized_l1=float(normalized.mean()), first_native_mae=float(np.abs(native_delta).mean()),
        first_raw_clipping_rate=float((np.abs(prediction[:, 0]) > 1+1e-6).any(-1).mean()))
    for key, mask in (('all', np.ones(len(truth), bool)), ('startup', startup)):
        values[key] = dict(xyz_pd_p95_m=float(np.percentile(pd_error[mask, :3].max(-1), 95)),
            rotation_pd_p95_rad=float(np.percentile(pd_error[mask, 3:6].max(-1), 95)),
            fingers_pd_p95_rad=float(np.percentile(pd_error[mask, 6:].max(-1), 95)))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    args.fit, args.output = args.fit.resolve(), args.output.resolve()
    if args.output.exists() or ROOT/'outputs/trajectory-policy' not in args.output.parents:
        raise ValueError('fresh task-owned audit output required')
    before = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], text=True).strip()
    utilization, memory = map(int, before.split(','))
    if utilization > 10 or memory > 512:
        raise ValueError('audit GPU not idle: '+before)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONDONTWRITEBYTECODE='1')
    import torch
    from trajectory_policy.learned_retargeter import load_retargeter
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    args.output.mkdir(parents=True)
    started = time.monotonic()
    audit = dict(status='RUNNING', git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        fit=str(args.fit), physical_gpu=args.gpu, gpu_before=before)
    errors = {}

    def check(key, actual, expected, atol):
        actual, expected = np.asarray(actual), np.asarray(expected)
        if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise ValueError('shape/nonfinite: '+key)
        delta = float(np.max(np.abs(actual.astype(np.float64)-expected.astype(np.float64))))
        errors[key] = delta
        if delta > atol:
            raise ValueError('%s error %g exceeds %g' % (key, delta, atol))

    try:
        fit = json.loads((args.fit/'manifest.json').read_text())
        result = json.loads((args.fit/'result.json').read_text())
        if fit['status'] != 'COMPLETED' or fit['train_samples'] != 66304 or fit['validation_samples'] != 16576:
            raise ValueError('fixed completed fit required')
        if fit['schema'] == 'ref2dex.local-motion-retargeter.v1':
            from trajectory_policy.local_retargeter import load_retargeter as local_loader
            model_loader = local_loader
        elif fit['schema'] == 'ref2dex.learned-native-retargeter.v1':
            model_loader = load_retargeter
        else:
            raise ValueError('unknown fit schema')
        bound = dict(fit['input_sha256'])
        for name in ('manifest.json', 'result.json', 'validation_predictions.npz', 'monitor.jsonl', 'tau-best.pt', 'state_only-best.pt'):
            bound[str(args.fit/name)] = sha(args.fit/name)
        bound[str(Path(__file__))] = sha(__file__)
        if any(sha(path) != digest for path, digest in bound.items()):
            raise ValueError('frozen source/fit hash mismatch')
        if any('teacher' in path for path in fit['input_sha256']):
            raise ValueError('prohibited teacher packet included')
        pieces = {'train': [], 'validation': []}
        shuffled, startup_rows = [], []
        if len(fit['metadata']) != 2:
            raise ValueError('fixed two-source design differs')
        for source_index, metadata in enumerate(fit['metadata']):
            folder = Path(metadata['source'])
            with np.load(folder/'low.npz', allow_pickle=False) as f:
                data = {key: f[key] for key in ('q', 'dq', 'hand', 'obj', 'velocity', 'action', 'applied', 'pd_targets', 'episode_start', 'episode_tick')}
            check('requested_applied_%d' % source_index, data['action'], data['applied'], 0)
            check('passive_labels_%d' % source_index, data['applied'][..., [7, 9, 11, 13, 16, 17]], np.zeros_like(data['applied'][..., :6]), 0)
            if np.abs(data['applied']).max() > 1+1e-6 or np.unique(data['episode_start']).tolist() != [0, 542, 1084, 1626, 2168, 2710]:
                raise ValueError('native limits or episode layout differs')
            controller = json.loads((folder/'manifest.json').read_text())['native_controller']
            if controller != fit['native_controller'] or metadata['train_reset_starts'] != [0, 542, 1084, 1626] or metadata['validation_reset_start'] != 2168 or metadata['excluded_partial_start'] != 2710:
                raise ValueError('controller/split differs')
            pd = data['applied'].copy()
            pd[..., 6:] = (pd[..., 6:]+1)*.5
            pd = np.asarray(controller['offset'], np.float32)+np.asarray(controller['scale'], np.float32)*pd
            pd[..., :6] += data['q'][..., :6]
            for distal, parent, ratio in ((7, 6, 1.05), (9, 8, 1.05), (11, 10, 1.05), (13, 12, 1.05), (16, 15, .6), (17, 15, .8)):
                pd[..., distal] = pd[..., parent]*ratio
            check('actual_native_pd_%d' % source_index, pd, data['pd_targets'], 1e-6)
            for split, starts in (('train', [0, 542, 1084, 1626]), ('validation', [2168])):
                state, tau, action, indices = reconstruct(data, starts)
                if len(indices) != (2072 if split == 'train' else 518):
                    raise ValueError('whole-episode sample count differs')
                pieces[split].append((state.reshape(-1, 87), tau.reshape(-1, 24, 33), action.reshape(-1, 8, 12)))
                if split == 'validation':
                    ticks = np.repeat(data['episode_tick'][indices], 16)
                    check('validation_ticks_%d' % source_index, ticks, metadata['validation_ticks'], 0)
                    order = np.random.default_rng(297+source_index).permutation(16)
                    donors = np.empty(16, int)
                    donors[order] = np.roll(order, 1)
                    check('same_tick_donors_%d' % source_index, donors, metadata['validation_donor_envs'], 0)
                    if np.any(donors == np.arange(16)):
                        raise ValueError('shuffle donor equals target episode')
                    shuffled.append(tau[:, donors].reshape(-1, 24, 33))
                    startup_rows.append(ticks < 64)
        arrays = {split: [np.concatenate([part[k] for part in pieces[split]]) for k in range(3)] for split in pieces}
        if arrays['train'][0].shape != (66304, 87) or arrays['validation'][0].shape != (16576, 87):
            raise ValueError('reconstructed sample count differs')
        vx, vtau, truth = arrays['validation']
        startup = np.concatenate(startup_rows)
        models, reports = {}, {}
        with np.load(args.fit/'validation_predictions.npz', allow_pickle=False) as f:
            saved = {key: f[key] for key in f.files}
        for name in ('tau', 'state_only'):
            packet = torch.load(args.fit/(name+'-best.pt'), map_location='cpu', weights_only=False)
            if sha(args.fit/(name+'-best.pt')) != fit['checkpoint_sha256'][name] or packet['use_tau'] != (name == 'tau') or packet['git_commit'] != fit['git_commit'] or packet['schema'] != fit['schema']:
                raise ValueError('checkpoint identity/control mode differs')
            if packet['input_sha256'] != fit['input_sha256']:
                raise ValueError('checkpoint input binding differs')
            models[name] = model_loader(packet, 'cuda').eval()
            for prefix, raw, floor in zip(('state', 'tau', 'action'), arrays['train'], (.001, .001, .01)):
                for suffix, expected in (('mean', raw.mean(axis=0, dtype=np.float64)), ('scale', np.maximum(raw.std(axis=0, dtype=np.float64), floor))):
                    check(name+'_'+prefix+'_'+suffix, packet['model'][prefix+'_'+suffix].numpy(), expected, 1e-5)
            check(name+'_selected_step', packet['step'], result['selected_steps'][name], 0)
        with torch.no_grad():
            for role, name, future in (('tau', 'tau', vtau), ('state_only', 'state_only', vtau), ('shuffled', 'tau', np.concatenate(shuffled))):
                model = models[name]
                predictions = []
                for index in range(0, len(vx), 512):
                    state_tensor = torch.as_tensor(vx[index:index+512], device='cuda')
                    tau_tensor = torch.as_tensor(future[index:index+512], device='cuda')
                    raw = model(state_tensor, tau_tensor)
                    native = model.native_action(state_tensor, tau_tensor)
                    if not torch.isfinite(raw).all() or not torch.isfinite(native).all() or native.abs().max() > 1 or native[..., [7, 9, 11, 13, 16, 17]].abs().max() != 0:
                        raise ValueError('nonfinite/native/passive inference differs')
                    predictions.append(raw.cpu().numpy())
                prediction = np.concatenate(predictions)
                if role in saved:
                    check(role+'_replayed_predictions', prediction, saved[role], 2e-5)
                reports[role] = report(prediction, truth, model.action_scale.cpu().numpy(), fit['native_controller'], startup)
                processes = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True).strip()
                for process in processes.splitlines():
                    pid, used = map(int, process.split(','))
                    if pid != os.getpid() and used > 512:
                        raise RuntimeError('foreign GPU compute detected during audit')
                for key, value in reports[role].items():
                    if isinstance(value, dict):
                        for child, number in value.items():
                            check(role+'_'+key+'_'+child, number, result[role][key][child], 1e-5)
                    else:
                        check(role+'_'+key, value, result[role][key], 1e-5)
        full, only, shuffled_report = [reports[key] for key in ('tau', 'state_only', 'shuffled')]
        bounds = all(full[key]['xyz_pd_p95_m'] <= .005 and full[key]['rotation_pd_p95_rad'] <= .05 and full[key]['fingers_pd_p95_rad'] <= .05 for key in ('all', 'startup'))
        promising = full['normalized_l1'] <= .8*only['normalized_l1'] and shuffled_report['normalized_l1'] >= 1.1*full['normalized_l1'] and bounds
        screen = 'PROMISING' if promising else ('UNPROMISING' if full['normalized_l1'] >= only['normalized_l1'] or shuffled_report['normalized_l1'] <= 1.02*full['normalized_l1'] else 'UNCLEAR')
        if screen != result['status'] or bounds != result['physical_error_screen']:
            raise ValueError('fixed screen differs')
        monitor = [json.loads(line) for line in (args.fit/'monitor.jsonl').read_text().splitlines()]
        if [row['step'] for row in monitor] != [250, 500, 750, 1000, 1250, 1500]:
            raise ValueError('fixed update schedule differs')
        for name in models:
            selected = min(monitor, key=lambda row: row['validation'][name]['normalized_l1'])
            check(name+'_best_monitor_step', result['selected_steps'][name], selected['step'], 0)
        if any(sha(path) != digest for path, digest in bound.items()):
            raise ValueError('source drift during audit')
        audit.update(status='COMPLETED', audit='PASS', input_sha256=bound, errors=errors, metrics=reports,
            screen=screen, elapsed_s=time.monotonic()-started, torch_peak_bytes=torch.cuda.max_memory_allocated(),
            audited='Actual measured inputs/future-hand/applied labels; whole-reset split; FP64 independent train statistics; native PD/couplings; complete saved predictions and same-tick shuffle; fixed selection and screen. No physics or training rerun.')
        write(args.output/'manifest.json', audit)
        print(json.dumps(audit, indent=2), flush=True)
    except BaseException as error:
        audit.update(status='FAILED', error=repr(error), errors=errors, elapsed_s=time.monotonic()-started)
        write(args.output/'manifest.json', audit)
        raise


if __name__ == '__main__':
    def deadline(signum, frame):
        raise TimeoutError('learned retargeter audit exceeded120s')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(120)
    main()

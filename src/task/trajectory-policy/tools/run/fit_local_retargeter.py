"""Fixed action-aligned local-motion inverse Probe with matched state-only control."""
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
sys.path.insert(0, str(TASK/'src'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists() or ROOT/'outputs/trajectory-policy' not in args.output.parents:
        raise ValueError('fresh task-owned output required')
    before = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], text=True).strip()
    utilization, memory = map(int, before.split(','))
    if utilization > 10 or memory > 512:
        raise ValueError('GPU not idle: '+before)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONDONTWRITEBYTECODE='1')
    import torch
    from trajectory_policy.retargeter_data import ACTIVE, build_windows, valid_windows
    from trajectory_policy.local_retargeter import LocalMotionRetargeter as LearnedRetargeter, SCHEMA
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(296)
    inventory = json.loads((args.inventory/'inventory.json').read_text())
    audit_manifest = json.loads((args.inventory/'manifest.json').read_text())
    if audit_manifest['status'] != 'COMPLETED' or inventory['total_windows'] != 93376:
        raise ValueError('fixed complete data contract inventory required')
    # Do not open/use the engineering-only teacher packet as training data.
    hashes = {str(p.resolve()): sha(p) for p in (args.inventory/'inventory.json', args.inventory/'manifest.json', Path(__file__),
        TASK/'src/trajectory_policy/retargeter_data.py', TASK/'src/trajectory_policy/learned_retargeter.py', TASK/'src/trajectory_policy/local_retargeter.py')}
    pieces = {key:[] for key in ('train', 'validation', 'shuffle')}
    metadata = []
    controller = None
    for source_index, source in enumerate(inventory['sources']):
        folder = Path(source['folder'])
        for name in ('low.npz', 'manifest.json', 'result.json', 'final.pt'):
            path = folder/name
            expected = audit_manifest['input_sha256'][str(path.resolve())]
            if sha(path) != expected:
                raise ValueError('fixed inventory input drift: '+str(path))
            hashes[str(path.resolve())] = expected
        manifest = json.loads((folder/'manifest.json').read_text())
        if controller is None:
            controller = manifest['native_controller']
        elif controller != manifest['native_controller']:
            raise ValueError('native control units differ')
        with np.load(folder/'low.npz', allow_pickle=False) as f:
            data = {key:f[key].copy() for key in ('q', 'dq', 'hand', 'obj', 'velocity', 'action', 'applied', 'episode_start', 'episode_tick')}
        starts = np.unique(data['episode_start'])
        if starts.tolist() != [0, 542, 1084, 1626, 2168, 2710]:
            raise ValueError('fixed complete/partial episode layout differs')
        valid = valid_windows(data)
        train = valid[np.isin(data['episode_start'][valid], starts[:4])]
        validation = valid[data['episode_start'][valid] == starts[4]]
        if len(train) != 2072 or len(validation) != 518:
            raise ValueError('whole reset-wave split/window boundary differs')
        for split, indices in (('train', train), ('validation', validation)):
            state, tau, action = build_windows(data, indices)
            pieces[split].append((state.reshape(-1, 87), tau.reshape(-1, 24, 33), action.reshape(-1, 8, 12)))
            if split == 'validation':
                order = np.random.default_rng(297+source_index).permutation(16)
                donors = np.empty(16, int)
                donors[order] = np.roll(order, 1)
                if np.any(donors == np.arange(16)):
                    raise ValueError('shuffle must use a different source episode')
                pieces['shuffle'].append(tau[:, donors].reshape(-1, 24, 33))
                metadata.append(dict(source=str(folder), train_reset_starts=starts[:4].tolist(),
                    validation_reset_start=int(starts[4]), excluded_partial_start=int(starts[5]),
                    validation_donor_envs=donors.tolist(),
                    validation_ticks=np.repeat(data['episode_tick'][validation], 16).tolist()))
    arrays = {split: [np.concatenate([part[k] for part in pieces[split]]) for k in range(3)] for split in ('train', 'validation')}
    shuffled_tau = np.concatenate(pieces['shuffle'])
    if arrays['train'][0].shape != (66304, 87) or arrays['validation'][0].shape != (16576, 87):
        raise ValueError('fixed training/validation sample counts differ')
    if not all(np.isfinite(x).all() for parts in arrays.values() for x in parts):
        raise ValueError('nonfinite actual data')
    x, tau, target = [torch.as_tensor(value, device='cuda') for value in arrays['train']]
    vx, vtau, vtarget = [torch.as_tensor(value, device='cuda') for value in arrays['validation']]
    stau = torch.as_tensor(shuffled_tau, device='cuda')
    startup = np.concatenate([np.asarray(row['validation_ticks']) < 64 for row in metadata])
    base = LearnedRetargeter(128).cuda()
    with torch.no_grad():
        for key, value in dict(state_mean=x.mean(0), state_scale=x.std(0, unbiased=False).clamp_min(.001),
            tau_mean=tau.mean(0), tau_scale=tau.std(0, unbiased=False).clamp_min(.001),
            action_mean=target.mean(0), action_scale=target.std(0, unbiased=False).clamp_min(.01)).items():
            getattr(base, key).copy_(value)
    models = {'tau':base, 'state_only':copy.deepcopy(base)}
    models['state_only'].use_tau = False
    if any(not torch.equal(v, models['state_only'].state_dict()[k]) for k,v in base.state_dict().items()):
        raise ValueError('matched initialization differs')
    optimizers = {name:torch.optim.Adam(model.parameters(), lr=.0003) for name,model in models.items()}
    generator = torch.Generator(device='cuda').manual_seed(296)
    args.output.mkdir(parents=True)
    run = dict(status='RUNNING', schema=SCHEMA, git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        input_sha256=hashes, physical_gpu=args.gpu, gpu_before=before, seeds=[296,297,298],
        local_query='concat desired hand displacement and first differences / .01m; action slot k aligns with hand[k+1]', width=128, updates=1500, batch_size=256, learning_rate=.0003, dropout=0., grad_norm=1.,
        train_samples=len(x), validation_samples=len(vx), split='first4 complete reset waves train, fifth entire wave validation, partial excluded in both sources',
        metadata=metadata, native_controller=controller, normalization='train only; per horizon tau/action coordinates',
        loss='L1 in standardized action units; all eight actual active-action labels',
        checkpoint_selection='own lowest validation normalized L1 at fixed250-update intervals; Probe validation used for selection',
        inference='measured current s87 + hand24 displacements in current object frame -> active native command8; no future q/object/phase/force',
        claim='Offline inverse/trajectory-use mechanism only, not physical grasp or Cm utility')
    write(args.output/'manifest.json', run)
    started = time.monotonic()
    metrics, best = [], {name:float('inf') for name in models}
    best_paths = {}

    def evaluate(model, future):
        model.eval()
        outputs = []
        with torch.no_grad():
            for index in range(0, len(vx), 512):
                outputs.append(model(vx[index:index+512], future[index:index+512]).cpu().numpy())
        prediction = np.concatenate(outputs)
        truth = arrays['validation'][2]
        action_scale = model.action_scale.cpu().numpy()
        error = np.abs(prediction-truth)
        delta = (prediction.clip(-1,1)-truth)[:, 0]
        pd_scale = np.asarray(controller['scale'])[list(ACTIVE)].copy()
        pd_scale[6:] /= 2
        pd_error = np.abs(delta*pd_scale)
        report = dict(normalized_l1=float((error/action_scale).mean()),
            first_native_mae=float(np.abs(delta).mean()),
            first_raw_clipping_rate=float((np.abs(prediction[:,0]) > 1+1e-6).any(-1).mean()))
        for mask_label, mask in (('all', np.ones(len(vx),bool)), ('startup', startup)):
            report[mask_label] = dict(xyz_pd_p95_m=float(np.quantile(pd_error[mask,:3].max(-1),.95)),
                rotation_pd_p95_rad=float(np.quantile(pd_error[mask,3:6].max(-1),.95)),
                fingers_pd_p95_rad=float(np.quantile(pd_error[mask,6:].max(-1),.95)))
        return report, prediction

    try:
        for step in range(1500):
            indices = torch.randint(len(x), (256,), generator=generator, device='cuda')
            losses = {}
            for name,model in models.items():
                model.train()
                loss = ((model(x[indices], tau[indices])-target[indices])/model.action_scale).abs().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite inverse action loss')
                optimizers[name].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizers[name].step()
                losses[name] = float(loss)
            if (step+1) % 250 == 0:
                evaluated = {}
                for name,model in models.items():
                    score,_ = evaluate(model, vtau)
                    evaluated[name] = score
                    if score['normalized_l1'] < best[name]:
                        best[name] = score['normalized_l1']
                        packet = dict(schema=SCHEMA, width=128, use_tau=model.use_tau,
                            model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                            step=step+1, git_commit=run['git_commit'], input_sha256=hashes)
                        path = args.output/(name+'-u%04d.pt' % (step+1))
                        if path.exists():
                            raise ValueError('preserve existing snapshot')
                        torch.save(packet, path)
                        best_paths[name] = path
                state = subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip()
                processes = subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip()
                for process in processes.splitlines():
                    pid, used = map(int, process.split(','))
                    if pid != os.getpid() and used > 512:
                        raise RuntimeError('foreign GPU compute detected')
                record = dict(step=step+1, train_loss=losses, validation=evaluated, gpu=state,
                    elapsed_s=time.monotonic()-started, eta_s=(time.monotonic()-started)/(step+1)*(1500-step-1))
                metrics.append(record)
                print(json.dumps(record),flush=True)
                with (args.output/'monitor.jsonl').open('a') as f:
                    f.write(json.dumps(record)+'\n')
        for name,path in best_paths.items():
            (args.output/(name+'-best.pt')).symlink_to(path.name)
        from trajectory_policy.local_retargeter import load_retargeter
        selected_models = {name:load_retargeter(torch.load(args.output/(name+'-best.pt'),map_location='cuda',weights_only=False),'cuda') for name in models}
        full,prediction = evaluate(selected_models['tau'], vtau)
        only,baseline = evaluate(selected_models['state_only'], vtau)
        shuffled,_ = evaluate(selected_models['tau'], stau)
        bounds = all(full[key]['xyz_pd_p95_m'] <= .005 and full[key]['rotation_pd_p95_rad'] <= .05
            and full[key]['fingers_pd_p95_rad'] <= .05 for key in ('all','startup'))
        promising = full['normalized_l1'] <= .8*only['normalized_l1'] and shuffled['normalized_l1'] >= 1.1*full['normalized_l1'] and bounds
        status = 'PROMISING' if promising else ('UNPROMISING' if full['normalized_l1'] >= only['normalized_l1'] or shuffled['normalized_l1'] <= 1.02*full['normalized_l1'] else 'UNCLEAR')
        if any(sha(path) != digest for path,digest in hashes.items()):
            raise ValueError('frozen fit data/source drift')
        np.savez_compressed(args.output/'validation_predictions.npz', tau=prediction, state_only=baseline)
        result = dict(status=status, tau=full, state_only=only, shuffled=shuffled, physical_error_screen=bounds,
            tau_l1_ratio=full['normalized_l1']/only['normalized_l1'], shuffle_l1_ratio=shuffled['normalized_l1']/full['normalized_l1'],
            selected_steps={name:int(torch.load(args.output/(name+'-best.pt'),map_location='cpu',weights_only=False)['step']) for name in models},
            model_parameter_count=sum(p.numel() for p in base.parameters()),
            elapsed_s=time.monotonic()-started, claim=run['claim'])
        write(args.output/'result.json', result)
        run.update(status='COMPLETED', elapsed_s=result['elapsed_s'], torch_peak_bytes=torch.cuda.max_memory_allocated(),
            checkpoint_sha256={name:sha(args.output/(name+'-best.pt')) for name in models})
        write(args.output/'manifest.json',run)
        print(json.dumps(result,indent=2),flush=True)
    except BaseException as error:
        run.update(status='FAILED',elapsed_s=time.monotonic()-started,error=repr(error))
        write(args.output/'manifest.json',run)
        raise


if __name__ == '__main__':
    def deadline(signum, frame):
        raise TimeoutError('learned retargeter fit exceeded300s')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(300)
    main()

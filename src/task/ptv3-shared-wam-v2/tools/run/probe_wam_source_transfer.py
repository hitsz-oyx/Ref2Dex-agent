"""Bounded source/readout probe; reuses the completed teacher temporal split."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[5]
TASK = ROOT / 'src/task/ptv3-shared-wam-v2'
ACTIVE = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15]


def write(path, value):
    part = path.with_suffix('.json.part')
    part.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    part.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'extract', 'fit', 'summarize'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--source-name', choices=['parent', 'robot'])
    p.add_argument('--arm', choices=['none', 'parent-mean', 'robot-mean', 'parent-wrist', 'robot-wrist', 'random'])
    p.add_argument('--gpu', type=int, default=5)
    p.add_argument('--steps', type=int, default=24000)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    out = a.output.resolve()
    if out.exists() or ROOT / 'outputs/ptv3-shared-wam-v2' not in out.parents:
        raise ValueError('Fresh Task output required')
    if shutil.disk_usage(ROOT).free < 20 * 2**30:
        raise RuntimeError('Disk reserve')
    if a.stage in ['extract', 'fit']:
        occupied = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if occupied:
            raise RuntimeError('Assigned GPU occupied: ' + occupied)
        idle = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
        if sum(int(line.split(',')[1]) < 100 and int(line.split(',')[0]) != a.gpu for line in idle.splitlines()) < 2:
            raise RuntimeError('Keep two local GPUs idle')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', TMPDIR=str(ROOT / 'tmp'))
    sys.path.insert(0, str(TASK / 'src'))
    import numpy as np
    import torch
    from ptv3_shared_wam_v2.feature_transfer import FeatureControl, readouts, train_moments, random_projection
    from ptv3_shared_wam_v2.teacher_generalization import summarize
    from ptv3_shared_wam_v2.wrist_shared_probe import phase_draw
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    out.mkdir(parents=True)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('1800s stage budget')))
    signal.alarm(1800)
    status = dict(status='RUNNING', stage=a.stage, pid=os.getpid(), gpu=a.gpu,
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    write(out / 'process.json', status)
    def resource():
        consumers = set(map(int, subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
            '--query-compute-apps=pid', '--format=csv,noheader'], text=True).split()))
        if consumers - {os.getpid()}:
            raise RuntimeError('Foreign GPU consumer: ' + str(sorted(consumers - {os.getpid()})))
        value = dict(elapsed_s=time.monotonic() - started,
                     gpu=subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader'], text=True).strip())
        with (out / 'resources.jsonl').open('a') as stream:
            stream.write(json.dumps(value) + '\n')
        if shutil.disk_usage(ROOT).free < 20 * 2**30:
            raise RuntimeError('Disk reserve')
        if sum(p.stat().st_size for p in out.parent.rglob('*') if p.is_file()) > 2 * 2**30:
            raise RuntimeError('Probe artifact budget')
        return value
    try:
        if a.stage == 'prepare':
            previous = ROOT / 'outputs/ptv3-shared-wam-v2/teacher-sequence-generalization-20261016-r1'
            membership = json.loads((previous / 'membership.json').read_text())
            cache = ROOT / 'outputs/ptv3-shared-wam-v2/full-teacher-cache-20261015-r1'
            meta = json.loads((cache / 'manifest.json').read_text())
            assert meta['teacher_pd_xyz_max_mm'] < .2
            groups = {}
            for name, entry in membership['groups'].items():
                split = entry['cache_split']
                ids = np.asarray(entry['cache_indices'])
                sequence = np.load(cache / (split + '-sequence.npy'))[ids]
                ticks = np.load(cache / (split + '-tick.npy'))[ids]
                records = meta['metadata'][split]['records']
                keys = [[records[int(s)]['sequence'], int(t)] for s, t in zip(sequence, ticks)]
                q = np.load(cache / (split + '-q.npy'), mmap_mode='r')[ids]
                pd = np.load(cache / (split + '-pd.npy'), mmap_mode='r')[ids][:, ACTIVE]
                y = pd.copy()
                y[:, :6] -= q[:, :6]
                np.savez(out / (name + '.npz'), x=np.load(cache / (split + '-full.npy'), mmap_mode='r')[ids],
                         y=y, q=q, pd=pd, sequence=sequence, tick=ticks, cache_indices=ids)
                write(out / (name + '-keys.json'), keys)
                groups[name] = dict(split=split, rows=len(ids))
            rms = torch.load('/home2/wyy/oyx_ws/_external/dexplore_official_v120/checkpoint/inspire.pth', map_location='cpu', weights_only=False)['running_mean_std']
            np.savez(out / 'input-statistics.npz', mean=rms['running_mean'].float().numpy(),
                     scale=torch.sqrt(rms['running_var'].float() + 1e-5).numpy())
            write(out / 'manifest.json', dict(source=meta['source'], groups=groups, membership=str(previous / 'membership.json'),
                controller=meta['controller'], selection='exact original membership; no quality reselection'))
        elif a.stage == 'extract':
            from ptv3_shared_wam_v2.model import SharedWAMV2
            from ptv3_shared_wam_v2.robot_data import V2RLRobotWindows, robot_collate
            from ptv3_shared_wam_v2.robot_workspace import observed_workspace_mask
            from ptv3_shared_wam.loading import to_device
            from ptv3_shared_wam.action_targets import GEOMETRY_LIFTED
            packet = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
            config = packet['config']
            model = SharedWAMV2(packet['stats'], config)
            weights = packet['model']
            if any(k.startswith('forecaster.') for k in weights):
                weights = {k[len('forecaster.'):]: v for k, v in weights.items() if k.startswith('forecaster.')}
            model.load_state_dict(weights, strict=True)
            del packet, weights
            model.cuda().eval().requires_grad_(False)
            meta = json.loads((a.cache / 'manifest.json').read_text())
            checks = {}
            completed_rows = 0
            total_rows = sum(g['rows'] for g in meta['groups'].values())
            for group, entry in meta['groups'].items():
                ds = V2RLRobotWindows(Path(meta['source']), entry['split'], 32, GEOMETRY_LIFTED)
                lookup = {r['sequence']: i for i, r in enumerate(ds.records)}
                keys = json.loads((a.cache / (group + '-keys.json')).read_text())
                wanted = [(t, lookup[s]) for s, t in keys]
                assert set(wanted).issubset(set(ds.keys))
                for seq in set(s for _, s in wanted):
                    ticks = [t for t, s in wanted if s == seq]
                    data, canonical = ds.sequence(seq)
                    if not observed_workspace_mask(data, canonical, ticks, config['voxel_origin_m'], config['voxel_m']).all():
                        raise ValueError('Original panel outside encoder support; no silent filtering')
                ds.keys = wanted
                loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=robot_collate)
                stores = {mode: np.lib.format.open_memmap(out / (group + '-' + mode + '.npy'), mode='w+',
                          dtype=np.float32, shape=(len(ds), 2304)) for mode in ['mean', 'wrist']}
                cursor = 0
                with torch.inference_mode():
                    for batch in loader:
                        batch = to_device(batch, 'cuda')
                        tokens, valid, scene = model.encode(batch)
                        for mode, values in readouts(tokens, valid, scene['tokens'].shape[1]).items():
                            stores[mode][cursor:cursor + len(values)] = values.cpu().numpy()
                        cursor += len(tokens)
                        if cursor == len(tokens) or cursor % 1024 == 0:
                            timing = resource()
                            row = dict(group=group, rows=cursor, total=len(ds),
                                eta_s=timing['elapsed_s'] / (completed_rows + cursor) * (total_rows - completed_rows - cursor), **timing)
                            write(out / 'progress.json', row)
                            print(json.dumps(row), flush=True)
                assert cursor == len(ds)
                for store in stores.values():
                    store.flush()
                    assert np.isfinite(store).all()
                checks[group] = dict(rows=cursor, exact_key_order=True, workspace_passed=True)
                completed_rows += cursor
            write(out / 'manifest.json', dict(source=a.source_name, checkpoint=str(a.checkpoint.resolve()),
                config=config, groups=checks, frozen=True, grad_absent=all(p.grad is None for p in model.parameters()),
                feature_contract='scene mean + hand mean or right wrist + goal mean; all2304D'))
        elif a.stage == 'fit':
            if not a.smoke and a.steps != 24000:
                raise ValueError('Scientific fit uses fixed24000 endpoint')
            meta = json.loads((a.cache / 'data/manifest.json').read_text())
            input_stats = dict(np.load(a.cache / 'data/input-statistics.npz'))
            # Match the prior baseline's CUDA float32 RMS arithmetic exactly.
            rms = torch.load('/home2/wyy/oyx_ws/_external/dexplore_official_v120/checkpoint/inspire.pth',
                map_location='cpu', weights_only=False)['running_mean_std']
            xmean = rms['running_mean'].float().cuda()
            xscale = torch.sqrt(rms['running_var'].float().cuda() + 1e-5)
            input_stats = dict(mean=xmean.cpu().numpy(), scale=xscale.cpu().numpy())
            data = {}
            for group in meta['groups']:
                d = dict(np.load(a.cache / ('data/' + group + '.npz')))
                d['x'] = ((torch.tensor(d['x'], device='cuda') - xmean) / xscale).clamp(-5, 5)
                data[group] = d
            y = torch.tensor(data['seen']['y'], device='cuda')
            ymean, yscale = y.mean(0), y.std(0, unbiased=False).clamp_min(1e-3)
            if a.arm in ['none', 'random']:
                matrix = random_projection(data['seen']['x']) if a.arm == 'random' else None
                projection = torch.tensor(matrix, device='cuda') if matrix is not None else None
                z = {g: (d['x'] @ projection).cpu().numpy() if projection is not None else np.zeros((len(d['x']), 2304), np.float32)
                     for g, d in data.items()}
            else:
                source, mode = a.arm.split('-')
                z = {g: np.load(a.cache / ('features-' + source) / (g + '-' + mode + '.npy')) for g in data}
            zmean, zscale = train_moments(z['seen'])
            for g, d in data.items():
                d['z'] = torch.tensor((z[g] - zmean) / zscale, device='cuda')
                d['y'] = (torch.tensor(d['y'], device='cuda') - ymean) / yscale
            torch.manual_seed(180)
            model = FeatureControl().cuda()
            original = torch.load(ROOT / 'outputs/ptv3-shared-wam-v2/teacher-sequence-generalization-20261016-r1/step0.pt', map_location='cpu', weights_only=False)
            assert all(torch.equal(v.cpu(), original['model'][k]) for k, v in model.raw.state_dict().items())
            del original
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
            draw = phase_draw(data['seen']['tick'], data['seen']['sequence'], 512, np.random.default_rng(181))
            write(out / 'manifest.json', dict(arm=a.arm, steps=a.steps, engineering_only=a.smoke, seed=180,
                draw_seed=181, raw_initialization_exact=True, feature_stats_fit_group='seen',
                selection='fixed endpoint; no heldout model selection', parameters=sum(p.numel() for p in model.parameters())))
            curve = []
            def evaluate(save=False):
                reports, predictions = {}, {}
                with torch.no_grad():
                    for g, d in data.items():
                        pred = torch.cat([model(d['x'][i:i+512], d['z'][i:i+512]) for i in range(0, len(d['x']), 512)])
                        pd = (pred * yscale + ymean).cpu().numpy()
                        pd[:, :6] += d['q'][:, :6]
                        delta = pd - d['pd']
                        error = np.linalg.norm(delta[:, :3], axis=1) * 1000
                        reports[g] = summarize(error, d['sequence'])
                        for phase, mask in [('startup', d['tick'] < 16), ('later', d['tick'] >= 16)]:
                            reports[g][phase] = summarize(error[mask], d['sequence'][mask])
                        reports[g]['rotation_p95_rad'] = float(np.percentile(abs(delta[:, 3:6]).max(1), 95))
                        predictions[g + '_pd'] = pd
                        predictions[g + '_cache_indices'] = d['cache_indices']
                if save:
                    np.savez_compressed(out / 'predictions-final.npz', **predictions)
                return reports
            for step in range(a.steps + 1):
                if step:
                    if step > 12000:
                        frac = (step - 12000) / (a.steps - 12000)
                        optimizer.param_groups[0]['lr'] = 3e-6 + (3e-4 - 3e-6) * .5 * (1 + math.cos(math.pi * frac))
                    ids = draw()
                    if step == 1:
                        np.save(out / 'first-draw.npy', ids)
                    d = data['seen']
                    loss = (model(d['x'][ids], d['z'][ids]) - d['y'][ids]).abs().mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    if not torch.isfinite(loss) or not torch.isfinite(norm):
                        raise FloatingPointError('Nonfinite fit')
                    optimizer.step()
                if step % 3000 == 0 or step == a.steps:
                    timing = resource()
                    row = dict(step=step, groups=evaluate(save=step == a.steps),
                        eta_s=timing['elapsed_s'] / step * (a.steps - step) if step else None, **timing)
                    curve.append(row)
                    write(out / 'curve.json', curve)
                    print(json.dumps(row), flush=True)
            torch.save(dict(model=model.state_dict(), xmean=input_stats['mean'], xscale=input_stats['scale'],
                ymean=ymean.cpu(), yscale=yscale.cpu(), zmean=zmean, zscale=zscale,
                arm=a.arm, steps=a.steps), out / 'final.pt')
            write(out / 'result.json', dict(status='COMPLETED', arm=a.arm, final=curve[-1]))
        else:
            rows = []
            for name in ['none', 'parent-mean', 'robot-mean', 'random', 'parent-wrist', 'robot-wrist']:
                f = a.cache / ('fit-' + name) / 'result.json'
                if f.exists():
                    d = json.loads(f.read_text())
                    rows.append(dict(arm=name, **{g: v['sequence_mean_mm'] for g, v in d['final']['groups'].items()}))
            write(out / 'results.json', dict(rows=rows, scope='single-seed offline source/readout decision probe'))
            print(json.dumps(rows, indent=2))
        status['status'] = 'COMPLETED'
    except BaseException as exc:
        status.update(status='FAILED', error=repr(exc))
        raise
    finally:
        status['elapsed_s'] = time.monotonic() - started
        write(out / 'process.json', status)
        signal.alarm(0)


if __name__ == '__main__':
    main()

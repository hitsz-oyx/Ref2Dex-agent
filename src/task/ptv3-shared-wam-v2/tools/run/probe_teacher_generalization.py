"""Bounded full-teacher BC: seen states, isolated same-sequence blocks, new sequences."""
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
EXPERIMENT = 'P-20261016-teacher-sequence-generalization'


def write(path, value):
    temporary = path.with_suffix('.json.part')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--updates', type=int, default=24000)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.updates != 24000 and not args.smoke:
        raise ValueError('scientific run requires the preregistered 24000 updates')
    out, cache = args.output.resolve(), args.cache.resolve()
    if out.exists() or ROOT / 'outputs/ptv3-shared-wam-v2' not in out.parents:
        raise ValueError('fresh task output required')
    if subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
                                '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('assigned GPU occupied')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='1', TMPDIR=str(ROOT / 'tmp'))
    sys.path.insert(0, str(TASK / 'src'))
    import numpy as np
    import torch
    from ptv3_shared_wam_v2.teacher_generalization import (
        isolated_middle, assert_support_disjoint, summarize,
    )
    torch.set_num_threads(2)
    torch.manual_seed(180)
    torch.backends.cuda.matmul.allow_tf32 = True
    meta = json.loads((cache / 'manifest.json').read_text())
    if meta.get('schema') != 'ref2dex.full-teacher-input-cache.v1' or meta.get('status') != 'COMPLETED':
        raise ValueError('audited full teacher cache required')
    if meta['teacher_pd_xyz_max_mm'] > .2 or meta['future_reference_deltas'] != [1, 16]:
        raise ValueError('teacher reconstruction gate failed')
    out.mkdir(parents=True)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('1200s budget')))
    signal.alarm(1200)
    manifest = dict(status='RUNNING', experiment_id=EXPERIMENT, run_id=out.name,
                    engineering_only=args.smoke, cache=str(cache), physical_gpu=args.gpu,
                    pid=os.getpid(), git_commit=subprocess.check_output(
                        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    input='full1442 delta1+delta16; fixed teacher RMS/clamp',
                    output='relative wrist6 + absolute active fingers6; fixed live-q decoder',
                    architecture=[1442, 1024, 1024, 1024, 512, 12],
                    objects=['cup', 'mug'], seeds=[180, 181], updates=args.updates,
                    batch=512, learning_rate=3e-4, tail_start=12000, final_lr=3e-6,
                    loss='TRAIN-standardized active12 L1', weight_decay=1e-4, clip_grad=1,
                    selection='fixed final step; no heldout model selection',
                    limits=dict(seconds=1200, bytes=512_000_000, min_free_bytes=20 * 2**30),
                    split_support=dict(history=3, future=24, minimum_query_separation=28))
    write(out / 'manifest.json', manifest)
    try:
        data, membership = {}, dict(groups={}, sequences=[])
        for split in ['train', 'val']:
            sequence = np.load(cache / (split + '-sequence.npy'))
            tick = np.load(cache / (split + '-tick.npy'))
            records = meta['metadata'][split]['records']
            selected = [i for i, r in enumerate(records) if r['object'] in ['cup', 'mug']]
            ids = np.flatnonzero(np.isin(sequence, selected))
            local_sequence, local_tick = sequence[ids], tick[ids]
            fit = np.zeros(len(ids), dtype=bool)
            held = np.zeros(len(ids), dtype=bool)
            eligible = np.zeros(len(ids), dtype=bool)
            middle = np.zeros(len(ids), dtype=bool)
            for seq in selected:
                rows = np.flatnonzero(local_sequence == seq)
                ticks = local_tick[rows]
                if split == 'train':
                    keep, hold, interval = isolated_middle(ticks)
                    fit[rows], held[rows] = keep, hold
                    separation = assert_support_disjoint(ticks[keep], ticks[hold]) if interval else None
                    if interval:
                        eligible[rows] = True
                else:
                    interval, separation = None, None
                record = records[seq]
                # An additional phase-matched view; not used to filter the all-DEV result.
                middle[rows] = ((ticks >= .4 * (record['frames'] - 1)) &
                                (ticks < .6 * (record['frames'] - 1)))
                membership['sequences'].append(dict(
                    split=split, sequence=record['sequence'], object=record['object'],
                    rows=len(rows), fit_rows=int(fit[rows].sum()),
                    interpolation_rows=int(held[rows].sum()), interval=interval,
                    minimum_query_separation=separation,
                    excluded_buffer_rows=int((~fit[rows] & ~held[rows]).sum()) if split == 'train' else 0))
            groups = dict(seen=fit, interpolation=held) if split == 'train' else dict(unseen=np.ones(len(ids), dtype=bool))
            q = np.load(cache / (split + '-q.npy'))[ids].copy()
            pd = np.load(cache / (split + '-pd.npy'))[ids][:, ACTIVE].copy()
            target = pd.copy()
            target[:, :6] -= q[:, :6]
            x = np.load(cache / (split + '-full.npy'), mmap_mode='r')[ids].copy()
            data[split] = dict(x=torch.tensor(x, device='cuda'),
                               target=torch.tensor(target, device='cuda'),
                               q=q, pd=pd, tick=local_tick, sequence=local_sequence,
                               ids=ids, groups=groups, eligible=eligible, middle=middle)
            for group, mask in groups.items():
                membership['groups'][group] = dict(cache_split=split, cache_indices=ids[mask].tolist())
        train_names = {r['sequence'] for r in membership['sequences'] if r['split'] == 'train'}
        unseen_names = {r['sequence'] for r in membership['sequences'] if r['split'] == 'val'}
        if train_names & unseen_names:
            raise ValueError('sequence leakage')
        if not data['train']['groups']['interpolation'].any():
            raise ValueError('no eligible interpolation blocks')
        write(out / 'membership.json', membership)
        fit_mask = data['train']['groups']['seen']
        fit_ids = np.flatnonzero(fit_mask)
        fit_target = data['train']['target'][torch.tensor(fit_ids, device='cuda')]
        ymean, yscale = fit_target.mean(0), fit_target.std(0, unbiased=False).clamp(min=1e-3)
        rms = torch.load('/home2/wyy/oyx_ws/_external/dexplore_official_v120/checkpoint/inspire.pth',
                         map_location='cpu', weights_only=False)['running_mean_std']
        xmean = rms['running_mean'].float().cuda()
        xscale = torch.sqrt(rms['running_var'].float().cuda() + 1e-5)
        for d in data.values():
            d['x'] = ((d['x'] - xmean) / xscale).clamp(-5, 5)
            d['target'] = (d['target'] - ymean) / yscale
        layers = []
        for n, m in zip([1442, 1024, 1024, 1024], [1024, 1024, 1024, 512]):
            layers.extend([torch.nn.Linear(n, m), torch.nn.ReLU()])
        head = torch.nn.Linear(512, 12)
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
        model = torch.nn.Sequential(*layers, head).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        manifest['parameters'] = sum(p.numel() for p in model.parameters())
        manifest['target_normalization_rows'] = len(fit_ids)
        write(out / 'manifest.json', manifest)
        rng = np.random.default_rng(181)
        phases = []
        d = data['train']
        for phase_mask in [d['tick'] < 16, d['tick'] >= 16]:
            pools = [np.flatnonzero(fit_mask & phase_mask & (d['sequence'] == s))
                     for s in np.unique(d['sequence'])]
            pools = [pool for pool in pools if len(pool)]
            if not pools:
                raise ValueError('both training phase pools required')
            lengths = np.array([len(pool) for pool in pools])
            phases.append((np.concatenate(pools), np.r_[0, np.cumsum(lengths)[:-1]], lengths))

        def draw():
            batches = []
            for flat, offsets, lengths in phases:
                seq = rng.integers(len(lengths), size=256)
                local = (rng.random(256) * lengths[seq]).astype(np.int64)
                batches.append(flat[offsets[seq] + local])
            return torch.tensor(np.concatenate(batches), device='cuda')

        @torch.no_grad()
        def evaluate(save=False):
            model.eval()
            reports = {}
            predictions = {}
            for split, d in data.items():
                predicted = torch.cat([model(x) for x in d['x'].split(512)]) * yscale + ymean
                pd = predicted.cpu().numpy()
                pd[:, :6] += d['q'][:, :6]
                delta = pd - d['pd']
                error = np.linalg.norm(delta[:, :3], axis=1) * 1000
                scale = np.array(meta['controller']['scale'], dtype=np.float32)[ACTIVE]
                offset = np.array(meta['controller']['offset'], dtype=np.float32)[ACTIVE]
                raw = (pd - offset) / scale
                raw[:, :6] = (pd[:, :6] - d['q'][:, :6] - offset[:6]) / scale[:6]
                raw[:, 6:] = raw[:, 6:] * 2 - 1
                clipped = raw.clip(-1, 1)
                executed = offset + scale * np.concatenate([clipped[:, :6], (clipped[:, 6:] + 1) / 2], axis=1)
                executed[:, :6] += d['q'][:, :6]
                executable_error = np.linalg.norm(executed[:, :3] - d['pd'][:, :3], axis=1) * 1000
                for name, mask in d['groups'].items():
                    result = {}
                    for phase, subset in dict(all=mask, startup=mask & (d['tick'] < 16),
                                               later=mask & (d['tick'] >= 16),
                                               middle_phase=mask & d['middle']).items():
                        result[phase] = summarize(error[subset], d['sequence'][subset])
                    result['executable'] = summarize(executable_error[mask], d['sequence'][mask])
                    result['clipping'] = float(np.mean(np.max(abs(raw[mask] - clipped[mask]), axis=1) > 1e-6))
                    result['rotation_p95_rad'] = float(np.percentile(abs(delta[mask, 3:6]).max(1), 95))
                    result['finger_p95_rad'] = float(np.percentile(abs(delta[mask, 6:]).max(1), 95))
                    result['per_sequence'] = {
                        meta['metadata'][split]['records'][int(s)]['sequence']:
                            summarize(error[mask & (d['sequence'] == s)], d['sequence'][mask & (d['sequence'] == s)])
                        for s in np.unique(d['sequence'][mask])}
                    if name == 'seen':
                        subset = mask & d['eligible'] & (d['tick'] >= 16)
                        result['interpolation_eligible_later'] = summarize(error[subset], d['sequence'][subset])
                    reports[name] = result
                if save:
                    predictions[split + '_pd'] = pd
                    predictions[split + '_cache_indices'] = d['ids']
            if save:
                np.savez_compressed(out / 'predictions-final.npz', **predictions)
            model.train()
            return reports

        curve = []
        for step in range(args.updates + 1):
            if step:
                if step > 12000:
                    fraction = (step - 12000) / max(args.updates - 12000, 1)
                    optimizer.param_groups[0]['lr'] = 3e-6 + (3e-4 - 3e-6) * .5 * (1 + math.cos(math.pi * fraction))
                ids = draw()
                optimizer.zero_grad(set_to_none=True)
                loss = (model(data['train']['x'][ids]) - data['train']['target'][ids]).abs().mean()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(loss) or not torch.isfinite(norm):
                    raise FloatingPointError('nonfinite supervised fit')
                optimizer.step()
            if step % 1000 == 0 or step == args.updates:
                reports = evaluate(save=step == args.updates)
                row = dict(step=step, elapsed_s=time.monotonic() - started,
                           learning_rate=optimizer.param_groups[0]['lr'], groups=reports)
                curve.append(row)
                write(out / 'curve.json', curve)
                print(json.dumps(dict(step=step, elapsed_s=row['elapsed_s'],
                                      means={k: v['all']['sequence_mean_mm'] for k, v in reports.items()})), flush=True)
            if step % 500 == 0 or step == args.updates:
                consumers = set(map(int, subprocess.check_output([
                    'nvidia-smi', '-i', str(args.gpu), '--query-compute-apps=pid', '--format=csv,noheader'], text=True).split()))
                if consumers - {os.getpid()}:
                    raise RuntimeError('foreign GPU consumer')
                elapsed = time.monotonic() - started
                gpu = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
                    '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader'], text=True).strip()
                write(out / 'progress.json', dict(step=step, elapsed_s=elapsed,
                    eta_s=elapsed / max(step, 1) * (args.updates - step), gpu=gpu, pid=os.getpid()))
                if shutil.disk_usage(ROOT).free < 20 * 2**30:
                    raise RuntimeError('20GiB disk reserve')
                if sum(f.stat().st_size for f in out.iterdir() if f.is_file()) > 512_000_000:
                    raise RuntimeError('512MB artifact budget')
            if step in [0, 12000, args.updates]:
                torch.save(dict(schema='ref2dex.full-teacher-MLP.v1', step=step,
                    model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    xmean=xmean.cpu(), xscale=xscale.cpu(), ymean=ymean.cpu(), yscale=yscale.cpu(),
                    manifest=manifest), out / ('step%d.pt' % step))
        write(out / 'result.json', dict(status='COMPLETED', final=curve[-1],
            scope='offline full-input control generalization; no closed-loop grasp claim'))
        manifest.update(status='COMPLETED', elapsed_s=time.monotonic() - started)
    except BaseException as error:
        manifest.update(status='FAILED', error=repr(error))
        raise
    finally:
        write(out / 'manifest.json', manifest)
        signal.alarm(0)


if __name__ == '__main__':
    main()

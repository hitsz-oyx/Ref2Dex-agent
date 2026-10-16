"""Fixed-state coverage decision: scratch BC on cups versus all legal TRAIN."""
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
BASE = ROOT / 'outputs/ptv3-shared-wam-v2'
EXPERIMENT = 'P-20261016-teacher-state-coverage'
ACTIVE = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15]


def write(path, value):
    part = path.with_suffix(path.suffix + '.part')
    part.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    part.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arm', choices=['original', 'expanded', 'diverse', 'balanced'], required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    out = a.output.resolve()
    if out.exists() or BASE not in out.parents:
        raise ValueError('Fresh Task output required')
    if subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-compute-apps=pid',
                                '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('Assigned GPU occupied')
    idle = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                    '--format=csv,noheader,nounits'], text=True)
    if sum(int(s.split(',')[1]) < 100 and int(s.split(',')[0]) != a.gpu
           for s in idle.splitlines()) < 2:
        raise RuntimeError('Keep two other local GPUs idle')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), OMP_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='1', TMPDIR=str(ROOT / 'tmp'))
    sys.path.insert(0, str(TASK / 'src'))
    import numpy as np
    import torch
    from ptv3_shared_wam_v2.feature_transfer import FeatureControl
    from ptv3_shared_wam_v2.teacher_coverage import coverage_indices, phase_sampler, diverse_subset, balanced_sampler
    from ptv3_shared_wam_v2.teacher_generalization import summarize
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(180)
    cache = BASE / 'full-teacher-cache-20261015-r1'
    previous = BASE / 'teacher-sequence-generalization-20261016-r1'
    meta = json.loads((cache / 'manifest.json').read_text())
    membership = json.loads((previous / 'membership.json').read_text())['groups']
    if meta['status'] != 'COMPLETED' or meta['teacher_pd_xyz_max_mm'] > .2:
        raise ValueError('Audited teacher cache required')
    init = torch.load(previous / 'step0.pt', map_location='cpu', weights_only=False)
    if init['step'] != 0:
        raise ValueError('Scratch initialization required')
    data = {}
    for split in ('train', 'val'):
        data[split] = {k: np.load(cache / (split + '-' + k + '.npy'), mmap_mode='r')
                       for k in ('full', 'q', 'pd', 'sequence', 'tick')}
    seen = np.array(membership['seen']['cache_indices'])
    held = np.array(membership['interpolation']['cache_indices'])
    expanded = coverage_indices(data['train']['sequence'], data['train']['tick'], seen, held,
                                meta['metadata']['train']['records'])
    train_ids = seen if a.arm == 'original' else expanded
    if a.arm == 'diverse':
        train_ids = diverse_subset(expanded, data['train']['sequence'], len(seen))
    train_names = {r['sequence'] for r in meta['metadata']['train']['records']}
    val_names = {r['sequence'] for r in meta['metadata']['val']['records']}
    if train_names & val_names:
        raise ValueError('TRAIN/DEV sequence leakage')
    out.mkdir(parents=True)
    start = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('1200s budget')))
    signal.alarm(1200)
    manifest = dict(status='RUNNING', experiment_id=EXPERIMENT, run_id=out.name, arm=a.arm,
        engineering_only=a.smoke, cache=str(cache), membership=str(previous / 'membership.json'),
        initialization=str(previous / 'step0.pt'), physical_gpu=a.gpu, pid=os.getpid(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        work_version='git commit', train_rows=len(train_ids),
        train_sequences=len(np.unique(data['train']['sequence'][train_ids])),
        normalization='fixed original seen-only targets and original teacher RMS; no refit',
        fixed_seen_panel='Original cup training states; only partially trained in diverse arm',
        sampling='half cup / half noncup, each half startup' if a.arm == 'balanced' else 'sequence uniform / half startup',
        updates=2 if a.smoke else 24000, batch=512, seeds=[180, 181],
        optimizer='AdamW lr3e-4 wd1e-4 clip1; cosine12k->24k final3e-6',
        limits=dict(seconds=1200, bytes=512_000_000, minimum_free_bytes=20 * 2**30))
    write(out / 'manifest.json', manifest)
    np.save(out / 'train-cache-indices.npy', train_ids)
    try:
        xmean, xscale, ymean, yscale = [init[k].cuda() for k in ('xmean', 'xscale', 'ymean', 'yscale')]
        xtrain = torch.tensor(np.array(data['train']['full']), device='cuda')
        xtrain = ((xtrain - xmean) / xscale).clamp(-5, 5)
        target = np.array(data['train']['pd'][:, ACTIVE])
        target[:, :6] -= data['train']['q'][:, :6]
        ytrain = (torch.tensor(target, device='cuda') - ymean) / yscale
        model = FeatureControl().raw.cuda()
        model.load_state_dict(init['model'], strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sample = phase_sampler(train_ids, data['train']['sequence'], data['train']['tick'])
        if a.arm == 'balanced':
            sample = balanced_sampler(seen, expanded, data['train']['sequence'], data['train']['tick'])
        evaluation = {}
        for group, entry in membership.items():
            split = entry['cache_split']
            ids = np.array(entry['cache_indices'])
            x = ((torch.tensor(np.array(data[split]['full'][ids]), device='cuda') - xmean) / xscale).clamp(-5, 5)
            evaluation[group] = (split, ids, x)

        def resources(step):
            consumers = set(map(int, subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
                '--query-compute-apps=pid', '--format=csv,noheader'], text=True).split()))
            if consumers - {os.getpid()}:
                raise RuntimeError('Foreign GPU consumer')
            elapsed = time.monotonic() - start
            value = dict(step=step, elapsed_s=elapsed,
                eta_s=elapsed / max(step, 1) * (manifest['updates'] - step),
                gpu=subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
                    '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader'], text=True).strip())
            with (out / 'resources.jsonl').open('a') as stream:
                stream.write(json.dumps(value) + '\n')
            if shutil.disk_usage(ROOT).free < 20 * 2**30:
                raise RuntimeError('Disk reserve')
            if sum(f.stat().st_size for f in out.iterdir() if f.is_file()) > 512_000_000:
                raise RuntimeError('Artifact budget')
            return value

        @torch.no_grad()
        def evaluate(save=False):
            reports, arrays = {}, {}
            for group, (split, ids, x) in evaluation.items():
                prediction = (torch.cat([model(b) for b in x.split(512)]) * yscale + ymean).cpu().numpy()
                prediction[:, :6] += data[split]['q'][ids, :6]
                delta = prediction - data[split]['pd'][ids][:, ACTIVE]
                error = np.linalg.norm(delta[:, :3], axis=1) * 1000
                seq, tick = data[split]['sequence'][ids], data[split]['tick'][ids]
                result = summarize(error, seq)
                result['startup'] = summarize(error[tick < 16], seq[tick < 16])
                result['later'] = summarize(error[tick >= 16], seq[tick >= 16])
                result['rotation_p95_rad'] = float(np.percentile(abs(delta[:, 3:6]).max(1), 95))
                result['finger_p95_rad'] = float(np.percentile(abs(delta[:, 6:]).max(1), 95))
                result['per_sequence_mean_mm'] = {meta['metadata'][split]['records'][int(s)]['sequence']:
                    float(error[seq == s].mean()) for s in np.unique(seq)}
                reports[group] = result
                arrays.update({group + '_pd': prediction, group + '_cache_indices': ids})
            if save:
                np.savez_compressed(out / 'predictions-final.npz', **arrays)
                own = (torch.cat([model(xtrain[torch.tensor(b, device='cuda')])
                    for b in np.array_split(train_ids, max(1, math.ceil(len(train_ids) / 512)))])
                    * yscale + ymean).cpu().numpy()
                err = np.linalg.norm(own[:, :3] - target[train_ids, :3], axis=1) * 1000
                reports['all_fit_states'] = summarize(err, data['train']['sequence'][train_ids])
            return reports

        curve = []
        for step in range(manifest['updates'] + 1):
            if step:
                if step > 12000:
                    fraction = (step - 12000) / 12000
                    optimizer.param_groups[0]['lr'] = 3e-6 + (3e-4 - 3e-6) * .5 * (1 + math.cos(math.pi * fraction))
                ids = sample()
                if step == 1:
                    np.save(out / 'first-draw.npy', ids)
                ids = torch.tensor(ids, device='cuda')
                optimizer.zero_grad(set_to_none=True)
                loss = (model(xtrain[ids]) - ytrain[ids]).abs().mean()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(loss) or not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite training')
                optimizer.step()
            if step % 3000 == 0 or step == manifest['updates']:
                row = dict(step=step, elapsed_s=time.monotonic() - start,
                           groups=evaluate(step == manifest['updates']))
                curve.append(row)
                write(out / 'curve.json', curve)
                print(json.dumps(dict(step=step, arm=a.arm, elapsed_s=row['elapsed_s'],
                    means={g: v['sequence_mean_mm'] for g, v in row['groups'].items()})), flush=True)
            if step % 1000 == 0 or step == manifest['updates']:
                resources(step)
        torch.save(dict(schema='ref2dex.full-teacher-MLP.v1', step=step, manifest=manifest,
            model={k: v.cpu() for k, v in model.state_dict().items()},
            **{k: init[k] for k in ('xmean', 'xscale', 'ymean', 'yscale')}), out / 'final.pt')
        write(out / 'result.json', dict(status='COMPLETED', final=curve[-1], elapsed_s=time.monotonic()-start))
        manifest['status'] = 'COMPLETED'
    except BaseException as error:
        manifest.update(status='FAILED', error=repr(error))
        raise
    finally:
        write(out / 'manifest.json', manifest)
        signal.alarm(0)


if __name__ == '__main__':
    main()

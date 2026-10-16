"""Audit release-window coverage and teacher-distribution fit using legal TRAIN only."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
BASE = ROOT / 'outputs' / TASK.name


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    a = p.parse_args()
    output = a.output.resolve()
    if output.exists() or BASE not in output.parents:
        raise ValueError('Fresh Task output required')
    if subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-compute-apps=pid',
                                '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU occupied')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), TMPDIR=str(ROOT / 'tmp'), OMP_NUM_THREADS='2')
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('600s audit budget')))
    signal.alarm(600)
    started = time.monotonic()
    output.mkdir(parents=True)
    meta = json.loads((BASE / 'full-teacher-cache-20261015-r1/manifest.json').read_text())
    cache = BASE / 'full-teacher-cache-20261015-r1'
    source = Path(meta['source'])
    fit = BASE / 'teacher-state-coverage-20261016-r1'
    ids = np.load(fit / 'expanded/train-cache-indices.npy')
    if not np.array_equal(ids, np.load(fit / 'balanced/train-cache-indices.npy')):
        raise ValueError('Shared legal train membership required')
    seq = np.load(cache / 'train-sequence.npy')[ids]
    ticks = np.load(cache / 'train-tick.npy')[ids]
    records = meta['metadata']['train']['records']
    release = np.zeros(len(ids), bool)
    cups = np.zeros(len(ids), bool)
    per_sequence = []
    for index in np.unique(seq):
        record = records[index]
        if record['split'] != 'train':
            raise ValueError('TRAIN only')
        rows = np.flatnonzero(seq == index)
        with np.load(source / record['path'] / 'trajectory.npz') as data:
            reference = data['reference_current']
        if reference.shape != (record['frames'], 428):
            raise ValueError('Native Inspire reference shape')
        contact = reference[:, 167]
        off = np.flatnonzero((contact[:-1] > .5) & (contact[1:] <= .5)) + 1
        if len(off):
            delta = ticks[rows, None] - off[None, :]
            release[rows] = ((delta >= -12) & (delta < 24)).any(1)
        cups[rows] = record['object'] in ('cup', 'mug')
        per_sequence.append(dict(sequence=record['sequence'], object=record['object'],
            legal_rows=len(rows), contact_off_ticks=off.tolist(), release_rows=int(release[rows].sum())))
    phases = dict(startup=ticks < 16, release=release & (ticks >= 16), other=(ticks >= 16) & ~release)
    raw = np.load(cache / 'train-full.npy', mmap_mode='r')[ids]
    pd = np.load(cache / 'train-pd.npy', mmap_mode='r')[ids]
    q = np.load(cache / 'train-q.npy', mmap_mode='r')[ids]
    target = pd[:, [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15]].copy()
    target[:, :6] -= q[:, :6]
    results = {}
    for arm in ('expanded', 'balanced'):
        packet = torch.load(fit / arm / 'final.pt', map_location='cpu', weights_only=False)
        layers = []
        for i in (0, 2, 4, 6, 8):
            w = packet['model'][str(i) + '.weight']
            layers.append(torch.nn.Linear(w.shape[1], w.shape[0]))
            if i < 8:
                layers.append(torch.nn.ReLU())
        model = torch.nn.Sequential(*layers).cuda().eval()
        model.load_state_dict(packet['model'], strict=True)
        norm = {k: packet[k].cuda() for k in ('xmean', 'xscale', 'ymean', 'yscale')}
        with torch.inference_mode():
            x = ((torch.tensor(raw, device='cuda') - norm['xmean']) / norm['xscale']).clamp(-5, 5)
            prediction = (torch.cat([model(v) for v in x.split(512)]) * norm['yscale'] + norm['ymean']).cpu().numpy()
        xyz = np.linalg.norm(prediction[:, :3] - target[:, :3], axis=-1) * 1000
        fingers = abs(prediction[:, 6:] - target[:, 6:]).mean(-1)
        result = {}
        for group, mask in [('all', np.ones(len(ids), bool)), ('cup_mug', cups)]:
            result[group] = {}
            for phase, pmask in phases.items():
                keep = mask & pmask
                sids = np.unique(seq[keep])
                result[group][phase] = dict(rows=int(keep.sum()), sequences=len(sids),
                    xyz_mean_mm=float(xyz[keep].mean()), xyz_p95_mm=float(np.percentile(xyz[keep], 95)),
                    xyz_sequence_mean_mm=float(np.mean([xyz[keep & (seq == s)].mean() for s in sids])),
                    finger_l1_rad=float(fingers[keep].mean()))
        results[arm] = result
    report = dict(status='COMPLETED', class_='Decision', training_performed=False,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        torch_version=torch.__version__, physical_gpu=a.gpu, elapsed_s=time.monotonic() - started,
        rows=len(ids), sequences=per_sequence, results=results,
        scope='TRAIN-only phase description; no claim that a contact-off event is necessarily table placement')
    (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(elapsed_s=report['elapsed_s'], results=results), indent=2))
    signal.alarm(0)


if __name__ == '__main__':
    main()
